from pathlib import Path
import json
import sys
import importlib

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "dataset_traffic_weather.csv"
LSTM_PATH = ROOT / "data" / "model_lstm.pt"
LSTM_META_PATH = ROOT / "data" / "model_lstm_meta.json"
LSTM_SCALER_X_PATH = ROOT / "data" / "scaler_lstm_x.joblib"
LSTM_SCALER_Y_PATH = ROOT / "data" / "scaler_lstm_y.joblib"
HGB_PATH = ROOT / "data" / "model_hgb.joblib"
HGB_META_PATH = ROOT / "data" / "model_hgb_meta.json"
GRU_PATH = ROOT / "data" / "model_gru.pt"
GRU_META_PATH = ROOT / "data" / "model_gru_meta.json"
GRU_SCALER_X_PATH = ROOT / "data" / "scaler_gru_x.joblib"
GRU_SCALER_Y_PATH = ROOT / "data" / "scaler_gru_y.joblib"

LOCAL_TZ = "Europe/Paris"
TRAFFIC_API = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-multimodal-comptages/records"

try:
    from fetch_weather import fetch_weather_window
except ImportError:
    from src.fetch_weather import fetch_weather_window


def ajouter_variables_temporelles(df: pd.DataFrame, time_col: str = "hour") -> pd.DataFrame:
    """Ajoute les mêmes variables temporelles que celles utilisées pendant l'entraînement."""
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["hour_of_day"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["day_of_month"] = t.dt.day
    df["month"] = t.dt.month
    df["year"] = t.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


@st.cache_data
def load_data(path: str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hour" in df.columns:
        df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=900)
def fetch_live_traffic(id_site: str, limit: int = 100) -> pd.DataFrame:
    """
    Télécharge les dernières observations d'un site depuis Paris Open Data,
    puis agrège par heure pour retrouver le même format que le pipeline local.
    Essaie plusieurs syntaxes de filtre pour éviter les erreurs ODSQL.
    """
    candidate_params = []

    if str(id_site).isdigit():
        candidate_params.append({
            "select": "id_site,t,nb_usagers",
            "where": f"id_site = {id_site}",
            "order_by": "t desc",
            "limit": limit,
        })

    candidate_params.append({
        "select": "id_site,t,nb_usagers",
        "where": f"id_site = '{str(id_site)}'",
        "order_by": "t desc",
        "limit": limit,
    })

    last_error = None

    for params in candidate_params:
        try:
            response = requests.get(TRAFFIC_API, params=params, timeout=60)
            if not response.ok:
                last_error = f"HTTP {response.status_code} - {response.text[:500]}"
                continue

            payload = response.json()
            rows = payload.get("results", [])
            if not rows:
                continue

            df = pd.DataFrame(rows)
            if "id_site" not in df.columns:
                continue

            time_col = None
            for col in ["t", "hour", "date", "datetime"]:
                if col in df.columns:
                    time_col = col
                    break

            value_col = None
            for col in ["nb_usagers", "count", "volume"]:
                if col in df.columns:
                    value_col = col
                    break

            if time_col is None or value_col is None:
                continue

            df = df[["id_site", time_col, value_col]].copy()
            df["id_site"] = df["id_site"].astype(str)
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df = df.dropna(subset=[time_col, value_col])

            if df.empty:
                continue

            df["hour"] = df[time_col].dt.floor("h")
            out = (
                df.groupby(["id_site", "hour"], as_index=False)[value_col]
                .sum()
                .rename(columns={value_col: "nb_usagers"})
                .sort_values("hour")
                .reset_index(drop=True)
            )
            return out

        except Exception as e:
            last_error = str(e)

    raise RuntimeError(
        f"Impossible de récupérer le trafic en ligne pour id_site={id_site}. "
        f"Dernière erreur API : {last_error}"
    )


@st.cache_resource
def load_lstm_bundle():
    if not (LSTM_PATH.exists() and LSTM_META_PATH.exists() and LSTM_SCALER_X_PATH.exists() and LSTM_SCALER_Y_PATH.exists()):
        return None
    import torch
    module = importlib.import_module("train_lstm")
    with open(LSTM_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("feature_names", [])
    seq_len = int(meta.get("seq_len", 24))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.LSTMModel(
        input_size=len(feats),
        hidden_size=meta.get("hidden_size", 64),
        num_layers=meta.get("num_layers", 2),
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()
    return {
        "torch": torch,
        "model": model,
        "meta": meta,
        "feats": feats,
        "seq_len": seq_len,
        "device": device,
        "scaler_x": joblib.load(LSTM_SCALER_X_PATH),
        "scaler_y": joblib.load(LSTM_SCALER_Y_PATH),
    }


@st.cache_resource
def load_gru_bundle():
    if not (GRU_PATH.exists() and GRU_META_PATH.exists() and GRU_SCALER_X_PATH.exists() and GRU_SCALER_Y_PATH.exists()):
        return None
    import torch
    module = importlib.import_module("train_gru")
    with open(GRU_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("feature_names", [])
    seq_len = int(meta.get("seq_len", 24))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.GRUModel(
        input_size=len(feats),
        hidden_size=meta.get("hidden_size", 128),
        num_layers=meta.get("num_layers", 2),
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(GRU_PATH, map_location=device))
    model.eval()
    return {
        "torch": torch,
        "model": model,
        "meta": meta,
        "feats": feats,
        "seq_len": seq_len,
        "device": device,
        "scaler_x": joblib.load(GRU_SCALER_X_PATH),
        "scaler_y": joblib.load(GRU_SCALER_Y_PATH),
    }


@st.cache_resource
def load_hgb_bundle():
    if not (HGB_PATH.exists() and HGB_META_PATH.exists()):
        return None
    with open(HGB_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(HGB_PATH)
    return {"model": model, "meta": meta}


def get_baseline_predictor(df: pd.DataFrame):
    if df is None or df.empty or "nb_usagers" not in df.columns:
        return None
    if "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
        base = ajouter_variables_temporelles(df, "hour")
    else:
        base = df.copy()
    return base.groupby(["id_site", "hour_of_day", "day_of_week"], as_index=False)["nb_usagers"].mean()


def predict_baseline(agg: pd.DataFrame, id_site: str, target_dt_utc: pd.Timestamp) -> float:
    if agg is None or agg.empty:
        return float("nan")
    hour = int(target_dt_utc.hour)
    weekday = int(target_dt_utc.dayofweek)
    m = agg[(agg["id_site"].astype(str) == str(id_site)) & (agg["hour_of_day"] == hour) & (agg["day_of_week"] == weekday)]
    if m.empty:
        site_mean = agg[agg["id_site"].astype(str) == str(id_site)]["nb_usagers"].mean()
        return float(site_mean) if not np.isnan(site_mean) else float("nan")
    return float(m["nb_usagers"].iloc[0])


def build_online_state(id_site: str, target_dt_utc: pd.Timestamp, history_hours: int = 120):
    """
    Construit une table horaire continue contenant:
    - le trafic observé récent,
    - la météo historique/prévisionnelle,
    - les variables temporelles.
    """
    traffic = fetch_live_traffic(id_site)
    if traffic.empty:
        return None, None

    latest_obs = traffic["hour"].max().floor("h")
    start_dt_utc = latest_obs - pd.Timedelta(hours=history_hours - 1)
    end_dt_utc = max(latest_obs, pd.Timestamp(target_dt_utc).tz_convert("UTC").floor("h"))
    full_hours = pd.date_range(start=start_dt_utc, end=end_dt_utc, freq="h", tz="UTC")

    state = pd.DataFrame({"hour": full_hours})
    state["id_site"] = str(id_site)
    traffic = traffic[(traffic["hour"] >= start_dt_utc) & (traffic["hour"] <= latest_obs)].copy()
    state = state.merge(traffic, on=["id_site", "hour"], how="left")
    state = state.sort_values("hour").reset_index(drop=True)
    weather = fetch_weather_window(start_dt_utc, end_dt_utc)
    if not weather.empty:
        weather = weather.reset_index().rename(columns={weather.index.name or "index": "hour"})
        state = state.merge(weather, on="hour", how="left")

    state = ajouter_variables_temporelles(state, "hour")
    return state.sort_values("hour").reset_index(drop=True), latest_obs


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def predict_lstm_next_hour(state: pd.DataFrame, latest_obs: pd.Timestamp, bundle: dict):
    state = ensure_columns(state, bundle["feats"])
    hist = state[state["hour"] <= latest_obs].copy().sort_values("hour")
    if len(hist) < bundle["seq_len"]:
        return None, latest_obs + pd.Timedelta(hours=1)
    tail = hist.tail(bundle["seq_len"])
    X = tail[bundle["feats"]].fillna(0).values.astype(np.float32)
    X = bundle["scaler_x"].transform(X.reshape(-1, len(bundle["feats"]))).reshape(1, bundle["seq_len"], len(bundle["feats"]))
    with bundle["torch"].no_grad():
        x_t = bundle["torch"].from_numpy(X).to(bundle["device"])
        out = bundle["model"](x_t).cpu().numpy()
    pred = float(bundle["scaler_y"].inverse_transform(out)[0, 0])
    return max(pred, 0.0), latest_obs + pd.Timedelta(hours=1)


def predict_hgb_recursive(state: pd.DataFrame, latest_obs: pd.Timestamp, target_dt_utc: pd.Timestamp, bundle: dict):
    meta = bundle["meta"]
    model = bundle["model"]
    site_col = meta.get("site_col", "id_site")
    n_lags = int(meta.get("n_lags", 24))
    feat_cols = meta.get("feature_cols", [])
    time_cols = meta.get("time_feature_cols", [])
    weather_cols = meta.get("weather_feature_cols", [])

    state = ensure_columns(state, feat_cols + time_cols + weather_cols + [site_col, "nb_usagers"])
    current = latest_obs.floor("h")
    target_dt_utc = pd.Timestamp(target_dt_utc).tz_convert("UTC").floor("h")

    while current < target_dt_utc:
        next_time = current + pd.Timedelta(hours=1)
        hist_vals = state.loc[state["hour"] <= current, "nb_usagers"].dropna().tail(n_lags).astype(float).to_numpy()
        if len(hist_vals) < n_lags:
            return None

        row_match = state[state["hour"] == next_time]
        row = {site_col: str(state[site_col].iloc[0])}
        for lag in range(1, n_lags + 1):
            row[f"lag_{lag}"] = float(hist_vals[-lag])
        row["lag_mean_24"] = float(hist_vals.mean())
        row["lag_std_24"] = float(hist_vals.std(ddof=0))
        row["lag_last"] = float(hist_vals[-1])
        if not row_match.empty:
            for c in time_cols + weather_cols:
                row[c] = row_match.iloc[0][c] if c in row_match.columns else np.nan

        X = pd.DataFrame([row])
        X = ensure_columns(X, feat_cols)
        pred = float(model.predict(X[feat_cols])[0])
        pred = max(pred, 0.0)
        state.loc[state["hour"] == next_time, "nb_usagers"] = pred
        current = next_time

    final_pred = float(state.loc[state["hour"] == target_dt_utc, "nb_usagers"].iloc[0])
    return max(final_pred, 0.0)


def predict_gru_recursive(state: pd.DataFrame, latest_obs: pd.Timestamp, target_dt_utc: pd.Timestamp, bundle: dict):
    feats = bundle["feats"]
    seq_len = int(bundle["seq_len"])
    state = ensure_columns(state, feats + ["nb_usagers"])
    current = latest_obs.floor("h")
    target_dt_utc = pd.Timestamp(target_dt_utc).tz_convert("UTC").floor("h")

    while current < target_dt_utc:
        seq = state[state["hour"] <= current].sort_values("hour").tail(seq_len).copy()
        if len(seq) < seq_len:
            return None
        X = seq[feats].fillna(0).values.astype(np.float32).reshape(1, seq_len, len(feats))
        Xs = bundle["scaler_x"].transform(X.reshape(-1, len(feats))).reshape(1, seq_len, len(feats))
        with bundle["torch"].no_grad():
            x_t = bundle["torch"].from_numpy(Xs).to(bundle["device"])
            out = bundle["model"](x_t).cpu().numpy().ravel()[0]
        pred = float(bundle["scaler_y"].inverse_transform(np.array([[out]], dtype=np.float32))[0, 0])
        pred = max(pred, 0.0)
        next_time = current + pd.Timedelta(hours=1)
        state.loc[state["hour"] == next_time, "nb_usagers"] = pred
        current = next_time

    final_pred = float(state.loc[state["hour"] == target_dt_utc, "nb_usagers"].iloc[0])
    return max(final_pred, 0.0)


def main():
    st.set_page_config(page_title="Prédiction trafic Paris", page_icon="🚦", layout="wide")
    st.title("🚦 Prédiction du trafic urbain à Paris")
    st.caption("Comparaison de modèles : LSTM en ligne (heure suivante), HGB et GRU avec horizon libre jusqu'à 3 jours.")

    data_file = st.sidebar.file_uploader("Fichier dataset local (CSV)", type=["csv"], help="Optionnel : dataset_traffic_weather.csv pour la liste des sites et la baseline.")
    if data_file is not None:
        df_local = pd.read_csv(data_file)
        if "hour" in df_local.columns:
            df_local["hour"] = pd.to_datetime(df_local["hour"], utc=True, errors="coerce")
    else:
        df_local = load_data(str(DATA_PATH))

    if df_local is None or df_local.empty or "id_site" not in df_local.columns:
        st.warning("Aucun dataset local disponible. Déposez `dataset_traffic_weather.csv` pour afficher les sites et la baseline.")
        return

    df_local = df_local.copy()
    df_local["id_site"] = df_local["id_site"].astype(str).str.strip()

    # Ne garder que les identifiants de site valides : uniquement des chiffres
    valid_site_mask = df_local["id_site"].str.fullmatch(r"\d+")
    nb_invalid_sites = int((~valid_site_mask).sum())
    if nb_invalid_sites > 0:
        st.sidebar.warning(f"{nb_invalid_sites} lignes avec id_site invalide ont été ignorées dans l'interface.")

    df_local = df_local[valid_site_mask].copy()

    sites = sorted(df_local["id_site"].dropna().unique().tolist())
    if not sites:
        st.error("Aucun site disponible dans le dataset local après filtrage des id_site invalides.")
        return

    available_models = ["Baseline"]
    lstm_bundle = load_lstm_bundle()
    if lstm_bundle is not None:
        available_models.append("LSTM en ligne")
    hgb_bundle = load_hgb_bundle()
    if hgb_bundle is not None:
        available_models.append("HGB")
    gru_bundle = load_gru_bundle()
    if gru_bundle is not None:
        available_models.append("GRU")

    st.sidebar.subheader("Paramètres")
    id_site = st.sidebar.selectbox("Site de comptage", sites, index=0)
    model_name = st.sidebar.radio("Modèle", available_models, index=0)

    with st.spinner("Chargement de l'historique récent du site..."):
        try:
            live_traffic = fetch_live_traffic(id_site)
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API Paris Open Data : {e}")
            st.info("Vérifiez la connexion réseau, la syntaxe du filtre id_site, ou utilisez temporairement la baseline / le dataset local.")
            return

    if live_traffic.empty:
        st.error("Impossible de récupérer l'historique récent pour ce site depuis l'API Paris Open Data.")
        return

    latest_obs = live_traffic["hour"].max().floor("h")
    latest_obs_local = latest_obs.tz_convert(LOCAL_TZ)
    st.sidebar.caption(f"Dernière heure observée : {latest_obs_local.strftime('%Y-%m-%d %H:%M')} ({LOCAL_TZ})")

    if model_name == "LSTM en ligne":
        target_dt_utc = latest_obs + pd.Timedelta(hours=1)
        st.sidebar.info("Le LSTM n'est autorisé qu'en mode temps réel : prévision de l'heure suivante à partir du dernier historique disponible.")
    else:
        min_dt_local = (latest_obs + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ)
        max_dt_local = (latest_obs + pd.Timedelta(days=3)).tz_convert(LOCAL_TZ)
        default_dt_local = min_dt_local
        pred_date = st.sidebar.date_input(
            "Date cible",
            value=default_dt_local.date(),
            min_value=min_dt_local.date(),
            max_value=max_dt_local.date(),
        )
        pred_hour = st.sidebar.number_input("Heure cible (0–23)", min_value=0, max_value=23, value=int(default_dt_local.hour), step=1)
        target_dt_local = pd.Timestamp(pred_date.year, pred_date.month, pred_date.day, int(pred_hour), tz=LOCAL_TZ)
        target_dt_utc = target_dt_local.tz_convert("UTC").floor("h")
        if target_dt_utc < latest_obs + pd.Timedelta(hours=1):
            st.error("La date cible doit être strictement postérieure à la dernière observation disponible.")
            return
        if target_dt_utc > latest_obs + pd.Timedelta(days=3):
            st.error("L'horizon maximal autorisé est de 3 jours après la dernière observation disponible.")
            return

    history_hours = 168 if model_name in {"HGB", "GRU"} else 72
    with st.spinner("Construction de la fenêtre en ligne (trafic + météo)..."):
        state, latest_from_state = build_online_state(id_site, target_dt_utc, history_hours=history_hours)
    if state is None or latest_from_state is None:
        st.error("Impossible de construire la fenêtre en ligne pour la prédiction.")
        return

    baseline_agg = get_baseline_predictor(df_local)
    pred_value = np.nan
    note = ""

    if model_name == "Baseline":
        pred_value = predict_baseline(baseline_agg, id_site, target_dt_utc)
        note = "Baseline calculée sur le dataset local historique."
    elif model_name == "LSTM en ligne":
        pred_value, target_dt_utc = predict_lstm_next_hour(state, latest_from_state, lstm_bundle)
        note = "Le LSTM utilise l'historique récent et prédit uniquement l'heure suivante."
    elif model_name == "HGB":
        pred_value = predict_hgb_recursive(state, latest_from_state, target_dt_utc, hgb_bundle)
        if pred_value is None or np.isnan(pred_value):
            pred_value = predict_baseline(baseline_agg, id_site, target_dt_utc)
            note = "Le HGB n'avait pas assez d'historique récent ; fallback vers la baseline."
        else:
            note = "Le HGB utilise les lags récents et la météo/temps de l'heure cible."
    elif model_name == "GRU":
        pred_value = predict_gru_recursive(state, latest_from_state, target_dt_utc, gru_bundle)
        note = "Le GRU procède de manière récursive jusqu'à l'horizon demandé."

    if pred_value is None or np.isnan(pred_value):
        st.error("La prédiction a échoué pour ce modèle et cet horizon.")
        return

    target_local = target_dt_utc.tz_convert(LOCAL_TZ)
    st.metric("Prédiction de flux (nb_usagers)", f"{pred_value:.0f}")
    st.caption(f"Modèle utilisé : **{model_name}**")
    st.caption(f"Heure cible : **{target_local.strftime('%Y-%m-%d %H:%M')}** ({LOCAL_TZ})")
    st.caption(note)

    st.subheader("Historique récent du site")
    if not live_traffic.empty:
        hist_chart = live_traffic.sort_values("hour").tail(168).copy()
        hist_chart["hour_local"] = hist_chart["hour"].dt.tz_convert(LOCAL_TZ)
        st.line_chart(hist_chart[["hour_local", "nb_usagers"]].set_index("hour_local"))

    st.subheader("Dernières observations utilisées")
    preview = state[state["hour"] <= latest_from_state].tail(24).copy()
    preview["hour_local"] = preview["hour"].dt.tz_convert(LOCAL_TZ)
    cols = [c for c in ["hour_local", "nb_usagers", "temperature_2m", "precipitation", "cloud_cover"] if c in preview.columns]
    st.dataframe(preview[cols], use_container_width=True)


if __name__ == "__main__":
    main()
