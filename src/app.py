from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset_traffic_weather.csv"
LSTM_PATH = Path(__file__).resolve().parent / "data" / "model_lstm.pt"
LSTM_META_PATH = Path(__file__).resolve().parent / "data" / "model_lstm_meta.json"
LSTM_SCALER_X_PATH = Path(__file__).resolve().parent / "data" / "scaler_lstm_x.joblib"
LSTM_SCALER_Y_PATH = Path(__file__).resolve().parent / "data" / "scaler_lstm_y.joblib"


@st.cache_data
def load_data(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df


def get_baseline_predictor(df: pd.DataFrame):
    if df is None or "nb_usagers" not in df.columns:
        return None
    return df.groupby(["id_site", "hour_of_day", "day_of_week"], as_index=False)["nb_usagers"].mean()


def predict_baseline(agg: pd.DataFrame, id_site: str, hour: int, weekday: int) -> float:
    if agg is None or agg.empty:
        return np.nan
    m = agg[(agg["id_site"].astype(str) == str(id_site)) & (agg["hour_of_day"] == hour) & (agg["day_of_week"] == weekday)]
    if m.empty:
        return agg[agg["id_site"].astype(str) == str(id_site)]["nb_usagers"].mean()
    return float(m["nb_usagers"].iloc[0])


def predict_lstm(df_site, pred_dt, feats, seq_len, model, scaler_x, scaler_y, device):
    df_site = df_site.sort_values("hour").reset_index(drop=True)
    df_site = df_site[df_site["hour"] < pred_dt]
    if len(df_site) < seq_len:
        return None
    tail = df_site.tail(seq_len)
    X = tail[feats].fillna(0).values.astype(np.float32)
    X = scaler_x.transform(X.reshape(-1, len(feats))).reshape(1, seq_len, len(feats))
    with __import__("torch").no_grad():
        x_t = __import__("torch").from_numpy(X).to(device)
        out = model(x_t).cpu().numpy()
    return float(scaler_y.inverse_transform(out)[0, 0])


def main():
    st.set_page_config(page_title="Prédiction trafic Paris", page_icon="🚦", layout="wide")
    st.title("🚦 Prédiction du trafic urbain à Paris")
    st.caption("Projet NN2026 — Données comptages multimodaux + météo Open-Meteo")

    data_file = st.sidebar.file_uploader("Fichier dataset (CSV)", type=["csv"], help="Optionnel : dataset_traffic_weather.csv")
    if data_file is not None:
        df = pd.read_csv(data_file)
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
    else:
        df = load_data(DATA_PATH)

    if df is None or df.empty:
        st.warning("Aucun jeu de données chargé. Déposez un CSV ou placez `data/dataset_traffic_weather.csv`.")
        st.info("Pipeline : process_traffic_data → fetch_weather → build_dataset")
        return

    sites = sorted(df["id_site"].astype(str).unique().tolist())
    if not sites:
        st.error("Aucun site (id_site) dans le dataset.")
        return

    st.sidebar.subheader("Paramètres de prédiction")
    id_site = st.sidebar.selectbox("Site de comptage", sites, index=0)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pred_date = st.date_input("Date")
    with col2:
        pred_hour = st.number_input("Heure (0–23)", min_value=0, max_value=23, value=12, step=1)

    weekday = pred_date.weekday()
    baseline_agg = get_baseline_predictor(df)
    pred_value = np.nan
    model_label = "baseline"

    use_lstm = LSTM_PATH.exists() and LSTM_META_PATH.exists() and LSTM_SCALER_X_PATH.exists() and LSTM_SCALER_Y_PATH.exists()
    if use_lstm:
        try:
            import torch
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from src.train_lstm import LSTMModel
            with open(LSTM_META_PATH) as f:
                meta = json.load(f)
            feats = meta.get("feature_names", [])
            seq_len = meta.get("seq_len", 24)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMModel(
                input_size=len(feats),
                hidden_size=meta.get("hidden_size", 64),
                num_layers=meta.get("num_layers", 2),
                dropout=0.2,
            ).to(device)
            model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
            model.eval()
            scaler_x = __import__("joblib").load(LSTM_SCALER_X_PATH)
            scaler_y = __import__("joblib").load(LSTM_SCALER_Y_PATH)
            site_df = df[df["id_site"].astype(str) == str(id_site)].copy()
            pred_dt = pd.Timestamp(pred_date.year, pred_date.month, pred_date.day, pred_hour, tz="UTC")
            pred_value = predict_lstm(site_df, pred_dt, feats, seq_len, model, scaler_x, scaler_y, device)
            if pred_value is not None:
                model_label = "LSTM"
            else:
                pred_value = predict_baseline(baseline_agg, id_site, pred_hour, weekday)
                st.sidebar.caption("LSTM : pas assez d’historique → baseline.")
        except Exception as e:
            pred_value = predict_baseline(baseline_agg, id_site, pred_hour, weekday)
            st.sidebar.caption("LSTM invalide → baseline.")

    if model_label == "baseline" and not use_lstm:
        pred_value = predict_baseline(baseline_agg, id_site, pred_hour, weekday)
        st.sidebar.caption("LSTM non trouvé → prédiction baseline (moyenne historique).")

    st.metric("Prédiction flux (nb_usagers)", f"{pred_value:.0f}" if not np.isnan(pred_value) else "—")
    st.caption("Modèle utilisé : **" + model_label + "**")

    site_df = df[df["id_site"].astype(str) == str(id_site)].copy()
    if len(site_df) > 0:
        site_df = site_df.sort_values("hour")
        st.subheader("Historique récent (ce site)")
        st.line_chart(site_df[["hour", "nb_usagers"]].tail(168).set_index("hour"))

    st.sidebar.divider()
    st.sidebar.markdown("**Modèle** : LSTM (`python src/train_lstm.py`) ; sinon baseline.")


if __name__ == "__main__":
    main()
