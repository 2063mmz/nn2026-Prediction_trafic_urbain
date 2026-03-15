"""
Entraînement d'un modèle HistGradientBoosting pour la prédiction du trafic.
Ce script construit un jeu supervisé à partir des lags, des variables
temporelles et des variables météo, puis entraîne un modèle tabulaire
servant de baseline forte pour la comparaison avec les modèles neuronaux.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_traffic_weather.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = MODEL_DIR / "model_hgb.joblib"
META_PATH = MODEL_DIR / "model_hgb_meta.json"
TARGET_COL = "nb_usagers"
SITE_COL = "id_site"
TIME_COL = "hour"
N_LAGS = 24
TRAIN_RATIO = 0.8
TIME_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "hour_of_day", "day_of_week", "day_of_month", "month", "year", "is_weekend",
]
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
    "weather_code", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "cloud_cover",
]


def main():
    """Prépare les variables supervisées à partir du dataset fusionné, entraîne le modèle HGB puis sauvegarde le modèle et les métadonnées."""
    if not DATASET_PATH.exists():
        print(f"Jeu de données introuvable : {DATASET_PATH}")
        print("Exécutez d'abord : process_traffic_data -> fetch_weather -> build_dataset")
        return

    # 1) Chargement du dataset déjà fusionné (trafic + météo + variables temporelles).
    df = pd.read_csv(DATASET_PATH)
    if TARGET_COL not in df.columns:
        print(f"Colonne cible absente : {TARGET_COL}")
        return

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df[SITE_COL] = df[SITE_COL].astype(str)
    df = df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)

    usable_time_feats = [c for c in TIME_FEATURES if c in df.columns]
    usable_weather = [c for c in WEATHER_COLS if c in df.columns]

    # 2) Construction d'un jeu supervisé tabulaire avec des lags de trafic.
    rows = []
    for sid, g in df.groupby(SITE_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        vals = g[TARGET_COL].astype(float)
        for i in range(N_LAGS, len(g)):
            row = {
                SITE_COL: sid,
                TIME_COL: g.loc[i, TIME_COL],
                TARGET_COL: float(vals.iloc[i]),
                "lag_mean_24": float(vals.iloc[i - N_LAGS:i].mean()),
                "lag_std_24": float(vals.iloc[i - N_LAGS:i].std(ddof=0)),
                "lag_last": float(vals.iloc[i - 1]),
            }
            for lag in range(1, N_LAGS + 1):
                row[f"lag_{lag}"] = float(vals.iloc[i - lag])
            for c in usable_time_feats + usable_weather:
                row[c] = g.loc[i, c] if c in g.columns else np.nan
            rows.append(row)

    if not rows:
        print("Aucune ligne supervisée n'a été générée.")
        return

    # 3) Découpage chronologique train/test pour éviter les fuites temporelles.
    sup = pd.DataFrame(rows).sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(sup) * TRAIN_RATIO)
    train_df = sup.iloc[:split_idx].copy()
    test_df = sup.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        print("Découpage train/test invalide.")
        return

    feature_cols = [c for c in sup.columns if c not in [TARGET_COL, TIME_COL]]
    lag_cols = [f"lag_{lag}" for lag in range(1, N_LAGS + 1)] + ["lag_mean_24", "lag_std_24", "lag_last"]
    cat_cols = [SITE_COL]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # 4) Définition du pipeline de prétraitement puis du modèle HGB.
    model = Pipeline([
        ("preproc", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ])),
        ("reg", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )),
    ])

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(float).values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(float).values

    # 5) Entraînement du modèle et évaluation sur le jeu de test.
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else float("nan")
    mape = float(np.mean(np.abs((y_test - pred) / (np.abs(y_test) + 1e-6))) * 100)
    rel_err = np.abs(y_test - pred) / (np.abs(y_test) + 1e-6)
    acc10 = float(np.mean(rel_err <= 0.10) * 100)
    acc20 = float(np.mean(rel_err <= 0.20) * 100)

    print("Métriques HGB (test) : MAE = {:.2f} | RMSE = {:.2f} | R² = {:.4f} | MAPE = {:.2f} %".format(mae, rmse, r2, mape))
    print("Accuracy à 10 % = {:.2f} % | Accuracy à 20 % = {:.2f} %".format(acc10, acc20))

    # 6) Sauvegarde du modèle entraîné et des métadonnées utiles pour l'inférence.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "HistGradientBoostingRegressor",
            "dataset_path": str(DATASET_PATH),
            "target_col": TARGET_COL,
            "site_col": SITE_COL,
            "time_col": TIME_COL,
            "n_lags": N_LAGS,
            "feature_cols": feature_cols,
            "lag_feature_cols": lag_cols,
            "time_feature_cols": usable_time_feats,
            "weather_feature_cols": usable_weather,
            "categorical_feature_cols": cat_cols,
            "train_ratio": TRAIN_RATIO,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
                "accuracy_within_10": acc10,
                "accuracy_within_20": acc20,
            },
        }, f, ensure_ascii=False, indent=2)

    pred_out = test_df[[SITE_COL, TIME_COL, TARGET_COL]].copy()
    pred_out["prediction_hgb"] = pred
    pred_out.to_csv(MODEL_DIR / "predictions_hgb.csv", index=False)


if __name__ == "__main__":
    main()
