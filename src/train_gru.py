"""
Entraînement d'un modèle GRU pour la prédiction horaire du trafic.
Le script prépare des séquences chronologiques par site, entraîne le modèle,
évalue les performances sur le jeu de test et sauvegarde les artefacts
nécessaires pour la réutilisation dans l'application Streamlit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("PyTorch requis : pip install torch")
    raise SystemExit(1)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_traffic_weather.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = MODEL_DIR / "model_gru.pt"
META_PATH = MODEL_DIR / "model_gru_meta.json"
SCALER_X_PATH = MODEL_DIR / "scaler_gru_x.joblib"
SCALER_Y_PATH = MODEL_DIR / "scaler_gru_y.joblib"
SEQ_LEN = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
TRAIN_RATIO = 0.8
FEATURE_COLS = [
    "nb_usagers",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "hour_of_day", "day_of_week", "month", "is_weekend",
]
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
    "weather_code", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "cloud_cover",
]
TARGET_COL = "nb_usagers"
SITE_COL = "id_site"
TIME_COL = "hour"


class GRUModel(nn.Module):
    """Architecture GRU simple pour la prédiction d'une valeur de trafic à partir d'une séquence horaire."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """Initialise les couches du réseau GRU et la couche linéaire de sortie."""
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Définit la propagation avant du réseau et renvoie la prédiction associée à la dernière étape temporelle."""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def main():
    """Prépare les séquences, entraîne le modèle GRU, évalue les performances puis sauvegarde les artefacts nécessaires."""
    if not DATASET_PATH.exists():
        print(f"Dataset non trouvé : {DATASET_PATH}")
        print("Exécutez d'abord : process_traffic_data → fetch_weather → build_dataset")
        return

    # 1) Chargement du dataset fusionné puis tri chronologique.
    df = pd.read_csv(DATASET_PATH)
    if TARGET_COL not in df.columns:
        print(f"Colonne cible absente : {TARGET_COL}")
        return
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df[SITE_COL] = df[SITE_COL].astype(str)
    df = df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)

    all_feats = [c for c in FEATURE_COLS + WEATHER_COLS if c in df.columns]
    if TARGET_COL not in all_feats:
        all_feats = [TARGET_COL] + [c for c in all_feats if c != TARGET_COL]

    # 2) Construction des séquences glissantes pour le modèle GRU.
    X_list, y_list, t_list = [], [], []
    for _, g in df.groupby(SITE_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        arr = g[all_feats].fillna(0).values.astype(np.float32)
        targets = g[TARGET_COL].values.astype(np.float32)
        times = pd.to_datetime(g[TIME_COL], utc=True)
        for i in range(SEQ_LEN, len(arr)):
            X_list.append(arr[i - SEQ_LEN:i])
            y_list.append(targets[i])
            t_list.append(times.iloc[i])

    if not X_list:
        print("Aucune séquence générée (pas assez de données par site).")
        return

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    t = pd.to_datetime(pd.Series(t_list), utc=True)
    order = np.argsort(t.values)
    X = X[order]
    y = y[order]
    t = t.iloc[order].reset_index(drop=True)

    # 3) Séparation chronologique entre apprentissage et test.
    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(X_train) == 0 or len(X_test) == 0:
        print("Split train/test invalide.")
        return

    # 4) Normalisation des variables d'entrée et de la cible.
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler_x.fit(X_train_flat)
    scaler_y.fit(y_train.reshape(-1, 1))
    X_train_s = scaler_x.transform(X_train_flat).reshape(X_train.shape)
    X_test_s = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(len(all_feats), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = None
    best_state = None
    patience = 5
    wait = 0
    X_test_t = torch.from_numpy(X_test_s).to(device)

    # 5) Boucle d'entraînement avec early stopping simple.
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pred_test_s = model(X_test_t).cpu().numpy().ravel()
        pred_test = scaler_y.inverse_transform(pred_test_s.reshape(-1, 1)).ravel()
        test_mae = float(np.abs(pred_test - y_test).mean())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("  Epoch {}  loss = {:.4f}  test MAE = {:.2f}".format(epoch + 1, total_loss / max(len(train_loader), 1), test_mae))

        if best_loss is None or test_mae < best_loss:
            best_loss = test_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping à l'epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_test_s = model(X_test_t).cpu().numpy().ravel()
    pred_test = np.clip(scaler_y.inverse_transform(pred_test_s.reshape(-1, 1)).ravel(), 0, None)
    mae = float(np.abs(y_test - pred_test).mean())
    rmse = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
    r2 = float(1 - np.sum((y_test - pred_test) ** 2) / (np.sum((y_test - y_test.mean()) ** 2) + 1e-12))
    mape = float(np.mean(np.abs((y_test - pred_test) / (np.abs(y_test) + 1e-6))) * 100)
    rel_err = np.abs(y_test - pred_test) / (np.abs(y_test) + 1e-6)
    acc10 = float(np.mean(rel_err <= 0.10) * 100)
    acc20 = float(np.mean(rel_err <= 0.20) * 100)

    print("Métriques GRU (test) : MAE = {:.2f}  RMSE = {:.2f}  R² = {:.4f}  MAPE = {:.2f} %".format(mae, rmse, r2, mape))
    print("Accuracy (within 10%) = {:.2f} %   Accuracy (within 20%) = {:.2f} %".format(acc10, acc20))

    # 6) Sauvegarde du modèle, des scalers et des métadonnées.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "GRU",
            "dataset_path": str(DATASET_PATH),
            "feature_names": all_feats,
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "train_ratio": TRAIN_RATIO,
            "train_sequences": int(len(X_train)),
            "test_sequences": int(len(X_test)),
            "metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
                "accuracy_within_10": acc10,
                "accuracy_within_20": acc20,
            },
        }, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({TIME_COL: t.iloc[split:].astype(str).tolist(), TARGET_COL: y_test, "prediction_gru": pred_test})
    pred_df.to_csv(MODEL_DIR / "predictions_gru.csv", index=False)


if __name__ == "__main__":
    main()
