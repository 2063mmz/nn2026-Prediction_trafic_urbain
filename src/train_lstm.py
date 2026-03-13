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
LSTM_PATH = MODEL_DIR / "model_lstm.pt"
LSTM_META_PATH = MODEL_DIR / "model_lstm_meta.json"
LSTM_SCALER_X_PATH = MODEL_DIR / "scaler_lstm_x.joblib"
LSTM_SCALER_Y_PATH = MODEL_DIR / "scaler_lstm_y.joblib"

SEQ_LEN = 24
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

FEATURE_COLS = [
    "nb_usagers",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "hour_of_day", "day_of_week", "month",
]
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "weather_code",
    "wind_speed_10m", "cloud_cover",
]
TARGET_COL = "nb_usagers"
SITE_COL = "id_site"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_sequences(df, site_col, feature_cols, target_col, seq_len):
    X_list, y_list = [], []
    for sid, g in df.groupby(site_col, sort=False):
        g = g.sort_values("hour").reset_index(drop=True)
        arr = g[feature_cols].fillna(0).values.astype(np.float32)
        targets = g[target_col].values.astype(np.float32)
        for i in range(seq_len, len(arr)):
            X_list.append(arr[i - seq_len : i])
            y_list.append(targets[i])
    if not X_list:
        return None, None
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset non trouvé : {DATASET_PATH}")
        print("Exécutez d'abord : process_traffic_data → fetch_weather → build_dataset")
        return

    df = pd.read_csv(DATASET_PATH)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df[SITE_COL] = df[SITE_COL].astype(str)

    all_feats = [c for c in FEATURE_COLS + WEATHER_COLS if c in df.columns]
    if TARGET_COL not in all_feats:
        all_feats = [TARGET_COL] + [c for c in all_feats if c != TARGET_COL]
    if TARGET_COL not in df.columns:
        print(f"Colonne cible '{TARGET_COL}' absente.")
        return

    df = df.sort_values([SITE_COL, "hour"]).reset_index(drop=True)
    X, y = build_sequences(df, SITE_COL, all_feats, TARGET_COL, SEQ_LEN)
    if X is None:
        print("Aucune séquence générée (pas assez de données par site).")
        return

    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler_x.fit(X_train_flat)
    scaler_y.fit(y_train.reshape(-1, 1))

    X_train_s = scaler_x.transform(X_train_flat).reshape(X_train.shape)
    X_val_s = scaler_x.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train_s).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(
        input_size=len(all_feats),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(X_val_s).to(device)
            val_pred = model(val_x).cpu().numpy().ravel()
        val_pred_inv = scaler_y.inverse_transform(val_pred.reshape(-1, 1)).ravel()
        val_mae = np.abs(val_pred_inv - y_val).mean()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("  Epoch {}  loss = {:.4f}  val MAE = {:.2f}".format(epoch + 1, total_loss / len(train_loader), val_mae))

    mae = float(np.abs(val_pred_inv - y_val).mean())
    rmse = float(np.sqrt(np.mean((val_pred_inv - y_val) ** 2)))
    r2 = float(1 - np.sum((y_val - val_pred_inv) ** 2) / (np.sum((y_val - y_val.mean()) ** 2) + 1e-12))
    mape = float(np.mean(np.abs((y_val - val_pred_inv) / (np.abs(y_val) + 1e-6))) * 100)
    rel_err = np.abs(y_val - val_pred_inv) / (np.abs(y_val) + 1e-6)
    accuracy_10 = float(np.mean(rel_err <= 0.10) * 100)
    accuracy_20 = float(np.mean(rel_err <= 0.20) * 100)
    print("Métriques (validation) : MAE = {:.2f}  RMSE = {:.2f}  R² = {:.4f}  MAPE = {:.2f} %".format(mae, rmse, r2, mape))
    print("Accuracy (within 10%) = {:.2f} %   Accuracy (within 20%) = {:.2f} %".format(accuracy_10, accuracy_20))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LSTM_PATH)
    joblib.dump(scaler_x, LSTM_SCALER_X_PATH)
    joblib.dump(scaler_y, LSTM_SCALER_Y_PATH)
    with open(LSTM_META_PATH, "w") as f:
        json.dump({
            "feature_names": all_feats,
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "accuracy_within_10": accuracy_10, "accuracy_within_20": accuracy_20},
        }, f, indent=2)

if __name__ == "__main__":
    main()
