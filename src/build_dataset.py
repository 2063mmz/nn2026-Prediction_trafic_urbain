"""
Construction du dataset final trafic + météo.
Ce script fusionne la série temporelle de trafic avec les données météo,
ajoute les variables temporelles cycliques et exporte un fichier prêt à être
utilisé par les scripts d'entraînement.
"""
from pathlib import Path
import pandas as pd
import numpy as np

LOCAL_TZ = "Europe/Paris"

def add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ajoute les variables temporelles classiques et cycliques à partir de la colonne horaire."""
    df = df.copy()
    t_utc = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    t_local = t_utc.dt.tz_convert(LOCAL_TZ)
    df[time_col] = t_utc
    df["hour_of_day"] = t_local.dt.hour
    df["day_of_week"] = t_local.dt.dayofweek
    df["day_of_month"] = t_local.dt.day
    df["month"] = t_local.dt.month
    df["year"] = t_local.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def merge_traffic_weather(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    traffic_time_col: str = "hour",
    weather_index_name: str = "time",
) -> pd.DataFrame:
    """Fusionne les données de trafic et de météo sur une granularité horaire commune."""
    weather_df = weather_df.copy()
    if weather_index_name in weather_df.columns:
        weather_df = weather_df.set_index(weather_index_name)
    weather_df.index = pd.to_datetime(weather_df.index, utc=True, errors="coerce")
    weather_df = weather_df[~weather_df.index.isna()].reset_index()
    weather_df["hour"] = weather_df[weather_index_name].dt.floor("h")

    traffic_df = traffic_df.copy()
    traffic_df[traffic_time_col] = pd.to_datetime(traffic_df[traffic_time_col], utc=True, errors="coerce")
    traffic_df = traffic_df.dropna(subset=[traffic_time_col])
    traffic_df["hour"] = traffic_df[traffic_time_col].dt.floor("h")

    merged = traffic_df.merge(weather_df.drop(columns=[weather_index_name]), on="hour", how="left")
    return merged


def build_dataset(
    traffic_path: str | Path,
    weather_path: str | Path,
    output_path: str | Path | None = None,
    traffic_time_col: str = "hour",
) -> pd.DataFrame:
    # Chargement des deux sources à fusionner.
    """Construit le dataset final en lisant les fichiers source, en les fusionnant et en exportant le résultat si demandé."""
    traffic = pd.read_csv(traffic_path)
    weather = pd.read_csv(weather_path)

    if "time" not in weather.columns and weather.index.name is None:
        if "Unnamed: 0" in weather.columns:
            weather = weather.rename(columns={"Unnamed: 0": "time"})
        else:
            raise ValueError("Le CSV météo doit avoir une colonne 'time' ou un index datetime.")

    # Fusion horaire entre le trafic et la météo.
    merged = merge_traffic_weather(traffic, weather, traffic_time_col=traffic_time_col)
    # Ajout des variables temporelles exploitées par les modèles.
    merged = add_temporal_features(merged, "hour")

    sort_cols = [c for c in ["id_site", "hour"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fusion trafic + météo + variables temporelles")
    parser.add_argument("--traffic", default="data/traffic_timeseries.csv")
    parser.add_argument("--weather", default="data/weather_paris.csv")
    parser.add_argument("--output", default="data/dataset_traffic_weather.csv")
    args = parser.parse_args()

    df = build_dataset(args.traffic, args.weather, args.output)
    print(f"Dataset: {len(df)} lignes, {len(df.columns)} colonnes -> {args.output}")
    print(df.columns.tolist())
    print(df.head())
