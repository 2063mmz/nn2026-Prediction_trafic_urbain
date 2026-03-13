from pathlib import Path
import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True)
    df["hour_of_day"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek  # 0 = lundi
    df["day_of_month"] = t.dt.day
    df["month"] = t.dt.month
    df["year"] = t.dt.year
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

    if weather_index_name in weather_df.columns:
        weather_df = weather_df.set_index(weather_index_name)
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)
    weather_df = weather_df.reset_index()
    weather_df["hour"] = weather_df["time"].dt.floor("h")

    traffic_df = traffic_df.copy()
    traffic_df[traffic_time_col] = pd.to_datetime(traffic_df[traffic_time_col], utc=True)
    traffic_df["hour"] = traffic_df[traffic_time_col].dt.floor("h")

    merged = traffic_df.merge(
        weather_df.drop(columns=["time"]),
        on="hour",
        how="left",
    )
    return merged


def build_dataset(
    traffic_path: str | Path,
    weather_path: str | Path,
    output_path: str | Path | None = None,
    traffic_time_col: str = "hour",
) -> pd.DataFrame:
    traffic = pd.read_csv(traffic_path)
    weather = pd.read_csv(weather_path)

    if "time" not in weather.columns and weather.index.name is None:
        if "Unnamed: 0" in weather.columns:
            weather = weather.rename(columns={"Unnamed: 0": "time"})
        else:
            raise ValueError("Le CSV météo doit avoir une colonne 'time' ou un index datetime.")

    merged = merge_traffic_weather(traffic, weather, traffic_time_col=traffic_time_col)
    merged = add_temporal_features(merged, traffic_time_col)

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
