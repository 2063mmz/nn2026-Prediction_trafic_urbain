from pathlib import Path
from datetime import datetime, date
import requests
import pandas as pd

PARIS_LAT = 48.8566
PARIS_LON = 2.3522
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    start_date: str | date,
    end_date: str | date,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> dict:

    if hourly is None:
        hourly = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(hourly),
    }
    r = requests.get(ARCHIVE_API, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def weather_response_to_dataframe(data: dict) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    return df


def fetch_and_save_weather(
    start_date: str | date,
    end_date: str | date,
    output_path: str | Path,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> pd.DataFrame:
    data = fetch_weather(start_date, end_date, latitude, longitude, hourly)
    df = weather_response_to_dataframe(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Récupération météo Open-Meteo pour Paris")
    parser.add_argument("--start", required=True, help="Date début YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Date fin YYYY-MM-DD")
    parser.add_argument("--output", default="data/weather_paris.csv", help="CSV de sortie")
    parser.add_argument("--lat", type=float, default=PARIS_LAT, help="Latitude")
    parser.add_argument("--lon", type=float, default=PARIS_LON, help="Longitude")
    args = parser.parse_args()

    df = fetch_and_save_weather(args.start, args.end, args.output, args.lat, args.lon)
    print(f"Enregistré: {len(df)} lignes -> {args.output}")
    print(df.head())
