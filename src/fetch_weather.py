"""
Utilitaires de récupération de la météo horaire via Open-Meteo.
Ce module permet de télécharger l'historique, la prévision, puis de construire
une fenêtre météo continue compatible avec le pipeline de préparation des données
et avec l'application de démonstration.
"""
from pathlib import Path
from datetime import date
import requests
import pandas as pd

PARIS_LAT = 48.8566
PARIS_LON = 2.3522
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
DEFAULT_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

def _normalize_hourly(hourly: list[str] | None) -> list[str]:
    """Normalise la liste des variables météo à demander à l'API et supprime les doublons éventuels."""
    if hourly is None:
        return DEFAULT_HOURLY.copy()
    return list(dict.fromkeys(hourly))


def weather_response_to_dataframe(data: dict) -> pd.DataFrame:
    """Convertit la réponse JSON d'Open-Meteo en DataFrame indexé par heure UTC."""
    hourly = data.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time")
    return df.sort_index()

def fetch_weather(
    start_date: str | date,
    end_date: str | date,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> dict:
    # Télécharge la météo historique horaire via l'API archive
    # Préparation des paramètres envoyés à l'API Open-Meteo
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(_normalize_hourly(hourly)),
        "timezone": "GMT",
    }
    r = requests.get(ARCHIVE_API, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_forecast_weather(
    start_date: str | date,
    end_date: str | date,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> dict:
    # Télécharge la météo prévisionnelle horaire via l'API forecast
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(_normalize_hourly(hourly)),
        "timezone": "GMT",
    }
    r = requests.get(FORECAST_API, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_weather_window(
    start_dt,
    end_dt,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> pd.DataFrame:
    """
    Construit une fenêtre météo horaire unique en combinant:
    - l'archive pour la partie passée,
    - la prévision pour aujourd'hui et le futur.
    Les timestamps retournés sont en UTC et indexés par heure.
    """
    hourly = _normalize_hourly(hourly)
    start_dt = pd.Timestamp(start_dt)
    end_dt = pd.Timestamp(end_dt)
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize("UTC")
    else:
        start_dt = start_dt.tz_convert("UTC")
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize("UTC")
    else:
        end_dt = end_dt.tz_convert("UTC")

    start_dt = start_dt.floor("h")
    end_dt = end_dt.floor("h")
    if end_dt < start_dt:
        return pd.DataFrame()

    today_utc = pd.Timestamp.now(tz="UTC").floor("D")
    yesterday_utc = today_utc - pd.Timedelta(days=1)
    # Concaténation progressive de la partie archive et de la partie prévision
    frames: list[pd.DataFrame] = []

    archive_end = min(end_dt, yesterday_utc + pd.Timedelta(hours=23))
    if start_dt <= archive_end:
        data_arch = fetch_weather(start_dt.date(), archive_end.date(), latitude, longitude, hourly)
        df_arch = weather_response_to_dataframe(data_arch)
        if not df_arch.empty:
            frames.append(df_arch)

    forecast_start = max(start_dt, today_utc)
    if end_dt >= forecast_start:
        data_fc = fetch_forecast_weather(forecast_start.date(), end_dt.date(), latitude, longitude, hourly)
        df_fc = weather_response_to_dataframe(data_fc)
        if not df_fc.empty:
            frames.append(df_fc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


def fetch_and_save_weather(
    start_date: str | date,
    end_date: str | date,
    output_path: str | Path,
    latitude: float = PARIS_LAT,
    longitude: float = PARIS_LON,
    hourly: list[str] | None = None,
) -> pd.DataFrame:
    # Télécharge la météo historique, la convertit en DataFrame puis l'enregistre au format CSV
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
    print(f"Enregistré : {len(df)} lignes -> {args.output}")
    print(df.head())
