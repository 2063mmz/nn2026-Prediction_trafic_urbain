from pathlib import Path
import pandas as pd

COLUMN_ALIASES = {
    "date": "t",
    "datetime": "t",
    "timestamp": "t",
    "time": "t",
    "site_id": "id_site",
    "site": "id_site",
    "count": "nb_usagers",
    "volume": "nb_usagers",
    "nb": "nb_usagers",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for col in df.columns:
        c = col.strip()
        if c in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[c]
        elif c not in ("id_site", "t", "nb_usagers", "mode", "label", "voie", "sens", "trajectoire", "lat", "lon", "id_trajectoire"):
            pass
    if rename:
        df = df.rename(columns=rename)
    return df


def load_traffic_csv(
    path: str | Path,
    *,
    chunksize: int | None = None,
    required_cols: tuple[str, ...] = ("id_site", "t", "nb_usagers"),
) -> pd.DataFrame:
   
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")

    if chunksize is None:
        df = pd.read_csv(path)
    else:
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            parts.append(chunk)
        df = pd.concat(parts, ignore_index=True)

    df = normalize_columns(df)
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Le CSV doit contenir la colonne '{c}' (ou un alias dans COLUMN_ALIASES). "
                f"Colonnes trouvées: {list(df.columns)}"
            )
    return df


def parse_time(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["hour"] = df[time_col].dt.floor("h")
    return df


def drop_invalid(df: pd.DataFrame, time_col: str = "t", value_col: str = "nb_usagers") -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=[time_col, value_col])
    df = df[df[value_col] >= 0]
    if len(df) < before:
        print(f"  Lignes supprimées (t ou {value_col} invalides): {before - len(df)}")
    return df


def aggregate_by_site_hour(
    df: pd.DataFrame,
    value_col: str = "nb_usagers",
    site_col: str = "id_site",
    hour_col: str = "hour",
) -> pd.DataFrame:
    return df.groupby([site_col, hour_col], as_index=False)[value_col].sum()


def aggregate_by_site_hour_mode(
    df: pd.DataFrame,
    value_col: str = "nb_usagers",
    site_col: str = "id_site",
    hour_col: str = "hour",
    mode_col: str = "mode",
) -> pd.DataFrame:
    if mode_col not in df.columns:
        return aggregate_by_site_hour(df, value_col, site_col, hour_col)
    return df.groupby([site_col, hour_col, mode_col], as_index=False)[value_col].sum()


def process_traffic_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    by_mode: bool = False,
    selected_sites: list[str] | None = None,
    chunksize: int | None = 100_000,
) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {input_path}")

    print(f"Lecture du CSV: {input_path}")
    df = load_traffic_csv(input_path, chunksize=chunksize)

    df["id_site"] = df["id_site"].astype(str).str.strip()
    if selected_sites:
        sites_set = {str(s).strip() for s in selected_sites}
        df = df[df["id_site"].isin(sites_set)]
        print(f"  Filtre sites: {len(sites_set)} sites")

    df = parse_time(df)
    df = drop_invalid(df)

    if by_mode:
        ts = aggregate_by_site_hour_mode(df)
    else:
        ts = aggregate_by_site_hour(df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ts.to_csv(output_path, index=False)
        print(f"Écrit: {output_path} ({len(ts)} lignes)")

    return ts


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CSV trafic -> série temporelle par site et par heure")
    p.add_argument("--input", default="data/traffic_raw_cleaned.csv", help="CSV des comptages")
    p.add_argument("--output", default="data/traffic_timeseries.csv", help="CSV de sortie (série temporelle)")
    p.add_argument("--by-mode", action="store_true", help="Agréger aussi par mode (vélos, véhicules, etc.)")
    p.add_argument("--sites", nargs="*", help="Ne garder que ces id_site (ex: 10022 10023 10077)")
    p.add_argument("--no-chunks", action="store_true", help="Tout charger en mémoire (si fichier petit)")
    args = p.parse_args()

    ts = process_traffic_csv(
        args.input,
        output_path=args.output,
        by_mode=args.by_mode,
        selected_sites=args.sites or None,
        chunksize=None if args.no_chunks else 100_000,
    )
    print(ts.head(10))
