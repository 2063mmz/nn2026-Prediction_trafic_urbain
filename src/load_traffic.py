from pathlib import Path

try:
    from .process_traffic_data import process_traffic_csv, load_traffic_csv, parse_time, aggregate_by_site_hour, aggregate_by_site_hour_mode
except ImportError:
    from process_traffic_data import process_traffic_csv, load_traffic_csv, parse_time, aggregate_by_site_hour, aggregate_by_site_hour_mode


def build_time_series(
    input_path: str | Path,
    output_path: str | Path | None = None,
    by_mode: bool = False,
    selected_sites: list[str] | None = None,
    chunksize: int | None = 100_000,
):
    return process_traffic_csv(
        input_path,
        output_path=output_path,
        by_mode=by_mode,
        selected_sites=selected_sites,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CSV trafic -> série temporelle")
    p.add_argument("--input", default="data/traffic_raw_cleaned.csv")
    p.add_argument("--output", default="data/traffic_timeseries.csv")
    p.add_argument("--by-mode", action="store_true")
    p.add_argument("--sites", nargs="*")
    p.add_argument("--no-chunks", action="store_true")
    args = p.parse_args()
    ts = build_time_series(
        args.input,
        output_path=args.output,
        by_mode=args.by_mode,
        selected_sites=args.sites or None,
        chunksize=None if args.no_chunks else 100_000,
    )
    print(f"Lignes: {len(ts)}")
    print(ts.head(10))
