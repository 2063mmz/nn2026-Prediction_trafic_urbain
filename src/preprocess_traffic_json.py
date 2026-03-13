from pathlib import Path
import csv
import json
import sys
from collections import deque

try:
    import ijson
except ImportError:
    ijson = None


def _stream_json_objects_no_ijson(file_handle):
    buf = ""
    in_array = False
    depth = 0
    start = -1
    for block in iter(lambda: file_handle.read(65536), ""):
        buf += block
        i = 0
        while i < len(buf):
            c = buf[i]
            if not in_array:
                if c == "[":
                    in_array = True
                i += 1
                continue
            if depth == 0:
                if c == "{":
                    depth = 1
                    start = i
                elif c == "]":
                    return
                i += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = buf[start : i + 1]
                    buf = buf[i + 1 :]
                    i = -1
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
            elif c == '"':
                i += 1
                while i < len(buf) and buf[i] != '"':
                    if buf[i] == "\\":
                        i += 1
                    i += 1
            i += 1
    if depth != 0 and start >= 0:
        try:
            yield json.loads(buf[start:])
        except json.JSONDecodeError:
            pass


def stream_json_objects(path: Path):
    path = Path(path)
    if ijson is not None:
        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                yield obj
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield from _stream_json_objects_no_ijson(f)


OUT_COLUMNS = [
    "id_trajectoire",
    "id_site",
    "label",
    "t",
    "mode",
    "nb_usagers",
    "voie",
    "sens",
    "trajectoire",
    "lat",
    "lon",
]


def _safe_str(v):
    if v is None:
        return ""
    return str(v).strip()


def _safe_int(v):
    if v is None:
        return 0
    try:
        x = int(v)
        return max(0, x)
    except (TypeError, ValueError):
        return 0


def process_record(raw: dict) -> dict:
    coords = raw.get("coordonnees_geo")
    if isinstance(coords, dict):
        lat = coords.get("lat")
        lon = coords.get("lon")
        lat_str = "" if lat is None else str(lat)
        lon_str = "" if lon is None else str(lon)
    else:
        lat_str = ""
        lon_str = ""

    return {
        "id_trajectoire": _safe_str(raw.get("id_trajectoire")),
        "id_site": _safe_str(raw.get("id_site")),
        "label": _safe_str(raw.get("label")),
        "t": _safe_str(raw.get("t")),
        "mode": _safe_str(raw.get("mode")),
        "nb_usagers": _safe_int(raw.get("nb_usagers")),
        "voie": _safe_str(raw.get("voie")),
        "sens": _safe_str(raw.get("sens")),
        "trajectoire": _safe_str(raw.get("trajectoire")),
        "lat": lat_str,
        "lon": lon_str,
    }


def stream_json_to_csv(
    json_path: str | Path,
    csv_path: str | Path,
    *,
    selected_sites: list[str] | None = None,
    skip_invalid_time: bool = True,
) -> int:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {json_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sites_set = None
    if selected_sites:
        sites_set = {str(s).strip() for s in selected_sites}

    count = 0
    skipped = 0
    stats = {"total": 0, "null_label": 0, "null_voie": 0, "null_sens": 0, "null_coords": 0, "nb_usagers_sum": 0, "nb_usagers_min": None, "nb_usagers_max": None, "sites": set(), "modes": set(), "t_min": None, "t_max": None}
    samples_first = []
    samples_last = deque(maxlen=5)
    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        try:
            for raw in stream_json_objects(json_path):
                if sites_set is not None and _safe_str(raw.get("id_site")) not in sites_set:
                    skipped += 1
                    continue
                if skip_invalid_time and not _safe_str(raw.get("t")):
                    skipped += 1
                    continue
                row = process_record(raw)
                writer.writerow(row)
                count += 1
                stats["total"] = count
                if not _safe_str(raw.get("label")):
                    stats["null_label"] += 1
                if not _safe_str(raw.get("voie")):
                    stats["null_voie"] += 1
                if not _safe_str(raw.get("sens")):
                    stats["null_sens"] += 1
                if not isinstance(raw.get("coordonnees_geo"), dict):
                    stats["null_coords"] += 1
                nu = row["nb_usagers"]
                stats["nb_usagers_sum"] += nu
                if stats["nb_usagers_min"] is None or nu < stats["nb_usagers_min"]:
                    stats["nb_usagers_min"] = nu
                if stats["nb_usagers_max"] is None or nu > stats["nb_usagers_max"]:
                    stats["nb_usagers_max"] = nu
                if row["id_site"]:
                    stats["sites"].add(row["id_site"])
                if row["mode"]:
                    stats["modes"].add(row["mode"])
                t_str = row.get("t", "")
                if t_str:
                    stats["t_min"] = t_str if stats["t_min"] is None else min(stats["t_min"], t_str)
                    stats["t_max"] = t_str if stats["t_max"] is None else max(stats["t_max"], t_str)
                if len(samples_first) < 5:
                    samples_first.append(dict(row))
                samples_last.append(dict(row))
                if count % 100_000 == 0:
                    print(f"  Écrit {count} lignes...", flush=True)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Erreur (après {count} lignes): {e}", file=sys.stderr)
            raise

    if skipped:
        print(f"Lignes ignorées (filtre sites ou t vide): {skipped}", file=sys.stderr)

    _print_stats_and_samples(stats, samples_first, list(samples_last), csv_path)
    return count


def _print_stats_and_samples(stats, first_rows, last_rows, csv_path):
    n = stats["total"]
    if n == 0:
        print("Aucune ligne écrite.")
        return

    print("STATISTIQUES (fichier prétraité)")
    print("  Fichier de sortie     :", csv_path)
    print("  Nombre de lignes      :", f"{n:,}")
    print("  Valeurs manquantes :")
    print("    label vide          :", f"{stats['null_label']:,}", f"({100*stats['null_label']/n:.1f} %)")
    print("    voie vide           :", f"{stats['null_voie']:,}", f"({100*stats['null_voie']/n:.1f} %)")
    print("    sens vide           :", f"{stats['null_sens']:,}", f"({100*stats['null_sens']/n:.1f} %)")
    print("    coordonnees_geo null :", f"{stats['null_coords']:,}", f"({100*stats['null_coords']/n:.1f} %)")
    print("  Variable cible nb_usagers :")
    print("    somme | min | max   :", stats["nb_usagers_sum"], "|", stats["nb_usagers_min"], "|", stats["nb_usagers_max"])
    if n:
        print("    moyenne             :", f"{stats['nb_usagers_sum']/n:.2f}")
    print("  Période (t)           :", stats["t_min"], "->", stats["t_max"])
    print("  Nombre de sites       :", len(stats["sites"]))
    print("  Nombre de modes       :", len(stats["modes"]))
    ms = sorted(stats["modes"])
    print("  Modes (liste)         :", ", ".join(ms[:15]) + (" ..." if len(ms) > 15 else ""))
    
    print("EXEMPLES (premières lignes)")
    for i, row in enumerate(first_rows, 1):
        print("  --- Ligne", i, "---")
        for k, v in row.items():
            print("   ", k + ":", v)

    print("EXEMPLES (dernières lignes)")
    for i, row in enumerate(last_rows, 1):
        print("  --- Dernière", i, "---")
        for k, v in row.items():
            print("   ", k + ":", v)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prétraitement stream du JSON comptages → CSV")
    parser.add_argument("--input", default="comptage_2024_2026.json")
    parser.add_argument("--output", default="data/traffic_raw_cleaned.csv")
    parser.add_argument("--sites", nargs="*", help="Filtrer par id_site (ex: 10022 10093)")
    parser.add_argument("--no-skip-invalid-time", action="store_true", help="Garder les lignes sans horodatage")
    args = parser.parse_args()

    n = stream_json_to_csv(
        args.input,
        args.output,
        selected_sites=args.sites or None,
        skip_invalid_time=not args.no_skip_invalid_time,
    )
    print(f"Terminé: {n} lignes → {args.output}")
