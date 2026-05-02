from __future__ import annotations

import csv
from pathlib import Path


def find_pairs(folder: Path) -> list[tuple[Path, Path, str]]:
    acc_files = list(folder.glob("*_Accelerometer.csv"))
    gyr_files = list(folder.glob("*_Gyroscope.csv"))

    gyr_by_prefix = {}
    for gyr in gyr_files:
        prefix = gyr.name[: -len("_Gyroscope.csv")]
        gyr_by_prefix[prefix] = gyr

    pairs = []
    for acc in acc_files:
        prefix = acc.name[: -len("_Accelerometer.csv")]
        gyr = gyr_by_prefix.get(prefix)
        if gyr:
            pairs.append((acc, gyr, prefix))

    return pairs


def merge_pair(acc_path: Path, gyr_path: Path, out_path: Path) -> int:
    with acc_path.open("r", newline="") as acc_file:
        acc_reader = csv.DictReader(acc_file)
        acc_rows = list(acc_reader)

    with gyr_path.open("r", newline="") as gyr_file:
        gyr_reader = csv.DictReader(gyr_file)
        gyr_rows = list(gyr_reader)

    gyr_by_epoch = {row["epoc (ms)"]: row for row in gyr_rows}

    out_fields = [
        "epoc (ms)",
        "timestamp (+0300)",
        "elapsed (s)",
        "x-axis (g)",
        "y-axis (g)",
        "z-axis (g)",
        "x-axis (deg/s)",
        "y-axis (deg/s)",
        "z-axis (deg/s)",
    ]

    merged_rows = []
    for acc_row in acc_rows:
        gyr_row = gyr_by_epoch.get(acc_row["epoc (ms)"])
        if not gyr_row:
            continue
        merged_rows.append(
            {
                "epoc (ms)": acc_row["epoc (ms)"],
                "timestamp (+0300)": acc_row["timestamp (+0300)"],
                "elapsed (s)": acc_row["elapsed (s)"],
                "x-axis (g)": acc_row["x-axis (g)"],
                "y-axis (g)": acc_row["y-axis (g)"],
                "z-axis (g)": acc_row["z-axis (g)"],
                "x-axis (deg/s)": gyr_row["x-axis (deg/s)"],
                "y-axis (deg/s)": gyr_row["y-axis (deg/s)"],
                "z-axis (deg/s)": gyr_row["z-axis (deg/s)"],
            }
        )

    with out_path.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    return len(merged_rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    if not data_root.exists():
        raise SystemExit(f"Data folder not found: {data_root}")

    total_pairs = 0
    total_rows = 0
    for gesture_dir in data_root.iterdir():
        if not gesture_dir.is_dir():
            continue
        for person_dir in gesture_dir.iterdir():
            if not person_dir.is_dir():
                continue
            pairs = find_pairs(person_dir)
            if not pairs:
                continue

            if len(pairs) == 1:
                acc_path, gyr_path, _ = pairs[0]
                out_path = person_dir / "merged.csv"
                rows = merge_pair(acc_path, gyr_path, out_path)
                total_pairs += 1
                total_rows += rows
            else:
                for acc_path, gyr_path, prefix in pairs:
                    out_path = person_dir / f"merged_{prefix}.csv"
                    rows = merge_pair(acc_path, gyr_path, out_path)
                    total_pairs += 1
                    total_rows += rows

    print(f"Merged {total_pairs} pair(s), wrote {total_rows} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
