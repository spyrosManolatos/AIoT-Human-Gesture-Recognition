from __future__ import annotations

import os
import shutil
from pathlib import Path

def parse_filename(filename: str) -> str | None:
    # The standardized filename schema has 8 parts separated by underscores:
    # [gesture_id]_[hand]_[sr]_[sensor]_[primary]_[spontaneous]_[user]_[unique_id_hash].csv
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 8:
        # The gesture_id is everything except the last 7 metadata fields
        gesture_id = "_".join(parts[:-7])
        return gesture_id
    return None

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    mongo_data_root = root / "mongo_data"
    
    if not data_root.exists():
        raise SystemExit(f"Data folder not found at: {data_root}")
        
    print(f"Scanning '{data_root}' recursively for renamed metadata-encoded CSV files...")
    
    mongo_data_root.mkdir(exist_ok=True)
    staged_count = 0
    
    # Recursively traverse the data/ directory
    for dirpath, _, filenames in os.walk(data_root):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
                
            # Skip unmerged raw logs and intermediate merge files
            if filename.endswith("_Accelerometer.csv") or filename.endswith("_Gyroscope.csv") or filename == "merged.csv":
                continue
                
            gesture_id = parse_filename(filename)
            if gesture_id:
                source_file = Path(dirpath) / filename
                dest_dir = mongo_data_root / gesture_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / filename
                
                # Copy the staged file flatly into mongo_data/
                shutil.copy2(source_file, dest_file)
                staged_count += 1
                print(f"  [Staged] {source_file.relative_to(root)} -> mongo_data/{gesture_id}/{filename}")
                
    print(f"\nStaging complete! Copied {staged_count} file(s) to 'mongo_data/'.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
