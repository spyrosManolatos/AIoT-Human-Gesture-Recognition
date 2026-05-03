# AIoT Human Gesture Recognition

Brief: tools and notebooks for building a human-gesture dataset from IMU sensor CSVs and inserting into MongoDB.

## Requirements
- Python 3.10+
- MongoDB (local or remote)

## Setup
1) Create a virtual environment and install deps:
   - `python -m venv .venv`
   - `pip install -r requirements.txt`

2) Configure the DB connection in `config.yml` (kept local, not committed):
   - `client`: MongoDB URI
   - `db`: database name (case-sensitive)
   - `col`: collection name

## Dataset layout
Expected directory structure under `data/` (example):

```
DataRoot/
  swipe-down-thumb/
    userA/
      file1_..._AccGyr_...csv
  swipe-left-thumb/
    userB/
      file2_..._AccGyr_...csv
```

## Mongo import notebook
Open `aiot_dataset_creation.ipynb` and run cells in order. The notebook:
- parses metadata from filenames
- loads CSVs from `mongo_data/`
- writes into MongoDB collection

## Feature Engineering & Pipeline
The project includes a robust pipeline for transforming raw sensor data into gesture classifications:
- **Notebook**: [aiot_project_feature_engineering.ipynb](file:///c:/Users/spman/OneDrive%20-%20University%20of%20Patras/CEID/8th%20Semester-CEID/IoT%20Project/github/AIoT-Human-Gesture-Recognition/aiot_project_feature_engineering.ipynb)
- **Architecture**: See [README_PIPELINE.md](file:///c:/Users/spman/OneDrive%20-%20University%20of%20Patras/CEID/8th%20Semester-CEID/IoT%20Project/github/AIoT-Human-Gesture-Recognition/README_PIPELINE.md) for details on cleaning, segmentation, and filtering.
- **Features**: Detailed list of extracted features in [README_FEATURES.md](file:///c:/Users/spman/OneDrive%20-%20University%20of%20Patras/CEID/8th%20Semester-CEID/IoT%20Project/github/AIoT-Human-Gesture-Recognition/README_FEATURES.md).

## Notes
- Database names are case-sensitive in MongoDB; keep a single casing.
- Large CSVs may take time to insert; consider batching if needed.
