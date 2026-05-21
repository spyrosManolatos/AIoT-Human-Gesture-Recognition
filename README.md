# AIoT Human Gesture Recognition

Tools, notebooks, and helper modules for building an IMU-based human gesture recognition dataset, inserting it into MongoDB, and training gesture classifiers from accelerometer and gyroscope time-series data.

---

## Requirements

- **Python**: 3.10+
- **MongoDB**: Local or remote instance (e.g., Docker or MongoDB Atlas)

---

## Setup & Installation

### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Settings
Copy the example configuration or edit the existing `config.yml` in the root directory:
- `client`: The MongoDB Connection URI (e.g., `"mongodb://localhost:27017"`).
- `db`: The target database name (case-sensitive).
- `col`: The collection name to store dataset instances.

### 3. Git Hygiene (Recommended)
Since executing Jupyter Notebooks (`.ipynb`) automatically updates cell execution timestamps and metadata, committing them directly can clutter your git history. To automatically strip execution metadata and outputs before committing, install `nbstripout`:
```bash
# Install nbstripout
pip install nbstripout

# Configure it for this repository
nbstripout --install
```

---

## Gesture Collection Procedure

The gesture data was collected using a watch-style **MetaMotionR** wristband logging 6-channel IMU data (3-axis Accelerometer, 3-axis Gyroscope) at **100 Hz** across **3 participants**. Each gesture was recorded continuously for a single **5-minute session**.

For a detailed breakdown of the hardware settings, participant styles, texting and swiping protocols, and orientation gravity biases, refer to the [Gesture Collection Procedure Documentation](docs/gesture_collection_procedure.md).

---

## Dataset Layout

The dataset creation notebook expects CSV files containing raw signal records grouped by gesture under a `mongo_data/` directory.

### Expected Directory Structure
```text
mongo_data/
  ├── swipe-down-thumb/
  │     ├── swipe-down-thumb_1_100_AccGyr_1_0_userA_e1c9f2b8.csv
  │     └── swipe-down-thumb_1_100_AccGyr_1_0_userB_f3a7d4a2.csv
  └── swipe-left-thumb/
        └── swipe-left-thumb_1_100_AccGyr_1_0_userC_d8e9c1a5.csv
```
> [!IMPORTANT]
> The database import script reads CSV files directly under each gesture directory. Do **not** place them inside extra nested folders (e.g., `mongo_data/swipe-down-thumb/userA/file.csv`), otherwise they will be skipped.

### Filename Schema
CSV filenames are parsed into metadata values using the schema:
`[gesture_id]_[hand]_[sr]_[sensor]_[primary]_[spontaneous]_[user]_[unique_id_hash].csv`

---

## Database Import

Open [aiot_dataset_creation.ipynb](aiot_dataset_creation.ipynb) and execute the cells. This notebook will:
1. Read gesture subdirectories in `mongo_data/`.
2. Parse metadata from the CSV filenames.
3. Automatically map sensor columns (supporting both standard names and unit-annotated headers).
4. Perform bulk insertion of raw series documents into MongoDB.

---

## Modeling Pipeline Overview

* For details on the hardware configuration, recording protocol, and participant gravity biases, see the [Gesture Collection Procedure Documentation](docs/gesture_collection_procedure.md).
* For a comprehensive guide detailing data ingestion, signal filtering, and the 1D CNN architecture, see the [Gesture Recognition Pipeline Documentation](docs/readme_pipeline.md).
* For a detailed breakdown of the Exploratory Data Analysis (EDA) steps, per-session preprocessing logic, and feature selection (ANOVA and multicollinearity checks), see the [Feature Engineering & EDA Documentation](docs/readme_features.md).

The project implements two modeling branches across four distinct experimental setups to study how classifiers generalize to unseen users.

### The Four Experimental Scenarios

| Preprocessing & Features | Data Split Type | Notebook | Key Generalization Takeaway |
| :--- | :--- | :--- | :--- |
| **Feature Engineering**<br>*(Statistical + Spectral)* | **Subject-Independent**<br>*(Split by user)* | [ml_subject_split.ipynb](ml_subject_split.ipynb) | Handcrafted features do not generalize well to completely unseen users (real-world performance drop). |
| **Feature Engineering**<br>*(Statistical + Spectral)* | **Stratified Random**<br>*(Windows mixed)* | [ml_stratified_split.ipynb](ml_stratified_split.ipynb) | Accuracy appears overly optimistic when the same users' windows leak into both train and test sets. |
| **1D CNN on Raw Series** | **Stratified Random**<br>*(Windows mixed)* | [cnn_stratified_split.ipynb](cnn_stratified_split.ipynb) | A lightweight 1D CNN performs very well when data from the same subject overlaps between sets. |
| **1D CNN on Raw Series** | **Subject-Wise Split** | [cnn_subject_split.ipynb](cnn_subject_split.ipynb) | Deep learning classification accuracy drops significantly on unseen subjects, reflecting true generalization limit. |

---

## Configurable Pipeline Steps (`config.yml`)

The preprocessing and model hyperparameters can be tuned in [config.yml](config.yml):

*   **`sliding_window`**:
    *   `ws`: Window size in samples (e.g., `150`).
    *   `overlap`: Overlap size between consecutive windows (e.g., `75`).
    *   `w_type`: Windowing function to apply (e.g., `"hann"`).
*   **`filter`**:
    *   `order`: The order of the Butterworth low-pass filter (e.g., `3`).
    *   `wn`: Critical normalized frequency cutoff (e.g., `0.17`).
*   **`classifier` / `fine_tune`**:
    *   Hyperparameter grid values for cross-validation search (e.g. SVM kernel, `C`, and `gamma` parameters).

---

## Code Modules Reference

*   [utils.py](utils.py): Provides signal filtering (Butterworth), sliding-window segmentation, dataset reshaping, and file helpers.
*   [utils_features.py](utils_features.py): Extracts statistical (mean, skewness, zero-crossing, etc.) and spectral descriptors from segmented windows.
*   [utils_visual.py](utils_visual.py): Visualizes raw/filtered signals, class distributions, and confusion matrices.
