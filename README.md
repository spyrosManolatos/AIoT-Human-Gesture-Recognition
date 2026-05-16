# AIoT Human Gesture Recognition

Tools, notebooks, and helper modules for building an IMU-based human gesture recognition dataset, inserting it into MongoDB, and training gesture classifiers from accelerometer and gyroscope time series.

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

## Pipeline overview
The full gesture-recognition workflow is implemented in the project notebooks and helper modules:

- [aiot_project_feature_engineering.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering.ipynb): main pipeline and subject-independent evaluation
- [aiot_project_feature_engineering_stratified_test_split.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering_stratified_test_split.ipynb): comparison experiment using a stratified random split
- [utils.py](/home/spman/ceid/Iot_TimeSeries/utils.py): filtering, windowing, and dataframe helpers
- [utils_features.py](/home/spman/ceid/Iot_TimeSeries/utils_features.py): feature extraction
- [utils_visual.py](/home/spman/ceid/Iot_TimeSeries/utils_visual.py): reusable plots for signal inspection and model analysis

### End-to-end flow
1. Load raw accelerometer and gyroscope samples from MongoDB.
2. Reformat the recordings into a consistent dataframe with sensor axes and metadata such as `gesture_id` and `user`.
3. Perform exploratory data analysis on the raw recordings:
   - duration per gesture and user
   - signal inspection
   - class and user balance checks
4. Apply a low-pass Butterworth filter for each user session to reduce high-frequency sensor noise.
5. Segment each filtered recording into fixed windows using a sliding-window procedure.
6. Re-check class balance after segmentation by counting the number of windows per gesture and user.
7. Split data for training and evaluation.
8. Extract handcrafted time-domain, spectral, and cross-channel features from each window with `extract_all_candidates(...)`.
9. Rank and prune features with univariate selection and correlation analysis.
10. Visualize the feature space with PCA.
11. Train and evaluate classifiers, mainly SVM and Random Forest.

### Feature set
The active feature extractor is `extract_all_candidates(window)` in [utils_features.py](/home/spman/ceid/Iot_TimeSeries/utils_features.py). It produces a structured feature pool from each segmented window, including:

- per-axis time-domain statistics such as mean, standard deviation, RMS, skewness, kurtosis, zero-crossing rate, and IQR
- spectral descriptors such as dominant frequency, spectral entropy, mean frequency, and band-energy ratios
- cross-channel information such as acceleration/gyroscope magnitudes, signal magnitude area, and pairwise axis correlations

## Experiments
We ran two main experiments to understand both raw model performance and generalization behavior.

### Experiment 1: subject-independent split
Notebook: [aiot_project_feature_engineering.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering.ipynb)

- Training users: `01`, `02`
- Test user: `03`
- Goal: evaluate whether handcrafted features generalize to a completely unseen subject

Observed outcome:

- SVM accuracy: about `0.15`
- Random Forest accuracy: about `0.15`

This is below the `0.20` random-guessing baseline for 5 classes, which shows that the initial feature pipeline is not robust enough for unseen-user generalization.

### Experiment 2: stratified random split
Notebook: [aiot_project_feature_engineering_stratified_test_split.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering_stratified_test_split.ipynb)

- Split: `train_test_split(..., test_size=0.25, stratify=y_all, random_state=42)`
- Goal: evaluate the same feature space when both train and test sets contain samples from all users

Observed outcome:

- SVC accuracy: about `0.78`
- Random Forest accuracy: about `0.84`
- Random Forest with GridSearchCV: about `0.85`

Best tuned Random Forest parameters from the notebook:

- `n_estimators = 200`
- `max_depth = 20`
- `min_samples_leaf = 1`

### Interpretation
The gap between the two experiments is the main result of the project so far:

- performance is strong when the train/test split contains the same users in both sets
- performance collapses when the model is evaluated on a completely unseen user

This suggests that the current handcrafted features capture user-specific execution patterns well, but still struggle with user-independent generalization.

## Repository structure
- `aiot_dataset_creation.ipynb`: imports local CSV recordings into MongoDB
- `aiot_project_feature_engineering.ipynb`: main pipeline notebook
- `aiot_project_feature_engineering_stratified_test_split.ipynb`: experimental split notebook
- `utils.py`: processing helpers
- `utils_features.py`: feature engineering helpers
- `utils_visual.py`: plotting helpers

## Notes
- Database names are case-sensitive in MongoDB; keep a single casing.
- Large CSVs may take time to insert; consider batching if needed.

