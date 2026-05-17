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

- [aiot_project_feature_engineering.ipynb](aiot_project_feature_engineering.ipynb): main pipeline and subject-independent evaluation
- [aiot_project_feature_engineering_stratified_test_split.ipynb](aiot_project_feature_engineering_stratified_test_split.ipynb): comparison experiment using a stratified random split
- [time_series_random_split.ipynb](time_series_random_split.ipynb): direct time-series classification with a lightweight 1D CNN fed by windowed 3D tensors
- [time_series_split_by_subject.ipynb](time_series_split_by_subject.ipynb): direct time-series classification with a subject-wise split
- [utils.py](utils.py): filtering, windowing, and dataframe helpers
- [utils_features.py](utils_features.py): feature extraction
- [utils_visual.py](utils_visual.py): reusable plots for signal inspection and model analysis

### End-to-end flow
1. Load raw accelerometer and gyroscope samples from MongoDB.
2. Reformat recordings into a consistent dataframe with sensor axes and metadata such as `gesture_id` and `user`.
3. Inspect class balance and raw signal behavior.
4. Apply a low-pass Butterworth filter to reduce sensor noise.
5. Segment each recording into fixed windows with a sliding-window procedure.
6. Split windows for training and evaluation.
7. Branch into two modeling paths:
   - feature engineering: flatten each window, extract handcrafted descriptors with `extract_all_candidates(...)`, then train classical models such as SVM and Random Forest
   - time series classification: feed the raw windowed signals directly to a lightweight 1D CNN as 3D tensors shaped like `(num_windows, window_size, 6)`
8. Evaluate each path and compare random-split performance with subject-wise generalization.

### Four stories
The project has two modeling paths, and each path has a random-split story plus a subject-wise story.

| Path | Split story | Notebook | Takeaway |
| --- | --- | --- | --- |
| Feature engineering | Subject-independent | [aiot_project_feature_engineering.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering.ipynb) | Handcrafted features do not generalize well to a completely unseen user. |
| Feature engineering | Stratified random split | [aiot_project_feature_engineering_stratified_test_split.ipynb](/home/spman/ceid/Iot_TimeSeries/aiot_project_feature_engineering_stratified_test_split.ipynb) | Accuracy looks much stronger when the same users appear in train and test. |
| Direct time series CNN | Random window split | [time_series_random_split.ipynb](/home/spman/ceid/Iot_TimeSeries/time_series_random_split.ipynb) | A lightweight 1D CNN performs well when windows from the same subject can leak into both sets. |
| Direct time series CNN | Subject-wise split | [time_series_split_by_subject.ipynb](/home/spman/ceid/Iot_TimeSeries/time_series_split_by_subject.ipynb) | The same CNN drops sharply on unseen subjects, showing the real generalization limit. |

### What the comparison shows
The random-split results are useful for checking whether the models can learn gesture structure under a mixed-user setting, but they are optimistic.

The subject-wise results are the more honest generalization test, and they show the same pattern in both ML branches:

- feature engineering works better when the user distribution overlaps between train and test
- the direct 1D CNN also looks strong under random window splitting
- both approaches degrade sharply when evaluated on a truly unseen subject

### Feature set
The active feature extractor is `extract_all_candidates(window)` in [utils_features.py](/home/spman/ceid/Iot_TimeSeries/utils_features.py). For each fixed window, it turns the raw sensor slice into a single feature vector by computing summary statistics over the accelerometer and gyroscope channels instead of feeding the raw samples directly to the model.

In practice, this means each window is described by:

- per-axis time-domain statistics such as mean, standard deviation, RMS, skewness, kurtosis, zero-crossing rate, and IQR
- spectral descriptors such as dominant frequency, spectral entropy, mean frequency, and band-energy ratios
- cross-channel information such as acceleration/gyroscope magnitudes, signal magnitude area, and pairwise axis correlations

Those handcrafted features are then used as input to classical machine-learning models.

### Models used
The project compares two model families:

- classical ML on handcrafted features: SVM and Random Forest in the feature-engineering notebooks
- direct deep learning on raw windows: a lightweight 1D CNN in the time-series notebooks

The Random Forest experiments are especially useful because they provide a strong non-neural baseline and are easy to interpret across the feature space. The 1D CNN, on the other hand, learns local motion patterns directly from the 3D window tensors shaped as `(num_windows, window_size, 6)`.

## Repository structure
- `aiot_dataset_creation.ipynb`: imports local CSV recordings into MongoDB
- `aiot_project_feature_engineering.ipynb`: main pipeline notebook
- `aiot_project_feature_engineering_stratified_test_split.ipynb`: experimental split notebook
- `time_series_random_split.ipynb`: direct 1D CNN time-series classifier
- `time_series_split_by_subject.ipynb`: direct 1D CNN time-series classifier with subject-wise split
- `utils.py`: processing helpers
- `utils_features.py`: feature engineering helpers
- `utils_visual.py`: plotting helpers

## Notes
- Database names are case-sensitive in MongoDB; keep a single casing.
- Large CSVs may take time to insert; consider batching if needed.
