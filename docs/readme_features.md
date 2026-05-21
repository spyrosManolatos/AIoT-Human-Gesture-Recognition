# Feature Engineering & Exploratory Data Analysis (EDA)

This document describes the Exploratory Data Analysis (EDA), per-session preprocessing rationale, and feature selection pipeline implemented in the classical machine learning notebooks: [ml_subject_split.ipynb](file:///home/spman/ceid/Iot_TimeSeries/ml_subject_split.ipynb) and [ml_stratified_split.ipynb](file:///home/spman/ceid/Iot_TimeSeries/ml_stratified_split.ipynb).

---

## 1. Per-Session Preprocessing Isolation

A critical design choice in the pipeline is that **preprocessing (filtering and windowing) is executed strictly on a per-session (recording) basis**. 

### Why Per-Session Preprocessing?
1. **Preventing Boundary Artifacts**: The dataset consists of independent recording trials (each mapped to a unique user and gesture). If we applied the Butterworth low-pass filter to the entire concatenated dataset, the filters would smooth over the sharp boundaries between completely unrelated gesture trials, producing artificial signal distortions.
2. **Avoiding Data Leakage**: Running window segmentation across session borders would result in "hybrid" windows containing frames from user A's trial and user B's trial. Grouping by `["gesture_id", "user"]` isolates each recording session, ensuring that windows are segmented purely within continuous, independent trials.

```python
# Grouping signals by trial session before filtering and windowing
list_of_signals = [value for key, value in dataframe.groupby(["gesture_id", "user"])]
```

---

## 2. Exploratory Data Analysis (EDA) Pipeline

Before training the classical classifiers, the notebooks execute a structured 8-step EDA pipeline to inspect the dataset characteristics and optimize the feature space.

### Step 1: Session Duration Analysis
*   **Method**: Group the samples by `["gesture_id", "user"]` and divide by the sampling rate (100Hz) to compute duration in seconds.
*   **Visual**: A barplot showing the duration of each recording trial. This ensures that the trials have consistent lengths and checks for class balance in terms of recording time.

### Step 2: Signal Filtering Inspection
*   **Method**: Plot the raw vs. filtered waveforms of the 3 accelerometer axes and 3 gyroscope axes using `plot_filtered_vs_raw_signal()`.
*   **Visual**: Inspects whether the low-pass Butterworth filter effectively removes high-frequency jitter/sensor noise without altering the overall motion signature.

### Step 3: Windowing Count Verification
*   **Method**: Segment the continuous sessions into windows (150 samples size, 75 overlap).
*   **Visual**: A barplot using `plot_window_counts_by_gesture_user()` verifying how many windows were successfully generated per gesture-user trial. This checks for representation imbalances in window counts.

### Step 4: ANOVA F-Test Feature Selection
*   **Method**: Generate all 134 candidate features (from `extract_all_candidates()`) for each training window, then run an **ANOVA F-test** (`SelectKBest(f_classif, k=12)`) to measure the relationship between each feature and the gesture target class.
*   **Goal**: Identifies the top $K$ features that have the highest discriminative power.

### Step 5: Multicollinearity Heatmap
*   **Method**: Plot a Pearson correlation matrix heatmap for the selected top $K$ features.
*   **Goal**: Handcrafted features often overlap in meaning (e.g. mean vs. RMS). The heatmap helps flag pairs of features with high correlation ($r > 0.9$).

### Step 6: Multicollinearity Reduction
*   **Method**: Manually drop highly redundant features to avoid overfitting.
*   **Example**: Keep `acc_y_rms` (due to high ANOVA F-score) and drop highly correlated features like `acc_y_mean` and `acc_mag_mean`.

### Step 7: Feature Distribution Boxplots
*   **Method**: Generate boxplots (`plot_feature_boxplots_by_class()`) of the final selected features grouped by gesture category.
*   **Goal**: Examines the variance, median differences, and outliers of the features to evaluate class separability.

### Step 8: Principal Component Analysis (PCA) 3D Projection
*   **Method**: Scale the features to unit variance and project them into a lower-dimensional space using PCA ($n\_components = 3$).
*   **Visual**: A 3D/2D scatter plot color-coded by gesture class. This helps inspect whether the classes form distinct clusters in the feature space prior to model training.
