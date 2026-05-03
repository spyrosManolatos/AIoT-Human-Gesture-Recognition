import numpy as np
import scipy.stats
import pandas as pd

def compute_window_features(window_df, sensor_cols):
    """
    Computes time-domain and frequency-domain features for a single segmented window.
    """
    feats = {}
    eps = 1e-12
    sr = window_df["sr"].iloc[0] if "sr" in window_df.columns else 100.0

    # Per-axis time-domain features
    for col in sensor_cols:
        x = window_df[col].to_numpy()
        feats[f"{col}_mean"] = np.mean(x)
        feats[f"{col}_std"] = np.std(x, ddof=0)
        feats[f"{col}_min"] = np.min(x)
        feats[f"{col}_max"] = np.max(x)
        feats[f"{col}_skew"] = scipy.stats.skew(x, bias=False)
        feats[f"{col}_kurt"] = scipy.stats.kurtosis(x, fisher=True, bias=False)
        feats[f"{col}_zcr"] = np.mean(np.diff(np.signbit(x)).astype(int))
        feats[f"{col}_ptp"] = np.ptp(x)
        feats[f"{col}_rms"] = np.sqrt(np.mean(x ** 2))
        feats[f"{col}_sma"] = np.mean(np.abs(x))

    # Magnitude features for acc and gyr
    acc_mag = np.sqrt(
        window_df["acc_x"] ** 2
        + window_df["acc_y"] ** 2
        + window_df["acc_z"] ** 2
    )
    gyr_mag = np.sqrt(
        window_df["gyr_x"] ** 2
        + window_df["gyr_y"] ** 2
        + window_df["gyr_z"] ** 2
    )
    feats["acc_mag_mean"] = np.mean(acc_mag)
    feats["acc_mag_std"] = np.std(acc_mag, ddof=0)
    feats["acc_mag_min"] = np.min(acc_mag)
    feats["acc_mag_max"] = np.max(acc_mag)
    feats["acc_mag_range"] = np.ptp(acc_mag)

    feats["gyr_mag_mean"] = np.mean(gyr_mag)
    feats["gyr_mag_std"] = np.std(gyr_mag, ddof=0)
    feats["gyr_mag_min"] = np.min(gyr_mag)
    feats["gyr_mag_max"] = np.max(gyr_mag)
    feats["gyr_mag_range"] = np.ptp(gyr_mag)

    # Correlation coefficients between axes
    acc_corr = np.corrcoef(window_df[["acc_x", "acc_y", "acc_z"]].to_numpy().T)
    gyr_corr = np.corrcoef(window_df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy().T)
    
    acc_corr = np.nan_to_num(acc_corr, nan=0.0, posinf=0.0, neginf=0.0)
    gyr_corr = np.nan_to_num(gyr_corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    feats["acc_corr_xy"] = acc_corr[0, 1]
    feats["acc_corr_xz"] = acc_corr[0, 2]
    feats["acc_corr_yz"] = acc_corr[1, 2]
    feats["gyr_corr_xy"] = gyr_corr[0, 1]
    feats["gyr_corr_xz"] = gyr_corr[0, 2]
    feats["gyr_corr_yz"] = gyr_corr[1, 2]

    # Tilt angles (accelerometer)
    pitch = np.arctan2(
        window_df["acc_x"],
        np.sqrt(window_df["acc_y"] ** 2 + window_df["acc_z"] ** 2) + eps
    )
    roll = np.arctan2(window_df["acc_y"], window_df["acc_z"] + eps)
    feats["pitch_mean"] = np.mean(pitch)
    feats["pitch_std"] = np.std(pitch, ddof=0)
    feats["roll_mean"] = np.mean(roll)
    feats["roll_std"] = np.std(roll, ddof=0)

    # Frequency-domain features
    def spectral_features(x, prefix):
        n = len(x)
        fft_vals = np.fft.rfft(x)
        psd = (np.abs(fft_vals) ** 2) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        energy = np.sum(psd)
        dominant_freq = freqs[np.argmax(psd)] if len(freqs) else 0.0
        p = psd / (np.sum(psd) + eps)
        spectral_entropy = -np.sum(p * np.log2(p + eps))
        feats[f"{prefix}_spec_energy"] = energy
        feats[f"{prefix}_dom_freq"] = dominant_freq
        feats[f"{prefix}_spec_entropy"] = spectral_entropy

    spectral_features(acc_mag.to_numpy(), "acc_mag")
    spectral_features(gyr_mag.to_numpy(), "gyr_mag")
    
    return feats

def extract_features_from_windows(windows_dict, sensor_cols):
    """
    High-level function to iterate over all windows and extract features.
    """
    feature_rows = []
    labels = []
    users = []

    for (user_id, gesture_id), windows in windows_dict.items():
        for window in windows:
            feats = compute_window_features(window, sensor_cols)
            feature_rows.append(feats)
            labels.append(gesture_id)
            users.append(user_id)

    X_features = pd.DataFrame(feature_rows)
    y_gesture = pd.Series(labels, name="gesture_id")
    y_user = pd.Series(users, name="user")
    
    return X_features, y_gesture, y_user
