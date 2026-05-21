"""
utils_features.py
─────────────────
Provides:
  extract_all_candidates(window)        – full structured pool  (105 features, per-axis)
"""

from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis as sp_kurtosis
from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd

FS = 100  # Hz – IMU sampling rate


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _safe_corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])










# ── Full structured candidate pool (103 features) ────────────────────────────

_CHANNELS     = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
_ACC_CHANNELS = ["acc_x", "acc_y", "acc_z"]
_GYR_CHANNELS = ["gyr_x", "gyr_y", "gyr_z"]


def extract_all_candidates(window, fs=FS):
    """
    Extracts the full structured 105-feature candidate pool per window.

    Structure
    ─────────
    Time domain   7 features × 6 channels = 42
      mean, std, rms, skewness, kurtosis, zcr, IQR

    Spectral      7 features × 6 channels = 42
      dominant_freq, spectral_entropy, mean_freq (centroid),
      energy_0_2hz, energy_2_5hz, energy_5_10hz, energy_10hz_plus

    Cross-channel                          = 21
      acc_mag (mean, std), gyr_mag (mean, std),
      acc_SMA, gyr_SMA,
      pairwise correlations C(6,2) = 15
    ──────────────────────────────────────────
    Total                                  = 105

    Notes
    ─────
    - Time/spectral features are per-axis (all 6 channels) → captures direction.
      Use this pool for direction-sensitive tasks (e.g. left vs right swipe).
    - For cross-subject, orientation-invariant tasks (e.g. texting vs swipe)
      prefer extract_features_v2() which operates on magnitude signals.
    - spectral_entropy is normalised to [0, 1] (divides by log2 N).

    Args:
        window : pandas DataFrame with columns
                 [user, gesture_id, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        fs     : int – sampling frequency in Hz (default 100)

    Returns:
        dict of feature values plus user / gesture_id meta-columns
    """
    feats = {
        "user":       window["user"].iloc[0],
        "gesture_id": window["gesture_id"].iloc[0],
    }

    arrays = {ch: window[ch].to_numpy(dtype=float) for ch in _CHANNELS}
    n = len(next(iter(arrays.values())))

    # ── Per-channel features (time + spectral) ────────────────────────────
    for ch, sig in arrays.items():
        centered = sig - np.mean(sig)

        # Time domain (7)
        q75, q25 = np.percentile(sig, [75, 25])
        feats[f"{ch}_mean"] = float(np.mean(sig))
        feats[f"{ch}_std"]  = float(np.std(sig))
        feats[f"{ch}_rms"]  = float(np.sqrt(np.mean(sig ** 2)))
        feats[f"{ch}_skew"] = float(skew(sig))
        feats[f"{ch}_kurt"] = float(sp_kurtosis(sig))
        feats[f"{ch}_zcr"]  = float(np.mean(np.diff(np.sign(centered)) != 0))
        feats[f"{ch}_iqr"]  = float(q75 - q25)

        # Spectral (7)
        freqs    = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(centered))
        psd      = fft_vals ** 2
        total    = psd.sum() + 1e-12

        # Dominant frequency (skip DC bin)
        feats[f"{ch}_dom_freq"] = float(freqs[np.argmax(fft_vals[1:]) + 1]) if n > 2 else 0.0

        # Spectral entropy – normalised to [0, 1]
        psd_norm = psd / total
        psd_pos  = psd_norm[psd_norm > 0]
        feats[f"{ch}_spectral_entropy"] = (
            float(-np.sum(psd_pos * np.log2(psd_pos)) / np.log2(len(psd_pos)))
            if len(psd_pos) > 1 else 0.0
        )

        # Mean frequency (spectral centroid)
        feats[f"{ch}_mean_freq"] = float(np.sum(freqs * psd) / total)

        # Band energies (fraction of total PSD)
        def _band(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(psd[mask].sum() / total)

        feats[f"{ch}_energy_0_2hz"]    = _band(0,  2)
        feats[f"{ch}_energy_2_5hz"]    = _band(2,  5)
        feats[f"{ch}_energy_5_10hz"]   = _band(5,  10)
        feats[f"{ch}_energy_10hz_plus"] = _band(10, fs / 2)

    # ── Cross-channel features ────────────────────────────────────────────
    acc_arr = np.stack([arrays[ch] for ch in _ACC_CHANNELS])   # 3 × N
    gyr_arr = np.stack([arrays[ch] for ch in _GYR_CHANNELS])   # 3 × N

    acc_mag = np.sqrt((acc_arr ** 2).sum(axis=0))  # N,
    gyr_mag = np.sqrt((gyr_arr ** 2).sum(axis=0))  # N,

    # Vector magnitude: mean & std
    feats["acc_mag_mean"] = float(np.mean(acc_mag))
    feats["acc_mag_std"]  = float(np.std(acc_mag))
    feats["gyr_mag_mean"] = float(np.mean(gyr_mag))
    feats["gyr_mag_std"]  = float(np.std(gyr_mag))

    # Signal Magnitude Area (SMA = sum|axes| / N)
    feats["acc_sma"] = float(np.abs(acc_arr).sum() / n)
    feats["gyr_sma"] = float(np.abs(gyr_arr).sum() / n)

    # Pairwise correlations between all 6 channels  C(6,2) = 15
    ch_list = _CHANNELS
    for i in range(len(ch_list)):
        for j in range(i + 1, len(ch_list)):
            feats[f"corr_{ch_list[i]}_{ch_list[j]}"] = _safe_corr(
                arrays[ch_list[i]], arrays[ch_list[j]]
            )

    return feats
