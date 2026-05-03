This document outlines the final set of **48 features** selected for the gesture recognition model. For the full end-to-end data processing flow, see [README_PIPELINE.md](file:///c:/Users/spman/OneDrive%20-%20University%20of%20Patras/CEID/8th%20Semester-CEID/IoT%20Project/github/AIoT-Human-Gesture-Recognition/README_PIPELINE.md).

We have intentionally pruned 20 noisy/redundant features to improve model generalization and stability.

## Gesture Classes
1. **Swipe Up**
2. **Swipe Down**
3. **Swipe Left**
4. **Swipe Right**
5. **Texting** (Index finger)

---

## Final Feature List (48 Features)

For each of the 6 sensor axes (`acc_x/y/z`, `gyr_x/y/z`):

### 1. Statistical Core (30 Features)
*   **Mean**: Static orientation (Gravity) and rotation bias.
*   **Standard Deviation (STD)**: Intensity of movement.
*   **Root Mean Square (RMS)**: Total power of the signal.
*   **Zero-Crossing Rate (ZCR)**: Frequency of oscillation (Key for **Texting**).
*   **Peak-to-Peak (PTP)**: Maximum range of motion.

### 2. Multi-Axis Dynamics (10 Features)
*   **Magnitude Mean/STD (Acc & Gyr)**: Captures the overall intensity regardless of hand orientation.
*   **Inter-Axis Correlation (Acc & Gyr)**: Captures the 3D shape/arc of the gesture.

### 3. Orientation & Frequency (8 Features)
*   **Pitch & Roll (Mean/STD)**: Detects the tilt of the hand (Key for **Direction**).
*   **Dominant Frequency**: Periodicity of the motion.
*   **Spectral Entropy**: Complexity/Chaos of the movement.

---

## Why 20 Features Were Dropped

We have manually removed the following to improve the Signal-to-Noise Ratio:

1.  **Skewness & Kurtosis (12 features)**: Too noisy in short windows; highly sensitive to minor jitter that doesn't represent the gesture.
2.  **SMA (6 features)**: Redundant with RMS/STD.
3.  **Spectral Energy (2 features)**: Redundant with time-domain RMS (Parseval's Theorem).

### Benefits of this Pruning:
*   **Reduced Overfitting**: The model focuses on universal patterns rather than user-specific noise.
*   **Stability**: Removing multicollinearity (SMA vs RMS) makes the feature importance more reliable.
*   **Interpretability**: Every remaining feature has a direct physical meaning in the context of human motion.
