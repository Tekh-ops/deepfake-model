"""
models/rppg_detector.py
────────────────────────
Remote PhotoPlethysmography (rPPG) biological signal detector.

Theory:
  Genuine human faces contain a subtle pulse signal (0.7–4.0 Hz / 42–240 BPM)
  embedded in the tiny skin colour variations caused by blood volume changes.
  Current deepfake generators do NOT model this physiological signal.
  A missing or incoherent pulse is a strong indicator of a synthetic face.

Algorithm: CHROM (Chrominance-based rPPG)
  de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG"
  IEEE Transactions on Biomedical Engineering, 2013.

Steps:
  1. Extract mean RGB from forehead ROI for each frame
  2. Temporally normalise each channel
  3. Apply CHROM projection:
       X = 3R - 2G
       Y = 1.5R + G - 1.5B
  4. Compute BVP signal: BVP = X - (std(X)/std(Y)) * Y
  5. Bandpass filter [0.7, 4.0] Hz
  6. Compute SNR = peak power in band / total power in band

SNR < 0.3 → no coherent pulse → likely synthetic face.
"""

import numpy as np
from PIL import Image
from scipy import signal as scipy_signal

from config import RPPG_BPM_LOW, RPPG_BPM_HIGH, RPPG_SNR_THRESHOLD


# ─────────────────────────────────────────────────────────────
# CHROM rPPG algorithm
# ─────────────────────────────────────────────────────────────

def _extract_roi_rgb(pil_img: Image.Image) -> np.ndarray | None:
    """
    Detect the forehead ROI in a PIL Image and return mean BGR.
    Returns None if no face is found.
    """
    import cv2
    arr  = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Forehead ROI: top 20% of face
    roi_h = max(1, int(h * 0.20))
    roi   = arr[y: y + roi_h, x: x + w]
    if roi.size == 0:
        return None

    return roi.reshape(-1, 3).mean(axis=0)   # mean [R, G, B]


def _bandpass_filter(signal: np.ndarray, fps: float,
                     low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter."""
    nyq  = fps / 2.0
    low  = max(0.01, low_hz  / nyq)
    high = min(0.99, high_hz / nyq)
    if low >= high:
        return signal
    try:
        b, a    = scipy_signal.butter(4, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, signal)
    except Exception:
        return signal


def _compute_bvp_snr(bvp: np.ndarray, fps: float,
                     low_hz: float, high_hz: float) -> float:
    """
    Compute the Signal-to-Noise Ratio of the BVP signal in the
    target frequency band.  Returns a value in [0, 1].
    """
    n   = len(bvp)
    if n < 8:
        return 0.0

    freqs  = np.fft.rfftfreq(n, d=1.0 / fps)
    power  = np.abs(np.fft.rfft(bvp)) ** 2

    in_band  = (freqs >= low_hz) & (freqs <= high_hz)
    total_p  = power.sum() + 1e-9
    band_p   = power[in_band].sum()

    return float(np.clip(band_p / total_p, 0.0, 1.0))


def extract_rppg_signal(
    pil_frames: list,
    fps: float = 25.0,
) -> dict:
    """
    Run CHROM rPPG analysis on a sequence of face-crop PIL Images.

    Args:
        pil_frames: list of PIL Images (face crops, consecutive frames)
        fps:        video frame rate (default 25 fps)

    Returns:
        {
            "has_pulse":   bool,   # True if coherent pulse is detected
            "bvp_snr":     float,  # signal-to-noise ratio [0,1]
            "available":   bool,   # False if < 10 frames or no face found
            "note":        str,
        }
    """
    MIN_FRAMES = 10
    if len(pil_frames) < MIN_FRAMES:
        return {
            "has_pulse": None,
            "bvp_snr":   None,
            "available": False,
            "note": f"rPPG requires >= {MIN_FRAMES} frames; got {len(pil_frames)}",
        }

    # Step 1: Extract mean ROI RGB per frame
    roi_rgb = []
    for f in pil_frames:
        mean_rgb = _extract_roi_rgb(f)
        if mean_rgb is not None:
            roi_rgb.append(mean_rgb)

    if len(roi_rgb) < MIN_FRAMES:
        return {
            "has_pulse": None,
            "bvp_snr":   None,
            "available": False,
            "note": "No face detected in enough frames for rPPG.",
        }

    rgb = np.array(roi_rgb, dtype=np.float64)   # (N, 3)
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # Step 2: Temporal normalisation per channel
    def _norm(c):
        mu = c.mean() + 1e-9
        return c / mu

    R_n, G_n, B_n = _norm(R), _norm(G), _norm(B)

    # Step 3: CHROM projection
    X = 3 * R_n - 2 * G_n
    Y = 1.5 * R_n + G_n - 1.5 * B_n

    std_x = X.std() + 1e-9
    std_y = Y.std() + 1e-9

    # Step 4: BVP signal
    bvp = X - (std_x / std_y) * Y

    # Step 5: Bandpass filter
    bvp_filtered = _bandpass_filter(bvp, fps, RPPG_BPM_LOW, RPPG_BPM_HIGH)

    # Step 6: SNR
    snr      = _compute_bvp_snr(bvp_filtered, fps, RPPG_BPM_LOW, RPPG_BPM_HIGH)
    has_pulse = snr >= RPPG_SNR_THRESHOLD

    return {
        "has_pulse": bool(has_pulse),
        "bvp_snr":   round(snr, 4),
        "available": True,
        "note": (
            f"CHROM rPPG  SNR={snr:.3f} "
            f"({'PULSE DETECTED' if has_pulse else 'NO PULSE — likely synthetic'})"
        ),
    }
