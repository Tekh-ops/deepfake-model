"""
models/ensemble.py
──────────────────
Orchestrates the full detection pipeline:

  1. Load media (image or video frames)
  2. Run face detection + crop
  3. Feed to primary (fine-tuned EfficientNet-B4) and secondary (Xception)
  4. Weighted ensemble of P(FAKE) scores
  5. Return structured result dict

The public interface is:

    from models.ensemble import run_detection
    result = run_detection(media_path)
"""

import os
import time
import numpy as np
import cv2
from PIL import Image

from config import (
    DEVICE,
    ENSEMBLE_WEIGHTS,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    XCEPTION_SIZE,
)
from models.detector import get_primary, get_secondary

# ── Media type helpers ────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


# ── Face detection / cropping ─────────────────────────────────
_cascade = None

def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def _crop_face(pil_img: Image.Image) -> Image.Image:
    """
    Detect the largest face in a PIL Image and crop to it.
    Returns the cropped face (with 20% padding) or the original
    image if no face is detected.
    """
    img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade = _get_cascade()
    faces   = cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        # No face found — return original
        return pil_img

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(max(w, h) * 0.20)
    x1  = max(0, x - pad);  y1 = max(0, y - pad)
    x2  = min(img_bgr.shape[1], x + w + pad)
    y2  = min(img_bgr.shape[0], y + h + pad)

    cropped = img_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


# ── Frame extraction for video ────────────────────────────────

def _extract_pil_frames(video_path: str, num_frames: int = 16) -> list:
    """
    Uniformly sample num_frames from a video.
    Returns a list of PIL Images.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )

    cap.release()
    if not frames:
        raise ValueError("No frames could be extracted from video.")
    return frames


# ── Core detection logic ──────────────────────────────────────

def _detect_single_frame(
    pil_img: Image.Image,
    primary,
    secondary,
    use_face_crop: bool = True,
) -> dict:
    """
    Run both detectors on a single PIL Image frame.
    Returns dict with per-model P(FAKE) values.
    """
    if use_face_crop:
        face_img = _crop_face(pil_img)
    else:
        face_img = pil_img

    p_fake_primary   = primary.predict(face_img)
    p_fake_secondary = secondary.predict(face_img)

    return {
        "primary":   p_fake_primary,
        "secondary": p_fake_secondary,
    }


def _ensemble_score(frame_scores: list) -> dict:
    """
    Aggregate per-frame per-model scores into a final ensemble result.

    frame_scores: list of dicts, each {"primary": float, "secondary": float}
    """
    w = ENSEMBLE_WEIGHTS

    # Average across frames first (temporal pooling)
    avg_primary   = float(np.mean([s["primary"]   for s in frame_scores]))
    avg_secondary = float(np.mean([s["secondary"] for s in frame_scores]))

    # Weighted ensemble
    ensemble_fake_prob = (
        w["primary"]   * avg_primary +
        w["secondary"] * avg_secondary
    )

    label      = "Fake" if ensemble_fake_prob >= CONFIDENCE_THRESHOLD else "Real"
    confidence = (
        ensemble_fake_prob
        if label == "Fake"
        else 1.0 - ensemble_fake_prob
    )

    return {
        "label":       label,
        "confidence":  round(confidence, 4),
        "fake_prob":   round(ensemble_fake_prob, 4),
        "per_model": {
            "efficientnet_b4_finetuned": {
                "label":     "Fake" if avg_primary   >= 0.5 else "Real",
                "fake_prob": round(avg_primary,   4),
                "note":      "Fine-tuned on FaceForensics++ (primary detector)",
            },
            "xception_imagenet": {
                "label":     "Fake" if avg_secondary >= 0.5 else "Real",
                "fake_prob": round(avg_secondary, 4),
                "note":      "ImageNet pretrained structural check (secondary)",
            },
        },
        "ensemble_weights": w,
    }


# ── Public API ────────────────────────────────────────────────

def run_detection(
    media_path: str,
    use_face_crop: bool = True,
    num_video_frames: int = 16,
) -> dict:
    """
    Full detection pipeline for an image or video file.

    Args:
        media_path:       Absolute path to the media file.
        use_face_crop:    Crop to face region before inference (recommended).
        num_video_frames: How many frames to sample from a video.

    Returns:
        {
            "label":       "Fake" | "Real",
            "confidence":  float,
            "fake_prob":   float,
            "per_model":   { ... },
            "frame_count": int,
            "media_type":  "image" | "video",
            "inference_ms": float,
        }
    """
    primary   = get_primary()
    secondary = get_secondary()

    t_start = time.time()

    if _is_video(media_path):
        # ── Video ──────────────────────────────────────────
        frames = _extract_pil_frames(media_path, num_frames=num_video_frames)
        frame_scores = [
            _detect_single_frame(f, primary, secondary, use_face_crop)
            for f in frames
        ]
        media_type  = "video"
        frame_count = len(frames)
    else:
        # ── Image ──────────────────────────────────────────
        pil_img = Image.open(media_path).convert("RGB")
        frame_scores = [
            _detect_single_frame(pil_img, primary, secondary, use_face_crop)
        ]
        media_type  = "image"
        frame_count = 1

    inference_ms = (time.time() - t_start) * 1000
    result = _ensemble_score(frame_scores)

    return {
        **result,
        "frame_count":  frame_count,
        "media_type":   media_type,
        "inference_ms": round(inference_ms, 1),
    }