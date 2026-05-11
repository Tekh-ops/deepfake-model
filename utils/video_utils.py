"""
utils/video_utils.py
────────────────────
Video and audio utility functions used across the pipeline.

Handles:
  • Frame extraction (uniform sampling or every-N-frames)
  • Audio extraction from video via ffmpeg
  • Audio resampling to 16kHz mono for Whisper / audio detector
  • Forehead ROI extraction for rPPG
"""

import os
import subprocess
import tempfile
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


# ── Video helpers ─────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """
    Return basic metadata about a video file using OpenCV.
    Returns dict with fps, frame_count, width, height, duration_s.
    """
    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps":         fps,
        "frame_count": frame_count,
        "width":       width,
        "height":      height,
        "duration_s":  frame_count / fps if fps > 0 else 0.0,
    }


def extract_frames_uniform(
    video_path: str,
    num_frames: int = 16,
    as_pil: bool = True,
) -> list:
    """
    Uniformly sample `num_frames` frames from a video.
    Returns list of PIL Images (as_pil=True) or np.ndarray BGR (False).
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
        if not ret:
            continue
        if as_pil:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from video.")
    return frames


def extract_frames_sequence(
    video_path: str,
    num_frames: int = 8,
    target_fps: float = 4.0,
    as_pil: bool = True,
) -> list:
    """
    Extract a contiguous sequence of `num_frames` starting from a
    random offset inside the video.  Used by Stream C (temporal model)
    which needs consecutive frames to detect motion artifacts.
    """
    info  = get_video_info(video_path)
    fps   = info["fps"] or 25.0
    total = info["frame_count"]

    # Step between frames (sub-sample to target_fps)
    step     = max(1, int(fps / target_fps))
    seq_len  = num_frames * step

    # Choose a random start that leaves room for the full sequence
    max_start = max(0, total - seq_len - 1)
    start     = np.random.randint(0, max_start + 1) if max_start > 0 else 0

    cap    = cv2.VideoCapture(video_path)
    frames = []

    for i in range(num_frames):
        frame_idx = start + i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        if as_pil:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(frame)

    cap.release()

    # Pad with last frame if we couldn't get enough
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])

    if not frames:
        raise ValueError("Could not extract sequence from video.")

    return frames


# ── Audio helpers ─────────────────────────────────────────────

def extract_audio(
    video_path: str,
    output_path: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Extract audio track from a video file using ffmpeg.
    Resamples to `sample_rate` Hz mono WAV.
    Returns path to the extracted WAV file.

    If the file is already audio-only, it is resampled in-place.
    Raises RuntimeError if ffmpeg is not installed or extraction fails.
    """
    if output_path is None:
        tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg",
        "-y",               # overwrite without asking
        "-i", video_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-vn",              # no video stream
        "-f", "wav",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}):\n{result.stderr[-500:]}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt-get install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    return output_path


def has_audio_track(video_path: str) -> bool:
    """Return True if the video file contains an audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return "audio" in result.stdout
    except Exception:
        return False


# ── Face / ROI helpers ────────────────────────────────────────

_cascade = None

def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def crop_face_pil(pil_img: Image.Image, padding: float = 0.30) -> Image.Image:
    """
    Detect the largest face in a PIL Image and crop to it with padding.
    Falls back to the full image if no face is found.
    padding=0.30 means 30% of the face size is added on each side.
    """
    arr  = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    cascade = _get_cascade()
    faces   = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return pil_img

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(max(w, h) * padding)
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(arr.shape[1], x + w + pad)
    y2  = min(arr.shape[0], y + h + pad)

    cropped = arr[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def extract_forehead_roi(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect face and return the forehead ROI (top 20% of face bbox).
    Used by the rPPG detector to find the forehead skin patch.
    Returns BGR crop or None if no face found.
    """
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces   = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Forehead = top 20% of face
    forehead_h = max(1, int(h * 0.20))
    roi = frame_bgr[y: y + forehead_h, x: x + w]
    return roi if roi.size > 0 else None


def frame_diff_map(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """
    Compute absolute difference between two BGR frames.
    Amplified 3× to make deepfake temporal artifacts visible.
    Returns a uint8 frame.
    """
    diff = cv2.absdiff(frame_a, frame_b).astype(np.float32) * 3.0
    return np.clip(diff, 0, 255).astype(np.uint8)
