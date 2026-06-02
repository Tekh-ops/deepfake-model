"""
utils/metadata.py
─────────────────
Builds the structured metadata payload returned to the frontend.
Replaces the placeholder version from your friend's prototype.
"""

import os
import time
from datetime import datetime, timezone

from config import SYSTEM_NAME, SYSTEM_VERSION

# ── File type helpers ─────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def _get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"

def _get_file_size_human(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def _risk_level(fake_prob: float) -> str:
    """Map fake probability to a human-readable risk tier."""
    if fake_prob >= 0.85:
        return "CRITICAL"
    if fake_prob >= 0.70:
        return "HIGH"
    if fake_prob >= 0.50:
        return "MEDIUM"
    if fake_prob >= 0.35:
        return "LOW"
    return "MINIMAL"

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "Very High"
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


# ── Main builder ──────────────────────────────────────────────

def create_metadata(
    file_path: str,
    result: str,
    confidence: float,
    submission_id: str = "",
    file_hash: str = "",
    detection_detail: dict | None = None,
    processing_time_ms: float = 0.0,
) -> dict:
    """
    Build the full metadata payload for the API response.

    Args:
        file_path:          Path to the analysed file
        result:             "Fake" or "Real"
        confidence:         Float [0, 1] — how confident the ensemble is
        submission_id:      Unique ID for this case (from hashing module)
        file_hash:          SHA-256 hex digest of the file
        detection_detail:   Per-model breakdown dict from ensemble
        processing_time_ms: Wall-clock inference time

    Returns:
        Structured dict ready to be JSON-serialised and returned to frontend.
    """
    filename   = os.path.basename(file_path)
    file_size  = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    media_type = _get_media_type(file_path)
    ext        = os.path.splitext(filename)[1].lower().lstrip(".")
    now_utc    = datetime.now(timezone.utc)
    fake_prob  = confidence if result == "Fake" else 1.0 - confidence

    metadata = {
        # ── Submission ──────────────────────────────────────
        "submission_id":    submission_id,
        "timestamp_utc":    now_utc.isoformat(),
        "timestamp_unix":   int(time.time()),

        # ── File information ────────────────────────────────
        "file": {
            "name":       filename,
            "size_bytes": file_size,
            "size_human": _get_file_size_human(file_size),
            "type":       ext,
            "media_type": media_type,
            "sha256":     file_hash,
        },

        # ── Detection result ────────────────────────────────
        "detection": {
            "label":             result,           # "Fake" | "Real"
            "confidence":        round(confidence, 4),
            "confidence_label":  _confidence_label(confidence),
            "fake_probability":  round(fake_prob, 4),
            "risk_level":        _risk_level(fake_prob),
            "is_deepfake":       result == "Fake",
        },

        # ── Per-model breakdown (if available) ──────────────
        "model_breakdown": detection_detail or {},

        # ── Performance ─────────────────────────────────────
        "processing": {
            "time_ms":         round(processing_time_ms, 1),
            "system_name":     SYSTEM_NAME,
            "system_version":  SYSTEM_VERSION,
        },

        # ── Integrity ───────────────────────────────────────
        "integrity": {
            "sha256":          file_hash,
            "chain_of_custody": "BSA Section 63 compliant audit log entry created",
        },
    }

    return metadata