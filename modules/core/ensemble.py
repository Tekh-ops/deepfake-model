"""
models/ensemble.py
──────────────────
BharatShield Three-Stream Ensemble — UPGRADED from weighted average.

Architecture:
  Stream A → P(FAKE) — spatial texture (EfficientNet-B4 + SRM)
  Stream B → P(FAKE) — frequency domain (DCT spectrum + EfficientNet-B0)
  Stream C → P(FAKE) — temporal consistency (R3D-18, video only)
  Audio    → P(FAKE) — voice synthesis detection (RawNet2-small)
  rPPG     → has_pulse — biological liveness signal

Fusion:
  Phase 1 (no trained meta-learner checkpoint):
    Weighted average of A, B, C with configurable weights.
  Phase 2 (after meta-learner training):
    XGBoost meta-learner trained on calibrated stream probabilities.
    Auto-activates when META_LEARNER_CKPT exists.

Public API:
    from modules.core.ensemble import run_detection
    result = run_detection(media_path)
"""

import os
import time
import pickle
import numpy as np
import cv2
from PIL import Image

from config import (
    DEVICE,
    CONFIDENCE_THRESHOLD,
    STREAM_WEIGHTS,
    ENSEMBLE_WEIGHTS,
    IMAGE_SIZE,
    TEMPORAL_FRAMES,
    VIDEO_EXTENSIONS,
    MODEL_CACHE_DIR,
)
from modules.core.video_utils import (
    extract_frames_uniform,
    extract_frames_sequence,
    extract_audio,
    has_audio_track,
    crop_face_pil,
)
from modules.image_analysis.spatial_detector import stream_a_predict
from modules.video_analysis.frequency_detector import stream_b_predict
from modules.video_analysis.temporal_detector import stream_c_predict
from modules.audio_analysis.audio_detector import audio_predict
from modules.video_analysis.rppg_detector   import extract_rppg_signal

# Path where the trained XGBoost meta-learner is saved
META_LEARNER_CKPT = os.path.join(MODEL_CACHE_DIR, "meta_learner.pkl")


# ── Helpers ───────────────────────────────────────────────────

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _safe_face_crop(pil_img: Image.Image) -> Image.Image:
    """Try face crop with 30% padding; fall back to full image."""
    try:
        return crop_face_pil(pil_img, padding=0.30)
    except Exception:
        return pil_img


# ── Meta-learner (XGBoost) ────────────────────────────────────

_meta_learner = None
_meta_learner_loaded = False


def _load_meta_learner():
    """Load XGBoost meta-learner if checkpoint exists."""
    global _meta_learner, _meta_learner_loaded
    if _meta_learner_loaded:
        return _meta_learner
    _meta_learner_loaded = True
    if os.path.exists(META_LEARNER_CKPT):
        try:
            with open(META_LEARNER_CKPT, "rb") as f:
                _meta_learner = pickle.load(f)
            print(f"[Ensemble] XGBoost meta-learner loaded from {META_LEARNER_CKPT}")
        except Exception as e:
            print(f"[Ensemble] Meta-learner load failed: {e}  — using weighted average.")
    else:
        print("[Ensemble] No meta-learner checkpoint found. Using weighted average.")
    return _meta_learner


def _build_feature_vector(sa: float, sb: float, sc: float,
                           audio: float, has_pulse: float) -> np.ndarray:
    """
    Build the feature vector for the XGBoost meta-learner.
    Features: [sa, sb, sc, audio, has_pulse,
               sa*sb, sa*sc, sb*sc, max(sa,sb,sc), mean(sa,sb,sc)]
    """
    modal_max  = max(sa, sb, sc)
    modal_mean = np.mean([sa, sb, sc])
    return np.array([
        sa, sb, sc, audio, has_pulse,
        sa * sb, sa * sc, sb * sc,
        modal_max, modal_mean,
    ], dtype=np.float32)


def _weighted_fusion(sa: float, sb: float, sc: float,
                     sc_weight_override: float | None = None) -> float:
    """
    Weighted average fusion of stream scores.
    If sc_weight_override is 0.0 (image input), weights are
    renormalised across A and B only.
    """
    w = STREAM_WEIGHTS.copy()
    if sc_weight_override is not None:
        w["temporal"] = sc_weight_override

    total = w["spatial"] + w["frequency"] + w["temporal"]
    return (
        w["spatial"]   / total * sa +
        w["frequency"] / total * sb +
        w["temporal"]  / total * sc
    )


def _risk_level(fake_prob: float) -> str:
    if fake_prob >= 0.85: return "CRITICAL"
    if fake_prob >= 0.70: return "HIGH"
    if fake_prob >= 0.50: return "MEDIUM"
    if fake_prob >= 0.35: return "LOW"
    return "MINIMAL"


# ── Core detection pipeline ───────────────────────────────────

def _detect_image(file_path: str) -> dict:
    """Run all streams on a single image file."""
    pil_img = Image.open(file_path).convert("RGB")
    face    = _safe_face_crop(pil_img)

    sa = stream_a_predict(face)
    sb = stream_b_predict(face)
    sc = 0.5   # no temporal signal for images

    # Fusion — temporal weight=0 → renormalise A+B
    meta = _load_meta_learner()
    if meta is not None:
        feat      = _build_feature_vector(sa, sb, sc, 0.5, 0.5)
        fake_prob = float(meta.predict_proba(feat.reshape(1, -1))[0, 1])
    else:
        fake_prob = _weighted_fusion(sa, sb, sc, sc_weight_override=0.0)

    return {
        "streams": {
            "spatial_texture":  {"fake_prob": round(sa, 4), "label": "Fake" if sa >= 0.5 else "Real"},
            "frequency_domain": {"fake_prob": round(sb, 4), "label": "Fake" if sb >= 0.5 else "Real"},
            "temporal":         {"fake_prob": None, "label": "N/A — image input"},
            "audio":            {"fake_prob": None, "label": "N/A — no audio", "available": False},
            "rppg":             {"has_pulse": None, "bvp_snr": None, "available": False,
                                 "note": "rPPG requires video frames"},
        },
        "ensemble_fake_prob": round(fake_prob, 4),
        "media_type": "image",
        "frame_count": 1,
    }


def _detect_video(file_path: str) -> dict:
    """Run all streams on a video file."""
    # ── Extract frames ────────────────────────────────────────
    try:
        uniform_frames = extract_frames_uniform(file_path, num_frames=16, as_pil=True)
    except Exception as e:
        raise RuntimeError(f"Frame extraction failed: {e}")

    face_frames = [_safe_face_crop(f) for f in uniform_frames]

    # ── Streams A + B (frame-level, averaged) ─────────────────
    sa_scores = []
    sb_scores = []
    for face in face_frames:
        try:
            sa_scores.append(stream_a_predict(face))
        except Exception:
            pass
        try:
            sb_scores.append(stream_b_predict(face))
        except Exception:
            pass

    sa = float(np.mean(sa_scores)) if sa_scores else 0.5
    sb = float(np.mean(sb_scores)) if sb_scores else 0.5

    # ── Stream C (temporal — needs 8-frame sequence) ──────────
    try:
        seq_frames = extract_frames_sequence(file_path, num_frames=TEMPORAL_FRAMES)
        seq_faces  = [_safe_face_crop(f) for f in seq_frames]
        sc = stream_c_predict(seq_faces)
    except Exception as e:
        print(f"[Ensemble] Stream C failed: {e}")
        sc = 0.5

    # ── rPPG (biological signal) ──────────────────────────────
    try:
        from config import RPPG_SNR_THRESHOLD
        import cv2
        cap   = cv2.VideoCapture(file_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        rppg_result = extract_rppg_signal(face_frames, fps=fps)
    except Exception as e:
        print(f"[Ensemble] rPPG failed: {e}")
        rppg_result = {"has_pulse": None, "bvp_snr": None,
                       "available": False, "note": str(e)}

    # ── Audio stream ──────────────────────────────────────────
    audio_result = {"fake_prob": 0.5, "label": "Unknown",
                    "available": False, "note": "No audio track detected."}
    tmp_audio_path = None
    try:
        if has_audio_track(file_path):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_audio_path = tmp.name
            tmp.close()
            extract_audio(file_path, output_path=tmp_audio_path)
            audio_result = audio_predict(tmp_audio_path)
    except Exception as e:
        print(f"[Ensemble] Audio stream failed: {e}")
        audio_result["note"] = str(e)
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except Exception:
                pass

    # ── Fusion ────────────────────────────────────────────────
    audio_score = audio_result.get("fake_prob", 0.5)
    has_pulse_f = 0.0 if (rppg_result.get("has_pulse") is True) else \
                  1.0 if (rppg_result.get("has_pulse") is False) else 0.5

    meta = _load_meta_learner()
    if meta is not None:
        feat      = _build_feature_vector(sa, sb, sc, audio_score, has_pulse_f)
        fake_prob = float(meta.predict_proba(feat.reshape(1, -1))[0, 1])
    else:
        visual_prob = _weighted_fusion(sa, sb, sc)
        # Blend visual + audio (80/20 if audio available, else 100% visual)
        if audio_result["available"]:
            fake_prob = 0.80 * visual_prob + 0.20 * audio_score
        else:
            fake_prob = visual_prob

        # rPPG adjustment: if no pulse detected, nudge P(FAKE) up
        if rppg_result.get("available") and not rppg_result.get("has_pulse"):
            fake_prob = min(1.0, fake_prob + 0.08)

    return {
        "streams": {
            "spatial_texture":  {"fake_prob": round(sa, 4),
                                 "label": "Fake" if sa >= 0.5 else "Real"},
            "frequency_domain": {"fake_prob": round(sb, 4),
                                 "label": "Fake" if sb >= 0.5 else "Real"},
            "temporal":         {"fake_prob": round(sc, 4),
                                 "label": "Fake" if sc >= 0.5 else "Real"},
            "audio":            audio_result,
            "rppg":             rppg_result,
        },
        "ensemble_fake_prob": round(fake_prob, 4),
        "media_type": "video",
        "frame_count": len(uniform_frames),
    }


# ── Public API ────────────────────────────────────────────────

def run_detection(
    media_path: str,
    use_face_crop: bool = True,
    num_video_frames: int = 16,
) -> dict:
    """
    Full detection pipeline for an image or video.

    Returns:
        {
            "label":       "Fake" | "Real",
            "confidence":  float,
            "fake_prob":   float,
            "risk_level":  str,
            "is_deepfake": bool,
            "streams":     { ... per-stream results ... },
            "media_type":  "image" | "video",
            "frame_count": int,
            "inference_ms": float,
            "per_model":   dict,   # legacy compat key
        }
    """
    t_start = time.time()
    _load_meta_learner()   # warm up meta-learner check

    if _is_video(media_path):
        result = _detect_video(media_path)
    else:
        result = _detect_image(media_path)

    fake_prob  = result["ensemble_fake_prob"]
    label      = "Fake" if fake_prob >= CONFIDENCE_THRESHOLD else "Real"
    confidence = fake_prob if label == "Fake" else 1.0 - fake_prob

    inference_ms = (time.time() - t_start) * 1000

    return {
        "label":        label,
        "confidence":   round(confidence, 4),
        "fake_prob":    fake_prob,
        "risk_level":   _risk_level(fake_prob),
        "is_deepfake":  label == "Fake",
        "streams":      result["streams"],
        "media_type":   result["media_type"],
        "frame_count":  result["frame_count"],
        "inference_ms": round(inference_ms, 1),
        # Legacy compat key expected by existing utils/metadata.py
        "per_model": {
            "stream_a_spatial": {
                "label":     result["streams"]["spatial_texture"]["label"],
                "fake_prob": result["streams"]["spatial_texture"]["fake_prob"],
                "note":      "EfficientNet-B4 + SRM branch",
            },
            "stream_b_frequency": {
                "label":     result["streams"]["frequency_domain"]["label"],
                "fake_prob": result["streams"]["frequency_domain"]["fake_prob"],
                "note":      "EfficientNet-B0 on DCT/FFT spectrum",
            },
            "stream_c_temporal": {
                "label":     result["streams"]["temporal"]["label"],
                "fake_prob": result["streams"]["temporal"].get("fake_prob"),
                "note":      "R3D-18 temporal consistency (video only)",
            },
        },
    }