import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from .config import VIDEO_MAX_FRAMES_TO_SCAN, VIDEO_SAMPLE_FRAMES
from .model_loader import get_device, get_model
from .preprocess import preprocess_image


LOGGER = logging.getLogger(__name__)
IDX_TO_LABEL = {0: "real", 1: "fake"}


def _predict_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    model = get_model()
    device = get_device()
    x = tensor.to(device)

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1)
        prob_fake = probs[0][1].item()

    THRESHOLD = 0.6

    label = "fake" if prob_fake > THRESHOLD else "real"
    confidence = prob_fake if label == "fake" else 1 - prob_fake

    return {
        "label": label,
        "confidence": confidence
    }

def predict(image: Image.Image) -> Dict[str, Any]:
    start = time.perf_counter()
    result = _predict_tensor(preprocess_image(image))
    elapsed_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("Image inference completed in %.2f ms", elapsed_ms)
    return {
        "label": result["label"],
        "confidence": result["confidence"]
    }


def _sample_frame_indices(total_frames: int, sample_count: int) -> List[int]:
    if total_frames <= 0:
        return []
    sample_count = max(1, min(sample_count, total_frames))
    return sorted(set(int(v) for v in torch.linspace(0, total_frames - 1, sample_count).tolist()))


def predict_video(video_path: str) -> Dict[str, Any]:
    start = time.perf_counter()
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_frames, VIDEO_MAX_FRAMES_TO_SCAN) if total_frames > 0 else VIDEO_MAX_FRAMES_TO_SCAN
        indices = _sample_frame_indices(total_frames, VIDEO_SAMPLE_FRAMES)
        if not indices:
            raise ValueError("Video has no readable frames.")

        idx_set = set(indices)
        frame_predictions: List[Dict[str, Any]] = []
        frame_idx = 0
        while cap.isOpened() and frame_idx <= max(indices):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in idx_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).convert("RGB")
                pred = _predict_tensor(preprocess_image(pil_img))
                frame_predictions.append(
                    {
                        "frame_index": frame_idx,
                        "prediction": pred["label"],
                        "confidence": pred["confidence"],
                    }
                )
            frame_idx += 1

        if not frame_predictions:
            raise ValueError("No frames could be analyzed from the video.")

        fake_scores = [fp["confidence"] if fp["prediction"] == "fake" else (1.0 - fp["confidence"]) for fp in frame_predictions]
        avg_fake_score = float(sum(fake_scores) / len(fake_scores))
        final_prediction = "fake" if avg_fake_score >= 0.5 else "real"
        confidence = avg_fake_score if final_prediction == "fake" else 1.0 - avg_fake_score

        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("Video inference completed in %.2f ms using %d frames", elapsed_ms, len(frame_predictions))
        return {
            "final_prediction": final_prediction,
            "confidence": float(confidence),
            "frame_predictions": frame_predictions,
            "frames_analyzed": len(frame_predictions),
        }
    finally:
        cap.release()

