import logging
import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "BHARATSHIELD_MODEL_PATH",
        PROJECT_ROOT / "ml_pipeline" / "models" / "frame_model.pth"
    )
).resolve()

UPLOAD_DIR = Path(os.getenv("BHARATSHIELD_UPLOAD_DIR", BACKEND_ROOT / "uploads")).resolve()

MAX_IMAGE_SIZE_MB = int(os.getenv("BHARATSHIELD_MAX_IMAGE_SIZE_MB", "20"))
MAX_VIDEO_SIZE_MB = int(os.getenv("BHARATSHIELD_MAX_VIDEO_SIZE_MB", "500"))
VIDEO_SAMPLE_FRAMES = int(os.getenv("BHARATSHIELD_VIDEO_SAMPLE_FRAMES", "16"))
VIDEO_MAX_FRAMES_TO_SCAN = int(os.getenv("BHARATSHIELD_VIDEO_MAX_FRAMES_TO_SCAN", "300"))

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

