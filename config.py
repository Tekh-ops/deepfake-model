import torch
import os

# ── Device ────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[config] Device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "uploads")
MODEL_CACHE_DIR   = os.path.join(BASE_DIR, ".model_cache")
AUDIT_LOG_FILE    = os.path.join(BASE_DIR, "audit_chain.jsonl")

os.makedirs(UPLOAD_FOLDER,   exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────
# Primary: fine-tuned deepfake classifier — loaded from HuggingFace.
# detector.py tries multiple verified model IDs in order.
# Secondary: EfficientNet-B4 via timm — ImageNet pretrained cross-check.

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "primary":   0.75,   # fine-tuned HF model
    "secondary": 0.25,   # timm EfficientNet-B4 cross-check
}

# Fake threshold — above this → FAKE
CONFIDENCE_THRESHOLD = 0.50

# Image preprocessing
IMAGE_SIZE    = 224
XCEPTION_SIZE = 299

# ── Flask ─────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 500
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "mp4", "mov", "avi"}
VIDEO_EXTENSIONS   = {"mp4", "mov", "avi", "mkv", "webm"}

# ── Legal / Compliance ────────────────────────────────────────
SYSTEM_NAME    = "BharatShield Deepfake Detection System"
SYSTEM_VERSION = "1.0.0-prototype"
LEGAL_AUTHORITY = "IT Rules 2021 (India) / BSA Section 63"