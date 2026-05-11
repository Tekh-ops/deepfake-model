"""
config.py
─────────
Inference-time configuration for BharatShield.
Loaded by app.py and all models/ modules.

Device priority: CUDA > MPS > CPU  (RTX 3070 first, M1 fallback, CPU last)
All secrets are read from .env — never hardcoded here.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()  # load .env before anything else

# ── Device ────────────────────────────────────────────────────
# CRITICAL: CUDA must be checked BEFORE MPS — the old code had this backwards.
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[config] CUDA GPU: {name}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[config] Apple MPS (M1/M2)")
        return torch.device("mps")
    print("[config] WARNING: No GPU — using CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
AUDIT_LOG_FILE  = os.path.join(BASE_DIR, "audit_chain.jsonl")

os.makedirs(UPLOAD_FOLDER,   exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Point HuggingFace and torch to the local cache so models
# only download once and never litter the home directory.
os.environ.setdefault("HF_HOME",    MODEL_CACHE_DIR)
os.environ.setdefault("TORCH_HOME", MODEL_CACHE_DIR)

# ── Detection thresholds ──────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.50"))

# ── Ensemble weights (legacy 2-model — kept for detector.py compat) ─
ENSEMBLE_WEIGHTS = {
    "primary":   0.75,
    "secondary": 0.25,
}

# ── Three-stream ensemble weights ────────────────────────────
# Tuned heuristically; will be replaced by trained XGBoost meta-learner
# once the meta-learner checkpoint exists.
STREAM_WEIGHTS = {
    "spatial":   0.40,   # Stream A: EfficientNet-B4 + SRM
    "frequency": 0.30,   # Stream B: DCT spectrum EfficientNet-B0
    "temporal":  0.30,   # Stream C: R3D-18 / frame-diff
}

# ── Inference ────────────────────────────────────────────────
IMAGE_SIZE         = 224
XCEPTION_SIZE      = 299
TEMPORAL_FRAMES    = 8          # frames fed to Stream C
INFERENCE_TIMEOUT_S = int(os.getenv("INFERENCE_TIMEOUT_S", "120"))

# ── Audio / video ─────────────────────────────────────────────
VIDEO_EXTENSIONS   = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS   = {".wav", ".mp3", ".aac", ".flac", ".m4a"}
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Flask ─────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 500
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "mp4", "mov", "avi", "wav", "mp3"}

# ── Fact-check ────────────────────────────────────────────────
NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
HARM_THRESHOLD     = float(os.getenv("HARM_THRESHOLD", "0.50"))
NEWS_DAYS_BACK     = int(os.getenv("NEWS_DAYS_BACK", "30"))
DDG_MAX_RESULTS    = int(os.getenv("DDG_MAX_RESULTS", "5"))
NEWS_MAX_ARTICLES  = int(os.getenv("NEWS_MAX_ARTICLES", "5"))

# ── rPPG ──────────────────────────────────────────────────────
RPPG_BPM_LOW  = 0.7   # Hz — 42 BPM
RPPG_BPM_HIGH = 4.0   # Hz — 240 BPM
RPPG_SNR_THRESHOLD = 0.3   # below this → likely synthetic face

# ── Legal / Compliance ────────────────────────────────────────
SYSTEM_NAME     = "BharatShield Deepfake Detection System"
SYSTEM_VERSION  = "2.0.0"
LEGAL_AUTHORITY = "IT Rules 2021 (India) / BSA Section 63"