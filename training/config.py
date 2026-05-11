"""
training/config.py
──────────────────
All hyperparameters and dataset paths in one place.
Edit this file before launching any training run.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────────────────────

def get_device():
    # Priority: CUDA (RTX 3070) > MPS (M1) > CPU
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        n_gpus   = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[config] GPU: {gpu_name}  ({n_gpus} device(s), {vram_gb:.1f} GB VRAM)")
        return device, n_gpus
    if torch.backends.mps.is_available():
        print("[config] Apple MPS (M1/M2 — dev mode)")
        return torch.device("mps"), 1
    print("[config] WARNING: No GPU — CPU only (training will be very slow)")
    return torch.device("cpu"), 1

DEVICE, NUM_GPUS = get_device()

# ─────────────────────────────────────────────────────────────
# Paths  — EDIT THESE to point at your data
# ─────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT     = os.path.join(BASE_DIR, "data")          # root for all datasets
OUTPUT_DIR    = os.path.join(BASE_DIR, "checkpoints")   # saved model weights
LOG_DIR       = os.path.join(BASE_DIR, "logs")          # tensorboard / CSV logs
CACHE_DIR     = os.path.join(BASE_DIR, ".model_cache")  # HF / timm cache

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Dataset paths  (set these after downloading)
# ─────────────────────────────────────────────────────────────
#
# Expected folder layout:
#   data/
#     ff_plus/                     ← FaceForensics++
#       real/   (extracted frames from pristine videos)
#       fake/   (extracted frames from all 4 manipulation types)
#     dfdc/                        ← DFDC (Kaggle)
#       real/
#       fake/
#     celebdf/                     ← CelebDF-v2
#       real/
#       fake/
#     wild/                        ← WildDeepfake (internet-sourced)
#       real/
#       fake/
#     extra_real/                  ← any additional real face images
#     extra_fake/                  ← diffusion / GAN generated faces
#
# Each leaf folder contains flat JPEG/PNG frames (no sub-folders).

DATASET_ROOTS = {
    "celebdf":    os.path.join(DATA_ROOT, "celebdf"),
    "celebdf_v2": os.path.join(DATA_ROOT, "celebdf_v2"),
    "dfdc":       os.path.join(DATA_ROOT, "dfdc"),
    "asvspoof":   os.path.join(DATA_ROOT, "asvspoof"),
    "ffhq":       os.path.join(DATA_ROOT, "ffhq"),
}

# Only include datasets whose folders actually exist
ACTIVE_DATASETS = [k for k, v in DATASET_ROOTS.items() if os.path.isdir(v)]
print(f"[config] Active datasets: {ACTIVE_DATASETS}")

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

MODEL_NAME      = "efficientnet_b4"   # timm identifier
PRETRAINED      = True                # start from ImageNet weights
NUM_CLASSES     = 2                   # 0=REAL, 1=FAKE
IMAGE_SIZE      = 224                 # input resolution
DROPOUT_RATE    = 0.4

# Path where the best fine-tuned checkpoint is saved
BEST_CKPT       = os.path.join(OUTPUT_DIR, "best_efficientnet_b4.pth")
LAST_CKPT       = os.path.join(OUTPUT_DIR, "last_efficientnet_b4.pth")

# ─────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────

# ── RTX 3070 (8 GB VRAM) optimised settings ─────────────────
# Memory budget: ~6.5 GB effective  (0.5 OS + 1 GB activations buffer)
# EfficientNet-B4 @ 224px: ~420 MB / batch of 16
EPOCHS              = 40
BATCH_SIZE          = 24         # per GPU — fits in 8 GB with AMP
ACCUMULATION_STEPS  = 4          # effective batch = 24 × 4 = 96 samples
NUM_WORKERS         = 8          # i9-10900X has 20 threads; 8 is sweet spot
PIN_MEMORY          = True

# Optimiser
LEARNING_RATE       = 2e-4       # new head (classifier + SRM branch)
BACKBONE_LR         = 1.5e-5    # pretrained backbone — 13× lower
WEIGHT_DECAY        = 1e-4
BETAS               = (0.9, 0.999)
GRAD_CLIP           = 1.0        # max gradient norm

# LR schedule
WARMUP_EPOCHS       = 4          # linear warmup
LR_MIN              = 5e-7       # cosine annealing floor

# Regularisation
LABEL_SMOOTHING     = 0.1        # prevents overconfidence
MIXUP_ALPHA         = 0.4        # strong MixUp — prevents overfit
CUTMIX_PROB         = 0.5        # probability of CutMix augmentation
DROPOUT_RATE        = 0.4

# Class weighting (addresses real/fake imbalance in some datasets)
# Set to None to disable (uses loss-level class weights instead)
CLASS_WEIGHTS       = None

# ─────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────

TRAIN_SPLIT     = 0.80
VAL_SPLIT       = 0.10
TEST_SPLIT      = 0.10
RANDOM_SEED     = 42

# Maximum samples per dataset (None = use all)
# Set lower during debugging, None for full training
MAX_PER_DATASET = None     # e.g. 10000

# ─────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────

# JPEG compression simulation (social media re-encoding)
JPEG_QUALITY_RANGE  = (40, 95)   # wider range than before — more realistic

# Horizontal flip probability
HFLIP_PROB          = 0.5

# RandAugment
RANDAUGMENT_N       = 2          # number of augmentation ops per image
RANDAUGMENT_M       = 9          # magnitude of each op

# Random erasing (simulates occlusion)
ERASE_PROB          = 0.2
ERASE_SCALE         = (0.02, 0.15)

# ── AMP (Automatic Mixed Precision) ──────────────────────────
# ALWAYS True on CUDA — critical for 8 GB VRAM
USE_AMP             = True

# ── Adversarial training ──────────────────────────────────────
ADV_TRAIN_PROB      = 0.20       # 20% of batches get PGD adversarial examples
ADV_EPS             = 4 / 255    # L∞ ball radius
ADV_ALPHA           = 1 / 255    # PGD step size
ADV_STEPS           = 7          # PGD steps per batch

# ── SWAD (domain generalisation) ─────────────────────────────
# Collect weights during the last 40% of training
SWAD_START_FRAC     = 0.60       # start collecting at 60% through training
SWAD_ENABLED        = True

# ─────────────────────────────────────────────────────────────
# Logging and checkpointing
# ─────────────────────────────────────────────────────────────

LOG_INTERVAL    = 50          # log every N batches
EVAL_INTERVAL   = 1           # evaluate on validation every N epochs
SAVE_TOP_K      = 3           # keep best K checkpoints (KEEP_BEST)
KEEP_BEST       = 3           # alias
PATIENCE        = 8           # early stopping patience (epochs)
SAVE_EVERY      = 2           # save checkpoint every N epochs

# Metrics to track
PRIMARY_METRIC  = "val_auc"   # metric for best model selection
TARGET_AUC      = 0.95        # early stop if AUC exceeds this