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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[config] GPU: {gpu_name}  ({n_gpus} device(s))")
        return device, n_gpus
    if torch.backends.mps.is_available():
        print("[config] Using Apple MPS (M1/M2)")
        return torch.device("mps"), 1
    print("[config] WARNING: No GPU found — using CPU")
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
    "ff_plus":  os.path.join(DATA_ROOT, "ff_plus"),
    "dfdc":     os.path.join(DATA_ROOT, "dfdc"),
    "celebdf":  os.path.join(DATA_ROOT, "celebdf"),
    "wild":     os.path.join(DATA_ROOT, "wild"),
    "extra":    os.path.join(DATA_ROOT, "extra_real"),   # real partner
    "extra_f":  os.path.join(DATA_ROOT, "extra_fake"),   # fake partner
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

EPOCHS              = 30
BATCH_SIZE          = 32         # per GPU — effective batch = BATCH_SIZE * NUM_GPUS
ACCUMULATION_STEPS  = 2          # gradient accumulation (effective batch *= this)
NUM_WORKERS         = 8          # DataLoader workers
PIN_MEMORY          = True

# Optimiser
LEARNING_RATE       = 3e-4       # initial LR for the new classification head
BACKBONE_LR         = 3e-5       # 10x lower for the pretrained backbone
WEIGHT_DECAY        = 1e-4
BETAS               = (0.9, 0.999)
GRAD_CLIP           = 1.0        # max gradient norm

# LR schedule
WARMUP_EPOCHS       = 3          # linear warmup
LR_MIN              = 1e-6       # cosine annealing floor

# Regularisation
LABEL_SMOOTHING     = 0.1        # prevents overconfidence
MIXUP_ALPHA         = 0.2        # MixUp augmentation strength (0 = off)
CUTMIX_PROB         = 0.5        # probability of CutMix augmentation

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

# JPEG compression simulation — deepfakes are always re-encoded on social media
JPEG_QUALITY_RANGE  = (50, 95)

# Horizontal flip probability
HFLIP_PROB          = 0.5

# Random erasing (simulates occlusion)
ERASE_PROB          = 0.2
ERASE_SCALE         = (0.02, 0.15)

# ─────────────────────────────────────────────────────────────
# Logging and checkpointing
# ─────────────────────────────────────────────────────────────

LOG_INTERVAL    = 50          # log every N batches
EVAL_INTERVAL   = 1           # evaluate on validation every N epochs
SAVE_TOP_K      = 3           # keep best K checkpoints
PATIENCE        = 7           # early stopping patience (epochs)

# Metrics to track
PRIMARY_METRIC  = "val_auc"   # metric for best model selection