"""
training/dataset.py
───────────────────
Multi-source deepfake detection dataset.

Handles:
  • FaceForensics++   (face swap, reenactment, neural textures)
  • DFDC              (diverse generation methods, demographics)
  • CelebDF-v2        (high-quality, hard-to-detect deepfakes)
  • WildDeepfake      (in-the-wild internet content)
  • Extra real/fake   (diffusion / GAN generated extras)

All sources are merged, shuffled, and split into train/val/test.
Class imbalance is addressed through WeightedRandomSampler.
"""

import os
import io
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from training.config import (
    DATASET_ROOTS,
    ACTIVE_DATASETS,
    IMAGE_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
    MAX_PER_DATASET,
    JPEG_QUALITY_RANGE,
    HFLIP_PROB,
    ERASE_PROB,
    ERASE_SCALE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True   # handle partially-downloaded images

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABEL_REAL = 0
LABEL_FAKE = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────
# JPEG compression augmentation
# ─────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """
    Simulate social media re-encoding.
    Deepfakes are almost always re-compressed at least once before
    reaching a detector — training without this augmentation causes
    a huge accuracy drop in production.
    """
    def __init__(self, quality_range=(50, 95)):
        self.qmin, self.qmax = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ─────────────────────────────────────────────────────────────
# Frequency domain augmentation (SRM-like high-pass filter)
# ─────────────────────────────────────────────────────────────

class SRMHighPassFilter:
    """
    Apply a Laplacian-style high-pass filter to expose noise residuals.
    This is a simplified version of the SRM (Steganalysis Rich Model)
    preprocessing used in forensic deepfake detection research.
    Applied with probability p during training.
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        import cv2
        arr = np.array(img).astype(np.float32)
        # 3x3 Laplacian kernel
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        channels = []
        for c in range(arr.shape[2]):
            filtered = cv2.filter2D(arr[:, :, c], -1, kernel)
            channels.append(np.clip(filtered, 0, 255))
        filtered_img = np.stack(channels, axis=2).astype(np.uint8)
        # Blend 30% filtered, 70% original to preserve visual quality
        blended = (0.7 * arr + 0.3 * filtered_img).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)


# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        # Geometric
        T.Resize((image_size + 32, image_size + 32)),   # slightly larger then crop
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=HFLIP_PROB),
        T.RandomApply([T.RandomRotation(degrees=10)], p=0.3),
        T.RandomApply([T.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.2, hue=0.05
        )], p=0.5),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),

        # Frequency domain / compression simulation
        RandomJPEGCompression(quality_range=JPEG_QUALITY_RANGE),
        SRMHighPassFilter(p=0.3),

        # Tensor + normalise
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Occlusion simulation
        T.RandomErasing(
            p=ERASE_PROB,
            scale=ERASE_SCALE,
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────

def _scan_class_folder(folder: str, label: int,
                       max_samples: int | None = None) -> list:
    """
    Recursively scan a folder and return a list of (path, label) tuples.
    Filters to IMAGE_EXTS only.
    """
    paths = []
    folder = Path(folder)
    if not folder.exists():
        return paths

    for p in folder.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append((str(p), label))

    if max_samples and len(paths) > max_samples:
        random.seed(RANDOM_SEED)
        paths = random.sample(paths, max_samples)

    return paths


def build_sample_list(max_per_dataset: int | None = MAX_PER_DATASET) -> list:
    """
    Scan all active dataset roots and return a merged list of
    (file_path, label) tuples.
    """
    all_samples = []

    for key in ACTIVE_DATASETS:
        root = DATASET_ROOTS[key]
        real_dir = os.path.join(root, "real")
        fake_dir = os.path.join(root, "fake")

        real_samples = _scan_class_folder(real_dir, LABEL_REAL, max_per_dataset)
        fake_samples = _scan_class_folder(fake_dir, LABEL_FAKE, max_per_dataset)

        print(f"  [{key:12s}]  real={len(real_samples):6d}  fake={len(fake_samples):6d}")
        all_samples.extend(real_samples)
        all_samples.extend(fake_samples)

    print(f"  Total: {len(all_samples)} samples")
    return all_samples


# ─────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for binary deepfake classification.

    Args:
        samples:   list of (file_path, label) tuples
        transform: torchvision transform to apply
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Corrupted or missing file — return a blank image
            print(f"[WARNING] Could not load {path}: {e}")
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# MixUp / CutMix
# ─────────────────────────────────────────────────────────────

def mixup_batch(images: torch.Tensor, labels: torch.Tensor,
                alpha: float = 0.2) -> tuple:
    """Apply MixUp to a batch. Returns mixed images and (lam, labels_a, labels_b)."""
    if alpha <= 0:
        return images, labels, None

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    rand_idx   = torch.randperm(batch_size)

    mixed = lam * images + (1 - lam) * images[rand_idx]
    return mixed, labels, (lam, labels[rand_idx])


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor,
                 prob: float = 0.5) -> tuple:
    """Apply CutMix to a batch with probability `prob`."""
    if random.random() > prob:
        return images, labels, None

    batch_size, _, H, W = images.shape
    rand_idx = torch.randperm(batch_size)

    # Random box
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed, labels, (lam_actual, labels[rand_idx])


# ─────────────────────────────────────────────────────────────
# Train/val/test split and DataLoader factory
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_per_dataset: int | None = MAX_PER_DATASET,
) -> tuple:
    """
    Build and return (train_loader, val_loader, test_loader, class_weights).

    class_weights is a [w_real, w_fake] tensor for weighted loss.
    """
    print("\n[Dataset] Scanning datasets...")
    samples = build_sample_list(max_per_dataset)

    if len(samples) == 0:
        raise RuntimeError(
            "No data found. Check DATASET_ROOTS in training/config.py\n"
            "and ensure your data folders follow the real/ fake/ layout."
        )

    # Stratified shuffle split
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n       = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    train_samples = samples[:n_train]
    val_samples   = samples[n_train:n_train + n_val]
    test_samples  = samples[n_train + n_val:]

    print(f"[Dataset] Split → train={len(train_samples)}, "
          f"val={len(val_samples)}, test={len(test_samples)}")

    # Class weights for loss function
    labels   = [s[1] for s in train_samples]
    n_real   = labels.count(LABEL_REAL)
    n_fake   = labels.count(LABEL_FAKE)
    total    = len(labels)
    w_real   = total / (2.0 * n_real) if n_real > 0 else 1.0
    w_fake   = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32)
    print(f"[Dataset] Class weights → real={w_real:.3f}  fake={w_fake:.3f}")

    # Weighted sampler for balanced mini-batches
    sample_weights = [w_real if s[1] == LABEL_REAL else w_fake
                      for s in train_samples]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_samples),
        replacement = True,
    )

    train_ds  = DeepfakeDataset(train_samples, get_train_transform())
    val_ds    = DeepfakeDataset(val_samples,   get_val_transform())
    test_ds   = DeepfakeDataset(test_samples,  get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, class_weights