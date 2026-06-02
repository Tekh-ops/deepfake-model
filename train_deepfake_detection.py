#!/usr/bin/env python3
"""
Production-grade binary deepfake detector training (EfficientNet-B4).

Dataset layout:
  final_dataset_aligned/
    real/
    fake/

Dependencies:
  pip install torch torchvision albumentations opencv-python tqdm scikit-learn

Run (example):
  python train_deepfake_detection.py --data_dir final_dataset_aligned --batch_size 32 --workers 4
  python train_deepfake_detection.py --device cuda:0 --epochs_head 5 --epochs_ft 50 --batch_size 64
"""

from __future__ import annotations

import argparse
import csv
import cv2
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torchvision
from torchvision.models import efficientnet_b4

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

def _check_torch_version():
    import warnings
    t_v = torch.__version__.split('+')[0]
    tv_v = torchvision.__version__.split('+')[0]
    compat_map = {
        "2.6.0": ["0.21.0"],
        "2.5.1": ["0.20.1", "0.20.0"],
        "2.5.0": ["0.20.0"],
        "2.4.1": ["0.19.1"],
        "2.4.0": ["0.19.0"],
        "2.3.1": ["0.18.1"],
        "2.3.0": ["0.18.0"],
        "2.2.2": ["0.17.2"],
        "2.2.1": ["0.17.1"],
        "2.2.0": ["0.17.0"],
        "2.1.2": ["0.16.2"],
        "2.1.1": ["0.16.1"],
        "2.1.0": ["0.16.0"],
    }
    t_base = ".".join(t_v.split('.')[:3])
    if t_base in compat_map:
        if tv_v not in compat_map[t_base]:
            warnings.warn(
                f"Version mismatch: torch {torch.__version__} with torchvision {torchvision.__version__}. "
                f"Expected torchvision to be one of {compat_map[t_base]} for torch {t_base}. "
                "Install matching versions from the official PyTorch selector.",
                RuntimeWarning
            )
    else:
        warnings.warn(
            f"Caution: Torch version {torch.__version__} is not in the known compatibility map. "
            f"Using torchvision {torchvision.__version__}. Proceed with caution.",
            RuntimeWarning
        )

_check_torch_version()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Label convention: REAL = 0.0, FAKE = 1.0 (sigmoid / BCE-positive = fake)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# simulate_social_media (optional augmentation path)
# ---------------------------------------------------------------------------
def simulate_social_media_numpy(image_rgb: np.ndarray) -> np.ndarray:
    """Downscale heavily, JPEG-compress artifacting, upscale back."""
    ds = A.Downscale(scale_range=(0.35, 0.9), p=1.0)
    try:
        comp = A.ImageCompression(quality_range=(40, 95), p=1.0)
    except TypeError:
        comp = A.ImageCompression(quality_lower=40, quality_upper=95, p=1.0)
    aug = A.Compose([
        ds,
        comp,
        A.SmallestMaxSize(max_size=380, interpolation=3, p=1.0)
    ])
    return aug(image=image_rgb)["image"]


def maybe_apply_social(image: np.ndarray, p_apply: float, **kwargs) -> np.ndarray:
    """Top-level callable for multiprocessing pickles (avoid nested lambdas inside Compose)."""
    if p_apply <= 0.0:
        return image
    if random.random() < p_apply:
        return simulate_social_media_numpy(image)
    return image


# ---------------------------------------------------------------------------
# Albumentations — exact pipeline (train)
# ---------------------------------------------------------------------------
def cv2_border_reflect() -> int:
    import cv2

    return cv2.BORDER_REFLECT_101


def jpeg_webp_oneof_compression(p_compress: float = 0.8):
    try:
        return A.OneOf(
            [
                A.ImageCompression(quality_range=(40, 95), p=1.0),
                A.ImageCompression(quality_range=(40, 95), p=1.0),
            ],
            p=p_compress,
        )
    except TypeError:
        return A.OneOf(
            [
                A.ImageCompression(quality_lower=40, quality_upper=95, p=1.0),
                A.ImageCompression(quality_lower=40, quality_upper=95, p=1.0),
            ],
            p=p_compress,
        )

def jpeg_webp_fallback_compression(quality_lower, quality_upper, p):
    try:
        return A.ImageCompression(quality_range=(quality_lower, quality_upper), p=p)
    except TypeError:
        return A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=p)


def _train_downscale() -> A.Downscale:
    return A.Downscale(scale_range=(0.35, 0.9), p=0.5)


def _train_coarse_dropout():
    return A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(24, 72),
        hole_width_range=(24, 72),
        p=0.3
    )


def build_train_compose_final(
    simulate_social: bool,
    simulate_social_p: float,
) -> A.Compose:
    """Full train pipeline ending with Normalize + ToTensorV2."""
    base = [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.0625, 0.0625),
            scale=(0.88, 1.12),
            rotate=(-18.0, 18.0),
            fit_output=False,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        jpeg_webp_oneof_compression(p_compress=0.8),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 9), p=1.0),
            ],
            p=0.5,
        ),
        _train_downscale(),
        A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.28, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=12,
            sat_shift_limit=22,
            val_shift_limit=12,
            p=0.6,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.6),
        A.OneOf(
            [
                A.GaussNoise(p=0.4),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.4), p=1.0),
            ],
            p=0.4,
        ),
        _train_coarse_dropout(),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    # Social media simulation is handled inside Dataset.__getitem__ for Windows stability.

    return A.Compose(base)


def build_tta_transforms_list() -> List[A.Compose]:
    """Five TTA variants on uint8 RGB before normalize."""
    return [
        A.Compose([]),

        A.Compose([
            A.HorizontalFlip(p=1.0)
        ]),

        A.Compose(
            [
                A.Affine(
                    translate_percent=(-0.02, 0.02),
                    scale=(0.96, 1.04),
                    rotate=(-4.0, 4.0),
                    fit_output=False,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
            ]
        ),

        A.Compose(
            [
                A.Affine(
                    translate_percent=(-0.02, 0.02),
                    scale=(0.96, 1.04),
                    rotate=(-8.0, -1.0),
                    fit_output=False,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
            ]
        ),

        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=1.0
            )
        ]),
    ]


def build_tta_post_normalize() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DeepfakeAlignedDataset(Dataset):
    """Loads images from real/ and fake/ subfolders. fake -> label 1.0, real -> 0.0."""

    SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        paths: List[Path],
        labels: List[float],
        transform: Optional[A.Compose] = None,
        simulate_social: bool = False,
        simulate_social_p: float = 0.0,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.simulate_social = simulate_social
        self.simulate_social_p = simulate_social_p

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        p = self.paths[idx]
        y = self.labels[idx]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                image = np.array(im, dtype=np.uint8)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to read {p}") from e

        if self.simulate_social and random.random() < self.simulate_social_p:
            image = simulate_social_media_numpy(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_efficientnet_b4_backbone(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import EfficientNet_B4_Weights

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        return efficientnet_b4(weights=weights)
    except Exception:
        return efficientnet_b4(pretrained=pretrained)


class EfficientNetB4Binary(nn.Module):
    """EfficientNet-B4 trunk + custom classification head (sigmoid via BCEWithLogits)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = load_efficientnet_b4_backbone(pretrained)
        self.features = backbone.features
        in_features = backbone.classifier[1].in_features

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    def set_backbone_requires_grad(self, requires: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = requires


# ---------------------------------------------------------------------------
# Focal Loss + label smoothing (binary, logits)
# ---------------------------------------------------------------------------
class FocalLossSigmoid(nn.Module):
    """
    Binary focal loss on logits with label smoothing (soft CE).
    gamma=2.0; alpha emphasizes the positive (fake=1.0) class vs negative.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,) or (B,1); targets: (B,) float in [0,1] (typically hard 0/1)
        """
        logit = logits.view(-1).float()
        t = targets.view(-1).float()
        ls = self.label_smoothing
        t_smooth = t * (1.0 - ls) + 0.5 * ls

        log_p = F.logsigmoid(logit)
        log_1_p = F.logsigmoid(-logit)
        bce = -(t_smooth * log_p + (1.0 - t_smooth) * log_1_p)

        p = torch.sigmoid(logit)
        p_t = p * t_smooth + (1.0 - p) * (1.0 - t_smooth)
        focal = torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma) * bce

        alpha_t = self.alpha * t_smooth + (1.0 - self.alpha) * (1.0 - t_smooth)
        return (alpha_t * focal).mean()


# ---------------------------------------------------------------------------
# MixUp (logits supervision)
# ---------------------------------------------------------------------------
def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        lam = 1.0
        return x, y, y, lam

    lam = float(np.random.beta(alpha, alpha))

    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]

    ya = y
    yb = y[idx]
    return mixed_x, ya, yb, lam


def focal_mixup_loss(
    focal: FocalLossSigmoid,
    logits: torch.Tensor,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """MixUp convex combination of focal losses (targets remain hard labels mixed via lam)."""
    if lam >= 1.0 - 1e-8:
        return focal(logits, ya)
    if lam <= 1e-8:
        return focal(logits, yb)
    return lam * focal(logits, ya) + (1.0 - lam) * focal(logits, yb)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of model parameters (trainable only)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.data.clone().detach()
                continue
            self.shadow[name] = (
                self.decay * self.shadow[name]
                + (1.0 - self.decay) * param.data.detach()
            )

    @torch.no_grad()
    def apply_to_model(self, model: nn.Module) -> None:
        self.backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], device: torch.device) -> None:
        if "shadow" in state_dict:
            self.shadow = {k: v.to(device) for k, v in state_dict["shadow"].items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_hat = (scores >= threshold).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    out: Dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true_i, scores))
    except ValueError:
        out["auc"] = float("nan")

    out["f1"] = float(f1_score(y_true_i, y_hat, zero_division=0))
    out["acc"] = float(accuracy_score(y_true_i, y_hat))
    out["eer"] = float(equal_error_rate(y_true_i.astype(np.float32), scores))
    return out


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    y_true = y_true.astype(np.int32)
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2.0)


# ---------------------------------------------------------------------------
# Data indexing
# ---------------------------------------------------------------------------
def collect_paths(data_dir: Path) -> Tuple[List[Path], List[float]]:
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    paths: List[Path] = []
    labels: List[float] = []

    for d, lab in [(real_dir, 0.0), (fake_dir, 1.0)]:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in DeepfakeAlignedDataset.SUPPORTED:
                paths.append(p)
                labels.append(lab)

    if len(paths) == 0:
        raise FileNotFoundError(f"No images under {real_dir} or {fake_dir}")

    return paths, labels


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    best_auc: float = -1.0
    patience_counter: int = 0


def freeze_backbone(model: EfficientNetB4Binary, freeze: bool) -> None:
    model.set_backbone_requires_grad(not freeze)


def optimizer_for_model(
    model: EfficientNetB4Binary,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def scheduler_one_cycle(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.1,
):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        final_div_factor=1000.0,
        div_factor=25.0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    focal: FocalLossSigmoid,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR],
    device: torch.device,
    epoch_max_norm: float,
    ema: ModelEMA,
    mixup_prob: float,
    mixup_alpha: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        apply_mixup = random.random() < mixup_prob and mixup_alpha > 0

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if apply_mixup:
                mixed_x, ya, yb, lam = mixup_batch(x, y, mixup_alpha)
                logits = model(mixed_x)
                loss = focal_mixup_loss(focal, logits, ya, yb, lam)
            else:
                logits = model(x)
                loss = focal(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=epoch_max_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        ema.update(model)
        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_list: List[A.Compose],
    tta_post: A.Compose,
    threshold: float,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_scores: List[float] = []
    all_true: List[float] = []

    for x_cpu, y in tqdm(loader, desc="val", leave=False):
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        batch_scores: List[np.ndarray] = []

        for img_b in x_cpu:
            if isinstance(img_b, torch.Tensor):
                img_b = img_b.numpy()
            if img_b.ndim == 3 and img_b.shape[0] == 3:
                img_b = _chw_to_hwc_uint8(img_b)

            per_image_logits: List[float] = []
            for tta_pre in tta_list:
                aug = tta_pre(image=img_b)["image"]
                aug = tta_post(image=aug)["image"]
                aug = aug.unsqueeze(0).to(device, non_blocking=True).float()
                logit = model(aug)
                per_image_logits.append(float(torch.sigmoid(logit).item()))
            batch_scores.append(np.mean(per_image_logits))

        all_scores.extend(batch_scores)
        all_true.extend(y.tolist())

    scores = np.asarray(all_scores, dtype=np.float64)
    y_true = np.asarray(all_true, dtype=np.float64)
    m = binary_metrics(y_true, scores, threshold=threshold)
    loss_proxy = float("nan")
    return loss_proxy, m, y_true, scores


def _chw_to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


@torch.no_grad()
def validate_batched(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_scores: List[float] = []
    all_true: List[float] = []

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True).float()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logit = model(x)
        scores = torch.sigmoid(logit).view(-1).cpu().numpy()
        all_scores.extend(scores.tolist())
        if isinstance(y, torch.Tensor):
            all_true.extend(y.numpy().tolist())
        else:
            all_true.extend(y.tolist())

    scores = np.asarray(all_scores, dtype=np.float64)
    y_true = np.asarray(all_true, dtype=np.float64)
    m = binary_metrics(y_true, scores, threshold=threshold)
    return float("nan"), m, y_true, scores


class DeepfakeAlignedRawValDataset(Dataset):
    """Validation: yield uint8 HWC RGB (no Albumentations)."""

    SUPPORTED = DeepfakeAlignedDataset.SUPPORTED

    def __init__(self, paths: List[Path], labels: List[float]) -> None:
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        p = self.paths[idx]
        y = self.labels[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            image = np.array(im, dtype=np.uint8)
        return image, torch.tensor(y, dtype=torch.float32)


def val_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return np.stack(images, axis=0), torch.stack(labels, dim=0)


def build_val_tta_dataloader(
    val_paths: List[Path],
    val_labels: List[float],
    workers: int,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    ds = DeepfakeAlignedRawValDataset(val_paths, val_labels)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        collate_fn=val_collate_fn,
    )


def build_val_dataloader(
    val_paths: List[Path],
    val_labels: List[float],
    batch_size: int,
    workers: int,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    ds = DeepfakeAlignedDataset(val_paths, val_labels, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )


def build_train_dataloader_only(
    train_paths: List[Path],
    train_labels: List[float],
    batch_size: int,
    workers: int,
    train_transform: A.Compose,
    simulate_social: bool = False,
    simulate_social_p: float = 0.0,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    train_ds = DeepfakeAlignedDataset(
        train_paths,
        train_labels,
        transform=train_transform,
        simulate_social=simulate_social,
        simulate_social_p=simulate_social_p,
    )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def run_phase(
    model: EfficientNetB4Binary,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    focal: FocalLossSigmoid,
    lr: float,
    weight_decay: float,
    epochs: int,
    grad_clip: float,
    ema: ModelEMA,
    state: TrainState,
    log_rows: List[Dict],
    mixup_prob: float,
    mixup_alpha: float,
    phase_name: str,
    thr: float,
    early_patience: int,
    enable_early_stop: bool,
    out_dir: Path,
    scaler: GradScaler,
    global_epoch_start: int,
    resume_ckpt: Optional[Dict] = None,
) -> Tuple[bool, int]:
    """
    Train one phase. Writes rows to log_rows.
    enable_early_stop: typically False during short warmup head training.
    Returns (stopped_early, next_global_epoch).
    """
    optimizer = optimizer_for_model(model, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * epochs
    scheduler = scheduler_one_cycle(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    start_ep = 0
    if resume_ckpt is not None and resume_ckpt.get("phase") == phase_name:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_ep = (resume_ckpt["epoch"] - global_epoch_start) + 1
        print(f"Resuming {phase_name} phase from epoch {start_ep}/{epochs}")

    for ep in range(start_ep, epochs):
        global_epoch = global_epoch_start + ep
        tr_loss = train_one_epoch(
            model,
            train_loader,
            focal,
            optimizer,
            scaler,
            scheduler,
            device,
            grad_clip,
            ema,
            mixup_prob=mixup_prob,
            mixup_alpha=mixup_alpha,
        )

        ema.apply_to_model(model)
        _, metrics, _, _ = validate_batched(model, val_loader, device, thr)
        ema.restore(model)

        row = {
            "phase": phase_name,
            "epoch_in_phase": ep,
            "global_epoch": global_epoch,
            "train_loss": tr_loss,
            "val_auc": metrics["auc"],
            "val_f1": metrics["f1"],
            "val_eer": metrics["eer"],
            "val_acc": metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)

        print(
            f"[{phase_name}] ep {ep+1}/{epochs} (global {global_epoch}) | train_loss={tr_loss:.4f} | "
            f"val_auc={metrics['auc']:.4f} val_f1={metrics['f1']:.4f} "
            f"val_eer={metrics['eer']:.4f} val_acc={metrics['acc']:.4f}"
        )

        auc = metrics["auc"]
        if not np.isnan(auc) and auc > state.best_auc:
            state.best_auc = auc
            state.patience_counter = 0
            ema.apply_to_model(model)
            torch.save(
                {
                    "model_state_dict": _cpu_state_dict(model),
                    "best_val_auc": auc,
                    "threshold": thr,
                    "label_real": 0.0,
                    "label_fake": 1.0,
                    "architecture": "efficientnet_b4_binary_sigmoid",
                },
                out_dir / "best_model.pth",
            )
            ema.restore(model)
        else:
            if enable_early_stop:
                state.patience_counter += 1

        torch.save(
            {
                "epoch": global_epoch,
                "phase": phase_name,
                "model_state_dict": _cpu_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "best_auc": state.best_auc,
            },
            out_dir / "latest_checkpoint.pth",
        )

        if enable_early_stop and state.patience_counter >= early_patience:
            print(f"Early stopping: no val_auc improvement for {early_patience} epochs.")
            return True, global_epoch_start + ep + 1

    return False, global_epoch_start + epochs


def append_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 deepfake detector")
    parser.add_argument("--data_dir", type=str, default="final_dataset_aligned")
    parser.add_argument("--out_dir", type=str, default="training_output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu; default auto")
    parser.add_argument("--epochs_head", type=int, default=5, help="Phase 1: frozen backbone")
    parser.add_argument("--epochs_ft", type=int, default=55, help="Phase 2: full fine-tune (40–60)")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_ft", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--early_patience", type=int, default=7)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--simulate_social", action="store_true")
    parser.add_argument("--simulate_social_p", type=float, default=0.05)
    parser.add_argument("--val_threshold", type=float, default=0.5)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--tta_final_eval", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv = out_dir / "training_logs.csv"
    if log_csv.exists() and not args.resume:
        log_csv.unlink()

    paths, labels = collect_paths(data_dir)
    y_int = np.asarray(labels, dtype=np.int32)
    train_p, val_p, train_y, val_y = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_int,
    )

    train_real = sum(1 for y in train_y if y < 0.5)
    train_fake = len(train_y) - train_real
    val_real = sum(1 for y in val_y if y < 0.5)
    val_fake = len(val_y) - val_real
    print(f"Train: {len(train_y)} (real={train_real}, fake={train_fake}) | Val: {len(val_y)} (real={val_real}, fake={val_fake})")

    train_transform = build_train_compose_final(
        simulate_social=args.simulate_social,
        simulate_social_p=args.simulate_social_p,
    )
    train_loader = build_train_dataloader_only(
        train_p,
        train_y,
        args.batch_size,
        args.workers,
        train_transform,
        simulate_social=args.simulate_social,
        simulate_social_p=args.simulate_social_p,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_val_dataloader(val_p, val_y, args.val_batch_size, args.workers, args.prefetch_factor)

    tta_list = build_tta_transforms_list()
    tta_post = build_tta_post_normalize()

    model = EfficientNetB4Binary(pretrained=not args.no_pretrained).to(device)
    focal = FocalLossSigmoid(gamma=2.0, alpha=0.75, label_smoothing=0.1).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    state = TrainState()
    log_rows: List[Dict] = []
    resume_ckpt = None
    skip_warmup = False

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if resume_ckpt.get("ema_state_dict"):
            ema.load_state_dict(resume_ckpt["ema_state_dict"], device)
        if resume_ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        state.best_auc = resume_ckpt.get("best_auc", -1.0)
        if resume_ckpt.get("phase") == "finetune":
            skip_warmup = True

    global_epoch = 0
    freeze_backbone(model, freeze=True)
    if not skip_warmup:
        _, global_epoch = run_phase(
            model,
            train_loader,
            val_loader,
            device,
            focal,
            lr=args.lr_head,
            weight_decay=args.weight_decay,
            epochs=args.epochs_head,
            grad_clip=args.grad_clip,
            ema=ema,
            state=state,
            log_rows=log_rows,
            mixup_prob=args.mixup_prob,
            mixup_alpha=args.mixup_alpha,
            phase_name="warmup",
            thr=args.val_threshold,
            early_patience=args.early_patience,
            enable_early_stop=False,
            out_dir=out_dir,
            scaler=scaler,
            global_epoch_start=global_epoch,
            resume_ckpt=resume_ckpt if resume_ckpt and resume_ckpt.get("phase") == "warmup" else None,
        )
    else:
        global_epoch = args.epochs_head

    freeze_backbone(model, freeze=False)
    if not (resume_ckpt and resume_ckpt.get("phase") == "finetune"):
        state.patience_counter = 0

    finetune_stopped_early, _ = run_phase(
        model,
        train_loader,
        val_loader,
        device,
        focal,
        lr=args.lr_ft,
        weight_decay=args.weight_decay,
        epochs=args.epochs_ft,
        grad_clip=args.grad_clip,
        ema=ema,
        state=state,
        log_rows=log_rows,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        phase_name="finetune",
        thr=args.val_threshold,
        early_patience=args.early_patience,
        enable_early_stop=True,
        out_dir=out_dir,
        scaler=scaler,
        global_epoch_start=global_epoch,
        resume_ckpt=resume_ckpt if resume_ckpt and resume_ckpt.get("phase") == "finetune" else None,
    )
    if finetune_stopped_early:
        print("Fine-tune stopped early (validation AUC patience exhausted).")

    append_csv(log_rows, log_csv)

    ema.apply_to_model(model)
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "best_val_auc": state.best_auc,
            "threshold": args.val_threshold,
            "label_real": 0.0,
            "label_fake": 1.0,
            "architecture": "efficientnet_b4_binary_sigmoid",
        },
        out_dir / "ema_model.pth",
    )
    
    if args.tta_final_eval:
        print("Running final evaluation with TTA...")
        val_tta_loader = build_val_tta_dataloader(val_p, val_y, args.workers, args.prefetch_factor)
        _, tta_metrics, _, _ = validate_tta(model, val_tta_loader, device, tta_list, tta_post, args.val_threshold)
        print(f"[TTA Final Eval] val_auc={tta_metrics['auc']:.4f} val_f1={tta_metrics['f1']:.4f} val_eer={tta_metrics['eer']:.4f} val_acc={tta_metrics['acc']:.4f}")

    ema.restore(model)

    print(f"Best val_auc (EMA snapshot when improved): {state.best_auc:.4f}")
    print(f"Saved: {out_dir / 'best_model.pth'}, {out_dir / 'ema_model.pth'}, {log_csv}")


if __name__ == "__main__":
    main()
