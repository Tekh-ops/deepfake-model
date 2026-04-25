"""
training/model.py
─────────────────
EfficientNet-B4 fine-tuned for deepfake detection.

Architecture choices:
  • EfficientNet-B4 backbone (ImageNet pretrained via timm)
  • Custom 3-layer classification head with:
      - Global average pool (built into backbone)
      - BatchNorm to stabilise fine-tuning
      - Dropout (0.4) before first FC — heavy regularisation needed
      - Intermediate 512-dim hidden layer with GELU
      - Dropout (0.2) before final FC
      - 2-class output (REAL=0, FAKE=1)
  • Differential learning rates:
      - New head  → full LR (3e-4)
      - Backbone  → 10× lower LR (3e-5)
        Fine-tuning all layers from the start, not frozen
  • Label smoothing 0.1 in the loss (see trainer.py)
  • Frequency-domain branch: optional SRM conv added as the
    first layer to force the model to learn from noise residuals,
    not just visual appearance
"""

import torch
import torch.nn as nn
import timm

from training.config import (
    DEVICE, MODEL_NAME, PRETRAINED, NUM_CLASSES,
    IMAGE_SIZE, DROPOUT_RATE, CACHE_DIR,
    LEARNING_RATE, BACKBONE_LR, WEIGHT_DECAY, BETAS,
)


# ─────────────────────────────────────────────────────────────
# SRM (Steganalysis Rich Model) first-layer filters
# ─────────────────────────────────────────────────────────────

def _make_srm_filters() -> torch.Tensor:
    """
    Build the 30 SRM filter kernels used in forensic steganalysis.
    These high-pass filters expose manipulation artefacts that are
    invisible to the human eye but leave detectable noise signatures.

    We use a simplified 5-filter version — the most discriminative
    kernels from the full 30-filter bank.
    Returns: (5, 3, 3, 3) tensor (5 filters, 3 channels each, 3×3 kernels)
    """
    filters_2d = torch.tensor([
        # Laplacian (sharpening)
        [[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]],
        # Edge horizontal
        [[-1,-1,-1],[ 2, 2, 2],[-1,-1,-1]],
        # Edge vertical
        [[-1, 2,-1],[-1, 2,-1],[-1, 2,-1]],
        # Diagonal
        [[-1,-1, 2],[-1, 2,-1],[ 2,-1,-1]],
        # Anti-diagonal
        [[ 2,-1,-1],[-1, 2,-1],[-1,-1, 2]],
    ], dtype=torch.float32) / 4.0   # normalise

    # Apply same filter to each RGB channel
    filters = filters_2d.unsqueeze(1).repeat(1, 3, 1, 1)
    return filters


class SRMLayer(nn.Module):
    """
    Fixed (non-trainable) SRM high-pass filter layer.
    Applied before the backbone to provide noise residual features.
    Output is concatenated with the original image: 3 + 5 = 8 channels.
    The backbone's first conv therefore needs in_channels=8.
    """
    def __init__(self):
        super().__init__()
        srm_w = _make_srm_filters()
        self.register_buffer("weight", srm_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = nn.functional.conv2d(x, self.weight, padding=1)
        residual = torch.tanh(residual)    # constrain to [-1, 1]
        return torch.cat([x, residual], dim=1)   # (B, 8, H, W)


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 with:
      • Optional SRM pre-processing branch
      • Custom deepfake-optimised classification head
    """

    def __init__(self, use_srm: bool = True):
        super().__init__()
        self.use_srm = use_srm

        # ── SRM branch ────────────────────────────────────────
        if use_srm:
            self.srm = SRMLayer()
            in_chans = 8       # 3 RGB + 5 SRM residual channels
        else:
            self.srm = None
            in_chans = 3

        # ── Backbone ──────────────────────────────────────────
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=0,          # remove original classifier
            in_chans=in_chans,      # adapt first conv for SRM input
            cache_dir=CACHE_DIR,
        )
        feature_dim = self.backbone.num_features   # 1792 for EfficientNet-B4

        # ── Classification head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, NUM_CLASSES),
        )

        # Initialise the new head properly
        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_srm and self.srm is not None:
            x = self.srm(x)
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (before classifier) — used for Grad-CAM."""
        if self.use_srm and self.srm is not None:
            x = self.srm(x)
        return self.backbone(x)

    def get_param_groups(self) -> list:
        """
        Return parameter groups with differential learning rates.
        Called by the trainer when building the optimizer.
        """
        return [
            {"params": self.classifier.parameters(), "lr": LEARNING_RATE},
            {"params": self.backbone.parameters(),   "lr": BACKBONE_LR},
        ] + (
            [{"params": self.srm.parameters(), "lr": 0.0}]
            if self.use_srm and self.srm else []
        )


# ─────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────

def build_loss(class_weights: torch.Tensor | None = None,
               label_smoothing: float = 0.1) -> nn.Module:
    """
    Cross-entropy loss with optional class weighting and label smoothing.
    Label smoothing 0.1 prevents overconfident predictions and improves
    generalisation to unseen deepfake types.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def build_model(use_srm: bool = True) -> DeepfakeDetector:
    model = DeepfakeDetector(use_srm=use_srm)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"[Model] Wrapping in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    return model.to(DEVICE)


def load_checkpoint(model: nn.Module, path: str,
                    strict: bool = True) -> dict:
    """Load weights from a checkpoint file."""
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    # Handle DataParallel prefix
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)
    print(f"[Model] Loaded checkpoint: {path}")
    return ckpt


def save_checkpoint(model: nn.Module, optimizer, scheduler,
                    epoch: int, metrics: dict, path: str):
    """Save a training checkpoint."""
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "epoch":            epoch,
        "model_state_dict": state,
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "metrics":          metrics,
    }, path)