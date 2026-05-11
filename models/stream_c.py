"""
models/stream_c.py
──────────────────
Stream C — Temporal Consistency (R3D-18 + FrameDiffCNN fallback)
Only runs on VIDEO inputs. Images return 0.5 (no temporal signal).
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

from config import DEVICE, TEMPORAL_FRAMES

_FRAME_SIZE    = 112
_KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
_KINETICS_STD  = [0.22803, 0.22145,  0.216989]


# ── Fallback: frame-difference CNN ───────────────────────────

class FrameDiffCNN(nn.Module):
    """
    Stacks absolute difference maps between consecutive frames,
    then runs a small CNN. Lightweight fallback for R3D-18.
    """
    def __init__(self, n_frames: int = TEMPORAL_FRAMES, dropout: float = 0.4):
        super().__init__()
        in_ch = (n_frames - 1) * 3  # diffs between adjacent frames
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ── R3D-18 wrapper ────────────────────────────────────────────

class R3D18Detector(nn.Module):
    """3D ResNet-18 with binary classification head."""
    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            weights  = R3D_18_Weights.DEFAULT if pretrained else None
            backbone = r3d_18(weights=weights)
        except (ImportError, AttributeError):
            import torchvision.models.video as vm
            backbone = vm.r3d_18(pretrained=pretrained)

        feat_dim    = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def get_param_groups(self) -> list:
        return [
            {"params": self.head.parameters(),     "lr": 2e-4},
            {"params": self.backbone.parameters(), "lr": 1.5e-5},
        ]


# ── Tensor builders ───────────────────────────────────────────

def _frame_tf():
    return transforms.Compose([
        transforms.Resize((_FRAME_SIZE, _FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_KINETICS_MEAN, std=_KINETICS_STD),
    ])

def _diff_tf():
    return transforms.Compose([
        transforms.Resize((_FRAME_SIZE, _FRAME_SIZE)),
        transforms.ToTensor(),
    ])


def frames_to_r3d_tensor(pil_frames: list) -> torch.Tensor:
    tf = _frame_tf()
    tensors = [tf(f) for f in pil_frames]
    while len(tensors) < TEMPORAL_FRAMES:
        tensors.append(tensors[-1])
    tensors = tensors[:TEMPORAL_FRAMES]
    stacked = torch.stack(tensors, dim=0).permute(1, 0, 2, 3)  # (C,T,H,W)
    return stacked.unsqueeze(0)  # (1,C,T,H,W)


def frames_to_diff_tensor(pil_frames: list) -> torch.Tensor:
    tf = _diff_tf()
    tensors = [tf(f) for f in pil_frames]
    diffs = [torch.abs(tensors[i] - tensors[i-1]) for i in range(1, len(tensors))]
    if not diffs:
        h, w = tensors[0].shape[1], tensors[0].shape[2]
        return torch.zeros(1, 3, h, w)
    return torch.cat(diffs, dim=0).unsqueeze(0)


# ── Singleton ─────────────────────────────────────────────────

_stream_c_instance: nn.Module | None = None
_use_r3d: bool = True


def get_stream_c() -> nn.Module:
    global _stream_c_instance, _use_r3d
    if _stream_c_instance is None:
        try:
            print("[StreamC] Loading R3D-18 ...")
            model    = R3D18Detector(pretrained=True).to(DEVICE)
            _use_r3d = True
            print("[StreamC] R3D-18 ready.")
        except Exception as e:
            print(f"[StreamC] R3D-18 failed ({e}). Using FrameDiffCNN.")
            model    = FrameDiffCNN(n_frames=TEMPORAL_FRAMES).to(DEVICE)
            _use_r3d = False
            print("[StreamC] FrameDiffCNN ready.")
        model.eval()
        _stream_c_instance = model
    return _stream_c_instance


@torch.no_grad()
def stream_c_predict(pil_frames: list) -> float:
    """Run Stream C on PIL frame list. Returns P(FAKE) in [0,1]."""
    if len(pil_frames) < 2:
        return 0.5
    model = get_stream_c()
    x = frames_to_r3d_tensor(pil_frames).to(DEVICE) if _use_r3d \
        else frames_to_diff_tensor(pil_frames).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    return float(probs[1].item())
