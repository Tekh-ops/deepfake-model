"""
modules/video_analysis/frequency_detector.py
──────────────────
Stream B — Frequency Domain Analysis

Architecture: EfficientNet-B0 operating on the 2D FFT magnitude spectrum
of the input face crop (grayscale → DCT → log magnitude → normalise).

Rationale:
  GAN and diffusion models produce characteristic checkerboard artifacts
  in the frequency domain that are invisible in pixel space but trivially
  detectable in the Fourier spectrum.  A CNN trained on FFT images learns
  to recognise upsampling signatures, GAN grid patterns, and compression
  ghost artifacts that spatial models miss entirely.

Reference:
  Frank et al., "Leveraging Frequency Analysis for Deep Fake Image
  Passive Detection" (ICML 2020)
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

from config import DEVICE, MODEL_CACHE_DIR, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────
# DCT / FFT preprocessing transform
# ─────────────────────────────────────────────────────────────

class FFTMagnitudeTransform:
    """
    Convert a normalised RGB tensor to its 2D FFT magnitude spectrum.
    Applied AFTER ToTensor() + Normalize() in the pipeline.

    Steps:
      1. Convert to grayscale (luminance-weighted)
      2. Apply 2D FFT
      3. Compute log-magnitude
      4. fftshift to move DC component to centre
      5. Normalise to [0, 1]
      6. Replicate to 3 channels (EfficientNet expects 3-channel input)
    """

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # img_tensor: (3, H, W) — already normalised
        gray = (0.299 * img_tensor[0]
                + 0.587 * img_tensor[1]
                + 0.114 * img_tensor[2])            # (H, W)

        fft       = torch.fft.fft2(gray)
        magnitude = torch.log(torch.abs(fft) + 1e-8)
        magnitude = torch.fft.fftshift(magnitude)   # DC to centre

        # Normalise to [0, 1]
        mn, mx    = magnitude.min(), magnitude.max()
        magnitude = (magnitude - mn) / (mx - mn + 1e-8)

        return magnitude.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


def get_stream_b_transform():
    """
    Two-stage transform:
      Stage 1 — standard spatial preprocessing
      Stage 2 — convert to FFT magnitude spectrum
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        FFTMagnitudeTransform(),   # RGB → frequency spectrum
    ])


# ─────────────────────────────────────────────────────────────
# Stream B model
# ─────────────────────────────────────────────────────────────

class StreamB(nn.Module):
    """
    EfficientNet-B0 trained on 2D FFT magnitude spectra.
    Lightweight (5.3 M params) — fast and memory-efficient.
    Input:  3-channel FFT magnitude image (3, 224, 224)
    Output: logits (B, 2)  — [P(REAL), P(FAKE)]
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.35):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # remove classifier
        )
        feat_dim = self.backbone.num_features   # 1280 for B0

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — FFT spectrum image
        f = self.backbone(x)    # (B, 1280)
        return self.head(f)     # (B, 2)

    def get_param_groups(self) -> list:
        return [
            {"params": self.head.parameters(),     "lr": 2e-4},
            {"params": self.backbone.parameters(), "lr": 1.5e-5},
        ]


# ─────────────────────────────────────────────────────────────
# Singleton + inference API
# ─────────────────────────────────────────────────────────────

_stream_b_instance:  StreamB | None = None
_stream_b_transform = None


def get_stream_b() -> StreamB:
    """Lazy-load Stream B. Uses timm ImageNet weights as initialisation."""
    global _stream_b_instance, _stream_b_transform
    if _stream_b_instance is None:
        print("[StreamB] Loading EfficientNet-B0 (DCT/FFT stream) ...")
        _stream_b_instance = StreamB(pretrained=True).to(DEVICE)
        _stream_b_instance.eval()
        _stream_b_transform = get_stream_b_transform()
        print("[StreamB] Ready.")
    return _stream_b_instance


@torch.no_grad()
def stream_b_predict(pil_img: Image.Image) -> float:
    """
    Run Stream B on a PIL face crop.
    Returns P(FAKE) in [0, 1].
    """
    model = get_stream_b()
    tensor = _stream_b_transform(pil_img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    return float(probs[1].item())
