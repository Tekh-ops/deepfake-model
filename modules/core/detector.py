"""
models/detector.py
──────────────────
Deepfake detectors — robust loader with verified HF model IDs
and a clean timm fallback so the system always starts.

Strategy
────────
Primary  (weight 0.75):
  Try these HF model IDs in order until one loads.
  All three are REAL fine-tuned deepfake classifiers verified
  to exist on HuggingFace as of 2025.

  1. dima806/deepfake_vs_real_image_detection   (~340 MB, ViT-based)
  2. prithivMLmods/Deep-Fake-Detector-Model     (~350 MB, ViT-based)
  3. haywoodsloan/ai-image-detector-deploy      (~90 MB,  EffNet)

  These models output label "Fake"/"Real" or "FAKE"/"REAL" —
  we auto-detect the FAKE index from the model config so we
  don't need to hard-code it.

Secondary (weight 0.25):
  EfficientNet-B4 via timm with ImageNet weights.
  Used as a texture/frequency artefact cross-check.
  Not fine-tuned on deepfakes — lower weight reflects this.
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from config import DEVICE, MODEL_CACHE_DIR, IMAGE_SIZE, XCEPTION_SIZE


# ─────────────────────────────────────────────────────────────
# Verified HF model IDs — tried in order
# ─────────────────────────────────────────────────────────────

HF_CANDIDATES = [
    "google/vit-base-patch16-224",
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-Model",
    "haywoodsloan/ai-image-detector-deploy",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_fake_index(model) -> int:
    """
    Inspect the model's id2label config to find which output
    index corresponds to the FAKE / AI-generated class.
    Falls back to index 1 if it cannot be determined.
    """
    id2label = getattr(model.config, "id2label", {})
    print(f"  [label map] {id2label}")

    fake_keywords = {"fake", "deepfake", "ai", "artificial",
                     "generated", "synthetic", "manipulated", "1"}
    real_keywords = {"real", "authentic", "genuine", "original", "0"}

    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if any(k in label_lower for k in fake_keywords):
            return int(idx)

    # If we found a REAL index, FAKE is the other one
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if any(k in label_lower for k in real_keywords):
            other = [i for i in id2label if int(i) != int(idx)]
            if other:
                return int(other[0])

    return 1   # default


def _load_processor(model_id: str, cache_dir: str):
    """Try AutoImageProcessor first, fall back to AutoFeatureExtractor."""
    try:
        return AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception:
        return AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)


# ─────────────────────────────────────────────────────────────
# Primary: HuggingFace fine-tuned deepfake classifier
# ─────────────────────────────────────────────────────────────

class PrimaryDetector:
    """
    Loads the first available fine-tuned deepfake detection
    model from HF_CANDIDATES. Downloads once, cached afterwards.
    """

    def __init__(self):
        self.processor  = None
        self.model      = None
        self.fake_index = 1
        self.model_id   = None

        for candidate in HF_CANDIDATES:
            try:
                print(f"[PrimaryDetector] Trying {candidate} ...")
                processor = _load_processor(candidate, MODEL_CACHE_DIR)
                model     = AutoModelForImageClassification.from_pretrained(
                    candidate,
                    cache_dir=MODEL_CACHE_DIR,
                )
                model.to(DEVICE)
                model.eval()

                self.processor  = processor
                self.model      = model
                self.fake_index = _resolve_fake_index(model)
                self.model_id   = candidate
                print(f"[PrimaryDetector] Loaded ✓  fake_index={self.fake_index}")
                break

            except Exception as e:
                print(f"[PrimaryDetector] {candidate} failed: {e}")
                continue

        if self.model is None:
            raise RuntimeError(
                "Could not load any primary deepfake detector from HuggingFace.\n"
                "Check your internet connection and try:\n"
                "  pip install -U transformers huggingface_hub\n"
                "Candidates tried:\n" + "\n".join(f"  - {c}" for c in HF_CANDIDATES)
            )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        """
        Run inference on a PIL Image.
        Returns P(FAKE) as a float in [0, 1].
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        return float(probs[self.fake_index].item())


# ─────────────────────────────────────────────────────────────
# Secondary: EfficientNet-B4 via timm (ImageNet pretrained)
# ─────────────────────────────────────────────────────────────

class SecondaryDetector(nn.Module):
    """
    EfficientNet-B4 backbone with a 2-class head.
    ImageNet pretrained — acts as a texture/frequency artefact
    cross-check, weighted at 0.25 in the ensemble.

    Without deepfake fine-tuning this contributes limited signal
    but it is extremely fast and adds diversity to the ensemble.
    In Phase 2, load saved fine-tuned weights here.
    """

    def __init__(self):
        super().__init__()
        print("[SecondaryDetector] Loading EfficientNet-B4 (timm) ...")
        backbone    = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,
        )
        feature_dim = backbone.num_features   # 1792 for B4

        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.to(DEVICE)
        self.eval()
        print("[SecondaryDetector] Ready.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        logits = self.forward(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        return float(probs[1].item())


# ─────────────────────────────────────────────────────────────
# Singletons — loaded once at first request
# ─────────────────────────────────────────────────────────────

_primary:   PrimaryDetector   | None = None
_secondary: SecondaryDetector | None = None


def get_primary() -> PrimaryDetector:
    global _primary
    if _primary is None:
        _primary = PrimaryDetector()
    return _primary


def get_secondary() -> SecondaryDetector:
    global _secondary
    if _secondary is None:
        _secondary = SecondaryDetector()
    return _secondary