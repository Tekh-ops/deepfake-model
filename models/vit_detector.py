import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from config import DEVICE, MODEL_CACHE_DIR
from PIL import Image

# We use google/vit-base-patch16-224-in21k and add a 2-class head.
# This is ~330MB — downloads once and is cached in MODEL_CACHE_DIR.
# Much lighter than DeiT-base and runs well on M1 MPS.

MODEL_NAME = "google/vit-base-patch16-224-in21k"

class ViTDetector(nn.Module):
    """
    Vision Transformer (ViT-B/16) with a binary classification head.
    The backbone weights are pretrained on ImageNet-21k.
    The classification head is randomly initialised (as if we were
    about to fine-tune — in production you would load saved weights).
    """

    def __init__(self):
        super().__init__()
        # Load backbone — hidden_size is 768 for ViT-base
        self.backbone = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True,   # replace the 21k-class head
            cache_dir=MODEL_CACHE_DIR,
        )
        # The HuggingFace model already has a .classifier linear layer
        # pointing to num_labels=2. We just use the whole model as-is.
        self.to(DEVICE)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.logits   # (batch, 2)

    def predict_proba(self, pixel_values: torch.Tensor) -> tuple[str, float]:
        self.eval()
        with torch.no_grad():
            pixel_values = pixel_values.to(DEVICE)
            logits = self.forward(pixel_values)
            probs  = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob
        return label, round(confidence, 4)


_vit_instance: ViTDetector | None = None

def get_vit() -> ViTDetector:
    global _vit_instance
    if _vit_instance is None:
        print("[ViT] Loading model (first run downloads ~330MB)...")
        _vit_instance = ViTDetector()
        _vit_instance.eval()
        print("[ViT] Ready.")
    return _vit_instance