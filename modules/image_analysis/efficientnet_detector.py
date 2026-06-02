import torch
import torch.nn as nn
import timm
from config import DEVICE, MODEL_CACHE_DIR

class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B0 fine-tuned head for binary deepfake classification.
    Uses timm's pretrained weights — downloads ~20MB on first run.
    Runs comfortably on M1 with 8GB RAM.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B0
        # num_classes=0 removes the original classifier head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,          # remove classifier
        )
        feature_dim = self.backbone.num_features  # 1280 for B0

        # Binary classification head: REAL (0) or FAKE (1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> tuple[str, float]:
        """
        Returns (label, confidence) where label is 'FAKE' or 'REAL'
        and confidence is a float in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            x = x.to(DEVICE)
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()   # index 1 = FAKE class

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob
        return label, round(confidence, 4)


# Singleton instance — loaded once at startup
_efficientnet_instance: EfficientNetDetector | None = None

def get_efficientnet() -> EfficientNetDetector:
    global _efficientnet_instance
    if _efficientnet_instance is None:
        print("[EfficientNet] Loading model...")
        _efficientnet_instance = EfficientNetDetector()
        _efficientnet_instance.eval()
        print("[EfficientNet] Ready.")
    return _efficientnet_instance