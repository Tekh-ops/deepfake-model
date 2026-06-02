import torch
import torch.nn as nn
import timm
from config import DEVICE, XCEPTION_SIZE

class XceptionDetector(nn.Module):
    """
    Xception pretrained on ImageNet via timm.
    Xception is the original model used in the FaceForensics++ paper
    and remains one of the most effective deepfake detectors.
    Input size: 299x299 (different from the other two models).
    Weight size: ~88MB — fast to download and load on M1.
    """

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,      # remove classifier
        )
        feature_dim = self.backbone.num_features  # 2048 for Xception

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> tuple[str, float]:
        self.eval()
        with torch.no_grad():
            x = x.to(DEVICE)
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else 1.0 - fake_prob
        return label, round(confidence, 4)


_xception_instance: XceptionDetector | None = None

def get_xception() -> XceptionDetector:
    global _xception_instance
    if _xception_instance is None:
        print("[Xception] Loading model...")
        _xception_instance = XceptionDetector()
        _xception_instance.eval()
        print("[Xception] Ready.")
    return _xception_instance