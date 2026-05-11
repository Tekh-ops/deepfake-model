# src/models/temporal/timesformer_detector.py

from transformers import TimesformerModel
import torch.nn as nn
import torch

class TemporalDetector(nn.Module):
    def __init__(self, num_frames=16, pretrained=True):
        super().__init__()
        
        model_name = "facebook/timesformer-base-finetuned-k400"
        self.backbone = TimesformerModel.from_pretrained(
            model_name if pretrained else None,
            num_frames=num_frames,
            image_size=224
        )
        hidden_size = self.backbone.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )
        
        self.num_frames = num_frames
    
    def forward(self, pixel_values):
        # pixel_values: [B, T, C, H, W]
        # T = num_frames = 16
        outputs = self.backbone(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits.squeeze(-1), cls_output