# src/models/spatial/efficientnet_detector.py

import torch
import torch.nn as nn
import timm

class SpatialDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b4', 
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0,  # remove classifier head
            global_pool='avg'
        )
        feature_dim = self.backbone.num_features  # 1792 for B4
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1)  # binary output
        )
        
        # SRM filters as fixed first layer for frequency-domain input
        self.srm_conv = self._build_srm_layer()
    
    def _build_srm_layer(self):
        """3-channel SRM filter bank for noise residual extraction."""
        # Simplified SRM: high-pass filters
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0,  0,  0,  0],
                   [0,-1,  2, -1,  0],
                   [0, 2, -4,  2,  0],
                   [0,-1,  2, -1,  0],
                   [0, 0,  0,  0,  0]]
        filter2 = [[-1, 2, -2,  2, -1],
                   [ 2,-6,  8, -6,  2],
                   [-2, 8,-12,  8, -2],
                   [ 2,-6,  8, -6,  2],
                   [-1, 2, -2,  2, -1]]
        filter3 = [[0,  0, 0,  0, 0],
                   [0,  0, 0,  0, 0],
                   [0,  1,-2,  1, 0],
                   [0,  0, 0,  0, 0],
                   [0,  0, 0,  0, 0]]
        
        filters = torch.tensor(
            [[filter1], [filter2], [filter3]], 
            dtype=torch.float32
        ) 
        # Normalize
        for i in range(3):
            filters[i] = filters[i] / q[i]
        
        conv = nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)
        conv.weight = nn.Parameter(filters, requires_grad=False)
        return conv
    
    def forward(self, x):
        # x: [B, 3, H, W] normalized RGB
        # Apply SRM on grayscale version
        gray = x.mean(dim=1, keepdim=True)
        noise = self.srm_conv(gray)  # [B, 3, H, W]
        # Concatenate RGB + SRM features (keep 3 channels, add noise)
        # Use only RGB for backbone (SRM used as auxiliary signal)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1), features  # return features for fusion