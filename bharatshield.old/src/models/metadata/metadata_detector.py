# src/models/metadata/metadata_detector.py

import torch.nn as nn

class MetadataDetector(nn.Module):
    """Lightweight MLP on extracted metadata features."""
    def __init__(self, input_dim=13, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), 
                      nn.BatchNorm1d(h),
                      nn.ReLU(inplace=True), 
                      nn.Dropout(0.2)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.net(x)
        return logits.squeeze(-1), x  # raw features as embedding