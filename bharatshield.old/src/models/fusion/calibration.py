# src/models/fusion/calibration.py

import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
import numpy as np

class TemperatureScaling(nn.Module):
    """Learnable temperature for post-hoc calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        """Optimize temperature on calibration set."""
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )
        criterion = nn.BCEWithLogitsLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled = self(logits)
            loss = criterion(scaled, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self.temperature.item()


def calibrate_with_isotonic(scores: np.ndarray, 
                             labels: np.ndarray) -> IsotonicRegression:
    """Isotonic regression calibration — more flexible than temperature."""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, labels)
    return ir