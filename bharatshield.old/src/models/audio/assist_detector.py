# src/models/audio/aasist_detector.py
# Simplified AASIST implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv(nn.Module):
    """Sinc-based convolution for raw waveform front-end."""
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Learnable low and high cutoff frequencies
        self.low_hz = nn.Parameter(torch.Tensor(out_channels, 1))
        self.band_hz = nn.Parameter(torch.Tensor(out_channels, 1))
        
        nn.init.uniform_(self.low_hz, 0, sample_rate / 2)
        nn.init.uniform_(self.band_hz, 0, sample_rate / 2)
    
    def forward(self, waveforms):
        # Build sinc filters and convolve with waveform
        low = torch.abs(self.low_hz)
        high = low + torch.abs(self.band_hz)
        high = torch.clamp(high, self.low_hz.min(), self.sample_rate / 2)
        # ... (full sinc filter construction)
        return F.conv1d(waveforms, self._compute_filters(), 
                       stride=1, padding=self.kernel_size // 2)

class AASISTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc_conv = SincConv(70, 128, 16000)
        # Graph attention layers for spectro-temporal modeling
        self.gat_spectral = GraphAttentionLayer(70, 70)
        self.gat_temporal = GraphAttentionLayer(70, 70)
        self.classifier = nn.Linear(140, 1)
    
    def forward(self, waveform):
        # waveform: [B, 1, 64000]
        x = self.sinc_conv(waveform)
        spec_feat = self.gat_spectral(x)
        temp_feat = self.gat_temporal(x.transpose(1, 2))
        fused = torch.cat([spec_feat.mean(-1), 
                          temp_feat.mean(-1)], dim=-1)
        return self.classifier(fused).squeeze(-1), fused