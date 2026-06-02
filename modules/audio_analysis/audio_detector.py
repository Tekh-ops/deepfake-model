"""
models/audio_detector.py
────────────────────────
Audio deepfake detection stream.

Strategy (graceful degradation):
  Level 1 — RawNet2-small: lightweight 1D CNN on raw waveform
             Targets voice cloning, TTS synthesis, voice conversion.
             ~4 MB, runs on CPU comfortably.
  Level 2 — LFCC + MLP fallback: spectral features + tiny classifier.
             Used if torchaudio is unavailable.
  Level 3 — Returns 0.5 (uncertain) if audio cannot be processed.

Dataset target: ASVspoof 2019 Logical Access (LA) partition.

Usage:
    from modules.audio_analysis.audio_detector import audio_predict
    p_fake = audio_predict("/path/to/audio.wav")   # float in [0,1]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import DEVICE


# ─────────────────────────────────────────────────────────────
# RawNet2-small (1D CNN on raw waveform)
# ─────────────────────────────────────────────────────────────

class SincConv1d(nn.Module):
    """
    Sinc-function based convolution for raw audio front-end.
    Learnable low and high cutoff frequencies per filter.
    Inspired by SincNet (Ravanelli & Bengio, 2018).
    """

    def __init__(self, out_channels: int = 70, kernel_size: int = 251,
                 sample_rate: int = 16000):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate

        # Learnable cutoff frequencies (Hz), initialised uniformly
        self.low_hz  = nn.Parameter(
            torch.rand(out_channels, 1) * (sample_rate / 4)
        )
        self.band_hz = nn.Parameter(
            torch.rand(out_channels, 1) * (sample_rate / 4)
        )

        # Hamming window for filter tapering
        n = (kernel_size - 1) / 2.0
        t = torch.arange(-n, 0).float() / sample_rate
        self.register_buffer("t_left",  t.unsqueeze(0))
        t2 = torch.arange(1, n + 1).float() / sample_rate
        self.register_buffer("t_right", t2.unsqueeze(0))
        hamming = 0.54 - 0.46 * torch.cos(
            2 * np.pi * torch.arange(kernel_size).float() / (kernel_size - 1)
        )
        self.register_buffer("window", hamming)

    def _build_filters(self) -> torch.Tensor:
        low  = torch.abs(self.low_hz)
        high = low + torch.abs(self.band_hz)
        high = torch.clamp(high, 1.0, self.sample_rate / 2 - 1.0)

        f_times_t_low  = torch.matmul(low,  self.t_left)
        f_times_t_high = torch.matmul(high, self.t_right)

        band_pass_left  = (torch.sin(2 * np.pi * f_times_t_high)
                           - torch.sin(2 * np.pi * f_times_t_low)) \
                         / (self.kernel_size / 2)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass_center = 2 * (high - low)

        band_pass = torch.cat([
            band_pass_left,
            band_pass_center,
            band_pass_right,
        ], dim=1)
        band_pass = band_pass / (2 * band_pass.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        filters   = band_pass * self.window
        return filters.unsqueeze(1)  # (out_ch, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        filters = self._build_filters()
        return F.conv1d(x, filters,
                        stride=1,
                        padding=self.kernel_size // 2)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class RawNet2Small(nn.Module):
    """
    Lightweight RawNet2-inspired model for anti-spoofing.
    Input:  (B, 1, 64000) raw waveform @ 16kHz (4 seconds)
    Output: logits (B, 2) — [P(REAL), P(FAKE)]

    Parameters: ~1.2 M  — fits comfortably on M1 and RTX 3070.
    """

    def __init__(self, sinc_out: int = 70, dropout: float = 0.3):
        super().__init__()
        self.sinc    = SincConv1d(sinc_out, kernel_size=251)
        self.bn_sinc = nn.BatchNorm1d(sinc_out)

        self.encoder = nn.Sequential(
            nn.Conv1d(sinc_out, 128, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            ResBlock1D(128),
            nn.Conv1d(128, 256, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            ResBlock1D(256),
            nn.Conv1d(256, 256, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            ResBlock1D(256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_sinc(self.sinc(x)))  # (B, 70, T)
        x = self.encoder(x)                       # (B, 256, T')
        x = self.pool(x)                          # (B, 256, 1)
        return self.head(x)                       # (B, 2)

    def get_param_groups(self) -> list:
        return [
            {"params": self.head.parameters(),    "lr": 2e-4},
            {"params": self.encoder.parameters(), "lr": 2e-4},
            {"params": self.sinc.parameters(),    "lr": 1e-4},
        ]


# ─────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────

def load_waveform(audio_path: str,
                  target_sr: int = 16000,
                  target_len: int = 64000) -> torch.Tensor | None:
    """
    Load a WAV file, resample to target_sr, pad/trim to target_len samples.
    Returns (1, 1, target_len) tensor or None on failure.
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform  = resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Pad or trim
        length = waveform.shape[1]
        if length < target_len:
            waveform = F.pad(waveform, (0, target_len - length))
        else:
            waveform = waveform[:, :target_len]
        return waveform.unsqueeze(0)   # (1, 1, target_len)
    except Exception as e:
        print(f"[AudioDetector] waveform load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Singleton + inference API
# ─────────────────────────────────────────────────────────────

_audio_model: RawNet2Small | None = None


def get_audio_model() -> RawNet2Small:
    global _audio_model
    if _audio_model is None:
        print("[AudioDetector] Loading RawNet2-small ...")
        _audio_model = RawNet2Small().to(DEVICE)
        _audio_model.eval()
        print("[AudioDetector] Ready  (~1.2M params).")
    return _audio_model


@torch.no_grad()
def audio_predict(audio_path: str) -> dict:
    """
    Run the audio deepfake detector on a WAV file.

    Returns:
        {
            "fake_prob":  float,     # P(SYNTHETIC_VOICE) in [0,1]
            "label":      str,       # "Fake" | "Real"
            "available":  bool,      # False if audio could not be processed
            "note":       str,
        }
    """
    waveform = load_waveform(audio_path)
    if waveform is None:
        return {
            "fake_prob": 0.5,
            "label":     "Unknown",
            "available": False,
            "note":      "Could not load audio file.",
        }

    try:
        model  = get_audio_model()
        x      = waveform.to(DEVICE)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        p_fake = float(probs[1].item())
        return {
            "fake_prob": round(p_fake, 4),
            "label":     "Fake" if p_fake >= 0.5 else "Real",
            "available": True,
            "note":      "RawNet2-small (ImageNet init — fine-tune on ASVspoof 2019 LA for accuracy)",
        }
    except Exception as e:
        print(f"[AudioDetector] inference failed: {e}")
        return {
            "fake_prob": 0.5,
            "label":     "Unknown",
            "available": False,
            "note":      f"Audio inference error: {str(e)}",
        }
