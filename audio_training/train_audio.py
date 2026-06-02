#!/usr/bin/env python3
"""
train_audio.py
──────────────
PyTorch script to train RawNet2 (Small or Large) on ASVspoof 2021 DF dataset (eval part00).
Supports GPU training on Nvidia RTX 3070, automatic key download, and validation EER calculation.
"""

import os
import sys
import argparse
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# SincConv1d & ResBlock1D (Matching Inference Architecture)
# ─────────────────────────────────────────────────────────────

class SincConv1d(nn.Module):
    """
    Sinc-function based convolution for raw audio front-end.
    Learnable low and high cutoff frequencies per filter.
    Inspired by SincNet (Ravanelli & Bengio, 2018).
    """
    def __init__(self, out_channels: int = 70, kernel_size: int = 251, sample_rate: int = 16000):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate

        # Learnable cutoff frequencies (Hz), initialised uniformly
        self.low_hz  = nn.Parameter(torch.rand(out_channels, 1) * (sample_rate / 4))
        self.band_hz = nn.Parameter(torch.rand(out_channels, 1) * (sample_rate / 4))

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
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class FMS(nn.Module):
    """Feature Map Scaling (Squeeze-and-Excitation block for 1D convolution)."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=-1) # Global average pooling over time
        s = self.fc(s).unsqueeze(-1) # (B, C, 1)
        return x * s


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, use_fms: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.fms = FMS(channels) if use_fms else None
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.net(x)
        if self.fms is not None:
            out = self.fms(out)
        return self.act(out)

# ─────────────────────────────────────────────────────────────
# Model Architectures (Small & Large)
# ─────────────────────────────────────────────────────────────

class RawNet2Small(nn.Module):
    """
    Lightweight RawNet2-inspired model.
    Fully compatible with inference module.
    Parameters: ~1.2 M
    """
    def __init__(self, sinc_out: int = 70, dropout: float = 0.3):
        super().__init__()
        self.sinc    = SincConv1d(sinc_out, kernel_size=251)
        self.bn_sinc = nn.BatchNorm1d(sinc_out)

        self.encoder = nn.Sequential(
            nn.Conv1d(sinc_out, 128, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            ResBlock1D(128, use_fms=False),
            nn.Conv1d(128, 256, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            ResBlock1D(256, use_fms=False),
            nn.Conv1d(256, 256, 3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            ResBlock1D(256, use_fms=False),
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
        # x: (B, 1, 64000)
        x = F.relu(self.bn_sinc(self.sinc(x)))  # (B, 70, T)
        x = self.encoder(x)                       # (B, 256, T')
        x = self.pool(x)                          # (B, 256, 1)
        return self.head(x)                       # (B, 2)


class RawNet2Large(nn.Module):
    """
    High-capacity RawNet2 architecture.
    Features Squeeze-and-Excitation (FMS) on 6 ResBlocks.
    Parameters: ~4.5 M
    """
    def __init__(self, sinc_out: int = 70, dropout: float = 0.4):
        super().__init__()
        self.sinc = SincConv1d(sinc_out, kernel_size=251)
        self.bn_sinc = nn.BatchNorm1d(sinc_out)
        
        self.conv1 = nn.Conv1d(sinc_out, 128, 3, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.block1 = ResBlock1D(128, use_fms=True)
        self.block2 = ResBlock1D(128, use_fms=True)
        
        self.conv2 = nn.Conv1d(128, 256, 3, stride=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.block3 = ResBlock1D(256, use_fms=True)
        self.block4 = ResBlock1D(256, use_fms=True)
        
        self.conv3 = nn.Conv1d(256, 256, 3, stride=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.block5 = ResBlock1D(256, use_fms=True)
        self.block6 = ResBlock1D(256, use_fms=True)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64000)
        x = F.relu(self.bn_sinc(self.sinc(x)))
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block3(x)
        x = self.block4(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.pool(x)
        return self.head(x)

# ─────────────────────────────────────────────────────────────
# Model Parameter EMA (Exponential Moving Average)
# ─────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential moving average of model parameters for improved validation/inference stability."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_val = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data.detach()
            self.shadow[name] = new_val

    @torch.no_grad()
    def apply_to_model(self, model: nn.Module):
        self.backup.clear()
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in self.backup.items():
            model.state_dict()[name].copy_(param)
        self.backup.clear()

# ─────────────────────────────────────────────────────────────
# Key Auto-Downloader
# ─────────────────────────────────────────────────────────────

def download_keys_if_missing(keys_dir: Path) -> Path:
    """Downloads and extracts ASVspoof 2021 keys to target directory if not present."""
    keys_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if a metadata file already exists
    protocol_file = find_protocol_file(keys_dir)
    if protocol_file:
        print(f"[Prep] Found existing trial keys file: {protocol_file.name}")
        return protocol_file
        
    url = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
    tar_path = keys_dir / "DF-keys-full.tar.gz"
    
    print(f"[Prep] Downloading ASVspoof 2021 keys from {url}...")
    try:
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading keys") as t:
            urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
            
        print(f"[Prep] Extracting keys archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=keys_dir)
            
        # Clean up downloaded tar file
        if tar_path.exists():
            tar_path.unlink()
            
        protocol_file = find_protocol_file(keys_dir)
        if protocol_file:
            print(f"[Prep] Keys downloaded and extracted: {protocol_file.name}")
            return protocol_file
        else:
            raise FileNotFoundError("Keys archive extracted but trial metadata file could not be found.")
            
    except Exception as e:
        print(f"[Prep] ERROR downloading keys: {e}")
        print("[Prep] Please manually download DF-keys-full.tar.gz and extract in the audio_training directory.")
        sys.exit(1)


def find_protocol_file(directory: Path) -> Optional[Path]:
    """Helper to locate keys file in a directory structure."""
    patterns = ["*DF.cm.eval.keys.txt", "*trial_metadata.txt", "*trl.txt"]
    for p in patterns:
        for file in directory.rglob(p):
            if file.is_file():
                return file
    return None

# ─────────────────────────────────────────────────────────────
# Data Loading & Augmentations
# ─────────────────────────────────────────────────────────────

def load_audio(file_path: Path, target_sr: int = 16000, target_len: int = 64000) -> torch.Tensor:
    """Robust audio loader trying torchaudio, falling back to soundfile + librosa."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(file_path))
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    except Exception:
        try:
            import soundfile as sf
            import librosa
            data, sr = sf.read(str(file_path))
            if len(data.shape) > 1:
                data = data.mean(axis=-1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Could not load audio file {file_path.name}: {e}")
            
    # Pad or Trim to fixed length (4s / 64,000 samples)
    length = waveform.shape[1]
    if length < target_len:
        waveform = F.pad(waveform, (0, target_len - length))
    else:
        waveform = waveform[:, :target_len]
        
    return waveform


def apply_spectral_mask(x: torch.Tensor, sr: int = 16000, num_masks: int = 1, max_mask_width: int = 800) -> torch.Tensor:
    """Zeroes out random frequency bands in the audio spectrum (simulates compression/channel dropouts)."""
    n = x.shape[-1]
    fft = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, d=1/sr)
    for _ in range(num_masks):
        width = np.random.randint(100, max_mask_width)
        start_freq = np.random.uniform(50, sr/2 - width - 50)
        mask = (freqs >= start_freq) & (freqs <= start_freq + width)
        fft[:, mask] = 0.0
    return torch.fft.irfft(fft, n=n)


def apply_speed_perturbation(x: torch.Tensor, speed_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """Changes the speed of the audio waveform via linear interpolation."""
    speed = np.random.uniform(*speed_range)
    if speed == 1.0:
        return x
    orig_len = x.shape[-1]
    new_len = int(orig_len / speed)
    x_recon = F.interpolate(x.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)
    if new_len < orig_len:
        x_recon = F.pad(x_recon, (0, orig_len - new_len))
    else:
        x_recon = x_recon[:, :orig_len]
    return x_recon


class ASVspoofDataset(Dataset):
    """ASVspoof dataset loading raw audio waveforms on the fly."""
    def __init__(self, file_paths: List[Path], labels: List[int], augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.file_paths[idx]
        y = self.labels[idx]
        
        try:
            x = load_audio(p)
        except Exception as e:
            # Return a zero tensor if file is corrupted during large training runs
            print(f"[Warning] Failed loading {p.name}, yielding zero tensor. Error: {e}")
            x = torch.zeros((1, 64000), dtype=torch.float32)
            
        if self.augment:
            # 50% chance to add slight noise
            if random.random() < 0.5:
                noise_level = np.random.uniform(1e-4, 1e-3)
                x = x + noise_level * torch.randn_like(x)
            # 30% chance for speed perturbation
            if random.random() < 0.3:
                x = apply_speed_perturbation(x)
            # 30% chance for frequency band masking
            if random.random() < 0.3:
                x = apply_spectral_mask(x)

        return x, y

# ─────────────────────────────────────────────────────────────
# EER Metric Calculation
# ─────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculates Equal Error Rate (EER) where FPR equals FNR."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)

# ─────────────────────────────────────────────────────────────
# Train & Validation Loops
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    loss_fn: nn.Module,
    device: torch.device,
    ema: Optional[ModelEMA],
    grad_clip: float = 1.0
) -> float:
    model.train()
    total_loss = 0.0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    ema: Optional[ModelEMA]
) -> Tuple[float, float, float]:
    """Evaluates model on validation loader and returns loss, accuracy, and EER."""
    if ema is not None:
        ema.apply_to_model(model)
        
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_scores = []
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        # Save scores representing probability of 'spoof' (class 1)
        all_targets.extend(y.cpu().numpy())
        all_scores.extend(probs[:, 1].cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    acc = correct / total
    eer = compute_eer(np.array(all_targets), np.array(all_scores))
    
    if ema is not None:
        ema.restore(model)
        
    return avg_loss, acc, eer

# ─────────────────────────────────────────────────────────────
# Main Pipeline Executer
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RawNet2 Training Script for Audio Deepfake Countermeasures")
    parser.add_argument("--data_dir", type=str, default="", 
                        help="Path to folder containing extracted dataset. If empty, auto-searches root workspace.")
    parser.add_argument("--keys_dir", type=str, default="audio_training/keys", 
                        help="Path to save / load protocol keys.")
    parser.add_argument("--model_size", type=str, choices=["small", "large"], default="small",
                        help="Model scale size. Choose 'small' for direct compatibility with app inference.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate for OneCycleLR.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader CPU multiprocessing workers.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples for quick sanity runs.")
    parser.add_argument("--val_split", type=float, default=0.20, help="Fraction of files reserved for validation.")
    parser.add_argument("--output_ckpt", type=str, default="best_audio.pth", help="Checkpoint output filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # ── 1. Device Resolution ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Running on device: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        
    # ── 2. Locate Dataset FLAC files ──
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir_path = Path(args.data_dir) if args.data_dir else workspace_root
    
    print(f"[Main] Scanning workspace for audio files under: {data_dir_path}")
    flac_files = {}
    for p in data_dir_path.rglob("*.flac"):
        flac_files[p.stem] = p
        
    n_total_flac = len(flac_files)
    print(f"[Main] Located {n_total_flac} FLAC audio files in workspace.")
    if n_total_flac == 0:
        print("[!] ERROR: No FLAC audio files found. Ensure the ASVspoof dataset tar is extracted inside the workspace.")
        sys.exit(1)
        
    # ── 3. Resolve / Download Keys ──
    keys_dir_path = workspace_root / args.keys_dir
    protocol_file = download_keys_if_missing(keys_dir_path)
    
    # ── 4. Parse Protocol & Match Files ──
    print(f"[Main] Loading labels from {protocol_file.name} ...")
    valid_paths = []
    labels = []
    
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            # Search parts for file ID matching our FLAC filenames
            file_id = None
            for part in parts:
                if part in flac_files:
                    file_id = part
                    break
                    
            if file_id is None:
                continue # Skip files not extracted in part00
                
            # Search parts for label
            label_str = None
            for part in parts:
                p_low = part.lower()
                if p_low in ["bonafide", "spoof"]:
                    label_str = p_low
                    break
                    
            if label_str:
                valid_paths.append(flac_files[file_id])
                labels.append(1 if label_str == "spoof" else 0)
                
    n_samples = len(valid_paths)
    print(f"[Main] Matched {n_samples} audio files with labels.")
    if n_samples == 0:
        print("[!] ERROR: No files matched between the protocol file and files on disk. Check filenames.")
        sys.exit(1)
        
    # Optional sample limiting for debugging
    if args.max_samples is not None and args.max_samples < n_samples:
        print(f"[Main] Limiting dataset to --max_samples={args.max_samples}")
        valid_paths = valid_paths[:args.max_samples]
        labels = labels[:args.max_samples]
        n_samples = len(valid_paths)
        
    # ── 5. Train / Val Split ──
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        valid_paths, labels, test_size=args.val_split, random_state=args.seed, stratify=labels
    )
    
    # Print class distribution
    train_fakes = sum(train_labels)
    train_reals = len(train_labels) - train_fakes
    print(f"[Main] Dataset split:")
    print(f"       Train:      {len(train_paths)} samples (Real: {train_reals}, Fake: {train_fakes})")
    print(f"       Validation: {len(val_paths)} samples")
    
    # ── 6. Create DataLoaders ──
    train_ds = ASVspoofDataset(train_paths, train_labels, augment=True)
    val_ds = ASVspoofDataset(val_paths, val_labels, augment=False)
    
    # Set persistent workers if workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0)
    )
    
    # ── 7. Initialize Model ──
    if args.model_size == "small":
        print("[Main] Instantiating RawNet2-small (~1.2M params)...")
        model = RawNet2Small(dropout=0.3).to(device)
    else:
        print("[Main] Instantiating RawNet2-large (~4.5M params)...")
        model = RawNet2Large(dropout=0.4).to(device)
        
    # Compute class weights for weighted CE loss (imbalance mitigation)
    n_train = len(train_labels)
    weight_real = n_train / (2.0 * train_reals)
    weight_fake = n_train / (2.0 * train_fakes)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    print(f"[Main] Class balancing weights: Real={weight_real:.3f}, Fake={weight_fake:.3f}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer (decay 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler: OneCycleLR
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.15, anneal_strategy="cos"
    )
    
    # Model Parameter EMA
    ema = ModelEMA(model, decay=0.999)
    
    # ── 8. Training Execution ──
    print(f"\n[Main] Starting training ({args.epochs} epochs)...")
    best_eer = 1.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_fn=loss_fn, device=device, ema=ema
        )
        
        val_loss, val_acc, val_eer = evaluate_model(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, ema=ema
        )
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"Val EER: {val_eer:.2%}")
              
        # Save best model checkpoint
        if val_eer < best_eer:
            best_eer = val_eer
            # Apply EMA parameters to checkpoint for inference excellence
            ema.apply_to_model(model)
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_size": args.model_size,
                "val_eer": val_eer,
                "val_acc": val_acc
            }
            
            torch.save(checkpoint, args.output_ckpt)
            ema.restore(model)
            print(f"  --> Saved new best checkpoint to '{args.output_ckpt}' with Val EER: {val_eer:.2%}")
            
    print(f"\n[Main] Training complete! Best Validation EER: {best_eer:.2%}")
    print(f"[Main] Saved best checkpoint: {os.path.abspath(args.output_ckpt)}")


if __name__ == "__main__":
    main()
