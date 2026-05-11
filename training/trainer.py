"""
training/trainer.py
───────────────────
Full training loop with:
  • Mixed precision (AMP) — halves memory, doubles throughput on CUDA
  • Gradient accumulation — simulate larger batch sizes
  • Cosine annealing LR with linear warmup
  • MixUp / CutMix augmentation applied at batch level
  • EWC (Elastic Weight Consolidation) for continual learning
  • Early stopping on validation AUC
  • TensorBoard + CSV logging
  • Checkpoint management (best + last + top-k)
"""

import os
import csv
import time
import math
import copy
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from training.config import (
    DEVICE, EPOCHS, ACCUMULATION_STEPS, GRAD_CLIP,
    LEARNING_RATE, WEIGHT_DECAY, BETAS,
    WARMUP_EPOCHS, LR_MIN,
    LABEL_SMOOTHING, MIXUP_ALPHA, CUTMIX_PROB,
    LOG_INTERVAL, EVAL_INTERVAL, PATIENCE,
    PRIMARY_METRIC, BEST_CKPT, LAST_CKPT,
    OUTPUT_DIR, LOG_DIR,
    ADV_TRAIN_PROB, ADV_EPS, ADV_ALPHA, ADV_STEPS,
    TARGET_AUC,
)
from training.model import build_loss, save_checkpoint
from training.dataset import mixup_batch, cutmix_batch, LABEL_FAKE


# ─────────────────────────────────────────────────────────────
# LR Schedule: linear warmup + cosine annealing
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_epochs: int,
                    warmup_epochs: int = WARMUP_EPOCHS,
                    eta_min: float = LR_MIN):
    warmup  = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ─────────────────────────────────────────────────────────────
# EWC (Elastic Weight Consolidation)
# ─────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation — prevents catastrophic forgetting
    when fine-tuning on new deepfake types while retaining performance
    on previously learned ones.

    Usage:
        ewc = EWC(model, old_dataloader)   # after training on old data
        loss = task_loss + ewc_lambda * ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, dataloader,
                 ewc_lambda: float = 400.0):
        self.ewc_lambda = ewc_lambda
        self.params   = {n: p.clone().detach()
                         for n, p in model.named_parameters() if p.requires_grad}
        self.fisher   = self._compute_fisher(model, dataloader)

    def _compute_fisher(self, model: nn.Module, dataloader) -> dict:
        """
        Compute diagonal Fisher information matrix via squared gradients.
        Represents how important each parameter is for the old task.
        """
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        model.zero_grad()

        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(images)
            loss   = nn.functional.cross_entropy(output, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        # Normalise by number of batches
        n_batches = max(1, len(dataloader))
        for n in fisher:
            fisher[n] /= n_batches

        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC penalty for the current model parameters."""
        penalty = torch.tensor(0.0, device=DEVICE)
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.ewc_lambda * penalty


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(all_labels: list, all_probs: list) -> dict:
    """Compute full evaluation metrics from collected labels + probabilities."""
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, zero_division=0)

    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0   # true positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0   # true negative rate

    return {
        "auc":          round(float(auc), 4),
        "accuracy":     round(float(acc), 4),
        "f1":           round(float(f1), 4),
        "sensitivity":  round(float(sensitivity), 4),  # fake detection rate
        "specificity":  round(float(specificity), 4),  # real preservation rate
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion,
    scaler: GradScaler,
    epoch: int,
    writer: SummaryWriter,
    ewc: EWC | None = None,
    adv_training: bool = False,
) -> dict:

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    all_labels, all_probs = [], []
    global_step = epoch * len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # ── Adversarial augmentation (PGD, 20% of batches) ──
        if adv_training and random.random() < ADV_TRAIN_PROB:
            from training.adversarial import pgd_attack
            # Must temporarily disable AMP and use float32 for PGD
            model.eval()  # no dropout during attack generation
            with torch.no_grad():
                pass
            model.train()
            images = pgd_attack(
                model, images, labels, criterion,
                eps=ADV_EPS, alpha=ADV_ALPHA, steps=ADV_STEPS,
            ).detach()

        # ── MixUp / CutMix (random choice per batch) ─────────
        mixup_data, cutmix_data = None, None
        if MIXUP_ALPHA > 0 and torch.rand(1).item() < 0.5:
            images, labels, mixup_data = mixup_batch(images, labels, MIXUP_ALPHA)
        elif CUTMIX_PROB > 0 and torch.rand(1).item() < CUTMIX_PROB:
            images, labels, cutmix_data = cutmix_batch(images, labels, CUTMIX_PROB)

        # ── Forward (AMP) ────────────────────────────────────
        use_amp = DEVICE.type == "cuda"
        with autocast(enabled=use_amp):
            logits = model(images)

            if mixup_data is not None:
                lam, labels_b = mixup_data
                loss = lam * criterion(logits, labels) + \
                       (1 - lam) * criterion(logits, labels_b)
            elif cutmix_data is not None:
                lam, labels_b = cutmix_data
                loss = lam * criterion(logits, labels) + \
                       (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

            # EWC penalty (only when fine-tuning on new data)
            if ewc is not None:
                loss = loss + ewc.penalty(model)

            loss = loss / ACCUMULATION_STEPS

        # ── Backward ─────────────────────────────────────────
        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Collect metrics ───────────────────────────────────
        total_loss += loss.item() * ACCUMULATION_STEPS
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, LABEL_FAKE].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())

        # ── Logging ───────────────────────────────────────────
        if batch_idx % LOG_INTERVAL == 0:
            step    = global_step + batch_idx
            cur_lr  = optimizer.param_groups[0]["lr"]
            avg_l   = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:5d}/{len(loader)} | "
                  f"Loss {avg_l:.4f} | LR {cur_lr:.2e}")
            writer.add_scalar("train/loss_step", avg_l, step)
            writer.add_scalar("train/lr", cur_lr, step)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = round(total_loss / len(loader), 4)
    return metrics


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, LABEL_FAKE].cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = round(total_loss / len(loader), 4)
    return metrics


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    class_weights: torch.Tensor | None = None,
    ewc: EWC | None = None,
    swad=None,
    run_name: str = "",
    adv_training: bool = False,
    target_auc: float = TARGET_AUC,
):
    """
    Full training loop.

    Args:
        model:          DeepfakeDetector (already on DEVICE)
        train_loader:   Training DataLoader
        val_loader:     Validation DataLoader
        class_weights:  Tensor [w_real, w_fake] for weighted loss
        ewc:            Optional EWC instance for continual learning
        run_name:       Identifier for logs/checkpoints
    """
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir  = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer   = SummaryWriter(os.path.join(LOG_DIR, run_name))
    csv_path = os.path.join(run_dir, "metrics.csv")

    # ── Optimiser ────────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    param_groups = raw_model.get_param_groups()
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY, betas=BETAS)

    # ── LR schedule ──────────────────────────────────────────
    scheduler = build_scheduler(optimizer, EPOCHS)

    # ── Loss ─────────────────────────────────────────────────
    criterion = build_loss(class_weights, LABEL_SMOOTHING)

    # ── AMP scaler ───────────────────────────────────────────
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # ── State ─────────────────────────────────────────────────
    best_metric   = -1.0
    patience_cnt  = 0
    best_ckpt_path = os.path.join(run_dir, "best.pth")
    last_ckpt_path = os.path.join(run_dir, "last.pth")

    # ── CSV header ────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "epoch", "train_loss", "train_auc", "train_acc",
            "val_loss", "val_auc", "val_acc", "val_f1",
            "val_sensitivity", "val_specificity", "lr", "time_s",
        ])

    print(f"\n[Trainer] Starting run: {run_name}")
    print(f"[Trainer] Epochs={EPOCHS}, accumulation={ACCUMULATION_STEPS}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch, writer, ewc, adv_training=adv_training,
        )
        scheduler.step()

        # ── SWAD: collect weights ─────────────────────────────
        if swad is not None:
            swad.update(model, epoch)

        # ── Validate ──────────────────────────────────────────
        if epoch % EVAL_INTERVAL == 0:
            val_metrics = evaluate(model, val_loader, criterion)
        else:
            val_metrics = {"auc": 0, "accuracy": 0, "loss": 0,
                           "f1": 0, "sensitivity": 0, "specificity": 0}

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        # ── Log ───────────────────────────────────────────────
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train AUC={train_metrics['auc']:.4f} Loss={train_metrics['loss']:.4f} | "
            f"Val AUC={val_metrics['auc']:.4f} Acc={val_metrics['accuracy']:.4f} | "
            f"Sensitivity={val_metrics['sensitivity']:.4f} "
            f"Specificity={val_metrics['specificity']:.4f} | "
            f"LR={cur_lr:.2e} | {elapsed:.1f}s"
        )

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_metrics["loss"], train_metrics["auc"], train_metrics["accuracy"],
                val_metrics["loss"],   val_metrics["auc"],   val_metrics["accuracy"],
                val_metrics["f1"],     val_metrics["sensitivity"],
                val_metrics["specificity"],
                cur_lr, round(elapsed, 1),
            ])

        # ── Checkpoint ───────────────────────────────────────
        save_checkpoint(model, optimizer, scheduler, epoch,
                        {**train_metrics, **{f"val_{k}": v
                                             for k, v in val_metrics.items()}},
                        last_ckpt_path)

        metric_val = val_metrics.get(PRIMARY_METRIC.replace("val_", ""), 0)
        if metric_val > best_metric:
            best_metric  = metric_val
            patience_cnt = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_metrics, best_ckpt_path)
            import shutil
            shutil.copy(best_ckpt_path, BEST_CKPT)
            print(f"  ✓ New best {PRIMARY_METRIC}={best_metric:.4f} — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Trainer] Early stopping after {epoch} epochs "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        # ── Target AUC early stop ─────────────────────────────
        if val_metrics.get("auc", 0) >= target_auc:
            print(f"\n[Trainer] Target AUC {target_auc:.3f} reached at epoch {epoch}. "
                  f"Stopping early.")
            break

    writer.close()
    print(f"\n[Trainer] Done. Best {PRIMARY_METRIC}={best_metric:.4f}")
    print(f"[Trainer] Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path