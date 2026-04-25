"""
train.py
────────
Main entry point for fine-tuning the deepfake detector.

Usage examples:

  # Basic training run (uses settings in training/config.py)
  python train.py

  # Name the run (creates checkpoints/<run_name>/)
  python train.py --run_name ff_plus_v1

  # Limit data for a quick sanity check
  python train.py --max_samples 500 --epochs 3

  # Disable SRM branch
  python train.py --no_srm

  # Continue from a checkpoint
  python train.py --resume checkpoints/myrun/last.pth

  # Fine-tune on new data with EWC (prevents forgetting old data)
  python train.py --ewc --old_checkpoint checkpoints/best.pth

  # Run evaluation only
  python train.py --eval_only --checkpoint checkpoints/best.pth

Multi-GPU:
  torchrun --nproc_per_node=4 train.py --run_name multigpu_v1
  (or just python train.py — it auto-detects and wraps in DataParallel)
"""

import os
import sys
import argparse
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.config import DEVICE, EPOCHS, OUTPUT_DIR, BEST_CKPT
from training.dataset import build_dataloaders
from training.model import (
    build_model, load_checkpoint, DeepfakeDetector, save_checkpoint
)
from training.trainer import train, EWC
from training.evaluate import run_full_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="BharatShield deepfake detector fine-tuning")
    p.add_argument("--run_name",    default="",    help="Name for this training run")
    p.add_argument("--epochs",      type=int, default=EPOCHS)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max samples per dataset (for quick testing)")
    p.add_argument("--no_srm",      action="store_true",
                   help="Disable SRM high-pass filter branch")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--ewc",         action="store_true",
                   help="Use EWC regularisation (requires --old_checkpoint)")
    p.add_argument("--old_checkpoint", default=None,
                   help="Checkpoint with old task weights for EWC")
    p.add_argument("--eval_only",   action="store_true",
                   help="Skip training and run evaluation only")
    p.add_argument("--checkpoint",  default=BEST_CKPT,
                   help="Checkpoint for --eval_only")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config if needed
    if args.epochs != EPOCHS:
        import training.config as cfg
        cfg.EPOCHS = args.epochs

    if args.max_samples:
        import training.config as cfg
        cfg.MAX_PER_DATASET = args.max_samples

    # ── Eval only ────────────────────────────────────────────
    if args.eval_only:
        print(f"[Main] Evaluation mode: {args.checkpoint}")
        run_full_evaluation(args.checkpoint)
        return

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        max_per_dataset=args.max_samples
    )

    # ── Model ─────────────────────────────────────────────────
    use_srm = not args.no_srm
    model   = build_model(use_srm=use_srm)

    start_epoch = 1

    if args.resume:
        print(f"[Main] Resuming from {args.resume}")
        ckpt = load_checkpoint(model, args.resume, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Main] Resuming from epoch {start_epoch}")

    # ── EWC setup ────────────────────────────────────────────
    ewc = None
    if args.ewc:
        if not args.old_checkpoint:
            raise ValueError("--ewc requires --old_checkpoint")
        old_model = DeepfakeDetector(use_srm=use_srm).to(DEVICE)
        load_checkpoint(old_model, args.old_checkpoint)
        print("[Main] Computing Fisher information matrix for EWC...")
        ewc = EWC(old_model, train_loader, ewc_lambda=400.0)
        del old_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("[Main] EWC ready.")

    # ── Print summary ─────────────────────────────────────────
    raw = model.module if hasattr(model, "module") else model
    total_params   = sum(p.numel() for p in raw.parameters())
    trainable_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    print(f"\n[Main] Model: EfficientNet-B4  SRM={use_srm}")
    print(f"[Main] Parameters: {total_params/1e6:.1f}M total, "
          f"{trainable_params/1e6:.1f}M trainable")
    print(f"[Main] Device: {DEVICE}")
    print(f"[Main] Training batches per epoch: {len(train_loader)}")

    # ── Train ─────────────────────────────────────────────────
    best_ckpt = train(
        model       = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        class_weights = class_weights,
        ewc          = ewc,
        run_name     = args.run_name,
    )

    # ── Final test-set evaluation ─────────────────────────────
    print("\n[Main] Running final evaluation on test set...")
    run_full_evaluation(best_ckpt)

    print("\n[Main] Training complete.")
    print(f"[Main] Best model: {best_ckpt}")
    print(f"[Main] Global inference checkpoint: {BEST_CKPT}")
    print("[Main] The Flask backend will automatically pick up the new weights.")


if __name__ == "__main__":
    main()