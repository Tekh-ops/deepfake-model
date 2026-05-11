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

from training.config import (
    DEVICE, EPOCHS, OUTPUT_DIR, BEST_CKPT,
    ADV_TRAIN_PROB, ADV_EPS, ADV_ALPHA, ADV_STEPS,
    SWAD_ENABLED, SWAD_START_FRAC, TARGET_AUC,
)
from training.dataset import build_dataloaders
from training.model import (
    build_model, load_checkpoint, DeepfakeDetector, save_checkpoint
)
from training.trainer import train, EWC
from training.evaluate import run_full_evaluation
from training.adversarial import SWADCollector


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
    p.add_argument("--adv",         action="store_true",
                   help="Enable PGD adversarial training (20%% of batches)")
    p.add_argument("--no_swad",     action="store_true",
                   help="Disable SWAD weight averaging (enabled by default)")
    p.add_argument("--target_auc",  type=float, default=TARGET_AUC,
                   help="Stop training early if val AUC exceeds this")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config if needed
    import training.config as cfg
    if args.epochs != EPOCHS:
        cfg.EPOCHS = args.epochs

    if args.max_samples:
        cfg.MAX_PER_DATASET = args.max_samples
    if args.adv:
        cfg.ADV_TRAIN_PROB = ADV_TRAIN_PROB   # use config default (0.20)
        print(f"[Main] Adversarial training enabled  "
              f"(PGD eps={ADV_EPS:.4f}, steps={ADV_STEPS}, "
              f"prob={ADV_TRAIN_PROB:.0%})")

    use_swad = SWAD_ENABLED and not args.no_swad

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
        print("[Main] Computing Fisher information matrix for EWC ...")
        ewc = EWC(old_model, train_loader, ewc_lambda=400.0)
        del old_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Main] EWC ready.")

    # ── SWAD setup ───────────────────────────────────────────
    swad = None
    if use_swad:
        epochs_total = cfg.EPOCHS
        swad_start   = max(1, int(epochs_total * SWAD_START_FRAC))
        swad         = SWADCollector(model,
                                     start_epoch=swad_start,
                                     end_epoch=epochs_total)
        print(f"[Main] SWAD enabled: collecting from epoch "
              f"{swad_start} → {epochs_total}")

    # ── Print summary ─────────────────────────────────────────
    raw = model.module if hasattr(model, "module") else model
    total_params     = sum(p.numel() for p in raw.parameters())
    trainable_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    print(f"\n[Main] Model: EfficientNet-B4  SRM={use_srm}")
    print(f"[Main] Parameters: {total_params/1e6:.1f}M total, "
          f"{trainable_params/1e6:.1f}M trainable")
    print(f"[Main] Device: {DEVICE}")
    print(f"[Main] Training batches per epoch: {len(train_loader)}")
    print(f"[Main] Adversarial training: {'YES' if args.adv else 'NO'}")
    print(f"[Main] SWAD: {'YES' if use_swad else 'NO'}")

    # ── Train ─────────────────────────────────────────────────
    best_ckpt = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        class_weights = class_weights,
        ewc          = ewc,
        swad         = swad,
        run_name     = args.run_name,
        adv_training = args.adv,
        target_auc   = args.target_auc,
    )

    # ── SWAD averaging ────────────────────────────────────────
    if use_swad and swad is not None and swad.n_collected > 0:
        print(f"\n[Main] Applying SWAD: averaging {swad.n_collected} snapshots ...")
        model = swad.get_averaged_model(model)
        swad_ckpt = best_ckpt.replace(".pth", "_swad.pth")
        save_checkpoint(model, None, None, cfg.EPOCHS, {"swad": True}, swad_ckpt)
        print(f"[Main] SWAD model saved to {swad_ckpt}")

    # ── Final test-set evaluation ─────────────────────────────
    print("\n[Main] Running final evaluation on test set ...")
    run_full_evaluation(best_ckpt)

    print("\n[Main] Training complete.")
    print(f"[Main] Best model: {best_ckpt}")
    print(f"[Main] Global inference checkpoint: {BEST_CKPT}")
    print("[Main] The Flask backend will automatically pick up the new weights.")


if __name__ == "__main__":
    main()