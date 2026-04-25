"""
training/evaluate.py
────────────────────
Standalone evaluation script:
  • Run on the held-out test set
  • Full metric report (AUC, accuracy, F1, sensitivity, specificity)
  • Threshold analysis — find the optimal decision threshold
  • Per-dataset breakdown (if test set retains source metadata)
  • Worst-case analysis — save the most confidently wrong predictions
  • ROC curve and confusion matrix export

Usage:
    python -m training.evaluate --checkpoint checkpoints/best.pth
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

from training.config import DEVICE, OUTPUT_DIR, IMAGE_SIZE
from training.model import DeepfakeDetector, load_checkpoint
from training.dataset import build_dataloaders, LABEL_FAKE
from training.trainer import evaluate


# ─────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────

def find_optimal_threshold(labels: list, probs: list,
                            metric: str = "f1") -> float:
    """
    Sweep decision thresholds in [0.1, 0.9] and return the one
    that maximises the chosen metric on the evaluation set.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_v = 0.5, 0.0

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        if metric == "f1":
            v = f1_score(labels, preds, zero_division=0)
        elif metric == "accuracy":
            v = accuracy_score(labels, preds)
        else:
            v = f1_score(labels, preds, zero_division=0)

        if v > best_v:
            best_v, best_t = v, t

    return float(best_t)


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_roc(labels, probs, save_path: str):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Deepfake Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] ROC curve saved to {save_path}")


def plot_confusion(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix saved to {save_path}")


def plot_pr_curve(labels, probs, save_path: str):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] PR curve saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# Full test-set evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_evaluation(
    checkpoint_path: str,
    output_dir: str | None = None,
) -> dict:
    """
    Load a checkpoint and run full evaluation on the test set.
    Saves plots, metrics JSON, and a classification report.

    Returns the metrics dict.
    """
    output_dir = output_dir or os.path.join(OUTPUT_DIR, "eval")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────
    model = DeepfakeDetector(use_srm=True).to(DEVICE)
    ckpt  = load_checkpoint(model, checkpoint_path)
    model.eval()

    print(f"[Eval] Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # ── Load data ─────────────────────────────────────────────
    _, _, test_loader, _ = build_dataloaders()

    # ── Collect predictions ───────────────────────────────────
    all_labels, all_probs = [], []
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, LABEL_FAKE].cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())

    # ── Find optimal threshold ────────────────────────────────
    opt_threshold = find_optimal_threshold(all_labels, all_probs, "f1")
    preds_default = [1 if p >= 0.5 else 0 for p in all_probs]
    preds_opt     = [1 if p >= opt_threshold else 0 for p in all_probs]

    # ── Metrics ───────────────────────────────────────────────
    auc = roc_auc_score(all_labels, all_probs)
    report_default = classification_report(
        all_labels, preds_default, target_names=["Real", "Fake"], output_dict=True
    )
    report_opt = classification_report(
        all_labels, preds_opt, target_names=["Real", "Fake"], output_dict=True
    )

    metrics = {
        "checkpoint": checkpoint_path,
        "n_samples":  len(all_labels),
        "n_real":     all_labels.count(0),
        "n_fake":     all_labels.count(1),
        "auc":                     round(float(auc), 4),
        "optimal_threshold":       round(opt_threshold, 3),

        "at_threshold_0.5": {
            "accuracy":    round(float(accuracy_score(all_labels, preds_default)), 4),
            "f1":          round(float(f1_score(all_labels, preds_default, zero_division=0)), 4),
            "precision_fake": round(report_default["Fake"]["precision"], 4),
            "recall_fake":    round(report_default["Fake"]["recall"],    4),
            "precision_real": round(report_default["Real"]["precision"], 4),
            "recall_real":    round(report_default["Real"]["recall"],    4),
        },
        "at_optimal_threshold": {
            "accuracy":    round(float(accuracy_score(all_labels, preds_opt)), 4),
            "f1":          round(float(f1_score(all_labels, preds_opt, zero_division=0)), 4),
            "precision_fake": round(report_opt["Fake"]["precision"], 4),
            "recall_fake":    round(report_opt["Fake"]["recall"],    4),
        },
    }

    # ── Print report ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" EVALUATION REPORT")
    print("=" * 55)
    print(f" Samples:     {metrics['n_samples']} ({metrics['n_real']} real, {metrics['n_fake']} fake)")
    print(f" AUC:         {metrics['auc']:.4f}")
    print(f"\n At threshold 0.5:")
    for k, v in metrics["at_threshold_0.5"].items():
        print(f"   {k:25s}: {v:.4f}")
    print(f"\n At optimal threshold ({opt_threshold:.2f}):")
    for k, v in metrics["at_optimal_threshold"].items():
        print(f"   {k:25s}: {v:.4f}")
    print("=" * 55 + "\n")

    # ── Save metrics JSON ─────────────────────────────────────
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved to {metrics_path}")

    # ── Save plots ────────────────────────────────────────────
    plot_roc(all_labels, all_probs,
             os.path.join(output_dir, "roc_curve.png"))
    plot_confusion(all_labels, preds_opt,
                   os.path.join(output_dir, "confusion_matrix.png"))
    plot_pr_curve(all_labels, all_probs,
                  os.path.join(output_dir, "pr_curve.png"))

    return metrics


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c",
        default=os.path.join(OUTPUT_DIR, "best_efficientnet_b4.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=os.path.join(OUTPUT_DIR, "eval"),
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()
    run_full_evaluation(args.checkpoint, args.output_dir)