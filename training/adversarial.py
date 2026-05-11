"""
training/adversarial.py
────────────────────────
Adversarial training utilities for robustness against input perturbations.

PGD (Projected Gradient Descent) Attack:
  Generates worst-case perturbations within an L∞ ball of radius eps.
  Used to harden the detector against adversaries who slightly modify
  deepfakes to evade detection (e.g., adding imperceptible noise).

Reference:
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial
  Attacks" (ICLR 2018)

SWAD (Stochastic Weight Averaging Densely):
  Collects model weights at every step in the dense averaging window
  (last 40% of training) to find flat minima that generalise to unseen
  deepfake methods.

Reference:
  Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima"
  (NeurIPS 2021)
"""

import copy
import random
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    eps: float    = 4 / 255,    # L∞ ball radius (4/255 is standard)
    alpha: float  = 1 / 255,    # step size
    steps: int    = 7,          # number of PGD steps
    rand_init: bool = True,     # random initialisation inside L∞ ball
) -> torch.Tensor:
    """
    PGD adversarial attack (L∞).

    Args:
        model:      The detector model (must be in train mode).
        images:     Clean input images (B, C, H, W) in [0, 1].
        labels:     Ground truth labels (B,).
        criterion:  Loss function.
        eps:        L∞ perturbation budget (default 4/255).
        alpha:      PGD step size (default 1/255).
        steps:      Number of gradient ascent steps (default 7).
        rand_init:  Randomise start within L∞ ball.

    Returns:
        Adversarially perturbed images tensor (same shape as images),
        detached from the computation graph.
    """
    images  = images.clone().detach()
    labels  = labels.clone().detach()

    if rand_init:
        delta = torch.empty_like(images).uniform_(-eps, eps)
        perturbed = torch.clamp(images + delta, 0, 1)
    else:
        perturbed = images.clone()

    for _ in range(steps):
        perturbed = perturbed.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = model(perturbed)
            loss    = criterion(outputs, labels)

        loss.backward()

        grad      = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad
        delta     = torch.clamp(perturbed - images, -eps, eps)
        perturbed = torch.clamp(images + delta, 0, 1)

    return perturbed.detach()


def apply_adversarial_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    prob: float = 0.20,
    eps: float  = 4 / 255,
    alpha: float = 1 / 255,
    steps: int  = 7,
) -> torch.Tensor:
    """
    Apply PGD to a random subset of the batch.
    prob=0.20 means 20% of each batch gets adversarial perturbations.
    The model stays in train mode throughout.

    Returns the (partially) perturbed batch.
    """
    if random.random() >= prob:
        return images   # skip adversarial augmentation for this batch

    was_training = model.training
    model.train()   # ensure grad flows through the model

    perturbed = pgd_attack(model, images, labels, criterion,
                           eps=eps, alpha=alpha, steps=steps)

    if not was_training:
        model.eval()

    return perturbed


# ─────────────────────────────────────────────────────────────
# SWAD — Stochastic Weight Averaging Densely
# ─────────────────────────────────────────────────────────────

class SWADCollector:
    """
    Densely collect model weights during the last 40% of training.
    Averaging over a dense collection of checkpoints finds flat loss
    minima that generalise better to unseen deepfake methods.

    Usage:
        swad = SWADCollector(model, start_epoch=25, end_epoch=40)
        for epoch in range(1, 41):
            train(...)
            swad.update(model, epoch)
        averaged_model = swad.get_averaged_model(model)
    """

    def __init__(self, model: nn.Module,
                 start_epoch: int,
                 end_epoch: int):
        """
        Args:
            model:       The model being trained.
            start_epoch: Epoch to start collecting (inclusive).
            end_epoch:   Epoch to stop collecting (inclusive).
        """
        self.start   = start_epoch
        self.end     = end_epoch
        self.weights = []
        print(f"[SWAD] Collecting weights from epoch {start_epoch} to {end_epoch}.")

    def update(self, model: nn.Module, epoch: int) -> bool:
        """
        Call at the END of each epoch.
        Returns True if weights were collected this epoch.
        """
        if self.start <= epoch <= self.end:
            # Store a deep copy of the current state dict
            self.weights.append(
                copy.deepcopy(model.state_dict())
            )
            print(f"[SWAD] Epoch {epoch}: collected "
                  f"({len(self.weights)} total snapshots)")
            return True
        return False

    def get_averaged_model(self, model: nn.Module) -> nn.Module:
        """
        Average all collected weight snapshots and load into model.
        Returns the model with SWAD-averaged weights.
        """
        if not self.weights:
            print("[SWAD] No weights collected — returning model unchanged.")
            return model

        avg_state = {}
        for key in self.weights[0]:
            avg_state[key] = torch.stack(
                [w[key].float() for w in self.weights], dim=0
            ).mean(dim=0)

        model.load_state_dict(avg_state)
        print(f"[SWAD] Averaged {len(self.weights)} snapshots into model.")
        return model

    @property
    def n_collected(self) -> int:
        return len(self.weights)


# ─────────────────────────────────────────────────────────────
# FGSM (Fast Gradient Sign Method) — single-step quick variant
# ─────────────────────────────────────────────────────────────

def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    eps: float = 4 / 255,
) -> torch.Tensor:
    """
    Single-step FGSM for fast adversarial data augmentation.
    Much cheaper than PGD (1 forward + 1 backward pass).
    Use when training speed is more important than attack strength.
    """
    images    = images.clone().detach().requires_grad_(True)
    outputs   = model(images)
    loss      = criterion(outputs, labels)
    loss.backward()

    perturbed = images.detach() + eps * images.grad.sign()
    return torch.clamp(perturbed, 0, 1).detach()
