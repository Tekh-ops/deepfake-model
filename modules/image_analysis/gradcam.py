"""
utils/gradcam.py
────────────────
Grad-CAM heatmap for CNN-based models.
Works with the secondary Xception detector.

For the primary HuggingFace EfficientNet-B4 we use
attention-map visualization instead since HF models
wrap the backbone differently.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from config import DEVICE, IMAGE_SIZE


IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _get_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv2d layer in a PyTorch model."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d found — Grad-CAM not applicable.")
    return last_conv


def generate_gradcam_secondary(
    image_path: str,
    save_path: str = "/tmp/gradcam_output.png",
) -> str:
    """
    Generate a Grad-CAM heatmap using the secondary (Xception) model
    and overlay it on the original image.

    Args:
        image_path: Path to the input image.
        save_path:  Where to save the overlay PNG.

    Returns:
        save_path (so Flask can send_file it).
    """
    from modules.core.detector import get_secondary

    model = get_secondary()
    model.eval()

    # ── Preprocessing ─────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        IMAGENET_NORM,
    ])
    pil_img = Image.open(image_path).convert("RGB")
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # ── Hook last conv layer ──────────────────────────────────
    activations = {}
    gradients   = {}

    last_conv = _get_last_conv(model.backbone)

    def fwd_hook(module, inp, out):
        activations["v"] = out

    def bwd_hook(module, gin, gout):
        gradients["v"] = gout[0]

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)

    # ── Forward + backward ────────────────────────────────────
    logits     = model(tensor)
    fake_score = logits[0, 1]   # FAKE class
    model.zero_grad()
    fake_score.backward()

    h1.remove()
    h2.remove()

    # ── Compute CAM ───────────────────────────────────────────
    grads   = gradients["v"][0]           # (C, H, W)
    acts    = activations["v"][0]         # (C, H, W)
    weights = grads.mean(dim=(1, 2))      # global avg pool

    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = F.relu(cam).detach().cpu().numpy()

    # Normalise and resize to input size
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    # ── Overlay ───────────────────────────────────────────────
    orig = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * orig + 0.45 * heat, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(save_path)
    return save_path