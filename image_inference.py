#!/usr/bin/env python3
"""
Unified demo inference: Spatial CNN + FFT branch + calibrated logistic fusion.

Usage:
    python image_inference.py --input_path path/to/image.jpg
    python image_inference.py --input_path test_images --face_crop
    python image_inference.py --input_path photo.jpg --device cuda:0

Recommended (Windows):
    D:\\envs\\gpu_env\\python.exe image_inference.py --input_path test_images/real --face_crop
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent
_FFT_DIR = _REPO_ROOT / "fft"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FFT_DIR) not in sys.path:
    sys.path.insert(0, str(_FFT_DIR))

from test_cnn import (
    SUPPORTED_EXTS,
    get_transform,
    iter_image_paths,
    load_image_pil,
    load_model as load_cnn_model,
    logit_to_fake_prob,
    predict_logit,
    preprocess_image as preprocess_cnn_image,
)

from fft_model import build_model as build_fft_model
from fft_preprocess import (
    build_radial_emphasis_mask,
    load_stats,
    preprocess_image as preprocess_fft_image,
)
from fusion_bundle import FusionBundle, apply_fusion_bundle, load_fusion_bundle

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CNN_MODEL = _REPO_ROOT / "training_output" / "best_model.pth"
DEFAULT_FFT_MODEL = _FFT_DIR / "fft_output" / "best_fft_model.pth"
DEFAULT_FFT_STATS = _FFT_DIR / "fft_output" / "fft_stats.json"
DEFAULT_FFT_RUN_CONFIG = _FFT_DIR / "fft_output" / "fft_run_config.json"
DEFAULT_FUSION_BUNDLE = _FFT_DIR / "fusion_bundle"
BRANCH_THRESHOLD = 0.5


@dataclass
class BranchResult:
    logit: float
    prob_fake: float
    label: str


@dataclass
class ImageInferenceResult:
    path: Path
    cnn: BranchResult
    fft: BranchResult
    prob_final: float
    label_final: str
    confidence: float
    reliability: str
    reason: str
    fusion_threshold: float
    ood_flags: List[str]
    warning: Optional[str] = None


@dataclass
class FFTPreprocessConfig:
    image_size: int = 224
    channel_mode: str = "ycbcr_y"
    norm_mode: str = "dataset"
    radial_emphasis: bool = True
    radial_emphasis_sigma: float = 0.3
    stats_file: Path = DEFAULT_FFT_STATS


def resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_fft_run_config(path: Path) -> FFTPreprocessConfig:
    cfg = FFTPreprocessConfig()
    if not path.is_file():
        return cfg
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    stats = raw.get("stats_file", str(DEFAULT_FFT_STATS))
    stats_path = Path(stats)
    if not stats_path.is_absolute():
        stats_path = (_FFT_DIR / stats_path).resolve()
    return FFTPreprocessConfig(
        image_size=int(raw.get("image_size", cfg.image_size)),
        channel_mode=str(raw.get("channel_mode", cfg.channel_mode)),
        norm_mode=str(raw.get("norm_mode", cfg.norm_mode)),
        radial_emphasis=bool(raw.get("radial_emphasis", cfg.radial_emphasis)),
        radial_emphasis_sigma=float(raw.get("radial_emphasis_sigma", cfg.radial_emphasis_sigma)),
        stats_file=stats_path,
    )


def load_fft_checkpoint(
    model_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"FFT model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = build_fft_model(arch).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, arch


@torch.no_grad()
def run_fft_forward(
    model: nn.Module,
    device: torch.device,
    spectrum: np.ndarray,
) -> float:
    tensor = torch.from_numpy(spectrum).unsqueeze(0).unsqueeze(0).to(device).float()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logit = model(tensor)
    return float(logit.view(-1).item())


def run_cnn_branch(
    pil_img,
    cnn_model: nn.Module,
    device: torch.device,
    transform,
    *,
    face_crop: bool,
    skip_no_face: bool,
) -> Tuple[Optional[BranchResult], Optional[str], Optional[bool]]:
    prep = preprocess_cnn_image(
        pil_img,
        device=device,
        face_crop=face_crop,
        skip_no_face=skip_no_face,
        resize_if_not_380=True,
    )
    if prep is None:
        return None, "No face detected (CNN skipped; use without --skip_no_face to resize full image)", False

    logit = predict_logit(cnn_model, device, prep.image_np, transform)
    prob = logit_to_fake_prob(logit)
    return BranchResult(logit=logit, prob_fake=prob, label=_label_from_prob(prob, BRANCH_THRESHOLD)), None, prep.face_detected


def run_fft_branch(
    image_path: Path,
    fft_model: nn.Module,
    device: torch.device,
    fft_cfg: FFTPreprocessConfig,
    dataset_mean: Optional[float],
    dataset_std: Optional[float],
    radial_mask: Optional[np.ndarray],
) -> BranchResult:
    spectrum = preprocess_fft_image(
        image_path,
        image_size=fft_cfg.image_size,
        channel_mode=fft_cfg.channel_mode,
        norm_mode=fft_cfg.norm_mode,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        use_radial_emphasis=fft_cfg.radial_emphasis,
        radial_mask=radial_mask,
    )
    logit = run_fft_forward(fft_model, device, spectrum)
    prob = logit_to_fake_prob(logit)
    return BranchResult(logit=logit, prob_fake=prob, label=_label_from_prob(prob, BRANCH_THRESHOLD))


def fuse_logistic(
    bundle: FusionBundle,
    logit_cnn: float,
    logit_fft: float,
) -> float:
    """Platt calibration + logistic stacking (bundle's selected method if logistic)."""
    spatial_logits = np.array([logit_cnn], dtype=np.float32)
    fft_logits = np.array([logit_fft], dtype=np.float32)
    probs = apply_fusion_bundle(bundle, spatial_logits, fft_logits)
    return float(probs[0])


def _label_from_prob(prob_fake: float, threshold: float) -> str:
    return "FAKE" if prob_fake >= threshold else "REAL"


def assess_reliability(
    prob_cnn: float,
    prob_fft: float,
    prob_final: float,
    confidence: float,
    fusion_threshold: float,
    ood_flags: List[str],
) -> Tuple[str, str]:
    """Assess reliability based on branch probabilities and overall confidence.
    `confidence` should be computed as prob_final if final prediction is FAKE, else 1 - prob_final.
    """
    # 1. OOD rules
    if "low_resolution" in ood_flags:
        return "LOW", "Low resolution image; may be out-of-distribution"

    if "no_face_detected" in ood_flags:
        return "LOW", "No face detected; input may be invalid or out-of-domain"

    if prob_cnn < 0.3 and prob_fft < 0.05:
        if "very_confident_real" not in ood_flags:
            ood_flags.append("very_confident_real")
        return "LOW", "Both models strongly predict REAL; possible out-of-distribution input"

    # 2. CNN vs FFT disagreement
    cnn_pred = prob_cnn >= BRANCH_THRESHOLD
    fft_pred = prob_fft >= BRANCH_THRESHOLD

    if cnn_pred != fft_pred:
        if "model_disagreement" not in ood_flags:
            ood_flags.append("model_disagreement")
        return "LOW", "CNN and FFT disagree"

    # 3. Confidence-based rules
    if abs(prob_final - fusion_threshold) < 0.05:
        if "near_threshold" not in ood_flags:
            ood_flags.append("near_threshold")
        return "LOW", "Prediction near decision threshold"

    if confidence >= 0.8:
        return "HIGH", "CNN and FFT agree with strong confidence"

    if 0.6 <= confidence < 0.8:
        return "MEDIUM", "CNN and FFT agree with moderate confidence"

    return "LOW", "CNN and FFT agree but confidence below moderate threshold"


def infer_image(
    image_path: Path,
    *,
    cnn_model: nn.Module,
    fft_model: nn.Module,
    bundle: FusionBundle,
    device: torch.device,
    cnn_transform,
    fft_cfg: FFTPreprocessConfig,
    dataset_mean: Optional[float],
    dataset_std: Optional[float],
    radial_mask: Optional[np.ndarray],
    face_crop: bool,
    skip_no_face: bool,
) -> ImageInferenceResult:
    fusion_threshold = float(bundle.thresholds.default)
    ood_flags = []

    try:
        pil_img = load_image_pil(image_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {exc}") from exc

    # Low image resolution check
    width, height = pil_img.size
    if width < 128 or height < 128:
        ood_flags.append("low_resolution")

    cnn_result, cnn_warn, face_detected = run_cnn_branch(
        pil_img,
        cnn_model,
        device,
        cnn_transform,
        face_crop=face_crop,
        skip_no_face=skip_no_face,
    )
    if cnn_result is None:
        if face_crop and face_detected is False:
            ood_flags.append("no_face_detected")
        raise RuntimeError(cnn_warn or "CNN inference failed")

    if face_crop and face_detected is False:
        ood_flags.append("no_face_detected")

    fft_result = run_fft_branch(
        image_path,
        fft_model,
        device,
        fft_cfg,
        dataset_mean,
        dataset_std,
        radial_mask,
    )

    prob_final = fuse_logistic(bundle, cnn_result.logit, fft_result.logit)
    label_final = _label_from_prob(prob_final, fusion_threshold)
    # Compute confidence based on final prediction
    confidence = prob_final if label_final == "FAKE" else 1 - prob_final
    reliability, reason = assess_reliability(
        cnn_result.prob_fake,
        fft_result.prob_fake,
        prob_final,
        confidence,
        fusion_threshold,
        ood_flags,
    )

    return ImageInferenceResult(
        path=image_path,
        cnn=cnn_result,
        fft=fft_result,
        prob_final=prob_final,
        label_final=label_final,
        confidence=confidence,
        reliability=reliability,
        reason=reason,
        fusion_threshold=fusion_threshold,
        ood_flags=ood_flags,
        warning=cnn_warn,
    )


def print_result(result: ImageInferenceResult) -> None:
    print("----------------------------------------")
    print(f"Image: {result.path.name}")
    print()
    print(f"CNN:    {result.cnn.label} (prob={result.cnn.prob_fake:.3f})")
    print(f"FFT:    {result.fft.label} (prob={result.fft.prob_fake:.3f})")
    print(f"Fusion: {result.label_final} (prob={result.prob_final:.3f})")
    print()
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Reliability: {result.reliability}")
    print(f"Reason: {result.reason}")
    flags_str = ", ".join(result.ood_flags)
    print(f"OOD Flags: [{flags_str}]")
    if result.warning:
        print(f"Note: {result.warning}")
    print(f"(Fusion threshold: {result.fusion_threshold:.3f})")
    print("----------------------------------------")


def collect_input_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        paths = iter_image_paths(input_path)
        if not paths:
            raise FileNotFoundError(f"No images found under {input_path}")
        return paths
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified CNN + FFT + fusion inference for deepfake detection"
    )
    p.add_argument("--input_path", type=str, required=True, help="Image file or folder")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda:0 if available, else cpu)",
    )
    p.add_argument("--cnn_model", type=str, default=str(DEFAULT_CNN_MODEL))
    p.add_argument("--fft_model", type=str, default=str(DEFAULT_FFT_MODEL))
    p.add_argument("--fft_stats", type=str, default=str(DEFAULT_FFT_STATS))
    p.add_argument("--fft_run_config", type=str, default=str(DEFAULT_FFT_RUN_CONFIG))
    p.add_argument("--fusion_bundle", type=str, default=str(DEFAULT_FUSION_BUNDLE))
    p.add_argument(
        "--face_crop",
        action="store_true",
        help="MTCNN face align to 380x380 for CNN (recommended for full-frame photos)",
    )
    p.add_argument(
        "--skip_no_face",
        action="store_true",
        help="Skip image if no face detected (only with --face_crop)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    input_path = Path(args.input_path).expanduser().resolve()

    print(f"Device: {device}")
    print(f"CNN model: {args.cnn_model}")
    print(f"FFT model: {args.fft_model}")
    print(f"Fusion bundle: {args.fusion_bundle}")
    print(f"Face crop: {args.face_crop}")
    print()

    # Load models and bundle once
    cnn_model, _cnn_meta = load_cnn_model(Path(args.cnn_model), device)
    fft_model, fft_arch = load_fft_checkpoint(Path(args.fft_model), device)
    bundle = load_fusion_bundle(Path(args.fusion_bundle))
    cnn_transform = get_transform()

    fft_cfg = load_fft_run_config(Path(args.fft_run_config))
    if args.fft_stats:
        fft_cfg.stats_file = Path(args.fft_stats).resolve()

    dataset_mean, dataset_std = None, None
    if fft_cfg.norm_mode == "dataset":
        if not fft_cfg.stats_file.is_file():
            raise FileNotFoundError(f"FFT stats not found: {fft_cfg.stats_file}")
        dataset_mean, dataset_std = load_stats(fft_cfg.stats_file)

    radial_mask = None
    if fft_cfg.radial_emphasis:
        radial_mask = build_radial_emphasis_mask(
            fft_cfg.image_size, fft_cfg.radial_emphasis_sigma
        )

    fusion_method = bundle.config.get("selected_fusion_method", bundle.selected_method)
    print(
        f"FFT arch: {fft_arch} | preprocess: {fft_cfg.channel_mode} @ {fft_cfg.image_size}px "
        f"| fusion: {fusion_method} | threshold: {bundle.thresholds.default:.3f}"
    )
    print()

    try:
        image_paths = collect_input_paths(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 1

    n_ok = 0
    n_err = 0
    for img_path in image_paths:
        try:
            result = infer_image(
                img_path,
                cnn_model=cnn_model,
                fft_model=fft_model,
                bundle=bundle,
                device=device,
                cnn_transform=cnn_transform,
                fft_cfg=fft_cfg,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                radial_mask=radial_mask,
                face_crop=args.face_crop,
                skip_no_face=args.skip_no_face,
            )
            print_result(result)
            print()
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print("----------------------------------------")
            print(f"Image: {img_path.name}")
            print(f"ERROR: {exc}")
            print("----------------------------------------")
            print()

    print(f"Done. {n_ok} succeeded, {n_err} failed, {len(image_paths)} total.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
