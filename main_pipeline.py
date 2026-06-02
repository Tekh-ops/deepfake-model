#!/usr/bin/env python3
"""
main_pipeline.py
Unified Multimodal Deepfake Detection Pipeline.
Handles Image, Audio, and Video files, routing them through visual, acoustic, and fact-checking models.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import torch

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Imports from Sub-Pipelines ───────────────────────────────────────
from metadata_analysis.metadata import create_metadata
from metadata_analysis.hashing import compute_sha256, generate_submission_id, append_audit_entry

from factcheck.pipeline import run_factcheck_pipeline
from audio_analysis.audio_detector import audio_predict

# We import the core image inference components directly
from image_inference import (
    resolve_device,
    load_cnn_model,
    load_fft_checkpoint,
    load_fusion_bundle,
    get_transform,
    load_fft_run_config,
    load_stats,
    build_radial_emphasis_mask,
    infer_image,
    DEFAULT_CNN_MODEL,
    DEFAULT_FFT_MODEL,
    DEFAULT_FUSION_BUNDLE,
    DEFAULT_FFT_RUN_CONFIG,
    DEFAULT_FFT_STATS
)
from test_cnn import SUPPORTED_EXTS as IMAGE_EXTS

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract audio from video using FFmpeg. Returns True if successful."""
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            output_audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0 and os.path.exists(output_audio_path)
    except Exception as e:
        print(f"[!] Audio extraction failed: {e}")
        return False


def extract_frames(video_path: str, out_dir: str) -> list[str]:
    """Extract frames based on video duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps > 0 and total_frames > 0:
        duration = total_frames / fps
        # Formula: 1 frame per second, max 32 frames, min 4 frames
        sample_count = max(4, min(int(duration), 32))
    else:
        sample_count = 32
        
    sample_count = min(sample_count, total_frames)
    if sample_count <= 0:
        return []

    indices = sorted(set(int(i) for i in torch.linspace(0, total_frames - 1, sample_count).tolist()))
    
    extracted_paths = []
    idx_set = set(indices)
    frame_idx = 0
    
    while cap.isOpened() and frame_idx <= max(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in idx_set:
            frame_path = os.path.join(out_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
        frame_idx += 1
        
    cap.release()
    return extracted_paths


def main():
    parser = argparse.ArgumentParser(description="Main Multimodal Pipeline for Deepfake Detection")
    parser.add_argument("--input_path", required=True, help="Path to input media (Image/Video/Audio)")
    parser.add_argument("--output_json", help="Path to output metadata JSON", default="final_report.json")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"[!] Error: File not found: {input_path}")
        sys.exit(1)

    ext = input_path.suffix.lower()
    media_type = "unknown"
    if ext in IMAGE_EXTS:
        media_type = "image"
    elif ext in VIDEO_EXTS:
        media_type = "video"
    elif ext in AUDIO_EXTS:
        media_type = "audio"
    else:
        print(f"[!] Error: Unsupported file extension {ext}")
        sys.exit(1)

    print(f"--- BharatShield Unified Pipeline ---")
    print(f"File: {input_path.name}")
    print(f"Type: {media_type.upper()}")
    print("Loading models (RTX 3070 VRAM is sufficient for simultaneous loading)...")

    device = resolve_device(None)
    
    # Pre-load Image Models (if image or video)
    cnn_model, fft_model, bundle, cnn_transform, fft_cfg = None, None, None, None, None
    dataset_mean, dataset_std, radial_mask = None, None, None
    
    if media_type in ("image", "video"):
        cnn_model, _ = load_cnn_model(Path(DEFAULT_CNN_MODEL), device)
        fft_model, _ = load_fft_checkpoint(Path(DEFAULT_FFT_MODEL), device)
        bundle = load_fusion_bundle(Path(DEFAULT_FUSION_BUNDLE))
        cnn_transform = get_transform()
        
        fft_cfg = load_fft_run_config(Path(DEFAULT_FFT_RUN_CONFIG))
        if fft_cfg.norm_mode == "dataset" and fft_cfg.stats_file.is_file():
            dataset_mean, dataset_std = load_stats(fft_cfg.stats_file)
            
        if fft_cfg.radial_emphasis:
            radial_mask = build_radial_emphasis_mask(fft_cfg.image_size, fft_cfg.radial_emphasis_sigma)

    start_time = time.time()
    
    # ── 1. Hashing & Identity ──
    file_hash = compute_sha256(str(input_path))
    sub_id = generate_submission_id()
    
    overall_confidence = 0.0
    overall_result = "Real"
    detection_detail = {}

    # ── 2. Image Pipeline ──
    if media_type == "image":
        img_res = infer_image(
            input_path,
            cnn_model=cnn_model,
            fft_model=fft_model,
            bundle=bundle,
            device=device,
            cnn_transform=cnn_transform,
            fft_cfg=fft_cfg,
            dataset_mean=dataset_mean,
            dataset_std=dataset_std,
            radial_mask=radial_mask,
            face_crop=True,
            skip_no_face=False
        )
        overall_confidence = img_res.confidence
        overall_result = img_res.label_final.capitalize()
        detection_detail["visual"] = {
            "cnn_prob": img_res.cnn.prob_fake,
            "fft_prob": img_res.fft.prob_fake,
            "fusion_prob": img_res.prob_final,
            "reliability": img_res.reliability,
            "ood_flags": img_res.ood_flags
        }

    # ── 3. Audio Pipeline ──
    elif media_type == "audio":
        audio_res = audio_predict(str(input_path))
        factcheck_res = run_factcheck_pipeline(str(input_path), media_type="audio")
        
        overall_result = audio_res.get("label", "Unknown")
        fake_prob = audio_res.get("fake_prob", 0.5)
        overall_confidence = fake_prob if overall_result == "Fake" else (1.0 - fake_prob)
        
        detection_detail["audio"] = {
            "fake_prob": fake_prob,
            "note": audio_res.get("note", "")
        }
        detection_detail["factcheck"] = factcheck_res

    # ── 4. Video Pipeline (Multimodal) ──
    elif media_type == "video":
        with tempfile.TemporaryDirectory() as tmpdir:
            print("Extracting frames...")
            frame_paths = extract_frames(str(input_path), tmpdir)
            print(f"Extracted {len(frame_paths)} frames for analysis.")
            
            frame_probs = []
            for fp in frame_paths:
                try:
                    res = infer_image(
                        Path(fp),
                        cnn_model=cnn_model,
                        fft_model=fft_model,
                        bundle=bundle,
                        device=device,
                        cnn_transform=cnn_transform,
                        fft_cfg=fft_cfg,
                        dataset_mean=dataset_mean,
                        dataset_std=dataset_std,
                        radial_mask=radial_mask,
                        face_crop=True,
                        skip_no_face=False
                    )
                    frame_probs.append(res.prob_final)
                except Exception as e:
                    print(f"Skipping frame due to error: {e}")
            
            if frame_probs:
                avg_visual_fake_prob = sum(frame_probs) / len(frame_probs)
                visual_label = "Fake" if avg_visual_fake_prob >= float(bundle.thresholds.default) else "Real"
            else:
                avg_visual_fake_prob = 0.5
                visual_label = "Unknown"
                
            detection_detail["visual"] = {
                "average_fake_prob": avg_visual_fake_prob,
                "frames_analyzed": len(frame_probs),
                "label": visual_label
            }

            print("Extracting audio for analysis...")
            tmp_audio = os.path.join(tmpdir, "extracted_audio.wav")
            audio_extracted = extract_audio_from_video(str(input_path), tmp_audio)
            
            audio_fake_prob = 0.5
            audio_label = "Unknown"
            if audio_extracted:
                audio_res = audio_predict(tmp_audio)
                audio_fake_prob = audio_res.get("fake_prob", 0.5)
                audio_label = audio_res.get("label", "Unknown")
                detection_detail["audio"] = {
                    "fake_prob": audio_fake_prob,
                    "label": audio_label,
                    "note": audio_res.get("note", "")
                }
                
                print("Running factcheck pipeline...")
                factcheck_res = run_factcheck_pipeline(str(input_path), media_type="video", existing_audio_path=tmp_audio)
                detection_detail["factcheck"] = factcheck_res
            else:
                print("No audio track found or extraction failed.")
                detection_detail["audio"] = {"available": False}
                detection_detail["factcheck"] = {"available": False, "warnings": ["No audio track found"]}
                
            # Unified Multimodal Score (Max of visual and audio probability to be sensitive to either manipulation)
            if audio_label != "Unknown" and visual_label != "Unknown":
                final_fake_prob = max(avg_visual_fake_prob, audio_fake_prob)
            elif visual_label != "Unknown":
                final_fake_prob = avg_visual_fake_prob
            elif audio_label != "Unknown":
                final_fake_prob = audio_fake_prob
            else:
                final_fake_prob = 0.5
                
            overall_result = "Fake" if final_fake_prob >= 0.5 else "Real"
            overall_confidence = final_fake_prob if overall_result == "Fake" else (1.0 - final_fake_prob)

    processing_time_ms = (time.time() - start_time) * 1000

    # ── 5. Build Metadata & Log ──
    metadata = create_metadata(
        file_path=str(input_path),
        result=overall_result,
        confidence=overall_confidence,
        submission_id=sub_id,
        file_hash=file_hash,
        detection_detail=detection_detail,
        processing_time_ms=processing_time_ms
    )

    append_audit_entry(
        submission_id=sub_id,
        file_hash=file_hash,
        action="MULTIMODAL_DETECTION",
        actor="main_pipeline",
        result={"label": overall_result, "confidence": overall_confidence}
    )

    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            print(f"\n[+] Full report saved to: {args.output_json}")
        except Exception as e:
            print(f"[!] Failed to save JSON report: {e}")

    print("\n" + "="*50)
    print("UNIFIED DEEPFAKE DETECTION SUMMARY")
    print("="*50)
    print(f"Submission ID: {sub_id}")
    print(f"Media Type:    {metadata['file']['media_type'].upper()}")
    print(f"Final Label:   {metadata['detection']['label']}")
    print(f"Confidence:    {metadata['detection']['confidence']:.4f}")
    print(f"Risk Level:    {metadata['detection']['risk_level']}")
    
    if "factcheck" in detection_detail and detection_detail["factcheck"].get("available"):
        fc = detection_detail["factcheck"]
        print(f"Misinfo Risk:  {fc.get('overall_misinfo_risk', 'UNKNOWN')}")
        claims = fc.get("claims", [])
        if claims:
            print(f"Claims Found:  {len(claims)}")

    print(f"Process Time:  {metadata['processing']['time_ms']} ms")
    print("="*50)


if __name__ == "__main__":
    main()
