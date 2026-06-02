"""
scripts/preprocess_dataset.py
──────────────────────────────
Extracts face crops and audio tracks from massive raw video datasets
(Celeb-DF, DFDC) and audio datasets (ASVspoof2021) into the standard
data/real and data/fake flat hierarchy expected by the PyTorch dataloader.

Features:
  - Multiprocessing (processes videos in parallel to utilize all CPU cores)
  - Face cropping (reduces storage footprint and speeds up training)
  - Automatic handling of DFDC metadata.json
  - Graceful error handling for corrupted videos

Usage:
  python scripts/preprocess_dataset.py --input_dir /path/to/dataset_raw --output_dir data/
"""

import os
import sys
import json
import glob
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.core.video_utils import extract_frames_uniform, extract_audio, crop_face_pil


def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Dataset Preprocessor")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Path to raw dataset directory (e.g. /dataset_raw)")
    p.add_argument("--output_dir", type=str, default="data",
                   help="Path to output extracted frames (e.g. data/)")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Number of frames to extract per video (32 provides great pose/lighting diversity without excessive redundancy)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max videos to process per dataset (for testing)")
    p.add_argument("--workers", type=int, default=cpu_count(),
                   help="Number of multiprocessing workers")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Worker Function (Runs in parallel)
# ─────────────────────────────────────────────────────────────

def process_video(task: dict) -> str:
    """
    Process a single video/audio file.
    task = {
        'filepath': str,
        'label': 'real' | 'fake',
        'dataset_name': str,
        'output_dir': str,
        'num_frames': int
    }
    """
    vid_path = task['filepath']
    label    = task['label']
    ds_name  = task['dataset_name']
    out_dir  = task['output_dir']
    num_f    = task['num_frames']

    vid_name = Path(vid_path).stem
    target_dir = os.path.join(out_dir, ds_name, label)
    os.makedirs(target_dir, exist_ok=True)

    # ── Handle Audio (ASVspoof) ─────────────────────────────
    if vid_path.lower().endswith(('.wav', '.flac', '.mp3')):
        try:
            out_audio = os.path.join(target_dir, f"{vid_name}.wav")
            if not os.path.exists(out_audio):
                extract_audio(vid_path, out_audio)
            return f"OK_AUDIO: {vid_name}"
        except Exception as e:
            return f"ERR_AUDIO: {vid_name} -> {e}"

    # ── Handle Video / Image (Celeb-DF, DFDC, FFHQ) ──────────────
    try:
        if vid_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # It's already an image (e.g. FFHQ)
            frames = [Image.open(vid_path).convert("RGB")]
        else:
            # Extract frames from video
            frames = extract_frames_uniform(vid_path, num_frames=num_f, as_pil=True)
        
        for i, frame_pil in enumerate(frames):
            out_img_path = os.path.join(target_dir, f"{vid_name}_f{i:03d}.jpg")
            if os.path.exists(out_img_path):
                continue
            
            # Crop to face (with 30% padding)
            face_img = crop_face_pil(frame_pil, padding=0.30)
            
            # Save as JPEG (quality=95 to preserve forensics)
            face_img.save(out_img_path, format="JPEG", quality=95)

        return f"OK_VIDEO: {vid_name}"

    except Exception as e:
        return f"ERR_VIDEO: {vid_name} -> {e}"


# ─────────────────────────────────────────────────────────────
# Parsing Logic per Dataset
# ─────────────────────────────────────────────────────────────

def scan_celebdf(base_path: str, dataset_name: str, limit: int = None) -> list:
    """Scan a Celeb-DF directory."""
    tasks = []
    
    # Real
    real_dirs = [os.path.join(base_path, "Celeb-real"), 
                 os.path.join(base_path, "YouTube-real")]
    for rd in real_dirs:
        for p in glob.glob(os.path.join(rd, "*.mp4")):
            tasks.append({'filepath': p, 'label': 'real', 'dataset_name': dataset_name})
            
    # Fake
    fake_dir = os.path.join(base_path, "Celeb-synthesis")
    for p in glob.glob(os.path.join(fake_dir, "*.mp4")):
        tasks.append({'filepath': p, 'label': 'fake', 'dataset_name': dataset_name})

    if limit:
        tasks = tasks[:limit]
    return tasks


def scan_dfdc(base_path: str, limit: int = None) -> list:
    """Scan DFDC directory (handling the metadata.json mapping)."""
    tasks = []
    part_folders = glob.glob(os.path.join(base_path, "dfdc_train_part_*"))
    
    for folder in part_folders:
        # Sometimes unzipped directly inside, sometimes one layer deep
        # Search for metadata.json recursively in this part folder
        metadata_files = list(Path(folder).rglob("metadata.json"))
        
        for meta_path in metadata_files:
            video_dir = meta_path.parent
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            for filename, info in metadata.items():
                vid_path = os.path.join(video_dir, filename)
                if not os.path.exists(vid_path):
                    continue
                    
                label_str = info.get("label", "").upper()
                label = "fake" if label_str == "FAKE" else "real"
                
                tasks.append({
                    'filepath': vid_path,
                    'label': label,
                    'dataset_name': 'dfdc'
                })

    if limit:
        tasks = tasks[:limit]
    return tasks


def scan_asvspoof(base_path: str, limit: int = None) -> list:
    """Scan ASVspoof2021 LA dataset based on protocol file."""
    tasks = []
    
    # Find protocol file (ignore LICENSE.txt and README.txt)
    protocol_path = None
    for p in Path(base_path).rglob("*.txt"):
        if "trl.txt" in p.name:
            protocol_path = p
            break
            
    if not protocol_path:
        print("[Prep] WARNING: No ASVspoof .trl.txt protocol file found.")
        return tasks

    
    # Search for FLAC files globally inside the base path
    flac_files = {p.stem: str(p) for p in Path(base_path).rglob("*.flac")}
    
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Protocol files can vary in column structure (4 to 6 columns).
            # We robustly search for the file ID and the label (bonafide/spoof).
            file_id = None
            label = "fake"  # Default
            
            for part in parts:
                if part in flac_files:
                    file_id = part
                part_lower = part.lower()
                if part_lower == "bonafide":
                    label = "real"
                elif part_lower == "spoof":
                    label = "fake"
            
            if file_id:
                tasks.append({
                    'filepath': flac_files[file_id],
                    'label': label,
                    'dataset_name': 'asvspoof'
                })

    import random
    random.seed(42)
    random.shuffle(tasks)
    # Downsample ASVspoof to 10%
    tasks = tasks[:max(1, int(len(tasks) * 0.10))]

    if limit:
        tasks = tasks[:limit]
    return tasks

def scan_ffhq(base_path: str, limit: int = None) -> list:
    """Scan FFHQ directory for high-quality real faces."""
    tasks = []
    # Recursively find all PNG and JPG images
    image_files = list(Path(base_path).rglob("*.png")) + list(Path(base_path).rglob("*.jpg"))
    for p in image_files:
        tasks.append({
            'filepath': str(p),
            'label': 'real',
            'dataset_name': 'ffhq'
        })
        
    if limit:
        tasks = tasks[:limit]
    return tasks


# ─────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_dir = args.input_dir
    out_dir = args.output_dir
    
    all_tasks = []

    print(f"[Prep] Scanning {input_dir} ...")

    # 1. Celeb-DF
    celeb_p = os.path.join(input_dir, "Celeb-DF")
    if os.path.exists(celeb_p):
        t = scan_celebdf(celeb_p, "celebdf", args.limit)
        all_tasks.extend(t)
        print(f"  -> Found {len(t)} Celeb-DF files.")

    # 2. Celeb-DF-v2
    celeb2_p = os.path.join(input_dir, "Celeb-DF-v2")
    if os.path.exists(celeb2_p):
        t = scan_celebdf(celeb2_p, "celebdf_v2", args.limit)
        all_tasks.extend(t)
        print(f"  -> Found {len(t)} Celeb-DF-v2 files.")

    # 3. DFDC
    # Note: Handles `dfdc_train_part_00` inside `dataset_raw` directly
    dfdc_t = scan_dfdc(input_dir, args.limit)
    if dfdc_t:
        all_tasks.extend(dfdc_t)
        print(f"  -> Found {len(dfdc_t)} DFDC files.")

    # 4. ASVSpoof2021
    asv_p = os.path.join(input_dir, "ASVspoof2021")
    if os.path.exists(asv_p):
        t = scan_asvspoof(asv_p, args.limit)
        all_tasks.extend(t)
        print(f"  -> Found {len(t)} ASVspoof files (10% sampled).")

    # 5. FFHQ
    ffhq_p = os.path.join(input_dir, "FFHQ")
    if os.path.exists(ffhq_p):
        t = scan_ffhq(ffhq_p, args.limit)
        all_tasks.extend(t)
        print(f"  -> Found {len(t)} FFHQ images.")



    print(f"\n[Prep] Total tasks: {len(all_tasks)}")
    if not all_tasks:
        print("[Prep] No tasks found. Check your folder structure.")
        return

    # Add standard args to tasks
    for t in all_tasks:
        t['output_dir'] = out_dir
        t['num_frames'] = args.num_frames

    print(f"[Prep] Starting multiprocess extraction ({args.workers} workers)...")
    
    # Process pool
    success_count = 0
    error_count = 0
    
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video, all_tasks), total=len(all_tasks)):
            if result.startswith("ERR"):
                error_count += 1
                # print(f"\n{result}") # Uncomment to debug errors
            else:
                success_count += 1

    print(f"\n[Prep] Extraction complete!")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Errors:  {error_count}")
    print(f"\nData is now ready in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
