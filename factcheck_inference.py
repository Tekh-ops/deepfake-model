#!/usr/bin/env python3
"""
factcheck_inference.py
CLI inference script for fact-checking audio/video files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the repository root is in the Python path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from factcheck.pipeline import run_factcheck_pipeline

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}

def safe_ascii(text: str) -> str:
    """Safely encode strings to ASCII to avoid Windows terminal print errors."""
    if not isinstance(text, str):
        return str(text)
    return text.encode("ascii", "replace").decode("ascii")

def main():
    parser = argparse.ArgumentParser(description="Fact-check inference for audio/video files.")
    parser.add_argument("--input_path", required=True, help="Path to the input audio/video file")
    parser.add_argument("--media_type", choices=["audio", "video"], help="Optional media type (auto-detected if not given)")
    parser.add_argument("--output_json", help="Optional path to save full report as JSON")
    parser.add_argument("--pretty", action="store_true", help="Print readable output (default is clean text summary)")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    
    if not input_path.exists():
        print(f"[!] Error: File not found: {input_path}")
        sys.exit(1)

    media_type = args.media_type
    if not media_type:
        ext = input_path.suffix.lower()
        if ext in VIDEO_EXTS:
            media_type = "video"
        elif ext in AUDIO_EXTS:
            media_type = "audio"
        else:
            print(f"[!] Error: Unsupported file type: {ext}. Please specify --media_type.")
            sys.exit(1)

    print(f"--- Fact-Check Inference ---")
    print(f"Input file: {input_path.name}")
    print(f"Media type: {media_type}")
    print("Running pipeline (this may take a while depending on hardware)...")
    
    try:
        result = run_factcheck_pipeline(str(input_path), media_type)
    except Exception as e:
        print(f"[!] Pipeline execution failed: {e}")
        sys.exit(1)

    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            print(f"Full report saved to: {args.output_json}")
        except Exception as e:
            print(f"[!] Failed to save JSON report: {e}")

    print("\n" + "="*50)
    print("FACT-CHECK SUMMARY")
    print("="*50)
    
    if not result.get("available"):
        print("[!] Fact-check pipeline was unable to process this file.")
    
    warnings = result.get("warnings", [])
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {safe_ascii(w)}")

    transcript = result.get("transcript", "")
    if transcript:
        preview = transcript[:150] + "..." if len(transcript) > 150 else transcript
        print(f"\nTRANSCRIPT PREVIEW:\n{safe_ascii(preview)}")
    else:
        print("\nTRANSCRIPT PREVIEW: None")

    claims = result.get("claims", [])
    print(f"\nCLAIMS EXTRACTED: {len(claims)}")
    for i, claim_dict in enumerate(claims, 1):
        claim_text = claim_dict.get("claim", "")
        verdict = claim_dict.get("combined_verdict", "UNKNOWN")
        
        # Get top evidence/source
        evidence = "None"
        newsapi_articles = claim_dict.get("newsapi_articles", [])
        ddg_articles = claim_dict.get("ddg_articles", [])
        sem_result = claim_dict.get("ddg_verdict", {})
        
        if sem_result.get("best_match") and sem_result.get("status") not in ("NO_RESULTS", "UNAVAILABLE", "ERROR"):
            match_txt = str(sem_result.get("best_match"))
            evidence = f"{match_txt[:100]}..."
        elif newsapi_articles:
            src = newsapi_articles[0].get("source", "News Source")
            title = newsapi_articles[0].get("title", "")
            evidence = f"[{src}] {title}"
        elif ddg_articles and isinstance(ddg_articles, list) and len(ddg_articles) > 0 and "title" in ddg_articles[0]:
            title = ddg_articles[0].get("title", "")
            evidence = f"[DDG] {title}"
            
        print(f"  {i}. [Verdict: {verdict}] {safe_ascii(claim_text)}")
        print(f"     Evidence: {safe_ascii(evidence)}")

    harm_analysis = result.get("harm_analysis", {})
    harm_label = harm_analysis.get("label", "UNKNOWN")
    harm_score = harm_analysis.get("harmful_score", 0.0)
    
    overall_risk = result.get("overall_misinfo_risk", "UNKNOWN")

    print(f"\nHARM ANALYSIS:")
    print(f"  Label: {harm_label}")
    print(f"  Harmful Score: {harm_score:.3f}")
    
    print(f"\nOVERALL MISINFORMATION RISK: {overall_risk}")
    print("="*50)


if __name__ == "__main__":
    main()

# Example usage:
# D:\envs\gpu_env\python.exe factcheck_inference.py --input_path sample.mp4
# D:\envs\gpu_env\python.exe factcheck_inference.py --input_path sample.wav --media_type audio --output_json factcheck_report.json
