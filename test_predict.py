"""
test_predict.py
───────────────
Full test suite for the BharatShield ML API.

Usage:
    # Start the server first:
    python app.py

    # Then in another terminal:
    python test_predict.py                          # run health + audit tests only
    python test_predict.py data/fake_face.jpg       # full test on an image
    python test_predict.py data/fake_face.jpg --batch  # also tests batch endpoint
"""

import argparse
import json
import os
import sys
import time

import requests

BASE_URL = "http://localhost:5001"


# ── Helpers ───────────────────────────────────────────────────


def _sep(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)


def _ok(label: str, extra: str = ""):
    print(f"  ✓  {label}" + (f"  →  {extra}" if extra else ""))


def _fail(label: str, reason: str = ""):
    print(f"  ✗  {label}" + (f"  →  {reason}" if reason else ""))
    sys.exit(1)


# ── Tests ─────────────────────────────────────────────────────


def test_health():
    _sep("GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok"
    _ok("health check", data.get("service", ""))


def test_analyze(image_path: str):
    _sep("POST /analyze")
    assert os.path.exists(image_path), f"File not found: {image_path}"

    with open(image_path, "rb") as f:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/analyze",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=120,
        )
        elapsed = time.time() - t0

    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()

    # ── Validate response shape ───────────────────────────────
    assert "detection" in data, "Missing 'detection' key"
    assert "label" in data["detection"], "Missing detection.label"
    assert "confidence" in data["detection"]
    assert "risk_level" in data["detection"]
    assert "file" in data
    assert "sha256" in data["file"]
    assert "legal_notice" in data
    assert "submission_id" in data

    det = data["detection"]
    _ok("response shape valid")
    _ok(f"label", det["label"])
    _ok(f"confidence", f"{det['confidence']:.4f}")
    _ok(f"fake_prob", f"{det['fake_probability']:.4f}")
    _ok(f"risk_level", det["risk_level"])
    _ok(f"sha256", data["file"]["sha256"][:16] + "...")
    _ok(f"submission_id", data["submission_id"])
    _ok(f"inference time", f"{elapsed:.2f}s wall-clock")

    # ── Model breakdown ───────────────────────────────────────
    if data.get("model_breakdown"):
        print("\n  Per-model breakdown:")
        for model_name, info in data["model_breakdown"].items():
            print(
                f"    {model_name:40s}  "
                f"label={info.get('label', '?'):5s}  "
                f"fake_prob={info.get('fake_prob', '?'):.4f}"
            )

    # ── Legal notice ──────────────────────────────────────────
    notice = data.get("legal_notice", {})
    if notice:
        print(f"\n  Legal notice:")
        print(f"    Title:    {notice.get('notice_title', '')}")
        print(f"    Escalation: {notice.get('escalation_level', '')}")
        print(f"    Takedown:   {notice.get('takedown_advised', '')}")

    print(f"\n  Full response:\n{json.dumps(data, indent=4)}")
    return data


def test_batch(image_path: str):
    _sep("POST /analyze/batch")

    with open(image_path, "rb") as f1, open(image_path, "rb") as f2:
        r = requests.post(
            f"{BASE_URL}/analyze/batch",
            files=[
                ("files[]", (os.path.basename(image_path), f1, "image/jpeg")),
                ("files[]", (os.path.basename(image_path), f2, "image/jpeg")),
            ],
            timeout=180,
        )

    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()

    assert "results" in data
    assert "batch_count" in data
    assert data["batch_count"] == 2

    for i, res in enumerate(data["results"]):
        label = res.get("detection", {}).get("label", "ERROR")
        conf = res.get("detection", {}).get("confidence", 0)
        _ok(f"File {i + 1}", f"{label} ({conf:.4f})")


def test_gradcam(image_path: str):
    _sep("POST /gradcam")

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/gradcam",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=120,
        )

    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    assert r.headers.get("Content-Type", "").startswith("image/png")

    out = "/tmp/test_gradcam_output.png"
    with open(out, "wb") as f:
        f.write(r.content)
    _ok("heatmap received", f"saved to {out}  ({len(r.content)} bytes)")


def test_audit_verify():
    _sep("GET /audit/verify")
    r = requests.get(f"{BASE_URL}/audit/verify", timeout=10)
    data = r.json()
    assert data.get("intact") is True, f"Chain broken: {data.get('message')}"
    _ok("HMAC chain intact", data.get("message", ""))


def test_audit_log():
    _sep("GET /audit/log?limit=5")
    r = requests.get(f"{BASE_URL}/audit/log?limit=5", timeout=10)
    assert r.status_code == 200
    data = r.json()
    _ok(f"total entries in log", str(data.get("total", 0)))
    for e in data.get("entries", []):
        print(
            f"    [{e.get('timestamp_utc', '')}] "
            f"{e.get('action', '?'):30s} "
            f"sub={e.get('submission_id', '?')}"
        )


def test_error_cases():
    _sep("Error handling")

    # No file
    r = requests.post(f"{BASE_URL}/analyze", timeout=10)
    assert r.status_code == 400
    _ok("no file → 400", r.json().get("error", ""))

    # Bad endpoint
    r = requests.get(f"{BASE_URL}/nonexistent", timeout=10)
    assert r.status_code == 404
    _ok("bad endpoint → 404")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BharatShield API test suite")
    parser.add_argument("image_path", nargs="?", help="Path to a test image/video")
    parser.add_argument("--batch", action="store_true", help="Also run batch test")
    parser.add_argument("--gradcam", action="store_true", help="Also run gradcam test")
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  BharatShield API Test Suite")
    print("═" * 50)

    test_health()
    test_error_cases()
    test_audit_verify()
    test_audit_log()

    if args.image_path:
        test_analyze(args.image_path)

        if args.batch:
            test_batch(args.image_path)

        if args.gradcam:
            test_gradcam(args.image_path)

    else:
        print("\n  ℹ  Pass an image path to run the full detection test:")
        print("     python test_predict.py data/fake_face.jpg --batch --gradcam")

    print("\n" + "═" * 50)
    print("  ✓  All tests passed.")
    print("═" * 50 + "\n")
