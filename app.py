"""
app.py
──────
BharatShield Deepfake Detection API
Built on top of the initial prototype by extending dummy_predict
with the real ML pipeline and adding hashing, audit, and legal layers.

Endpoints:
  POST /analyze          — main endpoint (matches friend's original)
  POST /analyze/batch    — multiple files in one request
  GET  /health           — liveness check
  POST /gradcam          — heatmap overlay for an image
  GET  /audit/verify     — verify HMAC chain integrity
  GET  /audit/log        — return last N audit entries
"""

import os
import time
import uuid
import json
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Config ────────────────────────────────────────────────────
from config import (
    UPLOAD_FOLDER,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    SYSTEM_NAME,
    SYSTEM_VERSION,
)

# ── Utils ─────────────────────────────────────────────────────
from utils.metadata import create_metadata
from utils.legal    import generate_legal_notice
from utils.hashing  import (
    compute_sha256,
    append_audit_entry,
    verify_audit_chain,
    generate_submission_id,
)

# ── ML pipeline ───────────────────────────────────────────────
from models.ensemble import run_detection

# ── Flask setup ───────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_upload(file) -> str:
    """Save uploaded file with collision-safe name. Returns abs path."""
    original  = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    filename  = f"{unique_id}_{original}"
    path      = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path


def _validate_file(file) -> tuple[bool, str, int]:
    """
    Validate the uploaded file object.
    Returns (ok, error_message, http_status).
    """
    if not file or file.filename == "":
        return False, "Empty filename.", 400

    if not allowed_file(file.filename):
        return False, (
            f"File type not allowed. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        ), 415

    # Check size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size == 0:
        return False, "Empty file.", 400

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size > max_bytes:
        return False, (
            f"File too large. Max: {MAX_FILE_SIZE_MB}MB, "
            f"received: {size / 1024 / 1024:.1f}MB"
        ), 413

    return True, "", 200


def _process_file(file_path: str) -> dict:
    """
    Full processing pipeline for a single file.
    Called by both /analyze and /analyze/batch.

    Returns the complete response dict.
    """
    t_start = time.time()

    # ── Step 1: Generate submission ID + SHA-256 hash ─────────
    submission_id = generate_submission_id()
    file_hash     = compute_sha256(file_path)

    # ── Step 2: Audit log — RECEIVED ─────────────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "MEDIA_RECEIVED",
        actor         = "api_gateway",
        extra         = {"file": os.path.basename(file_path)},
    )

    # ── Step 3: ML detection ──────────────────────────────────
    detection = run_detection(file_path, use_face_crop=True)

    result     = detection["label"]        # "Fake" | "Real"
    confidence = detection["confidence"]
    detail     = detection.get("per_model", {})

    # ── Step 4: Audit log — DETECTION COMPLETE ────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "DETECTION_COMPLETE",
        actor         = "ml_pipeline",
        result        = {
            "label":      result,
            "confidence": confidence,
            "fake_prob":  detection.get("fake_prob"),
        },
    )

    # ── Step 5: Build metadata ────────────────────────────────
    processing_ms = (time.time() - t_start) * 1000
    metadata      = create_metadata(
        file_path         = file_path,
        result            = result,
        confidence        = confidence,
        submission_id     = submission_id,
        file_hash         = file_hash,
        detection_detail  = detail,
        processing_time_ms = processing_ms,
    )

    # ── Step 6: Legal notice ──────────────────────────────────
    notice = generate_legal_notice(metadata)
    metadata["legal_notice"] = notice if notice else {}

    # ── Step 7: Audit log — REPORT GENERATED ─────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "REPORT_GENERATED",
        actor         = "compliance_layer",
        extra         = {
            "risk_level":     metadata["detection"]["risk_level"],
            "takedown_advised": notice.get("takedown_advised", False) if notice else False,
        },
    )

    return metadata


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — same path as friend's prototype."""
    return jsonify({
        "status":  "ok",
        "service": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint — direct replacement for the dummy_predict version.
    Accepts multipart/form-data with field 'file'.
    Returns the full structured JSON response.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    ok, err_msg, status = _validate_file(file)
    if not ok:
        return jsonify({"error": err_msg}), status

    file_path = save_upload(file)

    try:
        response = _process_file(file_path)
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error":   "Processing failed",
            "details": str(e),
        }), 500

    finally:
        # Clean up — same as friend's original pattern
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """
    Analyze multiple files in one request.
    Accepts multipart/form-data with multiple 'files[]' fields.
    Returns a list of results in the same order as submitted files.
    """
    files = request.files.getlist("files[]")

    if not files:
        return jsonify({"error": "No files provided. Use field name 'files[]'."}), 400

    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files per batch request."}), 400

    results      = []
    saved_paths  = []

    # Save all files first
    for file in files:
        ok, err_msg, status = _validate_file(file)
        if not ok:
            results.append({"error": err_msg, "filename": file.filename})
            saved_paths.append(None)
            continue
        saved_paths.append(save_upload(file))
        results.append(None)  # placeholder

    # Process each saved file
    for i, path in enumerate(saved_paths):
        if path is None:
            continue   # already has an error entry
        try:
            results[i] = _process_file(path)
        except Exception as e:
            traceback.print_exc()
            results[i] = {
                "error":    "Processing failed",
                "details":  str(e),
                "filename": files[i].filename,
            }
        finally:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    return jsonify({
        "batch_count": len(results),
        "results":     results,
    }), 200


@app.route("/gradcam", methods=["POST"])
def gradcam():
    """
    Generate a Grad-CAM heatmap overlay for an image.
    Returns the overlay PNG as a file download.
    Accepts multipart/form-data with field 'file'.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    ok, err_msg, status = _validate_file(file)
    if not ok:
        return jsonify({"error": err_msg}), status

    # Grad-CAM only supports images
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return jsonify({
            "error": "Grad-CAM only supports image files (jpg, png, webp)."
        }), 415

    file_path    = save_upload(file)
    heatmap_path = f"/tmp/gradcam_{uuid.uuid4().hex[:8]}.png"

    try:
        from utils.gradcam import generate_gradcam_secondary
        generate_gradcam_secondary(file_path, save_path=heatmap_path)
        return send_file(heatmap_path, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Grad-CAM failed: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.route("/audit/verify", methods=["GET"])
def audit_verify():
    """
    Verify the HMAC chain integrity of the audit log.
    Returns {"intact": true, "message": "..."}.
    """
    intact, message = verify_audit_chain()
    return jsonify({
        "intact":  intact,
        "message": message,
    }), 200 if intact else 409


@app.route("/audit/log", methods=["GET"])
def audit_log():
    """
    Return the last N audit log entries (default 20).
    Query param: ?limit=50
    """
    from config import AUDIT_LOG_FILE

    limit = min(int(request.args.get("limit", 20)), 200)

    if not os.path.exists(AUDIT_LOG_FILE):
        return jsonify({"entries": [], "total": 0})

    entries = []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return jsonify({
        "total":   len(entries),
        "showing": min(limit, len(entries)),
        "entries": entries[-limit:],  # most recent first
    })


# ─────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"  {SYSTEM_NAME}")
    print(f"  Version: {SYSTEM_VERSION}")
    print("  ML models load on first request.")
    print("  Endpoints:")
    print("    POST /analyze          — single file detection")
    print("    POST /analyze/batch    — batch detection (up to 10)")
    print("    POST /gradcam          — heatmap overlay")
    print("    GET  /health           — liveness check")
    print("    GET  /audit/verify     — chain integrity check")
    print("    GET  /audit/log        — view audit entries")
    print("=" * 60)
    app.run(debug=False, port=5001)