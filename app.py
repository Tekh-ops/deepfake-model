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
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    INFERENCE_TIMEOUT_S,
)

# ── Utils ─────────────────────────────────────────────────────
from modules.metadata_analysis.metadata import create_metadata
from modules.core.legal    import generate_legal_notice
from modules.metadata_analysis.hashing  import (
    compute_sha256,
    append_audit_entry,
    verify_audit_chain,
    generate_submission_id,
)

# ── ML pipeline ───────────────────────────────────────────────
from modules.core.ensemble import run_detection

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


def _get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "audio"


def _process_file(file_path: str) -> dict:
    """
    Full processing pipeline for a single file.
    Called by both /analyze and /analyze/batch.
    Returns the complete structured response dict.
    """
    t_start = time.time()

    # ── Step 1: Generate submission ID + SHA-256 hash ─────────
    submission_id = generate_submission_id()
    file_hash     = compute_sha256(file_path)
    media_type    = _get_media_type(file_path)

    # ── Step 2: Audit log — RECEIVED ─────────────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "MEDIA_RECEIVED",
        actor         = "api_gateway",
        extra         = {"file": os.path.basename(file_path),
                         "media_type": media_type},
    )

    # ── Step 3: ML detection ──────────────────────────────────
    detection = run_detection(file_path, use_face_crop=True)

    result     = detection["label"]         # "Fake" | "Real"
    confidence = detection["confidence"]
    fake_prob  = detection["fake_prob"]
    risk_level = detection.get("risk_level", "MEDIUM")

    if media_type == "image" and result == "Fake":
        try:
            from modules.image_analysis.gradcam import generate_gradcam_secondary
            gradcam_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{submission_id}.png")
            generate_gradcam_secondary(file_path, save_path=gradcam_path)
        except Exception as e:
            print("GradCAM auto-generation failed:", e)


    # ── Step 4: Audit log — DETECTION COMPLETE ────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "DETECTION_COMPLETE",
        actor         = "ml_pipeline",
        result        = {
            "label":      result,
            "confidence": confidence,
            "fake_prob":  fake_prob,
            "risk_level": risk_level,
        },
    )

    # ── Step 5: Fact-check pipeline (video/audio only) ────────
    fact_check_result = {"available": False,
                          "note": "Fact-check not run (image input)"}
    if media_type in ("video", "audio"):
        try:
            from modules.factcheck.pipeline import run_factcheck_pipeline
            fact_check_result = run_factcheck_pipeline(
                media_path = file_path,
                media_type = media_type,
            )
        except Exception as e:
            traceback.print_exc()
            fact_check_result = {
                "available": False,
                "note":      f"Fact-check pipeline error: {str(e)}",
            }

    # ── Step 6: Build metadata ────────────────────────────────
    processing_ms = (time.time() - t_start) * 1000
    file_size     = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    def _size_human(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024: return f"{b:.1f} {u}"
            b /= 1024
        return f"{b:.1f} TB"

    response = {
        "submission_id":  submission_id,
        "timestamp_utc":  __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),

        "file": {
            "name":       os.path.basename(file_path),
            "sha256":     file_hash,
            "media_type": media_type,
            "size_human": _size_human(file_size),
            "size_bytes": file_size,
        },

        "deepfake_detection": {
            "label":          result,
            "confidence":     round(confidence, 4),
            "risk_level":     risk_level,
            "is_deepfake":    result == "Fake",
            "fake_probability": round(fake_prob, 4),
            "streams":        detection.get("streams", {}),
            "metadata_flags": [],
            "gradcam_url":    f"/gradcam/{submission_id}"
                              if media_type == "image" else None,
            "processing_ms":  round(processing_ms, 1),
        },

        "fact_check": fact_check_result,

        "integrity": {
            "sha256":      file_hash,
            "audit_entry": f"BSA S.63 chain entry created (ID: {submission_id})",
        },
    }

    # ── Step 7: Legal report generation ───────────────────────
    try:
        import sys
        # Add the root directory to path to allow absolute imports
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
            
        from bharatshield_legal.legal2 import MediaEvidence, generate_all_documents
        
        evidence = MediaEvidence()
        evidence.case_id = submission_id
        evidence.media_filename = os.path.basename(file_path)
        evidence.media_sha256 = file_hash
        evidence.fusion_score = fake_prob
        evidence.verdict = "LIKELY SYNTHETIC" if fake_prob >= 0.50 else "LIKELY AUTHENTIC"
        evidence.model_version = "BharatShield-DetectCore v3.0 (Ensemble CBAM)"
        evidence.inference_duration = f"{processing_ms:.1f} ms"
        
        streams = detection.get("streams", {})
        evidence.cnn_score = streams.get("A_spatial", fake_prob)
        evidence.temporal_score = streams.get("C_temporal", fake_prob)
        evidence.audio_score = streams.get("D_audio", fake_prob)

        pdf_path = os.path.join(UPLOAD_FOLDER, f"BharatShield_Legal_{submission_id}.pdf")
        generate_all_documents(evidence=evidence, output_path=pdf_path)
        response["legal_report_url"] = f"/legal_report/{submission_id}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        response["legal_report_url"] = None
        response["legal_error"] = str(e)

    # ── Step 8: Audit log — REPORT GENERATED ─────────────────
    append_audit_entry(
        submission_id = submission_id,
        file_hash     = file_hash,
        action        = "REPORT_GENERATED",
        actor         = "compliance_layer",
        extra         = {
            "risk_level":       risk_level,
            "takedown_advised": risk_level == "HIGH",
            "fact_check_available": fact_check_result.get("available", False),
        },
    )

    return response


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
        from modules.image_analysis.gradcam import generate_gradcam_secondary
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

@app.route("/legal_report/<submission_id>", methods=["GET"])
def get_legal_report(submission_id):
    """Download the generated PDF legal report."""
    from flask import send_file
    pdf_path = os.path.join(UPLOAD_FOLDER, f"BharatShield_Legal_{submission_id}.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=True, download_name=f"BharatShield_Legal_{submission_id}.pdf")
    return jsonify({"error": "Report not found or not generated"}), 404

@app.route("/gradcam/<submission_id>", methods=["GET"])
def get_gradcam(submission_id):
    """Get the auto-generated GradCAM heatmap for an image."""
    from flask import send_file
    path = os.path.join(UPLOAD_FOLDER, f"gradcam_{submission_id}.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "GradCAM not found or not generated for this submission"}), 404

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