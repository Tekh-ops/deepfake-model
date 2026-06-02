# app.py

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename

# Import your modules
from modules.metadata_analysis.metadata import create_metadata
from modules.core.legal import generate_legal_notice
from ml.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    UPLOAD_DIR,
    setup_logging,
)
from ml.inference import predict, predict_video
from ml.model_loader import get_model

from flask_cors import CORS


# Create Flask app
app = Flask(__name__)
CORS(app)
setup_logging()
LOGGER = logging.getLogger(__name__)

# Folder to temporarily store uploaded files
UPLOAD_FOLDER = str(UPLOAD_DIR)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024


def _get_file_size(upload) -> int:
    upload.seek(0, os.SEEK_END)
    size = upload.tell()
    upload.seek(0)
    return size


def _validate_extension(filename: str, allowed_set) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in allowed_set


def _save_upload_to_temp(upload) -> str:
    safe_name = secure_filename(upload.filename)
    suffix = Path(safe_name).suffix
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_FOLDER)
    upload.save(tmp_file.name)
    tmp_file.close()
    return tmp_file.name


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main API endpoint:
    - Receives file
    - Runs prediction
    - Generates metadata
    - Generates legal notice (if needed)
    - Returns structured response
    """

    # Check if file is present in request
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _validate_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)):
        return jsonify({"error": "Unsupported file type"}), 400

    file_size = _get_file_size(file)
    if file_size > MAX_VIDEO_SIZE_BYTES:
        return jsonify({"error": f"File too large. Maximum size: {MAX_VIDEO_SIZE_MB}MB"}), 413
    if file_size == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        file_path = _save_upload_to_temp(file)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        ext = Path(file.filename).suffix.lower()
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            with Image.open(file_path) as image:
                pred = predict(image.convert("RGB"))
            result = pred["prediction"].capitalize()
            confidence = pred["confidence"]
        else:
            pred = predict_video(file_path)
            result = pred["final_prediction"].capitalize()
            confidence = pred["confidence"]

        # --- Step 2: Metadata ---
        metadata = create_metadata(file_path, result, confidence)

        # --- Step 3: Legal Notice ---
        notice = generate_legal_notice(metadata)

        # Add legal notice to response
        metadata["legal_notice"] = notice if notice else ""

        # Return final JSON response
        return jsonify(metadata)
    
    except Exception as e:
        LOGGER.exception("Processing failed for /analyze: %s", e)
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    finally:
        # Clean up uploaded file after processing
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass


@app.route("/predict/image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not _validate_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Unsupported image type"}), 400

    file_size = _get_file_size(file)
    if file_size == 0:
        return jsonify({"error": "Empty file"}), 400
    if file_size > MAX_IMAGE_SIZE_BYTES:
        return jsonify({"error": f"Image too large. Maximum size: {MAX_IMAGE_SIZE_MB}MB"}), 413

    file_path = None
    try:
        file_path = _save_upload_to_temp(file)
        
        print("🔥 Request received")
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            print("✅ Image loaded successfully")

            print("➡️ Running prediction...")
            result = predict(image)

            print("✅ Prediction completed:", result)

        return jsonify(result), 200
    except Exception as e:
        LOGGER.exception("Image prediction error: %s", e)
        return jsonify({"error": f"Image prediction failed: {str(e)}"}), 500
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.route("/predict/video", methods=["POST"])
def predict_video_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not _validate_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported video type"}), 400

    file_size = _get_file_size(file)
    if file_size == 0:
        return jsonify({"error": "Empty file"}), 400
    if file_size > MAX_VIDEO_SIZE_BYTES:
        return jsonify({"error": f"Video too large. Maximum size: {MAX_VIDEO_SIZE_MB}MB"}), 413

    file_path = None
    try:
        file_path = _save_upload_to_temp(file)
        result = predict_video(file_path)
        return jsonify(result), 200
    except Exception as e:
        LOGGER.exception("Video prediction error: %s", e)
        return jsonify({"error": f"Video prediction failed: {str(e)}"}), 500
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# Run the app
if __name__ == "__main__":
    # Warm up singleton model once at startup.
    get_model()
    app.run(debug=False, port=5000)