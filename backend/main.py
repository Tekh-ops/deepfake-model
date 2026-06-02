import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

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
from modules.core.legal import generate_legal_notice
from modules.metadata_analysis.metadata import create_metadata


setup_logging()
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="BharatShield Backend", version="1.0.0")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024


def _validate_extension(filename: str, allowed_set) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed_set:
        raise HTTPException(status_code=400, detail="Unsupported file type")


async def _read_upload_with_limit(upload: UploadFile, max_size_bytes: int) -> bytes:
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > max_size_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {max_size_bytes // 1024 // 1024}MB")
    return data


def _write_temp_file(content: bytes, suffix: str) -> str:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(UPLOAD_DIR))
    tmp_file.write(content)
    tmp_file.flush()
    tmp_file.close()
    return tmp_file.name


@app.on_event("startup")
def startup_event() -> None:
    get_model()


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    _validate_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS)

    content = await _read_upload_with_limit(file, MAX_IMAGE_SIZE_BYTES)
    tmp_path = None
    try:
        tmp_path = _write_temp_file(content, Path(file.filename).suffix)
        with Image.open(tmp_path) as image:
            result = predict(image.convert("RGB"))
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Image prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    _validate_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS)

    content = await _read_upload_with_limit(file, MAX_VIDEO_SIZE_BYTES)
    tmp_path = None
    try:
        tmp_path = _write_temp_file(content, Path(file.filename).suffix)
        result = predict_video(tmp_path)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Video prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    ext = Path(file.filename).suffix.lower()
    allowed = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)
    _validate_extension(file.filename, allowed)

    size_limit = MAX_IMAGE_SIZE_BYTES if ext in ALLOWED_IMAGE_EXTENSIONS else MAX_VIDEO_SIZE_BYTES
    content = await _read_upload_with_limit(file, size_limit)
    tmp_path = None
    try:
        tmp_path = _write_temp_file(content, ext)
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            with Image.open(tmp_path) as image:
                pred = predict(image.convert("RGB"))
            result = pred["label"].capitalize()
            confidence = pred["confidence"]
        else:
            pred = predict_video(tmp_path)
            result = pred["final_prediction"].capitalize()
            confidence = pred["confidence"]

        metadata = create_metadata(tmp_path, result, confidence)
        notice = generate_legal_notice(metadata)
        metadata["legal_notice"] = notice if notice else ""
        return JSONResponse(content=metadata)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Analyze endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

