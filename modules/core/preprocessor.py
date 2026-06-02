import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Standard ImageNet normalisation used by all three models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ── Image preprocessing ───────────────────────────────────────
def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Load an image from disk, convert to RGB, apply transforms.
    Returns a (1, 3, H, W) tensor ready for model inference.
    """
    img = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size)
    tensor = transform(img).unsqueeze(0)  # add batch dim
    return tensor


# ── Video preprocessing ───────────────────────────────────────
def extract_frames(
    video_path: str, num_frames: int = 16, image_size: int = 224
) -> list[torch.Tensor]:
    """
    Uniformly sample num_frames from a video.
    Returns a list of (1, 3, H, W) tensors — one per sampled frame.
    We process each frame independently through the image models.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    transform = get_transform(image_size)
    tensors = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV reads BGR → convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensors.append(transform(pil_img).unsqueeze(0))

    cap.release()

    if len(tensors) == 0:
        raise ValueError("No frames could be extracted from video.")

    return tensors


# ── Face detection (optional but improves accuracy) ───────────
def detect_and_crop_face(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Try to detect the largest face in the image and crop to it.
    Falls back to the full image if no face is found.
    Uses OpenCV Haar cascade — no extra dependencies needed.
    """
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) > 0:
        # Pick the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add 20% padding around the face
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        cropped = img_bgr[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.open(image_path).convert("RGB")

    transform = get_transform(image_size)
    return transform(pil_img).unsqueeze(0)
