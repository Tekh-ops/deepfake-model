import logging
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

LOGGER = logging.getLogger(__name__)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def to_pil_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb).convert("RGB")
        raise ValueError("Unsupported numpy image shape.")

    raise TypeError("Unsupported image type. Expected PIL.Image or OpenCV numpy array.")


def detect_and_crop_face(image: Image.Image) -> Image.Image:
    # 1. Convert PIL image to OpenCV format
    # Handle grayscale by ensuring it's RGB first
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(30, 30)
    )

    # 4. Logic & Logging
    if len(faces) == 0:
        LOGGER.warning("No face detected, using original image")
        return image

    if len(faces) > 1:
        LOGGER.info("Multiple faces detected, using largest")
        # Select the LARGEST face (based on area)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
    else:
        LOGGER.info("Face detected and cropped")
        x, y, w, h = faces[0]

    # 5. Crop with small padding (15% around face, staying inside bounds)
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)

    img_h, img_w = img_array.shape[:2]

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    cropped_bgr = img_bgr[y1:y2, x1:x2]

    if cropped_bgr.size == 0:
        LOGGER.warning("Invalid crop, using original image")
        return image

    # 6. Convert cropped image back to PIL and return
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def preprocess_image(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    pil_image = to_pil_image(image)
    pil_image = detect_and_crop_face(pil_image)
    tensor = _INFER_TRANSFORM(pil_image).unsqueeze(0)
    return tensor

