"""Util functions."""
import base64
import cv2
import numpy as np


cv2.setNumThreads(1)


def encode_img_base64(image: np.ndarray) -> str:
    """Decode byte image with base64 to np.ndarray."""
    return base64.b64encode(image.tobytes()).decode("utf8")


def decode_img(img: str) -> np.ndarray:
    """Decode byte image with base64 to np.ndarray."""
    img = np.frombuffer(base64.urlsafe_b64decode(img), dtype=np.uint8)
    img_dec = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    is_encoded = img_dec is not None
    return img_dec if is_encoded else img
