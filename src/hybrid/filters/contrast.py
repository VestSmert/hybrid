# --- file: hybrid/filters/contrast.py ---
"""
Local contrast enhancement utilities.
"""
from __future__ import annotations
import numpy as np

def clahe_u8(img: np.ndarray, clip_limit: float = 2.0, tile: int = 16) -> np.ndarray:
    """
    CLAHE on a single 2D image (expects float [0,1] or uint8).
    Returns uint8.
    """
    import cv2
    im = img
    if im.dtype != np.uint8:
        im = np.clip(im.astype(np.float32), 0, 1)
        im = (im * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(im)
