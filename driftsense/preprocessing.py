"""
Optional illumination normalization. OFF by default (see config.USE_CLAHE) --
the validated 100%/0px result did not use this. Provided as a knob for
harder/future styles, not applied silently.
"""

import cv2
import numpy as np

from driftsense import config


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to an 8-bit-range float image. Returns float32."""
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    return clahe.apply(img_u8).astype(np.float32)


def maybe_preprocess(img: np.ndarray) -> np.ndarray:
    """Applies CLAHE only if config.USE_CLAHE is True; otherwise passes through unchanged."""
    if config.USE_CLAHE:
        return normalize_illumination(img)
    return img