import time
import cv2
import numpy as np

from driftsense import config


def _rotate_template(template: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a template in place (same dimensions) around its own center.
    BORDER_REPLICATE avoids introducing black corner pixels that would
    otherwise corrupt the normalized cross-correlation score."""
    if angle_deg == 0.0:
        return template
    h, w = template.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def multi_scale_match(ref: np.ndarray, search: np.ndarray, scales, angles=None) -> list[dict]:
    """
    Slide `ref` (resized to each scale, then rotated to each angle) over
    `search` using normalized cross-correlation and return one candidate
    per (scale, angle) combination.

    Each candidate's (x, y) is the CENTER of the matched patch in `search`
    coordinates -- NOT the raw minMaxLoc top-left corner. cv2.matchTemplate
    always returns the top-left of the best window; forgetting the +w/2,+h/2
    offset produces a fixed error of exactly template_size/2 px on every
    prediction (confirmed empirically: 70.7px off at embed_size=100).

    angles defaults to config.DEFAULT_ANGLES (+-3 deg) -- added because the
    generators now apply rotation/scale jitter to the embedded patch per
    the hackathon spec ("Rotation 1-3 degrees to the polygons"), so a
    rotation-blind matcher would silently miss jittered samples.
    """
    angles = config.DEFAULT_ANGLES if angles is None else angles
    candidates = []
    for s in scales:
        w = max(1, int(round(ref.shape[1] * s)))
        h = max(1, int(round(ref.shape[0] * s)))
        if w >= search.shape[1] or h >= search.shape[0]:
            continue

        resized_ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)

        for angle in angles:
            template = _rotate_template(resized_ref, angle)
            corr = cv2.matchTemplate(search.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(corr)
            px, py = top_left

            candidates.append({
                "x": px + w / 2.0,   # top-left -> center fix, unchanged from before
                "y": py + h / 2.0,
                "scale": s,
                "angle": angle,
                "score": float(score),
                "template_w": w,
                "template_h": h,
            })
    return candidates


def match_with_timing(ref: np.ndarray, search: np.ndarray, scales, angles=None) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    candidates = multi_scale_match(ref, search, scales, angles)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return candidates, elapsed_ms