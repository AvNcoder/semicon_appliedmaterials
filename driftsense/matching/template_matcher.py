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


def _find_top_peaks(corr: np.ndarray, w: int, h: int, max_peaks: int, min_score_ratio: float) -> list[tuple[float, float, float]]:
    """
    Extract up to `max_peaks` local maxima from one correlation surface,
    instead of just the single global max cv2.minMaxLoc gives you.

    Why this exists: DRAM/FinFET search images are highly periodic by
    construction (per the hackathon spec). At a given (scale, angle), the
    single highest-scoring pixel is not guaranteed to be the true embedded
    patch -- it can land on a visually near-identical neighboring unit cell,
    with the correct location scoring a close second. Taking only
    cv2.minMaxLoc's one result silently discards that correct peak before
    tiebreak.py ever gets a chance to disambiguate -- no downstream logic
    can recover a candidate that was never added to the list.

    The first peak is always kept (matches the old always-one-candidate-
    per-scale/angle behavior exactly, so nothing downstream regresses).
    Additional peaks are only kept if they're credible: positive
    correlation, and not too far below the best score at this same
    scale/angle (min_score_ratio) -- otherwise they're just noise, not a
    genuine second instance.

    After accepting a peak, a window around it (sized off the template
    footprint, not a fixed pixel count) is suppressed so the next
    iteration finds a spatially distinct instance rather than a
    neighboring pixel of the same blob.

    Returns (score, x, y) tuples, highest score first, with x/y already
    center-corrected (top-left -> center, same fix as before).
    """
    work = corr.copy()
    peaks: list[tuple[float, float, float]] = []

    # Suppression window tied to template size: big enough to clear the
    # current match's own correlation blob, without a fixed pixel count
    # that would be wrong for a differently-scaled template. If you know
    # the true cell pitch (config.py / the generator's CELL_PITCH), tune
    # this against that instead -- this is a generic fallback.
    sup_h = max(3, h // 2)
    sup_w = max(3, w // 2)

    top_score = None
    for i in range(max(1, max_peaks)):
        _, score, _, top_left = cv2.minMaxLoc(work)

        if i == 0:
            top_score = score
        elif top_score <= 0 or score < top_score * min_score_ratio:
            break  # remaining peaks are too weak to be a credible 2nd instance

        px, py = top_left
        peaks.append((score, px + w / 2.0, py + h / 2.0))

        y0 = max(0, py - sup_h // 2)
        y1 = min(work.shape[0], py + sup_h // 2 + 1)
        x0 = max(0, px - sup_w // 2)
        x1 = min(work.shape[1], px + sup_w // 2 + 1)
        work[y0:y1, x0:x1] = -np.inf

    return peaks


def multi_scale_match(ref: np.ndarray, search: np.ndarray, scales, angles=None,
                       max_peaks_per_map: int = 5, min_score_ratio: float = 0.85) -> list[dict]:
    """
    Slide `ref` (resized to each scale, then rotated to each angle) over
    `search` using normalized cross-correlation and return candidates for
    each (scale, angle) combination -- at least one, up to
    `max_peaks_per_map` if the correlation surface has credible secondary
    peaks. (See _find_top_peaks: this is the fix for periodic layouts
    silently eating the correct match when it isn't the single
    global-max pixel at that scale/angle.)

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

            for score, cx, cy in _find_top_peaks(corr, w, h, max_peaks_per_map, min_score_ratio):
                candidates.append({
                    "x": cx,
                    "y": cy,
                    "scale": s,
                    "angle": angle,
                    "score": float(score),
                    "template_w": w,
                    "template_h": h,
                })
    return candidates


def match_with_timing(ref: np.ndarray, search: np.ndarray, scales, angles=None,
                       max_peaks_per_map: int = 5, min_score_ratio: float = 0.85) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    candidates = multi_scale_match(ref, search, scales, angles, max_peaks_per_map, min_score_ratio)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return candidates, elapsed_ms
