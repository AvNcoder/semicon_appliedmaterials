from driftsense import config
from driftsense.matching.template_matcher import match_with_timing
from driftsense.matching.tiebreak import suppress_duplicates, select_best
from driftsense.preprocessing import maybe_preprocess

# Kept as a module-level alias for backward compatibility with earlier code
# that imports DEFAULT_SCALES directly from this module.
DEFAULT_SCALES = config.DEFAULT_SCALES

def localize(ref, search, scales=None, angles=None, nms_radius_px: float = None,
             tie_tolerance: float = None, min_score: float = None,
             apply_preprocessing: bool = True) -> dict:
    """
    Single public entrypoint: (ref, search) -> predicted center + metadata.
    scales/angles default to config.DEFAULT_SCALES (+-20%) and
    config.DEFAULT_ANGLES (+-3 deg) -- widened to cover the rotation/scale
    jitter the generators now apply to the embedded patch.
    """
    scales = config.DEFAULT_SCALES if scales is None else scales
    angles = config.DEFAULT_ANGLES if angles is None else angles
    nms_radius_px = config.NMS_RADIUS_PX if nms_radius_px is None else nms_radius_px
    tie_tolerance = config.TIE_SCORE_TOLERANCE if tie_tolerance is None else tie_tolerance
    min_score = config.MIN_SCORE if min_score is None else min_score

    if apply_preprocessing:
        ref = maybe_preprocess(ref)
        search = maybe_preprocess(search)

    candidates, elapsed_ms = match_with_timing(ref, search, scales, angles)
    if not candidates:
        raise RuntimeError("no valid scale produced a template smaller than the search image")

    deduped = suppress_duplicates(candidates, radius_px=nms_radius_px)
    h, w = search.shape[:2]
    best = select_best(deduped, image_center=(w / 2.0, h / 2.0), score_tolerance=tie_tolerance)

    return {
        "x": best["x"],
        "y": best["y"],
        "scale": best["scale"],
        "angle": best["angle"],
        "score": best["score"],
        "matched": best["score"] >= min_score,
        "num_candidates": len(deduped),
        "time_ms": elapsed_ms,
    }