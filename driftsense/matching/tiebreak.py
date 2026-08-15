import math


def suppress_duplicates(candidates: list[dict], radius_px: float = 20.0) -> list[dict]:
    """Merge candidates that land on (roughly) the same physical location
    across adjacent scales, keeping the highest-scoring one per cluster."""
    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept: list[dict] = []
    for cand in ordered:
        if all(math.hypot(cand["x"] - k["x"], cand["y"] - k["y"]) > radius_px for k in kept):
            kept.append(cand)
    return kept


def select_best(candidates: list[dict], image_center=(500.0, 500.0), score_tolerance: float = 0.03) -> dict:
    """
    Score-first ranking. Distance-to-center is ONLY the tie-break among
    candidates within `score_tolerance` of the best score -- not the primary
    sort key. Using distance-to-center as the primary key silently prefers a
    central-but-wrong peak over an off-center-but-correct one whenever the
    true embed location isn't near (500,500), which is most samples given
    center_x/y are sampled uniformly over [150, 850].
    """
    if not candidates:
        raise ValueError("no candidates to select from")

    best_score = max(c["score"] for c in candidates)
    near_top = [c for c in candidates if c["score"] >= best_score - score_tolerance]

    if len(near_top) == 1:
        return near_top[0]

    cx, cy = image_center
    return min(near_top, key=lambda c: math.hypot(c["x"] - cx, c["y"] - cy))