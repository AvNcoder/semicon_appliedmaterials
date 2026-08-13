"""
Metrics for the localization evaluation loop. Pure functions, no I/O --
kept separate from evaluate.py so they're independently testable.
"""

import math

from driftsense import config


def pixel_error(pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> float:
    """Euclidean distance between predicted and true center, in pixels."""
    return math.hypot(pred_x - gt_x, pred_y - gt_y)


def success_at_tolerances(errors: list[float], tolerances=None) -> dict:
    """{tolerance_px: success_rate_percent} for each tolerance in config.SUCCESS_TOLERANCES_PX."""
    tolerances = config.SUCCESS_TOLERANCES_PX if tolerances is None else tolerances
    n = len(errors)
    if n == 0:
        return {t: 0.0 for t in tolerances}
    return {t: 100.0 * sum(1 for e in errors if e <= t) / n for t in tolerances}


def summarize(errors: list[float], times_ms: list[float], matched_flags: list[bool]) -> dict:
    """Aggregate per-sample results into the summary.json shape."""
    n = len(errors)
    if n == 0:
        return {"num_samples": 0}

    sorted_err = sorted(errors)
    mid = n // 2
    median = sorted_err[mid] if n % 2 else (sorted_err[mid - 1] + sorted_err[mid]) / 2.0

    return {
        "num_samples": n,
        "mean_error_px": sum(errors) / n,
        "median_error_px": median,
        "max_error_px": max(errors),
        "success_at_px": success_at_tolerances(errors),
        "mean_time_ms": sum(times_ms) / n,
        "num_matched_true": sum(1 for m in matched_flags if m),
        "num_matched_false": sum(1 for m in matched_flags if not m),
    }