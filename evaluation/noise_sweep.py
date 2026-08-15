"""
Precision-Recall vs Noise sweep, per the hackathon spec (slide: "Sweep, don't
guess: regenerate the same test cases at several noise levels... Plot
precision vs recall per level... Read the trade-off... Choose the threshold
with evidence").

Design, precisely matching the spec's own wording:
  - "Same test cases, everything else fixed": geometry (crop location,
    unique-feature placement, embed center, rotation, scale jitter) is drawn
    from the SAME seeded random stream as the real generators, in the SAME
    order, regardless of noise level. Only a multiplier applied AFTER each
    noise-intensity value is drawn changes -- so the random stream position
    and every geometric decision is bit-identical across Low/Medium/High;
    only noise strength differs.
  - "Precision vs recall per level": for each noise level, MIN_SCORE is swept
    across a range (not fixed at 0.35), and precision/recall computed at
    each threshold using a real positive+negative test set (reusing the
    confusion_matrix.py negative-pair construction) -- this traces an actual
    curve, not a single point.
  - "Choose the threshold with evidence": the printed/saved summary reports
    the best F1 threshold per noise level, which is the evidence-based
    justification for config.MIN_SCORE.

This does NOT touch driftsense/config.py or the real Fixed/data_N generators
-- it is a standalone diagnostic script, kept separate exactly as evaluate.py
and confusion_matrix.py are.
"""

import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from driftsense.localize import localize


# =============================================================================
# SHARED NOISE PRIMITIVES -- identical across all 4 of your generators,
# copied once here rather than importing 4 separate script files.
# =============================================================================

def create_smooth_charging_mask(shape, scale=120.0, threshold=0.55, feather_sigma=30.0):
    h, w = shape
    small_h, small_w = max(4, int(h / scale)), max(4, int(w / scale))
    raw_noise = np.random.rand(small_h, small_w).astype(np.float32)
    smooth_map = cv2.resize(raw_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    binary_mask = (smooth_map > threshold).astype(np.float32)
    return gaussian_filter(binary_mask, sigma=feather_sigma).astype(np.float32)


def apply_poisson_gaussian_noise(img, shot_scale=1.0, readout_sigma=6.0):
    normalized_signal = np.maximum(0.0, img)
    poisson_noisy = np.random.poisson(normalized_signal * shot_scale) / (shot_scale + 1e-5)
    gaussian_noise = np.random.normal(0, readout_sigma, img.shape)
    return (poisson_noisy + gaussian_noise).astype(np.float32)


def apply_edge_brightening(img, intensity=0.5):
    img_32f = img.astype(np.float32, copy=False)
    sobel_x = cv2.Sobel(img_32f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_32f, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.hypot(sobel_x, sobel_y)
    if edge_mag.max() > 0:
        edge_mag /= edge_mag.max()
    return (img_32f + (edge_mag * 90.0 * intensity)).astype(np.float32)


def apply_astigmatism_blur(img, sigma_x=1.1, sigma_y=0.55, angle_deg=30.0):
    img_32f = img.astype(np.float32, copy=False)
    ksize = int(max(sigma_x, sigma_y) * 5) | 1
    kernel_x = cv2.getGaussianKernel(ksize, sigma_x)
    kernel_y = cv2.getGaussianKernel(ksize, sigma_y)
    kernel = np.outer(kernel_x, kernel_y)
    M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle_deg, 1.0)
    rotated_kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    rotated_kernel /= rotated_kernel.sum()
    return cv2.filter2D(img_32f, -1, rotated_kernel).astype(np.float32)


def apply_line_scan_jitter(img, max_shift=2, jitter_prob=0.08):
    output = img.copy()
    h, w = output.shape[:2]
    for r in range(h):
        if np.random.rand() < jitter_prob:
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                output[r, shift:] = img[r, :-shift]
            elif shift < 0:
                output[r, :shift] = img[r, -shift:]
    return output.astype(np.float32)


def apply_stage_drift(img, max_drift=1.5):
    img_32f = img.astype(np.float32, copy=False)
    h, w = img_32f.shape[:2]
    mask = create_smooth_charging_mask((h, w), scale=130.0, threshold=0.55, feather_sigma=35.0)
    dx = gaussian_filter(np.random.uniform(-max_drift, max_drift, (h, w)), sigma=18.0).astype(np.float32) * mask
    dy = gaussian_filter(np.random.uniform(-max_drift, max_drift, (h, w)), sigma=18.0).astype(np.float32) * mask
    x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img_32f, x_grid + dx, y_grid + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def apply_charging_and_streaks(img, intensity=18.0):
    img_32f = img.astype(np.float32, copy=False)
    h, w = img_32f.shape[:2]
    halo_mask = create_smooth_charging_mask((h, w), scale=150.0, threshold=0.62, feather_sigma=40.0)
    streak_mask = np.zeros((h, w), dtype=np.float32)
    active_rows = np.where(halo_mask.max(axis=1) > 0.15)[0]
    if len(active_rows) > 0:
        r_start, r_end = active_rows[0], active_rows[-1]
        c_start = np.random.randint(w // 5, w // 2)
        streak_len = int(w * np.random.uniform(0.12, 0.20))
        actual_len = min(w - c_start, streak_len)
        if actual_len > 0:
            decay = np.exp(-np.linspace(0, 2.5, actual_len, dtype=np.float32))
            streak_mask[r_start:r_end, c_start:c_start+actual_len] = decay[np.newaxis, :] * (intensity * 0.25)
    return (img_32f + (halo_mask * intensity) + streak_mask).astype(np.float32)


def apply_ac_mains_ripple(img, amplitude=3.0, freq_cycles=6.0):
    h, w = img.shape[:2]
    y_indices = np.arange(h, dtype=np.float32)
    phase = np.random.uniform(0, 2 * np.pi)
    ripple = np.sin(2 * np.pi * freq_cycles * (y_indices / h) + phase) * amplitude
    return (img + ripple[:, np.newaxis]).astype(np.float32)


def apply_hydrocarbon_deposition(img, dark_attenuation=0.08):
    img_32f = img.astype(np.float32, copy=False)
    h, w = img_32f.shape[:2]
    dark_blob = create_smooth_charging_mask((h, w), scale=110.0, threshold=0.58, feather_sigma=45.0)
    return (img_32f * (1.0 - dark_blob * dark_attenuation)).astype(np.float32)


def tonemap_to_uint8(img_float):
    p1, p99 = np.percentile(img_float, (0.5, 99.5))
    norm = np.clip((img_float - p1) / (p99 - p1 + 1e-5), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def apply_light_reference_noise(img):
    output = img.copy()
    output = apply_edge_brightening(output, intensity=0.45)
    output = apply_astigmatism_blur(output, sigma_x=0.9, sigma_y=0.45, angle_deg=np.random.uniform(0, 180))
    output = apply_poisson_gaussian_noise(output, shot_scale=1.3, readout_sigma=4.0)
    return output


def apply_calibrated_search_noise_scaled(img, multiplier=1.0):
    """Same structure/order as your real apply_calibrated_search_noise, but
    every intensity value is drawn from its ORIGINAL fixed range FIRST, then
    multiplied -- so which noise types fire and their base magnitudes are
    identical to the real generator's random stream regardless of multiplier.
    Only the resulting strength differs. This is what makes Low/Medium/High
    genuinely 'the same test case at different noise levels' rather than
    three unrelated random draws."""
    output = img.copy()

    if np.random.rand() < 0.25:
        v = np.random.uniform(0.8, 1.8)
        output = apply_stage_drift(output, max_drift=v * multiplier)
    if np.random.rand() < 0.20:
        v = np.random.uniform(0.04, 0.09)
        output = apply_hydrocarbon_deposition(output, dark_attenuation=min(0.95, v * multiplier))
    if np.random.rand() < 0.25:
        v = np.random.uniform(10.0, 22.0)
        output = apply_charging_and_streaks(output, intensity=v * multiplier)
    if np.random.rand() < 0.20:
        v = np.random.uniform(1.5, 3.5)
        output = apply_ac_mains_ripple(output, amplitude=v * multiplier, freq_cycles=np.random.uniform(4.0, 8.0))

    sx = np.random.uniform(0.9, 1.3) * min(multiplier, 3.0)
    sy = np.random.uniform(0.45, 0.65) * min(multiplier, 3.0)
    output = apply_astigmatism_blur(output, sigma_x=sx, sigma_y=sy, angle_deg=np.random.uniform(0, 180))
    output = apply_edge_brightening(output, intensity=np.random.uniform(0.45, 0.70))

    base_shift = np.random.randint(1, 2)
    base_prob = np.random.uniform(0.04, 0.12)
    output = apply_line_scan_jitter(output, max_shift=max(1, int(base_shift * min(multiplier, 4))),
                                     jitter_prob=min(0.9, base_prob * multiplier))

    shot = np.random.uniform(0.9, 1.3)
    readout = np.random.uniform(4.0, 8.0)
    output = apply_poisson_gaussian_noise(output, shot_scale=max(0.1, shot / max(multiplier, 0.1)),
                                           readout_sigma=readout * multiplier)
    return output


# =============================================================================
# STYLE-SPECIFIC PATTERN GENERATORS -- exact logic from your pasted, current
# (phase-locked, feather-blended) fixed_noise_data_*.py files.
# =============================================================================

def _dram_octagonal_cell(unit_size=100):
    cell = np.full((unit_size, unit_size), 25.0, dtype=np.float32)
    cell[40:60, :] = 140.0
    cell[:, 40:60] = 160.0
    y_grid, x_grid = np.ogrid[:unit_size, :unit_size]
    via_mask = ((x_grid - 50)**2 + (y_grid - 50)**2) <= 12**2
    cell[via_mask] = 255.0
    return cell


def _dram_octagonal_features(base_pattern, seed):
    rng = np.random.default_rng(seed)
    pattern = base_pattern.copy()
    h, w = pattern.shape
    for _ in range(rng.integers(2, 4)):
        bx, by = rng.integers(180, w - 220), rng.integers(180, h - 220)
        bw, bh = rng.integers(90, 160), rng.integers(70, 130)
        pattern[by:by+bh, bx:bx+bw] = 25.0
    for _ in range(rng.integers(2, 3)):
        cx, cy = rng.integers(220, w - 220), rng.integers(220, h - 220)
        r = rng.integers(70, 110)
        cv2.circle(pattern, (cx, cy), r, 245.0, -1)
        cv2.circle(pattern, (cx, cy), r // 2, 40.0, -1)
    lx1, ly1 = rng.integers(120, w-120), rng.integers(120, h-120)
    lx2, ly2 = rng.integers(120, w-120), rng.integers(120, h-120)
    cv2.line(pattern, (lx1, ly1), (lx2, ly2), 200.0, thickness=rng.integers(20, 32))
    return pattern


def _dram_6f2_cell(unit_size=400):
    cell = np.full((unit_size, unit_size), 35.0, dtype=np.float32)
    y_grid, x_grid = np.ogrid[:unit_size, :unit_size]
    moat_mask = np.abs((x_grid - 0.4 * y_grid - 40) % unit_size - unit_size // 2) < (unit_size * 0.12)
    cell[moat_mask] = 100.0
    wl_y1, wl_y2 = int(unit_size * 0.3), int(unit_size * 0.7)
    wl_thick = max(1, int(unit_size * 0.08))
    cell[wl_y1 - wl_thick:wl_y1 + wl_thick, :] = 160.0
    cell[wl_y2 - wl_thick:wl_y2 + wl_thick, :] = 160.0
    wave = (unit_size * 0.08) * np.sin(2 * np.pi * y_grid / unit_size)
    bl_mask = np.abs((x_grid + wave - unit_size // 2) % unit_size) < (unit_size * 0.05)
    cell[bl_mask] = 210.0
    cx1, cy1 = int(unit_size * 0.25), int(unit_size * 0.25)
    cx2, cy2 = int(unit_size * 0.75), int(unit_size * 0.75)
    r = max(2, int(unit_size * 0.06))
    contact_mask = ((x_grid - cx1)**2 + (y_grid - cy1)**2 <= r**2) | ((x_grid - cx2)**2 + (y_grid - cy2)**2 <= r**2)
    cell[contact_mask] = 255.0
    return cell


def _pitch400_features(base_pattern, seed, fill_bg, fill_pad_inner):
    """Shared by dram_6f2/finfet_sram/beol_interconnect -- identical logic,
    only the fill colors differ per style's substrate value."""
    rng = np.random.default_rng(seed)
    pattern = base_pattern.copy()
    h, w = pattern.shape
    for _ in range(rng.integers(2, 4)):
        bx, by = rng.integers(180, w - 220), rng.integers(180, h - 220)
        bw, bh = rng.integers(120, 220), rng.integers(100, 180)
        pattern[by:by+bh, bx:bx+bw] = fill_bg
    for _ in range(rng.integers(2, 3)):
        cx, cy = rng.integers(220, w - 220), rng.integers(220, h - 220)
        r = rng.integers(90, 140)
        cv2.circle(pattern, (cx, cy), r, 245.0, -1)
        cv2.circle(pattern, (cx, cy), r // 2, fill_pad_inner, -1)
    lx1, ly1 = rng.integers(120, w-120), rng.integers(120, h-120)
    lx2, ly2 = rng.integers(120, w-120), rng.integers(120, h-120)
    cv2.line(pattern, (lx1, ly1), (lx2, ly2), 200.0, thickness=rng.integers(28, 45))
    return pattern


def _finfet_cell(unit_size=400):
    cell = np.full((unit_size, unit_size), 25.0, dtype=np.float32)
    y_grid, x_grid = np.ogrid[:unit_size, :unit_size]
    fin_pitch = unit_size // 4
    fin_thick = max(1, int(unit_size * 0.03))
    for f in range(4):
        fc = f * fin_pitch + fin_pitch // 2
        cell[:, max(0, fc - fin_thick):min(unit_size, fc + fin_thick)] = 90.0
    gate_pitch = unit_size // 2
    gate_thick = max(1, int(unit_size * 0.06))
    for g in range(2):
        gc = g * gate_pitch + gate_pitch // 2
        cell[max(0, gc - gate_thick):min(unit_size, gc + gate_thick), :] = 170.0
    rail_thick = int(unit_size * 0.07)
    cell[0:rail_thick, :] = 220.0
    cell[unit_size - rail_thick:unit_size, :] = 220.0
    c1x, c1y = int(unit_size * 0.25), int(unit_size * 0.5)
    c2x, c2y = int(unit_size * 0.75), int(unit_size * 0.5)
    r = int(unit_size * 0.04)
    mask = ((x_grid - c1x)**2 + (y_grid - c1y)**2 <= r**2) | ((x_grid - c2x)**2 + (y_grid - c2y)**2 <= r**2)
    cell[mask] = 255.0
    return cell


def _beol_cell(unit_size=400):
    cell = np.full((unit_size, unit_size), 15.0, dtype=np.float32)
    y_grid, x_grid = np.ogrid[:unit_size, :unit_size]
    m1_pitch = unit_size // 5
    m1_thick = max(1, int(unit_size * 0.04))
    for m in range(5):
        mc = m * m1_pitch + m1_pitch // 2
        cell[max(0, mc - m1_thick):min(unit_size, mc + m1_thick), :] = 100.0
    m2_pitch = unit_size // 5
    m2_thick = max(1, int(unit_size * 0.04))
    for m in range(5):
        mc = m * m2_pitch + m2_pitch // 2
        cell[:, max(0, mc - m2_thick):min(unit_size, mc + m2_thick)] = 170.0
    via_mask = np.zeros((unit_size, unit_size), dtype=bool)
    for row in range(5):
        for col in range(5):
            if (row + col) % 2 == 0:
                vx = col * m2_pitch + m2_pitch // 2
                vy = row * m1_pitch + m1_pitch // 2
                r = int(unit_size * 0.045)
                via_mask |= ((x_grid - vx)**2 + (y_grid - vy)**2 <= r**2)
    cell[via_mask] = 255.0
    return cell


STYLE_CONFIGS = {
    "dram_octagonal": {"cell_fn": _dram_octagonal_cell, "cell_pitch": 100,
                        "features_fn": _dram_octagonal_features, "seed_mult": 3037},
    "dram_6f2":        {"cell_fn": _dram_6f2_cell, "cell_pitch": 400,
                         "features_fn": lambda p, s: _pitch400_features(p, s, 35.0, 40.0), "seed_mult": 4096},
    "finfet_sram":     {"cell_fn": _finfet_cell, "cell_pitch": 400,
                         "features_fn": lambda p, s: _pitch400_features(p, s, 35.0, 40.0), "seed_mult": 3037},
    "beol_interconnect": {"cell_fn": _beol_cell, "cell_pitch": 400,
                           "features_fn": lambda p, s: _pitch400_features(p, s, 35.0, 40.0), "seed_mult": 4048},
}

_MASTER_DIE_CACHE = {}


def _get_master_die(style, master_size=10000):
    if style not in _MASTER_DIE_CACHE:
        cfg = STYLE_CONFIGS[style]
        tile = cfg["cell_fn"](unit_size=cfg["cell_pitch"])
        reps = master_size // cfg["cell_pitch"]
        _MASTER_DIE_CACHE[style] = np.tile(tile, (reps, reps)).astype(np.float32)
    return _MASTER_DIE_CACHE[style]


def generate_pair(style: str, sample_id: int, noise_multiplier: float, master_size: int = 10000):
    """Regenerate ONE (ref, search, gt) triplet, geometry identical across
    noise_multiplier values, noise strength scaled by noise_multiplier."""
    cfg = STYLE_CONFIGS[style]
    cell_pitch = cfg["cell_pitch"]
    master_die = _get_master_die(style, master_size)
    search_size, embedded_size = 1000, 100

    np.random.seed(sample_id * cfg["seed_mult"])

    crop_x = np.random.randint(500, master_size - 1500)
    crop_y = np.random.randint(500, master_size - 1500)
    crop_x = (crop_x // cell_pitch) * cell_pitch
    crop_y = (crop_y // cell_pitch) * cell_pitch
    clean_ref_1000 = master_die[crop_y:crop_y+1000, crop_x:crop_x+1000].copy()
    clean_ref_1000 = cfg["features_fn"](clean_ref_1000, sample_id * 999)

    base = cv2.resize(master_die, (search_size, search_size), interpolation=cv2.INTER_AREA)
    clean_search = base.copy()

    center_x = float(np.random.uniform(170.0, search_size - 170.0))
    center_y = float(np.random.uniform(170.0, search_size - 170.0))
    search_cell = cell_pitch / 10.0
    center_x = round(center_x / search_cell) * search_cell
    center_y = round(center_y / search_cell) * search_cell

    scale_jitter = float(np.random.uniform(0.8, 1.2))
    rotation_deg = float(np.random.uniform(-3.0, 3.0))
    jittered_size = max(20, int(round(embedded_size * scale_jitter)))

    patch = cv2.resize(clean_ref_1000, (jittered_size, jittered_size), interpolation=cv2.INTER_AREA)
    rc = ((jittered_size - 1) / 2.0, (jittered_size - 1) / 2.0)
    M = cv2.getRotationMatrix2D(rc, rotation_deg, 1.0)
    patch = cv2.warpAffine(patch, M, (jittered_size, jittered_size), borderMode=cv2.BORDER_REPLICATE)

    half = jittered_size // 2
    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2, y2 = x1 + jittered_size, y1 + jittered_size
    actual_cx, actual_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    feather = 4
    blend = np.ones((jittered_size, jittered_size), dtype=np.float32)
    ramp = np.linspace(0, 1, feather, dtype=np.float32)
    blend[:feather, :] *= ramp[:, None]
    blend[-feather:, :] *= ramp[::-1][:, None]
    blend[:, :feather] *= ramp[None, :]
    blend[:, -feather:] *= ramp[::-1][None, :]
    region = clean_search[y1:y2, x1:x2]
    clean_search[y1:y2, x1:x2] = patch * blend + region * (1 - blend)

    noisy_ref = apply_light_reference_noise(clean_ref_1000)
    noisy_search = apply_calibrated_search_noise_scaled(clean_search, multiplier=noise_multiplier)

    ref_u8 = tonemap_to_uint8(noisy_ref)
    search_u8 = tonemap_to_uint8(noisy_search)

    gt = {"center_x": round(actual_cx, 2), "center_y": round(actual_cy, 2)}
    return ref_u8.astype(np.float32), search_u8.astype(np.float32), gt


# =============================================================================
# PRECISION-RECALL-VS-NOISE SWEEP
# =============================================================================

NOISE_LEVELS = {"Low": 1.0, "Medium": 2.5, "High": 5.0, "Extreme": 10.0}
THRESHOLD_SWEEP = np.linspace(0.05, 0.85, 17)


def run_pr_sweep(style: str, num_samples: int = 30, num_negatives: int = 30) -> dict:
    results_by_level = {}

    for level_name, multiplier in NOISE_LEVELS.items():
        records = []  # (is_real_match, score, error_or_None)

        for sid in range(1, num_samples + 1):
            ref, search, gt = generate_pair(style, sid, multiplier)
            r = localize(ref, search, apply_preprocessing=False)
            err = ((r["x"] - gt["center_x"])**2 + (r["y"] - gt["center_y"])**2) ** 0.5
            records.append({"is_real_match": True, "score": r["score"], "error": err})

        # negatives: ref from sample A, search from DIFFERENT sample B, same noise level
        rng = np.random.RandomState(0)
        neg_ids = [(int(a), int(b)) for a, b in
                   zip(rng.randint(1, num_samples + 1, num_negatives),
                       rng.randint(1, num_samples + 1, num_negatives)) if a != b]
        for a, b in neg_ids:
            ref_a, _, _ = generate_pair(style, a, multiplier)
            _, search_b, _ = generate_pair(style, b, multiplier)
            r = localize(ref_a, search_b, apply_preprocessing=False)
            records.append({"is_real_match": False, "score": r["score"], "error": None})

        curve = []
        for thresh in THRESHOLD_SWEEP:
            tp = fp = fn = 0
            for rec in records:
                matched = rec["score"] >= thresh
                if rec["is_real_match"]:
                    if matched and rec["error"] <= 5.0:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if matched:
                        fp += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            curve.append({"threshold": round(float(thresh), 3), "precision": round(precision, 4), "recall": round(recall, 4)})

        f1s = [(2 * c["precision"] * c["recall"] / (c["precision"] + c["recall"]) if (c["precision"] + c["recall"]) > 0 else 0, c)
               for c in curve]
        best_f1, best_point = max(f1s, key=lambda x: x[0])

        results_by_level[level_name] = {
            "multiplier": multiplier,
            "curve": curve,
            "best_f1": round(best_f1, 4),
            "best_threshold": best_point["threshold"],
        }
        print(f"[{style}][{level_name} noise, x{multiplier}] best F1={best_f1:.3f} at threshold={best_point['threshold']}")

    return results_by_level


def plot_pr_curves(all_results: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5), squeeze=False)
    colors = {"Low": "tab:blue", "Medium": "tab:orange", "High": "tab:red"}

    for ax_idx, (style, by_level) in enumerate(all_results.items()):
        ax = axes[0][ax_idx]
        for level_name, data in by_level.items():
            recalls = [c["recall"] for c in data["curve"]]
            precisions = [c["precision"] for c in data["curve"]]
            order = np.argsort(recalls)
            recalls = np.array(recalls)[order]
            precisions = np.array(precisions)[order]
            ax.plot(recalls, precisions, marker="o", markersize=3,
                    label=f"{level_name} noise (x{data['multiplier']})", color=colors.get(level_name))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(style)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Precision-Recall vs Noise (MIN_SCORE threshold swept per point)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot saved -> {out_path}")


def run_all_styles(project_root: str = None, num_samples: int = 30, styles=None):
    project_root = _PROJECT_ROOT if project_root is None else project_root
    styles = list(STYLE_CONFIGS.keys()) if styles is None else styles

    all_results = {}
    for style in styles:
        all_results[style] = run_pr_sweep(style, num_samples=num_samples)

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "precision_recall_vs_noise.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\ndata -> {json_path}")

    plot_path = os.path.join(results_dir, "precision_recall_vs_noise.png")
    plot_pr_curves(all_results, plot_path)

    return all_results


if __name__ == "__main__":
    run_all_styles(num_samples=60)