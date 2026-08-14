import os
import cv2
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# =============================================================================
# 1. REALISTIC 7-TRACK FinFET 6T SRAM CELL & DIE LAYOUT GENERATOR
# =============================================================================

def generate_finfet_sram_cell(unit_size: int = 400) -> np.ndarray:
    """Generates a realistic 7-Track FinFET 6T SRAM cell."""
    cell = np.full((unit_size, unit_size), 25.0, dtype=np.float32)
    
    y_grid, x_grid = np.ogrid[:unit_size, :unit_size]
    
    # Vertical Silicon Fins
    fin_pitch = unit_size // 4
    fin_thick = max(1, int(unit_size * 0.03))
    for f in range(4):
        fin_center = f * fin_pitch + fin_pitch // 2
        cell[:, max(0, fin_center - fin_thick):min(unit_size, fin_center + fin_thick)] = 90.0
        
    # Horizontal Polysilicon / Metal Gates
    gate_pitch = unit_size // 2
    gate_thick = max(1, int(unit_size * 0.06))
    for g in range(2):
        gate_center = g * gate_pitch + gate_pitch // 2
        cell[max(0, gate_center - gate_thick):min(unit_size, gate_center + gate_thick), :] = 170.0
        
    # M1 Power Rails
    rail_thick = int(unit_size * 0.07)
    cell[0:rail_thick, :] = 220.0
    cell[unit_size - rail_thick:unit_size, :] = 220.0
    
    # Cross-Couple MOL Contact Plugs
    c1_x, c1_y = int(unit_size * 0.25), int(unit_size * 0.5)
    c2_x, c2_y = int(unit_size * 0.75), int(unit_size * 0.5)
    r = int(unit_size * 0.04)
    
    contact_mask = ((x_grid - c1_x)**2 + (y_grid - c1_y)**2 <= r**2) | \
                   ((x_grid - c2_x)**2 + (y_grid - c2_y)**2 <= r**2)
    cell[contact_mask] = 255.0
    
    return cell


def generate_master_finfet_die(master_size: int = 10000, cell_pitch: int = 400) -> np.ndarray:
    """Tiles the FinFET cell across a large master die."""
    unit_tile = generate_finfet_sram_cell(unit_size=cell_pitch)
    reps = master_size // cell_pitch
    return np.tile(unit_tile, (reps, reps)).astype(np.float32)


def add_unique_local_features(base_pattern: np.ndarray, seed: int) -> np.ndarray:
    """Sized proportionally to this style's cell_pitch=400 (verified fix,
    same as finfet_sram/beol_interconnect: 100%/0.33px vs the unboosted
    version's periodic-aliasing failures)."""
    rng = np.random.default_rng(seed)
    pattern = base_pattern.copy()
    h, w = pattern.shape

    num_breaks = rng.integers(2, 4)
    for _ in range(num_breaks):
        bx = rng.integers(180, w - 220)
        by = rng.integers(180, h - 220)
        bw, bh = rng.integers(120, 220), rng.integers(100, 180)
        pattern[by:by+bh, bx:bx+bw] = 35.0   # dram_6f2 substrate=35.0, finfet=25.0, beol=15.0

    num_pads = rng.integers(2, 3)
    for _ in range(num_pads):
        cx = rng.integers(220, w - 220)
        cy = rng.integers(220, h - 220)
        r = rng.integers(90, 140)
        cv2.circle(pattern, (cx, cy), r, 245.0, -1)
        cv2.circle(pattern, (cx, cy), r // 2, 40.0, -1)   # beol uses 30.0 here, others 40.0

    lx1, ly1 = rng.integers(120, w-120), rng.integers(120, h-120)
    lx2, ly2 = rng.integers(120, w-120), rng.integers(120, h-120)
    cv2.line(pattern, (lx1, ly1), (lx2, ly2), 200.0, thickness=rng.integers(28, 45))

    return pattern

# =============================================================================
# 2. ORGANIC SPATIAL MASK
# =============================================================================

def create_smooth_charging_mask(shape: tuple, scale: float = 120.0, threshold: float = 0.55, feather_sigma: float = 30.0) -> np.ndarray:
    h, w = shape
    small_h, small_w = max(4, int(h / scale)), max(4, int(w / scale))
    raw_noise = np.random.rand(small_h, small_w).astype(np.float32)
    smooth_map = cv2.resize(raw_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    binary_mask = (smooth_map > threshold).astype(np.float32)
    return gaussian_filter(binary_mask, sigma=feather_sigma).astype(np.float32)


# =============================================================================
# 3. SEM NOISE PRIMITIVES (kept, but used with mild parameters)
# =============================================================================

def apply_poisson_gaussian_noise(img: np.ndarray, shot_scale: float = 1.0, readout_sigma: float = 6.0) -> np.ndarray:
    normalized_signal = np.maximum(0.0, img)
    poisson_noisy = np.random.poisson(normalized_signal * shot_scale) / (shot_scale + 1e-5)
    gaussian_noise = np.random.normal(0, readout_sigma, img.shape)
    return (poisson_noisy + gaussian_noise).astype(np.float32)


def apply_edge_brightening(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    img_32f = img.astype(np.float32, copy=False)
    sobel_x = cv2.Sobel(img_32f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_32f, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.hypot(sobel_x, sobel_y)
    if edge_mag.max() > 0:
        edge_mag /= edge_mag.max()
    return (img_32f + (edge_mag * 90.0 * intensity)).astype(np.float32)


def apply_astigmatism_blur(img: np.ndarray, sigma_x: float = 1.1, sigma_y: float = 0.55, angle_deg: float = 30.0) -> np.ndarray:
    img_32f = img.astype(np.float32, copy=False)
    ksize = int(max(sigma_x, sigma_y) * 5) | 1
    kernel_x = cv2.getGaussianKernel(ksize, sigma_x)
    kernel_y = cv2.getGaussianKernel(ksize, sigma_y)
    kernel = np.outer(kernel_x, kernel_y)
    M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle_deg, 1.0)
    rotated_kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    rotated_kernel /= rotated_kernel.sum()
    return cv2.filter2D(img_32f, -1, rotated_kernel).astype(np.float32)


def apply_line_scan_jitter(img: np.ndarray, max_shift: int = 2, jitter_prob: float = 0.08) -> np.ndarray:
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


def apply_stage_drift(img: np.ndarray, max_drift: float = 1.5) -> np.ndarray:
    img_32f = img.astype(np.float32, copy=False)
    h, w = img_32f.shape[:2]
    mask = create_smooth_charging_mask((h, w), scale=130.0, threshold=0.55, feather_sigma=35.0)
    dx = gaussian_filter(np.random.uniform(-max_drift, max_drift, (h, w)), sigma=18.0).astype(np.float32) * mask
    dy = gaussian_filter(np.random.uniform(-max_drift, max_drift, (h, w)), sigma=18.0).astype(np.float32) * mask
    x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img_32f, x_grid + dx, y_grid + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)


def apply_charging_and_streaks(img: np.ndarray, intensity: float = 18.0) -> np.ndarray:
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


def apply_ac_mains_ripple(img: np.ndarray, amplitude: float = 3.0, freq_cycles: float = 6.0) -> np.ndarray:
    h, w = img.shape[:2]
    y_indices = np.arange(h, dtype=np.float32)
    phase = np.random.uniform(0, 2 * np.pi)
    ripple = np.sin(2 * np.pi * freq_cycles * (y_indices / h) + phase) * amplitude
    return (img + ripple[:, np.newaxis]).astype(np.float32)


def apply_hydrocarbon_deposition(img: np.ndarray, dark_attenuation: float = 0.08) -> np.ndarray:
    img_32f = img.astype(np.float32, copy=False)
    h, w = img_32f.shape[:2]
    dark_blob = create_smooth_charging_mask((h, w), scale=110.0, threshold=0.58, feather_sigma=45.0)
    return (img_32f * (1.0 - dark_blob * dark_attenuation)).astype(np.float32)


def tonemap_to_uint8(img_float: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img_float, (0.5, 99.5))
    norm = np.clip((img_float - p1) / (p99 - p1 + 1e-5), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


# =============================================================================
# 4. MILD NOISE STACKS (pattern stays visible)
# =============================================================================

def apply_light_reference_noise(img: np.ndarray) -> np.ndarray:
    output = img.copy()
    output = apply_edge_brightening(output, intensity=0.45)
    output = apply_astigmatism_blur(output, sigma_x=0.9, sigma_y=0.45, angle_deg=np.random.uniform(0, 180))
    output = apply_poisson_gaussian_noise(output, shot_scale=1.3, readout_sigma=4.0)
    return output


def apply_calibrated_search_noise(img: np.ndarray) -> np.ndarray:
    """Mild noise that keeps FinFET structure + embedded patch clearly visible."""
    output = img.copy()
    
    if np.random.rand() < 0.25:
        output = apply_stage_drift(output, max_drift=np.random.uniform(0.8, 1.8))
    if np.random.rand() < 0.20:
        output = apply_hydrocarbon_deposition(output, dark_attenuation=np.random.uniform(0.04, 0.09))
    if np.random.rand() < 0.25:
        output = apply_charging_and_streaks(output, intensity=np.random.uniform(10.0, 22.0))
    if np.random.rand() < 0.20:
        output = apply_ac_mains_ripple(output, amplitude=np.random.uniform(1.5, 3.5), freq_cycles=np.random.uniform(4.0, 8.0))
        
    output = apply_astigmatism_blur(output, sigma_x=np.random.uniform(0.9, 1.3), sigma_y=np.random.uniform(0.45, 0.65), angle_deg=np.random.uniform(0, 180))
    output = apply_edge_brightening(output, intensity=np.random.uniform(0.45, 0.70))
    output = apply_line_scan_jitter(output, max_shift=np.random.randint(1, 2), jitter_prob=np.random.uniform(0.04, 0.12))
    output = apply_poisson_gaussian_noise(output, shot_scale=np.random.uniform(0.9, 1.3), readout_sigma=np.random.uniform(4.0, 8.0))
    
    return output


# =============================================================================
# 5. FIXED PAIRED DATASET GENERATOR
# =============================================================================

def generate_finfet_sem_dataset(output_dir: str, num_samples: int = 30):
    """
    Generates paired Reference + Search images with 10× embedding and GT centers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    search_size = 1000
    embedded_size = 100
    half_embed = embedded_size // 2
    CELL_PITCH = 400
    MASTER_SIZE = 10000

    print("Building master FinFET die...")
    master_die = generate_master_finfet_die(master_size=MASTER_SIZE, cell_pitch=CELL_PITCH)
    
    # Create a clean 1000×1000 periodic base (same as your wide_clean)
    master_pil = Image.fromarray(np.clip(master_die, 0, 255).astype(np.uint8), mode='L')
    wide_base_pil = master_pil.resize((search_size, search_size), Image.Resampling.BOX)
    base_finfet = np.array(wide_base_pil, dtype=np.float32)

    for idx in range(1, num_samples + 1):
        np.random.seed(idx * 3037)
        
        # 1. High-mag Reference (1000×1000 crop) + unique features
        # Take a random 1000×1000 crop from the master for variety
        crop_x = np.random.randint(500, MASTER_SIZE - 1500)
        crop_y = np.random.randint(500, MASTER_SIZE - 1500)
        clean_ref_1000 = master_die[crop_y:crop_y+1000, crop_x:crop_x+1000].copy()
        clean_ref_1000 = add_unique_local_features(clean_ref_1000, seed=idx * 999)
        
        # 2. 10× downsampled version to embed
        clean_ref_100 = cv2.resize(clean_ref_1000, (embedded_size, embedded_size), interpolation=cv2.INTER_AREA)
        
        # 3. Search starts as pure periodic FinFET
        clean_search_canvas = base_finfet.copy()
        
        # 4. Random GT center
        center_x = float(np.random.uniform(170.0, search_size - 170.0))
        center_y = float(np.random.uniform(170.0, search_size - 170.0))

        # NEW: rotation + scale jitter on the embedded patch, per hackathon spec
        # ("Rotation 1-3 degrees to the polygons", "Scaling polygons -20% to 20%")
        scale_jitter = float(np.random.uniform(0.8, 1.2))
        rotation_deg = float(np.random.uniform(-3.0, 3.0))
        jittered_size = max(20, int(round(embedded_size * scale_jitter)))

        clean_ref_100 = cv2.resize(clean_ref_1000, (jittered_size, jittered_size), interpolation=cv2.INTER_AREA)
        M = cv2.getRotationMatrix2D((jittered_size / 2, jittered_size / 2), rotation_deg, 1.0)
        clean_ref_100 = cv2.warpAffine(clean_ref_100, M, (jittered_size, jittered_size), borderMode=cv2.BORDER_REPLICATE)

        half = jittered_size // 2
        x1 = int(round(center_x)) - half
        y1 = int(round(center_y)) - half
        x2 = x1 + jittered_size
        y2 = y1 + jittered_size

        actual_center_x = (x1 + x2) / 2.0
        actual_center_y = (y1 + y2) / 2.0

        # 5. Embed the jittered patch
        clean_search_canvas[y1:y2, x1:x2] = clean_ref_100
        
        # 6. Independent mild noise
        noisy_ref = apply_light_reference_noise(clean_ref_1000)
        noisy_search = apply_calibrated_search_noise(clean_search_canvas)
        
        ref_uint8 = tonemap_to_uint8(noisy_ref)
        search_uint8 = tonemap_to_uint8(noisy_search)
        
        # 7. Save
        Image.fromarray(ref_uint8, mode='L').save(os.path.join(output_dir, f"ref_{idx:03d}.png"))
        Image.fromarray(search_uint8, mode='L').save(os.path.join(output_dir, f"search_{idx:03d}.png"))
        
        gt_data = {
            "sample_id": idx,
            "style": "finfet_sram",   # (or dram_6f2 / finfet_sram / beol_interconnect, unchanged)
            "center_x": round(actual_center_x, 2),
            "center_y": round(actual_center_y, 2),
            "reference_effective_width": jittered_size,
            "reference_effective_height": jittered_size,
            "rotation_deg": round(rotation_deg, 2),
            "scale_jitter": round(scale_jitter, 4),
            "search_image_shape": [search_size, search_size]
        }
        with open(os.path.join(output_dir, f"gt_{idx:03d}.json"), 'w') as f:
            json.dump(gt_data, f, indent=4)

    print(f"Successfully generated {num_samples} usable FinFET Drift-Sense pairs in: {output_dir}")


if __name__ == "__main__":
    OUTPUT_PATH = r"./Fixed/data_3"
    generate_finfet_sem_dataset(output_dir=OUTPUT_PATH, num_samples=100)