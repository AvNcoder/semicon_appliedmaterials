# Drift-Sense
**AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**  
SEMICON India Hackathon 2026 – Applied Materials Problem Statement

---

## 1. Problem Description

In semiconductor manufacturing, SEM (Scanning Electron Microscope) inspection tools must repeatedly locate a known high-magnification reference pattern inside a wider, noisier, lower-magnification field of view. Stage drift, vibration and thermal effects cause the tool to land slightly off-target. 

**Drift-Sense** solves this “needle in a nanoscale haystack” problem:

- **Input**: a clean high-mag reference image (1000×1000) + a noisy wide-search image (1000×1000)
- **Output**: the exact center coordinates `(x, y)` of the reference pattern inside the search image

The system includes:
1. A physics-informed synthetic SEM dataset generator (DRAM-style and FinFET-style layouts)
2. A classical multi-scale / multi-angle Normalized Cross-Correlation (NCC) localizer
3. Full evaluation, visualization and noise-robustness analysis pipelines

---

## 2. System Requirements & Installation

### Requirements
- Python 3.10+
- OS: Windows / Linux / macOS
- See [`requirements.txt`](requirements.txt) for the full package list  
  (main libraries: `numpy`, `opencv-python`, `Pillow`, `scipy`, `matplotlib`)

### Installation

```bash
git clone https://github.com/AvNcoder/semicon_appliedmaterials.git
cd semicon_appliedmaterials
pip install -r requirements.txt
```


---


## 3. How the System Works

Drift-Sense follows the exact problem contract defined in the SEMICON / Applied Materials brief:

> Given a high-magnification reference image and a wider, noisier search image (both 1000×1000 grayscale), return the center `(x, y)` of the reference pattern inside the search image. If multiple matches exist, prefer the one closest to the search-image center.

The repository implements this end-to-end.

### 3.1 Repository Layout

semicon_appliedmaterials/
├── Fixed/                          # Synthetic SEM pair generators + data
│   ├── fixed_noise_data.py                 # dram_octagonal
│   ├── fixed_noise_data_dram.py            # dram_6f2
│   ├── fixed_noise_data_finfet6tsram.py    # finfet_sram
│   ├── fixed_noise_data_beol_interconnect.py
│   └── data_1 … data_4/                    # 400 pairs each (ref_XXX.png, search_XXX.png, gt_XXX.json)
├── driftsense/                     # Core localization package
│   ├── config.py                   # DEFAULT_SCALES, DEFAULT_ANGLES, MIN_SCORE, tolerances
│   ├── io_utils.py                 # iterate_dataset / load helpers
│   ├── preprocessing.py            # optional CLAHE
│   ├── localize.py                 # public API: (ref, search) → result dict
│   └── matching/
│       ├── template_matcher.py     # multi-scale + multi-angle NCC
│       └── tiebreak.py             # NMS + score-first / distance-to-center tie-break
├── evaluation/
│   ├── evaluate.py                 # full-style evaluation → results.csv + summary.json
│   ├── metrics.py                  # pixel_error, success@1/3/5 px, summarize
│   ├── visualize.py                # failure overlays + single-sample previews
│   └── noise_sweep.py              # Precision-Recall vs noise (1× / 2.5× / 5× / 10×)
├── predict.py                      # standalone predictor (hackathon contract)
├── cli.py                          # single entry-point (evaluate / visualize / show)
└── results/                        # auto-generated CSVs, summaries, PR plots



### 3.2 Synthetic Dataset Generation (`Fixed/`)

Each generator builds a realistic periodic die layout (DRAM-style or FinFET-style), embeds a unique high-mag patch, and applies a physics-informed SEM noise stack:

- **10× zoom ratio** – the reference is captured at higher magnification and appears shrunk inside the wide-search image (exactly as required by the problem statement).
- **Geometric jitter** – rotation ±3° and scale ±20 % applied to the embedded patch (PPT requirement).
- **Unique local features** – wire breaks, alignment pads and interconnect lines so that only one location is a true match (avoids pure periodic ambiguity).
- **Noise primitives** (independent on reference vs search):
  - Poisson–Gaussian (shot + readout)
  - Edge brightening (SE edge effect)
  - Astigmatism blur
  - Line-scan jitter
  - Stage drift
  - Charging & streaks
  - AC-mains ripple
  - Hydrocarbon deposition

Four styles are generated (400 pairs each):

| Style                | Folder   | Character |
|----------------------|----------|-----------|
| dram_octagonal       | data_1   | Orthogonal word-line / bit-line DRAM |
| dram_6f2             | data_2   | 6F² oblique active-moat DRAM |
| finfet_sram          | data_3   | Parallel fins + gate bars |
| beol_interconnect    | data_4   | Dual-layer M1/M2 + self-aligned vias |

### 3.3 Localization Algorithm (`driftsense/`)

1. **Multi-scale + multi-angle NCC** (`template_matcher.py`)  
   The reference is resized across `DEFAULT_SCALES` (covers the ±20 % generator jitter) and rotated across `DEFAULT_ANGLES` (±3°). At each pose, normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`) is computed.

2. **Center conversion**  
   `cv2.minMaxLoc` returns the top-left of the best window; we convert to center coordinates (`+ w/2, + h/2`) so the reported `(x, y)` matches the ground-truth convention.

3. **Non-maximum suppression** (`tiebreak.py`)  
   Nearby peaks that belong to the same physical location are collapsed.

4. **Score-first, distance-to-center tie-break**  
   Primary key = correlation score.  
   When scores are nearly equal, the candidate closest to the geometric center of the search image is chosen (exactly the rule stated in the problem brief).

5. **Confidence gate**  
   A match is accepted only if `score ≥ MIN_SCORE` (default 0.35, later justified by the PR-vs-noise sweep).

The public entry point is:

```python
from driftsense.localize import localize
result = localize(ref, search)
# result = {x, y, scale, angle, score, matched, time_ms, …}
```

### 3.4 Evaluation & Metrics (`evaluation/`)

- `cli.py evaluate` / `evaluate.py` runs the localizer over every sample of a style and writes:
  - `results/<style>/results.csv` – per-sample predictions, error, score, latency
  - `results/<style>/summary.json` – aggregate statistics + `failure_sample_ids`
  - `results/overall_summary.json`
- Success is defined at the tolerances required by the brief (1 px / 3 px / 5 px).
- A sample is counted as a **failure** when it is unmatched **or** its error exceeds 5 px.
- Live progress (percent, elapsed, ETA, last inference time) is printed while evaluating.

**Measured performance on the full 400-sample sets**

| Style                | Success @ 5 px | Mean Error | Failures | Mean Latency |
|----------------------|----------------|------------|----------|--------------|
| dram_6f2             | **100.0 %**    | 0.29 px    | 0        | ~2.3–2.5 s   |
| finfet_sram          | **96.0 %**     | 17.34 px   | 16       | ~2.4–2.5 s   |
| dram_octagonal       | **94.5 %**     | 24.61 px   | 22       | ~1.3–2.6 s   |
| beol_interconnect    | **91.5 %**     | 34.30 px   | 34       | ~2.2–2.3 s   |

Average success across all 1 600 images ≈ **95.5 %**.  
Computation time on a single 1000×1000 pair is reported as required by the problem statement.

### 3.5 Noise-Robustness Analysis (`noise_sweep.py`)

Following the PPT instruction “Sweep, don’t guess”:

- The **same geometry** is regenerated at four noise multipliers (1×, 2.5×, 5×, 10×).
- Precision–Recall curves are traced by sweeping the match-score threshold.
- Best-F1 thresholds and the usable operating region are recorded.

| Style                | Low (1×)     | Medium (2.5×) | High (5×)    | Extreme (10×) |
|----------------------|--------------|---------------|--------------|---------------|
| dram_octagonal       | 0.902 @ 0.75 | 0.891 @ 0.40  | 0.893 @ 0.25 | 0.844 @ 0.15  |
| dram_6f2             | 0.984 @ 0.80 | 0.945 @ 0.65  | 0.789 @ 0.45 | 0.716 @ 0.30  |
| finfet_sram          | 0.909 @ 0.80 | 0.800 @ 0.55  | 0.758 @ 0.35 | 0.641 @ 0.20  |
| beol_interconnect    | 0.837 @ 0.75 | 0.866 @ 0.35  | 0.958 @ 0.25 | 0.873 @ 0.15  |

The method stays usable up to 5× noise; clear degradation appears only at the Extreme (10×) tier. This supplies the evidence-based threshold justification required by the brief.

### 3.6 Standalone Predictor (`predict.py`)

Implements the exact hackathon contract:

```bash
python predict.py --search path/to/search.png --reference path/to/ref.png
```
