# Drift-Sense
**AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**  
SEMICON India Hackathon 2026 – Applied Materials Problem Statement

---

Repository: https://github.com/AvNcoder/semicon_appliedmaterials

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

```text
semicon_appliedmaterials/
├── Fixed/
│   ├── fixed_noise_data.py                 # dram_octagonal
│   ├── fixed_noise_data_dram.py            # dram_6f2
│   ├── fixed_noise_data_finfet6tsram.py    # finfet_sram
│   ├── fixed_noise_data_beol_interconnect.py
│   └── data_1 … data_4/                    # 400 pairs each
│       ├── ref_XXX.png
│       ├── search_XXX.png
│       └── gt_XXX.json
│
├── driftsense/
│   ├── config.py
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── localize.py
│   └── matching/
│       ├── template_matcher.py
│       └── tiebreak.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── visualize.py
│   └── noise_sweep.py
│
├── generate_dataset.py      # standalone generator (style / num / out)
├── predict.py               # standalone inference (hackathon contract)
├── cli.py
├── CITATIONS.md
├── requirements.txt
└── results/
```

### 3.2 Synthetic Dataset Generation (`Fixed/` + `generate_dataset.py`)

Each generator builds a realistic periodic die layout (DRAM-style or FinFET-style), embeds a unique high-mag patch, and applies a physics-informed SEM noise stack:

- **10× zoom ratio** – the reference appears shrunk inside the wide-search image (as required by the problem statement).
- **Geometric jitter** – rotation ±3° and scale ±20 % applied to the embedded patch.
- **Unique local features** – wire breaks, alignment pads and interconnect lines so that only one location is a true match.
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

| Style                | Folder   | Character                              |
|----------------------|----------|----------------------------------------|
| dram_octagonal       | data_1   | Orthogonal word-line / bit-line DRAM   |
| dram_6f2             | data_2   | 6F² oblique active-moat DRAM           |
| finfet_sram          | data_3   | Parallel fins + gate bars              |
| beol_interconnect    | data_4   | Dual-layer M1/M2 + self-aligned vias   |

### 3.3 Localization Algorithm (`driftsense/`)

1. **Multi-scale + multi-angle NCC** (`template_matcher.py`)  
   The reference is resized across `DEFAULT_SCALES` (covers ±20 % generator jitter) and rotated across `DEFAULT_ANGLES` (±3°).  
   At each pose, normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`) is computed.  
   **Up to 5 local peaks** are extracted from each correlation surface, rather than keeping only the single global maximum. This prevents the true match from being discarded when a neighboring period of a highly periodic layout scores slightly higher.

2. **Center conversion**  
   Each correlation peak is converted from the template's top-left position to center coordinates (`+ w/2, + h/2`).

3. **Non-maximum suppression** (`tiebreak.py`)  
   Nearby peaks belonging to the same physical location are collapsed.

4. **Score-first, distance-to-center tie-break**  
   Primary key = correlation score.  
   When scores are nearly equal, the candidate closest to the geometric center of the search image is chosen (exactly the rule stated in the problem brief).

5. **Confidence gate**  
   A match is accepted only if `score ≥ MIN_SCORE` (default 0.35, justified by the PR-vs-noise sweep).

### 3.4 Evaluation & Metrics (`evaluation/`)

- `cli.py evaluate` / `evaluate.py` runs the localizer over every sample of a style and writes:
  - `results/<style>/results.csv` – per-sample predictions, error, score, latency
  - `results/<style>/summary.json` – aggregate statistics + `failure_sample_ids`
  - `results/overall_summary.json`
- Success is defined at the tolerances required by the brief (1 px / 3 px / 5 px).
- A sample is counted as a **failure** when it is unmatched **or** its error exceeds 5 px.

**Measured performance on the full 400-sample sets**

| Style                | Success @ 5 px | Mean Error | Failures | Mean Latency |
|----------------------|----------------|------------|----------|--------------|
| dram_6f2             | **100.0 %**    | 0.66 px    | 0        | 2.29 s       |
| finfet_sram          | **92.2 %**     | 25.64 px   | 31       | 2.70 s       |
| dram_octagonal       | **89.5 %**     | 40.36 px   | 42       | 2.17 s       |
| beol_interconnect    | **87.5 %**     | 43.46 px   | 50       | 2.22 s       |

Average success across all 1 600 images ≈ **92.3 %**.  
Computation time on a single 1000×1000 pair is reported as required by the problem statement.

### 3.5 Noise-Robustness Analysis (`noise_sweep.py`)

Following the PPT instruction **“Sweep, don’t guess”**:

- The **same geometry** is regenerated at four noise multipliers: **1×, 2.5×, 5× and 10×**.
- Precision–Recall curves are obtained by sweeping the match-score threshold.
- The best-F1 threshold and usable operating region are recorded.

| Style                | Low (1×)     | Medium (2.5×) | High (5×)    | Extreme (10×) |
|----------------------|--------------|---------------|--------------|---------------|
| dram_octagonal       | 0.902 @ 0.75 | 0.891 @ 0.40  | 0.893 @ 0.25 | 0.844 @ 0.15  |
| dram_6f2             | 0.984 @ 0.80 | 0.945 @ 0.65  | 0.789 @ 0.45 | 0.716 @ 0.30  |
| finfet_sram          | 0.909 @ 0.80 | 0.800 @ 0.55  | 0.758 @ 0.35 | 0.641 @ 0.20  |
| beol_interconnect    | 0.837 @ 0.75 | 0.866 @ 0.35  | 0.958 @ 0.25 | 0.873 @ 0.15  |

The method remains usable up to **5× noise**, with clear degradation at the **10× Extreme** level.

### 3.6 Standalone Predictor (`predict.py`)

Implements the exact hackathon contract:

```bash
python predict.py --search path/to/search.png --reference path/to/ref.png
```

Returns `pred_x`, `pred_y`, `score`, `matched` and `time_ms` for any 1000×1000 pair with no dependency on the synthetic data folders.

---

## 4. Commands

Run all commands from the repository root:

```bash
cd semicon_appliedmaterials
```

### Generate datasets

**Recommended (standalone entry-point):**

```bash
python generate_dataset.py --style dram_6f2 --num 30 --out ./my_data
python generate_dataset.py --style finfet_sram --num 50 --out ./Fixed/data_3
python generate_dataset.py --style beol_interconnect --num 10 --out ./tmp_beol
python generate_dataset.py --style dram_octagonal --num 400 --out ./Fixed/data_1
```

Styles: `dram_octagonal` | `dram_6f2` | `finfet_sram` | `beol_interconnect`

The original per-style scripts under `Fixed/` remain available:

```bash
python Fixed/fixed_noise_data.py
python Fixed/fixed_noise_data_dram.py
python Fixed/fixed_noise_data_finfet6tsram.py
python Fixed/fixed_noise_data_beol_interconnect.py
```

Each dataset contains `ref_XXX.png`, `search_XXX.png`, and `gt_XXX.json`.

### Evaluate localization

```bash
python cli.py evaluate                          # all styles
python cli.py evaluate --style dram_octagonal   # one style
```

### Visualize failures / inspect a sample

```bash
python cli.py visualize
python cli.py visualize --style finfet_sram
python cli.py show --style dram_octagonal --sample 7
```

### Noise robustness

```bash
python evaluation/noise_sweep.py
```

### Standalone prediction

```bash
python predict.py --search search.png --reference ref.png
python predict.py --search search.png --reference ref.png --json
python predict.py --csv pairs.csv --out predictions.csv
```

### Recommended order

```bash
# Optional: regenerate datasets
python generate_dataset.py --style dram_6f2 --num 30 --out ./demo_data

# Evaluate
python cli.py evaluate

# Analyze failures
python cli.py visualize

# Noise analysis
python evaluation/noise_sweep.py
```

---

## 5. Results

Evaluation was performed on **400 samples per style (1 600 samples total)**.

| Style                  | Success @ 5 px | Mean Error | Failures | Mean Latency |
|------------------------|----------------|------------|----------|--------------|
| **dram_6f2**           | **100.0 %**    | 0.66 px    | 0        | 2.29 s       |
| **finfet_sram**        | **92.2 %**     | 25.64 px   | 31       | 2.70 s       |
| **dram_octagonal**     | **89.5 %**     | 40.36 px   | 42       | 2.17 s       |
| **beol_interconnect**  | **87.5 %**     | 43.46 px   | 50       | 2.22 s       |

**Overall:** ≈ 92.3 % average success across 1 600 samples.  
**Latency:** Mean inference time ≈ 2.2–2.7 s per pair (reported as required by the problem statement).

### Noise Robustness

Precision–Recall was evaluated at **1×, 2.5×, 5× and 10× noise**.

- The method remains usable up to **5× noise**.
- Clear degradation appears at the **10× (Extreme)** noise level.
- The optimal decision threshold decreases from approximately **0.75–0.80** at lower noise to **0.15–0.30** under extreme noise.

---

## 6. Sources & Citations

Full reference list: see [`CITATIONS.md`](CITATIONS.md).

### Die-Layout Generation

| Style | Source / Basis |
|---|---|
| **DRAM (octagonal / 6F²)** | US 7,349,232 B2 – “6F² DRAM Cell Design with 3F-Pitch Folded Digitline Sense Amplifier” (Micron). https://patents.google.com/patent/US7349232B2 |
| **FinFET SRAM** | US 9,012,287 B2 – “Cell Layout for SRAM FinFET Transistors”. https://patents.google.com/patent/US9012287 |
| **BEOL Interconnect** | imec – “Semi-damascene interconnects with fully self-aligned vias at 18 nm metal pitch”. https://www.imec-int.com/en/articles/imec-demonstrates-semi-damascene-interconnects-fully-self-aligned-vias-18nm-metal-pitch |

### SEM Noise Model

1. Timischl et al. (2012) – “A statistical model of signal-noise in scanning electron microscopy,” *Scanning* – Poisson + Gaussian shot/readout model.
2. Jin et al. (2015) – “Correction of image drift and distortion in a scanning electron microscopy,” *Journal of Microscopy* – stage drift and line-scan jitter.
3. Muller et al. (2006) – “Room design for high-performance electron microscopy,” *Ultramicroscopy* – AC-mains electromagnetic pickup.

### Classical Localization Algorithm

1. US 6,399,953 B1 – SEM feature matching using the normalized correlation coefficient method.
2. US 8,089,612 B2 – position detection using coarse correlation followed by local refinement.

---

## 7. License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for the full license text.
