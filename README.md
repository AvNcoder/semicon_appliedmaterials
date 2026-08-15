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
