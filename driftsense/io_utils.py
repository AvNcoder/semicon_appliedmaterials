"""
Loads (reference, search, ground_truth) triplets from a data_N folder.

Auto-detects the sample set by globbing gt_*.json rather than assuming a
fixed count -- validated against folders with both 30 and 100 samples.
"""

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Sample:
    sample_id: int
    ref: np.ndarray
    search: np.ndarray
    gt: dict
    ref_path: str
    search_path: str


def _load_gray(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"expected image not found: {path}")
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def list_sample_ids(data_dir: str) -> list[int]:
    """Discover every sample id present in a data_N folder via its gt_*.json files."""
    gt_files = sorted(glob.glob(os.path.join(data_dir, "gt_*.json")))
    if not gt_files:
        raise FileNotFoundError(f"no gt_*.json files found in {data_dir}")
    ids = []
    for path in gt_files:
        stem = os.path.splitext(os.path.basename(path))[0]  # "gt_001"
        ids.append(int(stem.split("_")[-1]))
    return sorted(ids)


def load_sample(data_dir: str, sample_id: int) -> Sample:
    """Load one (ref, search, gt) triplet by its zero-padded sample id."""
    ref_path = os.path.join(data_dir, f"ref_{sample_id:03d}.png")
    search_path = os.path.join(data_dir, f"search_{sample_id:03d}.png")
    gt_path = os.path.join(data_dir, f"gt_{sample_id:03d}.json")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"expected ground truth not found: {gt_path}")

    ref = _load_gray(ref_path)
    search = _load_gray(search_path)
    with open(gt_path, "r") as f:
        gt = json.load(f)

    return Sample(sample_id, ref, search, gt, ref_path, search_path)


def iterate_dataset(data_dir: str):
    """Yield every Sample in a data_N folder, in sample-id order."""
    for sid in list_sample_ids(data_dir):
        yield load_sample(data_dir, sid)