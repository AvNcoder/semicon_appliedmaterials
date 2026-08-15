"""
Drift-Sense standalone predictor.

Given ANY search image and reference image (1000x1000 grayscale each), this
prints/returns the predicted center (x, y) of the reference pattern inside
the search image, along with the confidence score, matched flag, and
inference time -- with no dependency on this project's own Fixed/data_N
folder layout or ground-truth files. This is the exact input/output contract
the hackathon FAQ specifies: "Well documented Python file generating the
center x,y results on a given pair of images 1k*1k (search, reference)."

Usage (command line):
    python predict.py --search path/to/search.png --reference path/to/ref.png
    python predict.py --search search.png --reference ref.png --json
    python predict.py --csv pairs.csv --out results.csv

CSV mode expects a header row:
    wide_search_image_path,reference_image_path
and writes:
    wide_search_image_path,reference_image_path,pred_x,pred_y,score,matched,time_ms

Coordinate convention: (0, 0) is the top-left pixel of the search image,
x increases rightward, y increases downward -- standard image-array
convention, matching the "[0,0 is Top Left]" spec.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from driftsense.localize import localize


def load_grayscale(path: str) -> np.ndarray:
    """Load any image file as a float32 grayscale array. Works on paths
    outside this project's own dataset folders -- no assumptions about
    directory structure, naming convention, or accompanying GT files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def predict_pair(search_path: str, reference_path: str) -> dict:
    """The core contract: (search image path, reference image path) -> result dict.
    This is the function a grading harness should import and call directly if
    it prefers not to shell out to the CLI."""
    search = load_grayscale(search_path)
    reference = load_grayscale(reference_path)

    result = localize(reference, search)

    return {
        "wide_search_image_path": search_path,
        "reference_image_path": reference_path,
        "pred_x": round(result["x"], 3),
        "pred_y": round(result["y"], 3),
        "score": round(result["score"], 4),
        "scale": round(result["scale"], 4),
        "angle": round(result["angle"], 2),
        "matched": result["matched"],
        "time_ms": round(result["time_ms"], 2),
    }


def run_csv(csv_in: str, csv_out: str) -> None:
    with open(csv_in, newline="") as f:
        reader = csv.DictReader(f)
        rows_out = []
        for row in reader:
            search_path = row["wide_search_image_path"].strip()
            ref_path = row["reference_image_path"].strip()
            try:
                rows_out.append(predict_pair(search_path, ref_path))
            except Exception as e:
                rows_out.append({
                    "wide_search_image_path": search_path,
                    "reference_image_path": ref_path,
                    "pred_x": None, "pred_y": None, "score": None,
                    "scale": None, "angle": None, "matched": False,
                    "time_ms": None, "error": str(e),
                })

    fieldnames = list(rows_out[0].keys()) if rows_out else [
        "wide_search_image_path", "reference_image_path", "pred_x", "pred_y",
        "score", "scale", "angle", "matched", "time_ms"
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {len(rows_out)} predictions -> {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense standalone predictor")
    parser.add_argument("--search", type=str, help="path to the wide search image")
    parser.add_argument("--reference", type=str, help="path to the high-mag reference image")
    parser.add_argument("--json", action="store_true", help="print result as JSON")
    parser.add_argument("--csv", type=str, help="batch mode: input CSV with wide_search_image_path,reference_image_path columns")
    parser.add_argument("--out", type=str, default="predictions.csv", help="output CSV path for --csv mode")
    args = parser.parse_args()

    if args.csv:
        run_csv(args.csv, args.out)
        return

    if not args.search or not args.reference:
        parser.error("either --csv, or both --search and --reference, are required")

    result = predict_pair(args.search, args.reference)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"center: ({result['pred_x']}, {result['pred_y']})")
        print(f"score: {result['score']}   matched: {result['matched']}   time_ms: {result['time_ms']}")


if __name__ == "__main__":
    main()