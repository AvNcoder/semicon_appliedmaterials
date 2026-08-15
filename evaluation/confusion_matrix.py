"""
Builds a genuine confusion matrix at each tolerance in config.SUCCESS_TOLERANCES_PX.

This is different from success_at_px in metrics.py: success_at_px only ever
sees POSITIVE pairs (a real match is guaranteed to exist), so it can't
report false positives or true negatives -- there's no negative class in
that test set. Here, negatives are constructed by pairing a reference from
one sample with the search image from a DIFFERENT sample (same style), so
the reference genuinely does not appear -- verified individually earlier
(a deliberately mismatched pair scored 0.13, well under MIN_SCORE=0.35).

Confusion categories, per pair, at tolerance T:
  TP: real match exists, matched=True,  error <= T   (found it correctly)
  FN: real match exists, matched=False OR error > T  (missed a real match)
  FP: no real match,     matched=True                (falsely claimed a match)
  TN: no real match,     matched=False               (correctly refused)
"""

import json
import os
import random
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from driftsense import config
from driftsense.io_utils import list_sample_ids, load_sample
from driftsense.localize import localize
from evaluation.metrics import pixel_error


def build_negative_pairs(data_dir: str, num_negatives: int, seed: int = 0) -> list[tuple[int, int]]:
    """Return (ref_id, search_id) pairs where ref_id != search_id -- the
    reference from sample A genuinely does not appear in sample B's search
    image, since each sample embeds a unique-featured patch at its own
    random location."""
    ids = list_sample_ids(data_dir)
    rng = random.Random(seed)
    pairs = set()
    attempts = 0
    while len(pairs) < num_negatives and attempts < num_negatives * 20:
        a, b = rng.choice(ids), rng.choice(ids)
        attempts += 1
        if a != b:
            pairs.add((a, b))
    return list(pairs)


def evaluate_confusion_matrix(style: str, project_root: str = None, num_negatives: int = None) -> dict:
    project_root = _PROJECT_ROOT if project_root is None else project_root
    data_dir = os.path.join(project_root, "Fixed", config.DATA_DIR_FOR_STYLE[style])
    ids = list_sample_ids(data_dir)
    num_negatives = len(ids) if num_negatives is None else num_negatives

    samples = {sid: load_sample(data_dir, sid) for sid in ids}

    records = []  # (is_real_match, result_dict, error_or_none)

    # Positives: genuine (ref_i, search_i) pairs
    for sid in ids:
        s = samples[sid]
        result = localize(s.ref, s.search)
        err = pixel_error(result["x"], result["y"], s.gt["center_x"], s.gt["center_y"])
        records.append({"is_real_match": True, "result": result, "error": err})

    # Negatives: (ref_a, search_b) where a != b -- no real match exists
    neg_pairs = build_negative_pairs(data_dir, num_negatives)
    for a, b in neg_pairs:
        ref = samples[a].ref
        search = samples[b].search
        result = localize(ref, search)
        records.append({"is_real_match": False, "result": result, "error": None})

    matrices = {}
    for tol in config.SUCCESS_TOLERANCES_PX:
        tp = fn = fp = tn = 0
        for r in records:
            matched = r["result"]["matched"]
            if r["is_real_match"]:
                found_correctly = matched and (r["error"] <= tol)
                if found_correctly:
                    tp += 1
                else:
                    fn += 1
            else:
                if matched:
                    fp += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        matrices[tol] = {
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
        }

    summary = {
        "style": style,
        "num_positive_pairs": len(ids),
        "num_negative_pairs": len(neg_pairs),
        "confusion_matrix_by_tolerance_px": matrices,
    }

    results_dir = os.path.join(project_root, "results", style)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "confusion_matrix.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{style}] {len(ids)} positive + {len(neg_pairs)} negative pairs -> {out_path}")
    for tol, m in matrices.items():
        print(f"    @{tol}px  TP={m['TP']:3d} FN={m['FN']:3d} FP={m['FP']:3d} TN={m['TN']:3d}  "
              f"precision={m['precision']}  recall={m['recall']}")

    return summary


def evaluate_all_styles(project_root: str = None) -> dict:
    project_root = _PROJECT_ROOT if project_root is None else project_root
    overall = {}
    for style in config.STYLES:
        data_dir = os.path.join(project_root, "Fixed", config.DATA_DIR_FOR_STYLE[style])
        if not os.path.isdir(data_dir):
            print(f"[{style}] skipped -- {data_dir} not found")
            continue
        overall[style] = evaluate_confusion_matrix(style, project_root)

    out_path = os.path.join(project_root, "results", "overall_confusion_matrix.json")
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\noverall confusion matrix -> {out_path}")
    return overall


if __name__ == "__main__":
    evaluate_all_styles()