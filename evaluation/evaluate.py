"""
Runs localize() over every sample in a data_N folder and writes:
  results/<style>/results.csv     -- per-sample predictions + error + timing
  results/<style>/summary.json    -- aggregate metrics, incl. failure_sample_ids
  results/<style>/failures/       -- created empty here; populated only by
                                      evaluation.visualize.render_failures(),
                                      never by this script directly
  results/<style>/previews/       -- created on demand by visualize.render_sample()
                                      (via `cli.py show`) for any single sample,
                                      success or failure -- kept separate from
                                      failures/ so that folder only ever holds
                                      genuine matched=False / error>5px cases

Run this directly (VS Code Run button included) OR as `python -m evaluation.evaluate`.
The sys.path bootstrap below makes both work regardless of the working directory
Code Runner launches with -- same class of fix as the earlier os.path.dirname
anchor used in fixed_noise_data.py, applied here to imports instead of file paths.
"""

import csv
import json
import os
import sys
import time

# --- sys.path bootstrap: make `driftsense` and `evaluation` importable no
# matter what directory this script is launched from. -----------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # SEMICON/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from driftsense import config
from driftsense.io_utils import iterate_dataset
from driftsense.localize import localize
from evaluation.metrics import pixel_error, summarize


def evaluate_style(style: str, project_root: str = None) -> dict:
    """Evaluate one style (e.g. 'dram_octagonal') and write results.csv + summary.json.
    Pre-creates an empty failures/ dir; visualize.render_failures() fills it later --
    this function never writes images into it itself."""
    """Evaluate one style (e.g. 'dram_octagonal') and write results/summary/failures."""
    project_root = _PROJECT_ROOT if project_root is None else project_root
    data_dir = os.path.join(project_root, "Fixed", config.DATA_DIR_FOR_STYLE[style])
    results_dir = os.path.join(project_root, "results", style)
    failures_dir = os.path.join(results_dir, "failures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)

    rows = []
    errors, times_ms, matched_flags = [], [], []
    failure_ids = []

    samples = list(iterate_dataset(data_dir))   # materialize once so we know N
    n = len(samples)
    t_start = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        result = localize(sample.ref, sample.search)
        err = pixel_error(result["x"], result["y"], sample.gt["center_x"], sample.gt["center_y"])

        errors.append(err)
        times_ms.append(result["time_ms"])
        matched_flags.append(result["matched"])

        is_failure = (not result["matched"]) or (err > config.SUCCESS_TOLERANCES_PX[-1])
        if is_failure:
            failure_ids.append(sample.sample_id)

        rows.append({
            "sample_id": sample.sample_id,
            "pred_x": round(result["x"], 2),
            "pred_y": round(result["y"], 2),
            "gt_x": sample.gt["center_x"],
            "gt_y": sample.gt["center_y"],
            "error_px": round(err, 3),
            "scale": round(result["scale"], 4),
            "score": round(result["score"], 4),
            "matched": result["matched"],
            "time_ms": round(result["time_ms"], 2),
        })

        # --- progress line (minimal) ---
        elapsed = time.perf_counter() - t_start
        pct = 100.0 * i / n
        eta = (elapsed / i) * (n - i) if i > 0 else 0.0
        print(f"\r[{style}] {i}/{n} ({pct:5.1f}%)  "
              f"elapsed={elapsed:6.1f}s  eta={eta:5.1f}s  "
              f"last={result['time_ms']:.0f}", end="", flush=True)

    print()  # newline after the \r progress line

    csv_path = os.path.join(results_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(errors, times_ms, matched_flags)
    summary["style"] = style
    summary["failure_sample_ids"] = failure_ids
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{style}] {summary['num_samples']} samples | "
      f"mean_err={summary['mean_error_px']:.2f}px | "
      f"success@5px={summary['success_at_px'][5]:.1f}% | "
      f"mean_time={summary['mean_time_ms']:.0f}ms | "
      f"failures={len(failure_ids)} | -> {results_dir}")
    
    return {"summary": summary, "failure_ids": failure_ids, "data_dir": data_dir, "results_dir": results_dir}


def evaluate_all_styles(project_root: str = None) -> dict:
    project_root = _PROJECT_ROOT if project_root is None else project_root
    overall = {}
    for style in config.STYLES:
        data_dir = os.path.join(project_root, "Fixed", config.DATA_DIR_FOR_STYLE[style])
        if not os.path.isdir(data_dir):
            print(f"[{style}] skipped -- {data_dir} not found")
            continue
        overall[style] = evaluate_style(style, project_root)

    overall_path = os.path.join(project_root, "results", "overall_summary.json")
    with open(overall_path, "w") as f:
        json.dump({s: r["summary"] for s, r in overall.items()}, f, indent=2)
    print(f"\noverall summary -> {overall_path}")
    return overall


if __name__ == "__main__":
    evaluate_all_styles()