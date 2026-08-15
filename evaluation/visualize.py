"""
Renders GT-vs-predicted overlay images for visual inspection.

Two entrypoints:
  render_failures(style)      -- draws every sample listed in that style's
                                  summary.json failure_sample_ids, saves into
                                  results/<style>/failures/
  render_sample(style, id)    -- draws any single sample on demand (success
                                  or failure), useful for spot-checking

Draws GT as a dashed green box (the answer key) and the prediction as a
solid box: red if matched=False or error > 5px, cyan if a correct match --
so a failure image is visually obvious even without reading numbers.
"""

import json
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from driftsense import config
from driftsense.io_utils import load_sample
from driftsense.localize import localize
from evaluation.metrics import pixel_error

GT_COLOR = (0, 200, 0)       # green, BGR
PRED_OK_COLOR = (255, 255, 0)   # cyan, BGR
PRED_FAIL_COLOR = (0, 0, 255)   # red, BGR


def _draw_dashed_rect(img, pt1, pt2, color, dash_len=8, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    for (a, b) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                   ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        dist = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        if dist == 0:
            continue
        for i in range(0, dist, dash_len * 2):
            t0, t1 = i / dist, min(1.0, (i + dash_len) / dist)
            p0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            p1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, p0, p1, color, thickness)


def render_overlay(search_gray: np.ndarray, gt: dict, result: dict, out_path: str, embed_size: int = None):
    embed_size = config.EMBED_SIZE if embed_size is None else embed_size
    half = embed_size / 2.0

    canvas = cv2.cvtColor(np.clip(search_gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    gx, gy = gt["center_x"], gt["center_y"]
    _draw_dashed_rect(canvas, (int(gx - half), int(gy - half)), (int(gx + half), int(gy + half)), GT_COLOR)
    cv2.drawMarker(canvas, (int(gx), int(gy)), GT_COLOR, cv2.MARKER_CROSS, 14, 2)

    err = pixel_error(result["x"], result["y"], gx, gy)
    is_fail = (not result["matched"]) or (err > config.SUCCESS_TOLERANCES_PX[-1])
    pred_color = PRED_FAIL_COLOR if is_fail else PRED_OK_COLOR

    px, py = result["x"], result["y"]
    cv2.rectangle(canvas, (int(px - half), int(py - half)), (int(px + half), int(py + half)), pred_color, 2)
    cv2.drawMarker(canvas, (int(px), int(py)), pred_color, cv2.MARKER_TILTED_CROSS, 14, 2)

    label = f"err={err:.1f}px score={result['score']:.2f} matched={result['matched']}"
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    return out_path

def render_sample(style: str, sample_id: int, out_path: str = None, project_root: str = None) -> str:
    """Render any single sample on demand -- success or failure. Used by `cli.py show`.
    Writes to results/<style>/previews/ by default, NOT failures/ -- a manually
    requested render is not necessarily a failure, and mixing the two folders
    makes failures/ misleading (a perfect match showing up where only genuine
    failures should live)."""
    project_root = _PROJECT_ROOT if project_root is None else project_root
    data_dir = os.path.join(project_root, "Fixed", config.DATA_DIR_FOR_STYLE[style])
    sample = load_sample(data_dir, sample_id)
    result = localize(sample.ref, sample.search)

    if out_path is None:
        out_path = os.path.join(project_root, "results", style, "previews", f"sample_{sample_id:03d}.png")
    return render_overlay(sample.search, sample.gt, result, out_path)


def render_failures(style: str, project_root: str = None) -> list[str]:
    project_root = _PROJECT_ROOT if project_root is None else project_root
    summary_path = os.path.join(project_root, "results", style, "summary.json")
    if not os.path.exists(summary_path):
        print(f"[{style}] no summary.json -- run evaluate.py first")
        return []

    with open(summary_path) as f:
        summary = json.load(f)

    failure_ids = summary.get("failure_sample_ids", [])
    if not failure_ids:
        print(f"[{style}] 0 failures -- nothing to render")
        return []

    written = []
    for sid in failure_ids:
        # explicit failures/ path here, independent of render_sample's own default
        out_path = os.path.join(project_root, "results", style, "failures", f"sample_{sid:03d}.png")
        written.append(render_sample(style, sid, out_path=out_path, project_root=project_root))
    print(f"[{style}] rendered {len(written)} failure overlay(s)")
    return written


if __name__ == "__main__":
    for style in config.STYLES:
        render_failures(style)