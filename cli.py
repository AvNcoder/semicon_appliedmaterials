"""
Single entrypoint for the Drift-Sense pipeline.

    python cli.py evaluate                    # run all 4 styles
    python cli.py evaluate --style dram_octagonal
    python cli.py visualize                   # render failure overlays, all styles
    python cli.py visualize --style dram_octagonal
    python cli.py show --style dram_octagonal --sample 1   # render one sample on demand

Run from anywhere -- same sys.path bootstrap as evaluate.py/visualize.py.
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from driftsense import config
from evaluation.evaluate import evaluate_all_styles, evaluate_style
from evaluation.visualize import render_failures, render_sample


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense localization pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="run localize() over dataset(s), write results.csv + summary.json")
    p_eval.add_argument("--style", choices=config.STYLES, default=None,
                         help="evaluate one style only; default: all 4")

    p_vis = sub.add_parser("visualize", help="render overlay images for failed samples")
    p_vis.add_argument("--style", choices=config.STYLES, default=None,
                        help="visualize one style only; default: all 4")

    p_show = sub.add_parser("show", help="render one specific sample, success or failure")
    p_show.add_argument("--style", choices=config.STYLES, required=True)
    p_show.add_argument("--sample", type=int, required=True)

    args = parser.parse_args()

    if args.command == "evaluate":
        if args.style:
            evaluate_style(args.style)
        else:
            evaluate_all_styles()

    elif args.command == "visualize":
        styles = [args.style] if args.style else config.STYLES
        for style in styles:
            render_failures(style)

    elif args.command == "show":
        out = render_sample(args.style, args.sample)
        print(f"written -> {out}")


if __name__ == "__main__":
    main()