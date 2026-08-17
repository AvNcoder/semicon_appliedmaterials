"""
Standalone dataset generator for Drift-Sense.

Creates synthetic SEM reference + search image pairs with ground-truth centers.

Usage:
    python generate_dataset.py --style dram_6f2 --num 30 --out ./my_data
    python generate_dataset.py --style finfet_sram --num 50 --out ./Fixed/data_3
    python generate_dataset.py --style beol_interconnect --num 10 --out ./tmp_beol
    python generate_dataset.py --style dram_octagonal --num 400 --out ./Fixed/data_1

Styles:
    dram_octagonal   - Orthogonal word-line / bit-line DRAM
    dram_6f2         - 6F² oblique active-moat DRAM
    finfet_sram      - Parallel fins + gate bars
    beol_interconnect- Dual-layer M1/M2 + self-aligned vias
"""

import argparse
import os
import sys

# Make the Fixed/ generators importable regardless of where this script is launched
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIXED_DIR = os.path.join(_THIS_DIR, "Fixed")
if _FIXED_DIR not in sys.path:
    sys.path.insert(0, _FIXED_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense synthetic SEM pair generator"
    )
    parser.add_argument(
        "--style",
        required=True,
        choices=["dram_octagonal", "dram_6f2", "finfet_sram", "beol_interconnect"],
        help="Architecture style to generate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="Number of reference/search pairs to generate (default: 30)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory (created if it does not exist)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.style == "dram_octagonal":
        from fixed_noise_data import generate_dram_sem_dataset
        generate_dram_sem_dataset(output_dir=args.out, num_samples=args.num)

    elif args.style == "dram_6f2":
        from fixed_noise_data_dram import generate_dram_sem_dataset
        generate_dram_sem_dataset(output_dir=args.out, num_samples=args.num)

    elif args.style == "finfet_sram":
        from fixed_noise_data_finfet6tsram import generate_finfet_sem_dataset
        generate_finfet_sem_dataset(output_dir=args.out, num_samples=args.num)

    elif args.style == "beol_interconnect":
        from fixed_noise_data_beol_interconnect import generate_beol_sem_dataset
        generate_beol_sem_dataset(output_dir=args.out, num_samples=args.num)

    print(f"\nDone. {args.num} pairs written to: {os.path.abspath(args.out)}")
    print("Each pair: ref_XXX.png  +  search_XXX.png  +  gt_XXX.json")


if __name__ == "__main__":
    main()
