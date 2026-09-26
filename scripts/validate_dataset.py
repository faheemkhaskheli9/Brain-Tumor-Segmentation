"""Validate a segmentation dataset before training."""
from __future__ import annotations

import argparse
import json

from src.data_validation import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate image/mask dataset integrity.")
    parser.add_argument("--images", required=True, help="Directory containing input images.")
    parser.add_argument("--masks", required=True, help="Directory containing segmentation masks.")
    parser.add_argument(
        "--allowed-mask-values",
        nargs="+",
        type=int,
        default=[0, 255],
        help="Allowed grayscale values in masks (default: 0 255).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(args.images, args.masks, args.allowed_mask_values)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
