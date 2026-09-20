#!/usr/bin/env python
"""Build the checked-in example-output preview shown in the README.

Combines the synthetic sample scan/mask (``assets/sample_images/``, always
present) with the model's predicted mask for that same sample
(``examples/output/predicted_mask.png``, produced by ``scripts/segment.py``
and gitignored -- regenerate it before running this script) into one
side-by-side montage, written to ``docs/sample_output.png`` (checked in, so
it renders on GitHub without anyone having to run the pipeline first).

    python scripts/segment.py               # writes examples/output/predicted_mask.png
    python scripts/generate_result_preview.py

Note: the model is built with ``encoder_weights=None`` and there is no
trained-weights checkpoint or real dataset in this repo (see README's
disclosure section and docs/evaluation.md) -- this preview demonstrates the
preprocess -> predict -> write pipeline running end to end, not
segmentation accuracy. A real Dice/IoU number requires training against the
actual (public) Kaggle dataset outside this repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preview import build_preview_montage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_PATH = REPO_ROOT / "assets" / "sample_images" / "sample_scan.png"
MASK_PATH = REPO_ROOT / "assets" / "sample_images" / "sample_mask.png"
PREDICTED_PATH = REPO_ROOT / "examples" / "output" / "predicted_mask.png"
OUTPUT_PATH = REPO_ROOT / "docs" / "sample_output.png"


def _atomic_write_png(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        if not cv2.imwrite(str(tmp_path), image):
            raise IOError(f"cv2.imwrite failed for {tmp_path}")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    if not PREDICTED_PATH.is_file():
        print(
            f"{PREDICTED_PATH} not found -- run `python scripts/segment.py` "
            "first to generate it.",
            file=sys.stderr,
        )
        return 1

    scan = cv2.imread(str(SCAN_PATH), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    predicted = cv2.imread(str(PREDICTED_PATH), cv2.IMREAD_GRAYSCALE)

    montage = build_preview_montage(scan, mask, predicted)
    _atomic_write_png(OUTPUT_PATH, montage)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
