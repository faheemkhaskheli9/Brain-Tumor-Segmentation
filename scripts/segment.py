#!/usr/bin/env python
"""CLI entrypoint: run the breast-ultrasound tumor segmentation pipeline on
one image.

    python scripts/segment.py
    python scripts/segment.py --image path/to/scan.png --output out/mask.png

With no arguments it runs against the tiny synthetic sample image checked
into ``assets/sample_images/`` (see ``scripts/generate_sample_assets.py`` --
there is no real patient data anywhere in this repo) and writes the
predicted mask to ``examples/output/predicted_mask.png``.

Building the real U-Net + ResNet backbone model requires the full
TensorFlow/segmentation_models stack pinned in requirements.txt, which only
installs on Python 3.9/3.10 (see the comment at the top of that file). This
script is a thin wrapper so ``src/cli.run()`` -- the actual preprocess/
predict/write logic -- stays importable and unit-testable without that
heavy stack (see tests/test_cli.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/segment.py` to find the `src` package without
# installing this repo as one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
