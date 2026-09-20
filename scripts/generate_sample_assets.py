"""Generate the tiny synthetic sample image/mask checked into
``assets/sample_images/``.

There is no real patient data anywhere in this repo (see README's
disclosure section) -- these two PNGs are a deterministically-seeded
synthetic grayscale "scan" with a synthetic circular "tumor" region, used
only as a default input for ``scripts/segment.py`` and in
``tests/test_cli.py`` / ``docs/`` so the CLI entrypoint has something to run
against without downloading the real (public) Kaggle dataset.

Re-run this script to regenerate the checked-in files if the synthetic
pattern below ever changes:

    python scripts/generate_sample_assets.py
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "sample_images"
IMAGE_PATH = ASSETS_DIR / "sample_scan.png"
MASK_PATH = ASSETS_DIR / "sample_mask.png"

SIZE = 128
SEED = 0


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Write ``image`` to ``path`` as a PNG via a temp file + rename, so a
    failed/interrupted write never leaves a truncated file at ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite dispatches on the file extension, so the temp file must
    # keep the real suffix (e.g. ``.tmp.png``, not ``.png.tmp``).
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        if not cv2.imwrite(str(tmp_path), image):
            raise IOError(f"cv2.imwrite failed for {tmp_path}")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def generate(size: int = SIZE, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic grayscale "scan" (speckle noise, like ultrasound)
    with one synthetic circular "tumor" region, and the matching binary
    mask for that region.
    """
    rng = np.random.default_rng(seed)

    # Speckle-noise background, roughly mimicking ultrasound texture.
    image = (rng.normal(loc=90, scale=25, size=(size, size))).clip(0, 255).astype("uint8")

    # One synthetic circular "tumor": a darker blob off-center.
    mask = np.zeros((size, size), dtype="uint8")
    center = (size // 2 + size // 8, size // 2 - size // 10)
    radius = size // 6
    cv2.circle(mask, center, radius, color=255, thickness=-1)

    tumor_region = mask == 255
    image[tumor_region] = (image[tumor_region].astype("int16") - 40).clip(0, 255).astype("uint8")

    return image, mask


def main() -> int:
    image, mask = generate()
    _atomic_write_png(IMAGE_PATH, image)
    _atomic_write_png(MASK_PATH, mask)
    print(f"Wrote {IMAGE_PATH} and {MASK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
