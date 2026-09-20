"""CLI orchestration for running the segmentation pipeline end-to-end on a
single image: load -> pad/resize/normalize (``src/preprocessing.py``) ->
predict -> threshold -> write mask PNG.

``run()`` takes the model as an injected ``predict_fn`` callable rather than
building one itself, so it (and the tests that exercise it) never need to
import ``tensorflow``/``segmentation_models`` -- only ``main()`` (the real
CLI path) does that, via a deferred import of :mod:`src.model`. This keeps
``pytest tests/`` green even in environments where the pinned TF/
segmentation_models stack can't be installed (see requirements.txt).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .preprocessing import load_image, padding, preprocess, resize

PredictFn = Callable[[np.ndarray], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "assets" / "sample_images" / "sample_scan.png"
DEFAULT_MASK = REPO_ROOT / "assets" / "sample_images" / "sample_mask.png"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "output" / "predicted_mask.png"


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Write ``image`` to ``path`` as a PNG via a temp file + ``os.replace``,
    so an interrupted write never leaves a truncated/partial file at the
    final path (see repo-wide robustness convention: atomic writes only).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        if not cv2.imwrite(str(tmp_path), image):
            raise IOError(f"cv2.imwrite failed for {tmp_path}")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def run(
    image_path: str | Path,
    mask_path: str | Path | None,
    output_path: str | Path,
    predict_fn: PredictFn,
    input_size: int = 256,
) -> np.ndarray:
    """Preprocess one image, run ``predict_fn`` on it, threshold the result
    into a binary tumor mask, and write that mask to ``output_path`` as an
    8-bit PNG.

    ``mask_path`` is optional and only used the way the notebook's own
    ``padding()`` needs a same-shape array to pad against; pass ``None`` to
    run against an image with no ground-truth mask available (the real
    inference use case).

    ``predict_fn`` must accept a ``(1, input_size, input_size, 1)`` float32
    batch and return a ``(1, input_size, input_size, C)`` array (this is
    exactly ``keras.Model.predict``'s signature for the model built by
    :func:`src.model.build_unet`, so the real CLI path passes
    ``model.predict`` directly).

    Returns the raw prediction array (before thresholding) for callers/tests
    that want to inspect it directly.
    """
    img = load_image(str(image_path))
    msk = load_image(str(mask_path)) if mask_path is not None else np.zeros_like(img)

    img, _ = padding(img, msk)
    img = resize(img, input_size)
    img = preprocess(img)

    batch = img[np.newaxis, ..., np.newaxis].astype("float32")
    prediction = predict_fn(batch)
    prediction = np.asarray(prediction)

    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise ValueError(
            f"predict_fn must return a (1, H, W, C) array, got shape {prediction.shape}"
        )

    # Merge all tumor-type channels into one binary mask for the PNG preview.
    binary_mask = (prediction[0].max(axis=-1) > 0.5).astype("uint8") * 255

    _atomic_write_png(Path(output_path), binary_mask)
    return prediction


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the breast-ultrasound tumor segmentation pipeline "
            "(U-Net + ResNet backbone) on one image and write the predicted "
            "tumor mask to disk."
        )
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help=(
            "Path to a grayscale scan image. Defaults to the tiny synthetic "
            "sample checked into assets/sample_images/ (no real patient "
            "data; see scripts/generate_sample_assets.py)."
        ),
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional path to a ground-truth mask, same shape as --image.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the predicted mask PNG.",
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--backbone", default="resnet34")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Deferred import: only the real CLI path needs the heavy TF/
    # segmentation_models stack (see module docstring).
    from .model import build_unet

    model = build_unet(
        input_size=args.input_size,
        channels=1,
        classes=2,
        backbone=args.backbone,
        encoder_weights=None,
    )

    run(
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.output,
        predict_fn=lambda batch: model.predict(batch, verbose=0),
        input_size=args.input_size,
    )
    print(f"Wrote predicted mask to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
