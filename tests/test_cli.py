"""Tests for src/cli.py -- the segment.py entrypoint's core logic.

These exercise ``run()`` with a fake ``predict_fn`` (a plain callable, no
tensorflow/segmentation_models import needed) so they stay green in every
environment, matching the module's design goal of keeping the CLI's I/O and
preprocessing wiring testable without the heavy pinned stack.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.cli import DEFAULT_IMAGE, DEFAULT_MASK, build_arg_parser, run


def _constant_predict_fn(value: float, classes: int = 2):
    def predict_fn(batch: np.ndarray) -> np.ndarray:
        n, h, w, _ = batch.shape
        return np.full((n, h, w, classes), value, dtype="float32")

    return predict_fn


def test_default_sample_assets_exist() -> None:
    """The CLI's --image/--mask defaults must point at real, checked-in
    files -- catches the sample assets being renamed/moved/deleted without
    updating src/cli.py.
    """
    assert DEFAULT_IMAGE.is_file(), f"missing default sample image: {DEFAULT_IMAGE}"
    assert DEFAULT_MASK.is_file(), f"missing default sample mask: {DEFAULT_MASK}"


def test_run_writes_thresholded_mask_png(tmp_path) -> None:
    """A predict_fn that always returns > 0.5 should produce an
    all-255 mask PNG; the reverse for < 0.5. Also proves run() returns the
    raw (pre-threshold) prediction array.
    """
    output_path = tmp_path / "mask.png"

    prediction = run(
        image_path=DEFAULT_IMAGE,
        mask_path=DEFAULT_MASK,
        output_path=output_path,
        predict_fn=_constant_predict_fn(0.9),
        input_size=32,
    )

    assert prediction.shape == (1, 32, 32, 2)
    assert output_path.is_file()

    import cv2

    written_mask = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert written_mask.shape == (32, 32)
    assert (written_mask == 255).all()


def test_run_without_ground_truth_mask(tmp_path) -> None:
    """mask_path=None (the real inference use case: no ground truth
    available) must still work end to end.
    """
    output_path = tmp_path / "mask.png"
    prediction = run(
        image_path=DEFAULT_IMAGE,
        mask_path=None,
        output_path=output_path,
        predict_fn=_constant_predict_fn(0.1),
        input_size=16,
    )
    assert prediction.shape == (1, 16, 16, 2)
    assert output_path.is_file()

    import cv2

    written_mask = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert (written_mask == 0).all()


def test_run_is_atomic_on_predict_failure(tmp_path) -> None:
    """If predict_fn raises mid-run, no partial/truncated output file (and
    no leftover temp file) should be left behind at the target path.
    """
    output_path = tmp_path / "mask.png"

    def failing_predict_fn(batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated model failure")

    with pytest.raises(RuntimeError):
        run(
            image_path=DEFAULT_IMAGE,
            mask_path=None,
            output_path=output_path,
            predict_fn=failing_predict_fn,
            input_size=16,
        )

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_run_rejects_wrong_shaped_prediction(tmp_path) -> None:
    """predict_fn returning something that isn't a (1, H, W, C) batch must
    raise loudly rather than writing a nonsense mask.
    """
    output_path = tmp_path / "mask.png"

    def bad_predict_fn(batch: np.ndarray) -> np.ndarray:
        return np.zeros((2, 3))  # wrong ndim

    with pytest.raises(ValueError):
        run(
            image_path=DEFAULT_IMAGE,
            mask_path=None,
            output_path=output_path,
            predict_fn=bad_predict_fn,
            input_size=16,
        )
    assert not output_path.exists()


def test_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])
    assert args.image == str(DEFAULT_IMAGE)
    assert args.mask is None
    assert args.input_size == 256
    assert args.backbone == "resnet34"


def test_arg_parser_overrides() -> None:
    args = build_arg_parser().parse_args(
        ["--image", "a.png", "--mask", "b.png", "--output", "c.png", "--input-size", "64"]
    )
    assert args.image == "a.png"
    assert args.mask == "b.png"
    assert args.output == "c.png"
    assert args.input_size == 64
