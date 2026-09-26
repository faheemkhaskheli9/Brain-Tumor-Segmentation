from __future__ import annotations

import cv2
import numpy as np

from src.data_validation import validate_dataset


def _write(path, value=0, shape=(8, 8)):
    image = np.full(shape, value, dtype="uint8")
    assert cv2.imwrite(str(path), image)


def test_validate_dataset_accepts_matching_pair(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write(images / "case1.png", 100)
    _write(masks / "case1.png", 255)

    report = validate_dataset(images, masks)

    assert report.valid
    assert report.images == 1
    assert report.masks == 1
    assert report.paired_samples == 1
    assert report.empty_masks == 0


def test_validate_dataset_reports_missing_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write(images / "case1.png", 100)

    report = validate_dataset(images, masks)

    assert not report.valid
    assert report.issues[0].kind == "missing_mask"


def test_validate_dataset_reports_shape_and_values(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write(images / "shape.png", 100, (8, 8))
    _write(masks / "shape.png", 255, (4, 4))
    _write(images / "values.png", 100)
    _write(masks / "values.png", 42)

    report = validate_dataset(images, masks)

    kinds = {issue.kind for issue in report.issues}
    assert "shape_mismatch" in kinds
    assert "unexpected_mask_values" in kinds


def test_validate_dataset_counts_empty_masks(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write(images / "normal.png", 100)
    _write(masks / "normal.png", 0)

    report = validate_dataset(images, masks)

    assert report.valid
    assert report.empty_masks == 1
