"""Unit tests for src/preview.py's montage-building logic.

No TensorFlow import needed here -- see src/preview.py's module docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preview import build_preview_montage


def test_montage_shape_matches_predicted_mask_height_and_combined_width():
    scan = np.full((16, 16), 50, dtype="uint8")
    mask = np.full((16, 16), 0, dtype="uint8")
    predicted = np.full((32, 32), 200, dtype="uint8")

    montage = build_preview_montage(scan, mask, predicted)

    gap = 8
    assert montage.shape == (32, 32 * 3 + gap * 2)


def test_montage_right_panel_is_predicted_mask_unchanged():
    scan = np.zeros((16, 16), dtype="uint8")
    mask = np.zeros((16, 16), dtype="uint8")
    predicted = np.arange(32 * 32, dtype="uint8").reshape(32, 32)

    montage = build_preview_montage(scan, mask, predicted)

    assert np.array_equal(montage[:, -32:], predicted)


def test_montage_left_panel_is_resized_scan():
    scan = np.full((16, 16), 123, dtype="uint8")
    mask = np.zeros((16, 16), dtype="uint8")
    predicted = np.zeros((32, 32), dtype="uint8")

    montage = build_preview_montage(scan, mask, predicted)

    # Uniform input resized stays uniform, regardless of interpolation.
    assert np.array_equal(montage[:, :32], np.full((32, 32), 123, dtype="uint8"))


def test_rejects_non_2d_input():
    scan = np.zeros((16, 16, 3), dtype="uint8")
    mask = np.zeros((16, 16), dtype="uint8")
    predicted = np.zeros((32, 32), dtype="uint8")

    with pytest.raises(ValueError):
        build_preview_montage(scan, mask, predicted)
