"""Build a side-by-side scan / ground-truth-mask / predicted-mask montage.

Pure array logic only (no file I/O, no TensorFlow import) so it stays
testable without the heavy TF/segmentation_models stack -- see
``src/cli.py``'s module docstring for why that separation matters in this
repo.
"""
from __future__ import annotations

import cv2
import numpy as np

_GAP_PX = 8
_GAP_VALUE = 255


def build_preview_montage(
    scan: np.ndarray,
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> np.ndarray:
    """Resize ``scan`` and ``ground_truth_mask`` to ``predicted_mask``'s
    shape and concatenate scan | ground_truth_mask | predicted_mask
    horizontally, with a thin white gap between panels, into one grayscale
    preview image.

    All three inputs must be single-channel (2D) grayscale arrays.
    """
    if scan.ndim != 2 or ground_truth_mask.ndim != 2 or predicted_mask.ndim != 2:
        raise ValueError("all three inputs must be 2D grayscale arrays")

    # cv2.resize takes (width, height); predicted_mask.shape is (height, width).
    target_size = (predicted_mask.shape[1], predicted_mask.shape[0])
    scan_resized = cv2.resize(scan, target_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(ground_truth_mask, target_size, interpolation=cv2.INTER_NEAREST)

    gap = np.full((predicted_mask.shape[0], _GAP_PX), _GAP_VALUE, dtype="uint8")
    panels = [
        scan_resized.astype("uint8"),
        gap,
        mask_resized.astype("uint8"),
        gap,
        predicted_mask.astype("uint8"),
    ]
    return np.concatenate(panels, axis=1)
