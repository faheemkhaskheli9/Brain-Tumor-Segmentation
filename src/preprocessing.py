"""Image/mask preprocessing pipeline ported from breast-tumor-segmentation.ipynb.

Pure numpy/opencv/scipy functions with no Kaggle-specific paths and no
module-level I/O or configuration loading (import is cheap and side-effect
free). Parameters that were hardcoded module-level globals in the notebook
(``input_images_size``, ``channel``) are explicit function arguments here so
callers/tests can exercise the pipeline on tiny synthetic arrays instead of
the full 256x256 real dataset.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage


def load_image(img_path: str) -> np.ndarray:
    """Load a single image from disk as grayscale (notebook cell 13)."""
    img = cv2.imread(img_path, 0)
    if img is None:
        raise FileNotFoundError(f"could not read image at {img_path!r}")
    return img


def padding(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad image and mask to a square, centered on the longer side."""
    if img.shape != msk.shape:
        raise ValueError(f"image shape {img.shape} != mask shape {msk.shape}")

    size = int(np.max(img.shape))
    offset_x = (size - img.shape[0]) // 2
    offset_y = (size - img.shape[1]) // 2

    blank_image = np.zeros((size, size), dtype=img.dtype)
    blank_mask = np.zeros((size, size), dtype=msk.dtype)

    blank_image[offset_x:offset_x + img.shape[0], offset_y:offset_y + img.shape[1]] = img
    blank_mask[offset_x:offset_x + img.shape[0], offset_y:offset_y + img.shape[1]] = msk
    return blank_image, blank_mask


def resize_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a mask with nearest/zoom interpolation (mask values must not
    be blended the way a normal image resize would blend them).

    Uses the same ``scipy.ndimage.interpolation.zoom`` call as the notebook;
    that submodule is deprecated (removal targeted for SciPy 2.0) but still
    functional on every SciPy pinned in requirements.txt.
    """
    new_size = np.array([target_size, target_size]) / mask.shape
    return scipy.ndimage.interpolation.zoom(mask, new_size)


def resize(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize an image to ``(target_size, target_size)``."""
    return cv2.resize(img, (target_size, target_size))


def preprocess(img: np.ndarray) -> np.ndarray:
    """Normalize an image from ``[0, 255]`` to ``[0, 1]``."""
    return img / 255.0


def inverse_preprocess(img: np.ndarray) -> np.ndarray:
    """Undo :func:`preprocess`."""
    return img * 255


def prepare_sample(
    img: np.ndarray,
    msk: np.ndarray,
    label_index: int,
    target_size: int = 256,
    num_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad, resize, and normalize one (image, mask) pair, and expand the
    mask into a one-hot-per-tumor-type channel layout.

    Mirrors notebook cell 13's ``load_data``: ``label_index == 0`` means
    "normal" (no tumor) and produces an all-zero mask; any other index
    activates that class's channel.
    """
    img, msk = padding(img, msk)

    msk = np.where(msk == 255, 1, msk).astype("uint8")
    img = resize(img, target_size)
    msk = resize_mask(msk, target_size)

    new_mask = np.zeros((target_size, target_size, num_classes))
    if label_index != 0:
        new_mask[:, :, label_index - 1] = msk

    img = preprocess(img)
    return img, new_mask
