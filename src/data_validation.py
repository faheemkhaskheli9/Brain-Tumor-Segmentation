"""Dataset validation utilities for image/mask segmentation datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ValidationIssue:
    sample_id: str
    kind: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    images: int
    masks: int
    paired_samples: int
    empty_masks: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict:
        data = asdict(self)
        data["valid"] = self.valid
        return data


def _index_files(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    return {
        p.stem: p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    allowed_mask_values: Iterable[int] = (0, 255),
) -> ValidationReport:
    """Validate image/mask pairing and basic segmentation invariants."""
    images = _index_files(images_dir)
    masks = _index_files(masks_dir)
    allowed = set(int(v) for v in allowed_mask_values)

    issues: list[ValidationIssue] = []
    empty_masks = 0

    for sample_id in sorted(set(images) - set(masks)):
        issues.append(ValidationIssue(sample_id, "missing_mask", "image has no matching mask"))
    for sample_id in sorted(set(masks) - set(images)):
        issues.append(ValidationIssue(sample_id, "missing_image", "mask has no matching image"))

    paired_ids = sorted(set(images) & set(masks))
    for sample_id in paired_ids:
        image = cv2.imread(str(images[sample_id]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(masks[sample_id]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            issues.append(ValidationIssue(sample_id, "unreadable_image", str(images[sample_id])))
            continue
        if mask is None:
            issues.append(ValidationIssue(sample_id, "unreadable_mask", str(masks[sample_id])))
            continue
        if image.shape != mask.shape:
            issues.append(
                ValidationIssue(
                    sample_id,
                    "shape_mismatch",
                    f"image shape {image.shape} != mask shape {mask.shape}",
                )
            )
            continue

        unique_values = set(int(v) for v in np.unique(mask))
        unexpected = sorted(unique_values - allowed)
        if unexpected:
            issues.append(
                ValidationIssue(
                    sample_id,
                    "unexpected_mask_values",
                    f"unexpected values: {unexpected}",
                )
            )

        if not np.any(mask):
            empty_masks += 1

    return ValidationReport(
        images=len(images),
        masks=len(masks),
        paired_samples=len(paired_ids),
        empty_masks=empty_masks,
        issues=tuple(issues),
    )
