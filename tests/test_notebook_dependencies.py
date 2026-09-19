"""Regression tests for the pinned dependency stack in requirements.txt.

These are not "does the file exist" checks. They:

1. Verify the versions actually installed in the current environment satisfy
   every pin in requirements.txt (catches requirements.txt drifting out of
   sync with what was actually verified to work).
2. Exercise the notebook's real preprocessing pipeline (src/preprocessing.py)
   on tiny synthetic arrays.
3. Build the notebook's real segmentation model (src/model.py, U-Net +
   ResNet34 backbone via `segmentation_models`) and run one forward pass and
   one training step on a tiny synthetic image, proving the pinned
   TensorFlow + segmentation_models versions actually work together end to
   end -- not just that each imports in isolation.

Test 3 needs the full TensorFlow/segmentation_models stack pinned in
requirements.txt, which (see the comment at the top of requirements.txt)
only installs together on Python 3.9/3.10. It is skipped -- not silently
passed -- when that stack is not importable, e.g. under a newer Python where
`segmentation_models` cannot be installed at all.
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement

from src.preprocessing import prepare_sample

REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"

# Maps a requirements.txt distribution name to the importlib.metadata
# distribution name, for the handful where they differ from the PyPI name
# used in requirements.txt (they already match, so this is only here in
# case a future edit adds one that doesn't).
_DIST_NAME_OVERRIDES: dict[str, str] = {}


def _parsed_requirements() -> list[Requirement]:
    reqs = []
    for line in REQUIREMENTS_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        reqs.append(Requirement(line))
    return reqs


@pytest.mark.parametrize("requirement", _parsed_requirements(), ids=lambda r: r.name)
def test_installed_version_satisfies_pin(requirement: Requirement) -> None:
    """Every pin in requirements.txt must be satisfied by what's installed.

    This fails loudly if requirements.txt is edited without re-verifying
    against a real install, instead of silently drifting.
    """
    dist_name = _DIST_NAME_OVERRIDES.get(requirement.name, requirement.name)
    try:
        installed_version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        pytest.fail(
            f"{dist_name!r} is pinned in requirements.txt but not installed. "
            f"Run `pip install -r requirements.txt` first."
        )
    assert installed_version in requirement.specifier, (
        f"installed {dist_name}=={installed_version} does not satisfy "
        f"pinned requirement {requirement}"
    )


def test_preprocessing_pipeline_on_synthetic_sample() -> None:
    """The real padding/resize/normalize pipeline on a tiny synthetic
    non-square image+mask, proving numpy/opencv/scipy work together as
    pinned (not just that each imports).
    """
    rng = np.random.default_rng(0)
    img = (rng.random((20, 12)) * 255).astype("uint8")
    msk = np.zeros((20, 12), dtype="uint8")
    msk[5:10, 3:7] = 255  # a synthetic "tumor" region

    target_size = 32
    out_img, out_mask = prepare_sample(img, msk, label_index=1, target_size=target_size, num_classes=2)

    assert out_img.shape == (target_size, target_size)
    assert out_mask.shape == (target_size, target_size, 2)
    # preprocess() normalizes to [0, 1]
    assert out_img.min() >= 0.0
    assert out_img.max() <= 1.0
    # label_index=1 activates channel 0 only, and the mask must have picked
    # up some of the synthetic tumor region rather than being all zero.
    assert out_mask[:, :, 0].sum() > 0
    assert out_mask[:, :, 1].sum() == 0


def test_preprocessing_normal_label_has_empty_mask() -> None:
    """label_index=0 ("normal", no tumor) must produce an all-zero mask,
    matching the notebook's class convention (0=normal, 1=benign, 2=malignant).
    """
    img = np.zeros((16, 16), dtype="uint8")
    msk = np.zeros((16, 16), dtype="uint8")
    _, out_mask = prepare_sample(img, msk, label_index=0, target_size=16, num_classes=2)
    assert out_mask.sum() == 0


def _segmentation_stack_unavailable_reason() -> str | None:
    if sys.version_info >= (3, 11):
        return (
            "segmentation_models hard-imports Keras-2-only internals that no "
            "longer exist on the Keras bundled with TensorFlow builds "
            "available for Python >= 3.11; see requirements.txt for the "
            "verified Python 3.9/3.10 + TensorFlow 2.9 combination."
        )
    try:
        import segmentation_models  # noqa: F401
        import tensorflow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"segmentation_models/tensorflow stack not importable: {exc!r}"
    return None


@pytest.mark.skipif(
    _segmentation_stack_unavailable_reason() is not None,
    reason=str(_segmentation_stack_unavailable_reason()),
)
def test_segmentation_model_forward_and_train_step() -> None:
    """Build the notebook's real U-Net (ResNet34 backbone, no pretrained
    weights downloaded) and run one predict + one train_on_batch step on a
    tiny synthetic image, proving the pinned TensorFlow + segmentation_models
    versions actually work together for the notebook's core training call.
    """
    from src.model import build_unet

    size = 32
    model = build_unet(input_size=size, channels=1, classes=2, encoder_weights=None)

    rng = np.random.default_rng(0)
    x = rng.random((1, size, size, 1)).astype("float32")
    y = model.predict(x, verbose=0)

    assert y.shape == (1, size, size, 2)
    assert np.isfinite(y).all()
    assert y.min() >= 0.0 and y.max() <= 1.0  # sigmoid activation

    y_true = rng.integers(0, 2, size=(1, size, size, 2)).astype("float32")
    loss = model.train_on_batch(x, y_true)
    loss_value = loss[0] if isinstance(loss, (list, tuple)) else loss
    assert np.isfinite(loss_value)
