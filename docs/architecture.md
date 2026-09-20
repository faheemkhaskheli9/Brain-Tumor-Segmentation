# Architecture

## Pipeline

```
input scan (grayscale PNG/JPG)
   |
   v
src/preprocessing.py
   load_image()          -- cv2.imread(..., grayscale)
   padding()              -- pad image+mask to a square, centered
   resize() / resize_mask() -- resize to model input size (default 256x256)
   preprocess()           -- normalize [0, 255] -> [0, 1]
   |
   v
src/model.py  build_unet()
   segmentation_models.Unet(backbone="resnet34", classes=2, activation="sigmoid")
   compiled with Dice + BinaryFocal loss, IoU/F-score metrics
   |
   v
src/cli.py  run()
   model.predict(batch) -> per-class sigmoid mask
   threshold > 0.5, merge tumor-type channels -> single binary mask
   atomic write to a PNG (temp file + os.replace)
   |
   v
scripts/segment.py  (CLI entrypoint, see README "Usage")
   predicted_mask.png
```

## Components

- **`src/preprocessing.py`** -- pure numpy/opencv/scipy functions ported
  from the notebook's cell 13 `load_data`. No module-level I/O; every
  parameter that was a notebook global (`input_images_size`, `channel`) is
  an explicit function argument so it's testable on tiny synthetic arrays.
- **`src/model.py`** -- `build_unet()`, the notebook's cell 18 model
  construction (U-Net + ResNet34 backbone via `segmentation_models`,
  `encoder_weights=None` by default so building the model never triggers a
  pretrained-weights download). Imports `tensorflow`/`segmentation_models`
  at module scope, deliberately kept out of `src/preprocessing.py` so that
  module stays importable without the heavy stack.
- **`src/cli.py`** -- `run()` (preprocess one image, call an injected
  `predict_fn`, threshold, atomically write the mask PNG) and `main()` (the
  real CLI: builds the actual model via `build_unet` and wires it into
  `run()`). `run()` takes `predict_fn` as a parameter specifically so
  `tests/test_cli.py` can exercise the whole I/O path with a fake predictor
  and never needs to import `tensorflow`.
- **`scripts/segment.py`** -- thin argv-parsing wrapper around
  `src.cli.main()`; this is what `.vscode/launch.json`'s "Run segmentation
  CLI" configuration points at.
- **`scripts/generate_sample_assets.py`** -- deterministically (seeded)
  generates the synthetic grayscale "scan" + circular "tumor" mask checked
  into `assets/sample_images/`, used as the CLI's default input. No real
  patient data anywhere in this repo -- see the README's Disclosure section.

## Design notes / boundaries mocked in tests

- The pinned TensorFlow 2.9 + `segmentation_models` stack (see
  `requirements.txt` for why it's pinned this tightly) only installs
  together on Python 3.9/3.10. `tests/test_cli.py` and
  `tests/test_notebook_dependencies.py`'s preprocessing tests run on any
  Python version by never importing that stack directly; only
  `test_segmentation_model_forward_and_train_step` (which does) is skipped
  -- not silently passed -- when the stack isn't importable.
- All tests run on tiny synthetic in-memory arrays or the checked-in sample
  image, CPU-only, no GPU, no pretrained-weight download, no paid API.

## Relationship to `medical-imaging-suite`

`medical-imaging-suite` (a sibling portfolio repo, a Django-based multi-app
medical imaging platform) is porting this repo's U-Net + ResNet-backbone
architecture into its own `breast_tumor_segmentation` feature app, registered
under that suite's `imaging_core` model registry as an `InputKind.IMAGE_2D`
option alongside its existing plain-UNet `xray_segmentation` feature. See
`medical-imaging-suite` issue #6 ("Port UNet+ResNet-backbone 2D architecture
from Brain-Tumor-Segmentation") for that work. This repo remains the
original, standalone notebook-derived implementation; the suite's version is
a from-scratch reimplementation against its own registry/config conventions,
not a copy of this repo's code.
