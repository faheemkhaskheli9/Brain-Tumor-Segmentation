# Brain-Tumor-Segmentation

https://www.kaggle.com/code/faheemkhaskheli9/breast-tumor-segmentation

In this code i have trained segmentation model to segment tumor in breast MRI, also tumor can be either benign or malignant.

This code first merse all masks and labels, to create single mask representing tumor and its type.

Segmentation model is used to segment the tumor in Breast MRI.

# Dataset

https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

# Preprocessing

I have used Square padding, and resize both images and mask, after which images were normalized.

# Architecture

I have used UNET Architecture, with resnet as backbone.

# Setup

The notebook's stack (`segmentation_models` + TensorFlow) only installs and
runs together on **Python 3.9 or 3.10** -- see the comment at the top of
`requirements.txt` for why. `src/preprocessing.py` and `src/model.py` extract
the notebook's core image-preprocessing and model-construction logic so it's
importable and testable outside Kaggle.

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.9 or 3.10
pip install -r requirements.txt
pytest tests/
```

# Usage

`scripts/segment.py` is a CLI entrypoint that runs the pipeline end to end
(`src/preprocessing.py` + `src/model.py`) on one image and writes the
predicted tumor mask to disk. With no arguments it runs against the tiny
synthetic sample image checked into `assets/sample_images/` (no real patient
data -- see `scripts/generate_sample_assets.py`):

```bash
python scripts/segment.py
# Wrote predicted mask to examples/output/predicted_mask.png

python scripts/segment.py --image path/to/scan.png --output out/mask.png
```

A VS Code launch configuration ("Run segmentation CLI (sample image)") runs
the same entrypoint; see `.vscode/launch.json`.

See `docs/architecture.md` for the full pipeline breakdown and
`docs/evaluation.md` for metrics/result log.
