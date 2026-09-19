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
