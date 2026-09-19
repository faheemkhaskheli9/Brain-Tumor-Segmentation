"""Model construction ported from breast-tumor-segmentation.ipynb cell 18.

Imports ``tensorflow``/``segmentation_models`` at module scope (a real,
necessary dependency of this module, not a hidden global side effect) so
that importing :mod:`src.preprocessing` alone never requires the heavy
TF/Keras/segmentation_models stack. Only import this module where you
actually need to build the model.
"""
from __future__ import annotations

import segmentation_models as sm
import tensorflow as tf

sm.set_framework("tf.keras")


def build_unet(
    input_size: int = 256,
    channels: int = 1,
    classes: int = 2,
    backbone: str = "resnet34",
    encoder_weights: str | None = None,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Build and compile the same U-Net + ResNet-backbone segmentation model
    as the notebook (``encoder_weights=None`` by default so building the
    model never triggers a pretrained-weights download).
    """
    model = sm.Unet(
        backbone,
        classes=classes,
        activation="sigmoid",
        input_shape=(input_size, input_size, channels),
        encoder_weights=encoder_weights,
    )

    optim = tf.keras.optimizers.Adam(learning_rate)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + focal_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, total_loss, metrics)
    return model
