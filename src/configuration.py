"""Configuration loading for reproducible segmentation experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "unet"
    backbone: str = "resnet34"
    input_size: int = 256
    channels: int = 1
    classes: int = 2
    encoder_weights: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-5
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    return data


def _validate(config: ExperimentConfig) -> None:
    if config.model.input_size <= 0:
        raise ValueError("model.input_size must be > 0")
    if config.model.channels <= 0:
        raise ValueError("model.channels must be > 0")
    if config.model.classes <= 0:
        raise ValueError("model.classes must be > 0")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a JSON experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    raw = _require_mapping(raw, "root")
    model_raw = _require_mapping(raw.get("model", {}), "model")
    training_raw = _require_mapping(raw.get("training", {}), "training")

    try:
        config = ExperimentConfig(
            model=ModelConfig(**model_raw),
            training=TrainingConfig(**training_raw),
        )
    except TypeError as exc:
        raise ValueError(f"invalid configuration key: {exc}") from exc

    _validate(config)
    return config
