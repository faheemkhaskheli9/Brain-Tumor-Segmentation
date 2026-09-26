from __future__ import annotations

import json

import pytest

from src.configuration import load_config


def test_load_config_reads_model_and_training_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {"backbone": "resnet50", "input_size": 128},
                "training": {"epochs": 5, "batch_size": 2, "seed": 7},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.backbone == "resnet50"
    assert config.model.input_size == 128
    assert config.model.classes == 2
    assert config.training.epochs == 5
    assert config.training.batch_size == 2
    assert config.training.seed == 7


def test_load_config_uses_safe_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    config = load_config(path)

    assert config.model.architecture == "unet"
    assert config.model.backbone == "resnet34"
    assert config.training.learning_rate == 1e-5


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"input_size": 0}},
        {"model": {"classes": 0}},
        {"training": {"epochs": 0}},
        {"training": {"batch_size": -1}},
        {"training": {"learning_rate": 0}},
    ],
)
def test_load_config_rejects_invalid_numeric_values(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"not_a_setting": 1}}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid configuration key"):
        load_config(path)
