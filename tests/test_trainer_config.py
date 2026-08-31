"""Unit tests for trainer configuration validation."""

import pytest

from src.config import Config
from src.trainer import validate_training_config


class TestTrainerConfigValidation:
    def test_valid_config(self):
        cfg = Config({
            "training": {
                "learning_rate": 1e-5,
                "batch_size": 4,
                "epochs": 5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "gradient_accumulation_steps": 2,
            },
            "data": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
            },
        })
        validate_training_config(cfg)  # Should not raise

    def test_rejects_negative_learning_rate(self):
        cfg = Config({
            "training": {"learning_rate": -1e-4, "batch_size": 4, "epochs": 5},
            "data": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        })
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_training_config(cfg)

    def test_rejects_zero_batch_size(self):
        cfg = Config({
            "training": {"learning_rate": 1e-4, "batch_size": 0, "epochs": 5},
            "data": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        })
        with pytest.raises(ValueError, match="Batch size must be a positive integer"):
            validate_training_config(cfg)

    def test_rejects_negative_epochs(self):
        cfg = Config({
            "training": {"learning_rate": 1e-4, "batch_size": 4, "epochs": -1},
            "data": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        })
        with pytest.raises(ValueError, match="Epochs must be a positive integer"):
            validate_training_config(cfg)

    def test_rejects_invalid_split_ratios(self):
        cfg = Config({
            "training": {"learning_rate": 1e-4, "batch_size": 4, "epochs": 5},
            "data": {"train_ratio": 0.6, "val_ratio": 0.1, "test_ratio": 0.1},
        })
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_training_config(cfg)
