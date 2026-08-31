"""Unit tests for src.dataset (validation, statistics, and deterministic splits)."""

import math
import numpy as np
import pytest
import torch

from src.dataset import (
    compute_dataset_statistics,
    create_deterministic_splits,
    validate_audio_tensor,
    validate_sample,
)


class TestAudioValidation:
    def test_valid_mono_audio(self):
        # 2 seconds 16kHz sine wave
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        wav = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        res = validate_audio_tensor(wav, sample_rate=16000)
        assert res.is_valid is True
        assert res.error_type is None
        assert math.isclose(res.duration, 2.0, rel_tol=1e-3)

    def test_rejects_empty_audio(self):
        res = validate_audio_tensor(np.array([], dtype=np.float32), sample_rate=16000)
        assert res.is_valid is False
        assert res.error_type == "EMPTY_AUDIO"

    def test_rejects_nan_inf_audio(self):
        wav = np.array([0.1, np.nan, 0.5, 0.2], dtype=np.float32)
        res = validate_audio_tensor(wav, sample_rate=16000)
        assert res.is_valid is False
        assert res.error_type == "NON_FINITE_VALUES"

    def test_rejects_too_short_audio(self):
        # 0.2s duration (below 0.5s min)
        wav = np.zeros(3200, dtype=np.float32) + 0.1
        res = validate_audio_tensor(wav, sample_rate=16000, min_duration=0.5)
        assert res.is_valid is False
        assert res.error_type == "TOO_SHORT"

    def test_rejects_too_long_audio(self):
        # 12s duration (above 10s max)
        wav = np.zeros(192000, dtype=np.float32) + 0.1
        res = validate_audio_tensor(wav, sample_rate=16000, max_duration=10.0)
        assert res.is_valid is False
        assert res.error_type == "TOO_LONG"

    def test_rejects_multichannel_stereo_audio(self):
        # 2 channels
        wav = np.zeros((2, 32000), dtype=np.float32) + 0.1
        res = validate_audio_tensor(wav, sample_rate=16000)
        assert res.is_valid is False
        assert res.error_type == "INVALID_CHANNELS"


class TestSampleValidation:
    def test_validates_clean_sample(self):
        t = np.linspace(0, 1.5, 24000, endpoint=False)
        wav = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        is_valid, norm_text, err_type, err_desc, dur = validate_sample(
            sample_id="test_01",
            raw_text="   Hello, this is a test transcript!   ",
            audio_path_or_array=wav,
            sample_rate=16000,
        )
        assert is_valid is True
        assert norm_text == "Hello, this is a test transcript!"
        assert err_type is None

    def test_rejects_empty_transcript(self):
        wav = np.zeros(16000, dtype=np.float32) + 0.1
        is_valid, norm_text, err_type, err_desc, dur = validate_sample(
            sample_id="test_02",
            raw_text="   \n\t   ",
            audio_path_or_array=wav,
            sample_rate=16000,
        )
        assert is_valid is False
        assert err_type in ("INVALID_TRANSCRIPT", "EMPTY_TRANSCRIPT")


class TestDeterministicSplits:
    def test_disjoint_and_deterministic_splits(self):
        records = [
            {"sample_id": f"S_{i:03d}", "normalized_text": f"Text {i}", "duration": 1.5}
            for i in range(100)
        ]

        splits1 = create_deterministic_splits(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        splits2 = create_deterministic_splits(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

        # Exact reproducibility
        assert [r["sample_id"] for r in splits1["train"]] == [r["sample_id"] for r in splits2["train"]]
        assert [r["sample_id"] for r in splits1["val"]] == [r["sample_id"] for r in splits2["val"]]
        assert [r["sample_id"] for r in splits1["test"]] == [r["sample_id"] for r in splits2["test"]]

        # Disjointness check
        train_ids = {r["sample_id"] for r in splits1["train"]}
        val_ids = {r["sample_id"] for r in splits1["val"]}
        test_ids = {r["sample_id"] for r in splits1["test"]}

        assert len(train_ids) == 80
        assert len(val_ids) == 10
        assert len(test_ids) == 10
        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0

    def test_rejects_invalid_split_sum(self):
        records = [{"sample_id": "s1", "normalized_text": "text", "duration": 1.0}]
        with pytest.raises(ValueError, match="sum to 1.0"):
            create_deterministic_splits(records, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


class TestDatasetStatistics:
    def test_computes_stats_correctly(self):
        records = [
            {"sample_id": "1", "normalized_text": "short", "duration": 1.0},
            {"sample_id": "2", "normalized_text": "medium text", "duration": 2.0},
            {"sample_id": "3", "normalized_text": "longer text sample", "duration": 3.0},
        ]
        stats = compute_dataset_statistics(records)
        assert stats["sample_count"] == 3
        assert stats["total_duration_seconds"] == 6.0
        assert stats["avg_duration_seconds"] == 2.0
        assert stats["min_duration_seconds"] == 1.0
        assert stats["max_duration_seconds"] == 3.0
