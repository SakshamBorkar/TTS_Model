"""Unit tests for src.audio_utils."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.audio_utils import (
    AudioValidationError,
    get_duration,
    load_audio,
    normalize_audio,
    save_audio,
    validate_audio,
)

SAMPLE_RATE = 16000


def _make_waveform(duration: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """Create a sine-wave test waveform."""
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# validate_audio
# ---------------------------------------------------------------------------


class TestValidateAudio:
    def test_passes_valid_waveform(self):
        waveform = _make_waveform()
        validate_audio(waveform, SAMPLE_RATE)  # should not raise

    def test_raises_empty_waveform(self):
        with pytest.raises(AudioValidationError, match="empty"):
            validate_audio(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_raises_nan_values(self):
        waveform = _make_waveform()
        waveform[10] = float("nan")
        with pytest.raises(AudioValidationError, match="NaN"):
            validate_audio(waveform, SAMPLE_RATE)

    def test_raises_inf_values(self):
        waveform = _make_waveform()
        waveform[10] = float("inf")
        with pytest.raises(AudioValidationError, match="Inf"):
            validate_audio(waveform, SAMPLE_RATE)

    def test_raises_invalid_sample_rate(self):
        waveform = _make_waveform()
        with pytest.raises(AudioValidationError, match="sample rate"):
            validate_audio(waveform, 0)

    def test_raises_amplitude_too_low(self):
        waveform = np.full(SAMPLE_RATE, 1e-10, dtype=np.float32)
        with pytest.raises(AudioValidationError, match="low"):
            validate_audio(waveform, SAMPLE_RATE)

    def test_raises_amplitude_too_high(self):
        waveform = np.full(SAMPLE_RATE, 5.0, dtype=np.float32)
        with pytest.raises(AudioValidationError, match="exceed"):
            validate_audio(waveform, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# save_audio / load_audio
# ---------------------------------------------------------------------------


class TestSaveLoadAudio:
    def test_save_creates_file(self):
        waveform = _make_waveform()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.wav"
            saved = save_audio(waveform, out, SAMPLE_RATE)
            assert saved.exists()

    def test_save_load_roundtrip(self):
        waveform = _make_waveform(duration=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.wav"
            save_audio(waveform, out, SAMPLE_RATE)
            loaded, sr = load_audio(out)
            assert sr == SAMPLE_RATE
            assert loaded.shape[0] == waveform.shape[0]
            # PCM_16 quantization introduces small error (~1e-5)
            np.testing.assert_allclose(loaded, waveform, atol=1e-4)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_save_creates_parent_dirs(self):
        waveform = _make_waveform()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c" / "test.wav"
            saved = save_audio(waveform, nested, SAMPLE_RATE)
            assert saved.exists()


# ---------------------------------------------------------------------------
# get_duration
# ---------------------------------------------------------------------------


class TestGetDuration:
    def test_correct_duration(self):
        waveform = _make_waveform(duration=2.5)
        assert abs(get_duration(waveform, SAMPLE_RATE) - 2.5) < 1e-6


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------


class TestNormalizeAudio:
    def test_peak_is_target(self):
        waveform = _make_waveform(amplitude=0.3)
        normalized = normalize_audio(waveform, target_peak=0.95)
        assert abs(float(np.max(np.abs(normalized))) - 0.95) < 1e-5

    def test_silent_input_returned_unchanged(self):
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = normalize_audio(silent)
        np.testing.assert_array_equal(result, silent)
