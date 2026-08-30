"""Unit tests for src.synthesizer — model is mocked so no GPU is required."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Config, load_config
from src.preprocessing import TextPreprocessingError
from src.synthesizer import Synthesizer, _sanitize_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(output_dir: str) -> Config:
    """Create a minimal Config pointing at a temp output directory."""
    return Config(
        {
            "model": {
                "name": "microsoft/speecht5_tts",
                "vocoder": "microsoft/speecht5_hifigan",
                "speaker_embeddings_dataset": "Matthijs/cmu-arctic-xvectors",
                "speaker_embeddings_split": "validation",
                "speaker_embeddings_index": 7306,
            },
            "audio": {
                "sample_rate": 16000,
                "output_format": "wav",
                "channels": 1,
            },
            "inference": {"device": "cpu", "batch_size": 1, "seed": 42},
            "evaluation": {
                "save_spectrograms": False,
                "calculate_latency": True,
                "calculate_rtf": True,
                "enable_asr": False,
            },
            "paths": {
                "output_audio": output_dir,
                "output_spectrograms": output_dir + "/spectrograms",
                "reports": output_dir + "/reports",
                "input_data": "data/input",
            },
        }
    )


def _make_mock_model(duration: float = 1.0, sr: int = 16000) -> MagicMock:
    """Build a TTSModel mock that returns a sine-wave waveform."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    mock_model = MagicMock()
    mock_model.is_loaded = True
    mock_model.get_device.return_value = "cpu"
    mock_model.synthesize.return_value = (
        waveform,
        {"acoustic_model_time_seconds": 0.1, "vocoder_time_seconds": 0.0},
    )
    return mock_model


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self):
        # Trailing underscores are stripped; "!" becomes "_" then stripped.
        assert _sanitize_filename("Hello world!") == "hello_world"

    def test_max_len(self):
        result = _sanitize_filename("a" * 100, max_len=10)
        assert len(result) <= 10

    def test_empty_fallback(self):
        assert _sanitize_filename("") == "audio"


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class TestSynthesizer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = _make_config(self.tmpdir)
        self.mock_model = _make_mock_model()
        self.synth = Synthesizer(self.mock_model, self.config)

    def test_synthesize_returns_metadata_dict(self):
        meta = self.synth.synthesize("Hello world!")
        required_keys = {
            "text",
            "preprocessed_text",
            "sample_rate",
            "audio_duration_seconds",
            "preprocessing_time_seconds",
            "acoustic_model_time_seconds",
            "vocoder_time_seconds",
            "total_inference_time_seconds",
            "real_time_factor",
            "device",
            "output_path",
        }
        assert required_keys.issubset(meta.keys())

    def test_output_wav_exists(self):
        meta = self.synth.synthesize("Hello world!")
        assert Path(meta["output_path"]).exists()

    def test_audio_duration_positive(self):
        meta = self.synth.synthesize("Hello world!")
        assert meta["audio_duration_seconds"] > 0

    def test_rtf_positive(self):
        meta = self.synth.synthesize("Hello world!")
        assert meta["real_time_factor"] > 0

    def test_sample_rate_matches_config(self):
        meta = self.synth.synthesize("Hello world!")
        assert meta["sample_rate"] == 16000

    def test_device_from_model(self):
        meta = self.synth.synthesize("Hello world!")
        assert meta["device"] == "cpu"

    def test_empty_text_raises(self):
        with pytest.raises(TextPreprocessingError):
            self.synth.synthesize("")

    def test_whitespace_only_raises(self):
        with pytest.raises(TextPreprocessingError):
            self.synth.synthesize("   ")

    def test_custom_output_path(self):
        out = Path(self.tmpdir) / "custom_output.wav"
        meta = self.synth.synthesize("Custom path test.", output_path=out)
        assert Path(meta["output_path"]).name == "custom_output.wav"
        assert Path(meta["output_path"]).exists()

    def test_counter_increments(self):
        self.synth.synthesize("First sentence.")
        self.synth.synthesize("Second sentence.")
        assert self.synth._counter == 2

    def test_preprocessed_text_in_metadata(self):
        meta = self.synth.synthesize("  Hello   world!  ")
        assert meta["preprocessed_text"] == "Hello world!"

    def test_original_text_preserved(self):
        original = "  Hello   world!  "
        meta = self.synth.synthesize(original)
        assert meta["text"] == original
