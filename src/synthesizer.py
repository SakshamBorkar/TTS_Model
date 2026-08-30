"""High-level synthesis pipeline for TTS Baseline.

Orchestrates preprocessing → acoustic model → vocoder → WAV.
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio_utils import get_duration, save_audio, validate_audio
from src.config import Config
from src.model import TTSModel
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    """Create a filesystem-safe slug from the first words of *text*.

    Parameters
    ----------
    text:
        Preprocessed text string.
    max_len:
        Maximum character length of the slug.

    Returns
    -------
    str
        Lowercase, underscore-separated slug.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = slug.strip("_")[:max_len].rstrip("_")
    return slug or "audio"


class Synthesizer:
    """End-to-end TTS pipeline wrapper.

    Parameters
    ----------
    model:
        Pre-loaded :class:`~src.model.TTSModel` instance.
    config:
        Project :class:`~src.config.Config` instance.
    """

    def __init__(self, model: TTSModel, config: Config) -> None:
        self._model = model
        self._config = config
        self._counter: int = 0  # Sequential sample counter for filenames.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        """Convert *text* to speech and save the result as a WAV file.

        Pipeline
        --------
        1. Validate and normalise text.
        2. Run acoustic model + vocoder.
        3. Validate waveform.
        4. Save WAV.
        5. Return metadata dict.

        Parameters
        ----------
        text:
            Raw input text (will be preprocessed internally).
        output_path:
            Optional explicit path for the output WAV file.  If omitted a
            filename is generated automatically under the configured output
            directory.

        Returns
        -------
        dict
            Metadata dictionary with the following keys:

            * ``text`` — original input text
            * ``preprocessed_text`` — text after normalization
            * ``sample_rate`` — audio sample rate (Hz)
            * ``audio_duration_seconds`` — duration of synthesised audio
            * ``preprocessing_time_seconds``
            * ``acoustic_model_time_seconds``
            * ``vocoder_time_seconds``
            * ``total_inference_time_seconds``
            * ``real_time_factor`` — inference_time / audio_duration
            * ``device`` — device used for inference
            * ``output_path`` — absolute path of the saved WAV

        Raises
        ------
        src.preprocessing.TextPreprocessingError
            If the input text is invalid.
        src.audio_utils.AudioValidationError
            If the generated waveform is invalid.
        """
        self._counter += 1
        original_text = text

        # 1. Preprocessing
        t_pre_start = time.perf_counter()
        preprocessed = preprocess_text(text)
        t_pre_end = time.perf_counter()
        preprocessing_time = t_pre_end - t_pre_start

        logger.info("[%04d] Synthesising: %r", self._counter, preprocessed)

        # 2. Model inference
        waveform, timings = self._model.synthesize(preprocessed)

        # 3. Validate
        sample_rate: int = int(self._config.audio.sample_rate)
        validate_audio(waveform, sample_rate)

        # 4. Save
        if output_path is None:
            output_path = self._build_output_path(preprocessed)

        saved_path = save_audio(waveform, output_path, sample_rate)

        # 5. Assemble metadata
        audio_duration = get_duration(waveform, sample_rate)
        acoustic_time = timings["acoustic_model_time_seconds"]
        vocoder_time = timings["vocoder_time_seconds"]
        total_inference = acoustic_time + vocoder_time
        rtf = total_inference / audio_duration if audio_duration > 0 else float("inf")

        metadata = {
            "text": original_text,
            "preprocessed_text": preprocessed,
            "sample_rate": sample_rate,
            "audio_duration_seconds": round(audio_duration, 4),
            "preprocessing_time_seconds": round(preprocessing_time, 6),
            "acoustic_model_time_seconds": round(acoustic_time, 6),
            "vocoder_time_seconds": round(vocoder_time, 6),
            "total_inference_time_seconds": round(total_inference, 6),
            "real_time_factor": round(rtf, 4),
            "device": self._model.get_device(),
            "output_path": str(saved_path),
        }

        logger.info(
            "[%04d] Done - duration=%.2f s, RTF=%.3f, saved to %s.",
            self._counter,
            audio_duration,
            rtf,
            saved_path,
        )
        return metadata

    def synthesize_waveform(self, text: str) -> tuple[np.ndarray, dict]:
        """Synthesize speech into in-memory waveform without writing to disk.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        tuple[np.ndarray, dict]
            (waveform, metadata)
        """
        self._counter += 1
        original_text = text

        t_pre_start = time.perf_counter()
        preprocessed = preprocess_text(text)
        t_pre_end = time.perf_counter()
        preprocessing_time = t_pre_end - t_pre_start

        logger.info("[%04d] Synthesising (in-memory): %r", self._counter, preprocessed)

        waveform, timings = self._model.synthesize(preprocessed)

        sample_rate: int = int(self._config.audio.sample_rate)
        validate_audio(waveform, sample_rate)

        audio_duration = get_duration(waveform, sample_rate)
        acoustic_time = timings["acoustic_model_time_seconds"]
        vocoder_time = timings["vocoder_time_seconds"]
        total_inference = acoustic_time + vocoder_time
        rtf = total_inference / audio_duration if audio_duration > 0 else float("inf")

        metadata = {
            "text": original_text,
            "preprocessed_text": preprocessed,
            "sample_rate": sample_rate,
            "audio_duration_seconds": round(audio_duration, 4),
            "preprocessing_time_seconds": round(preprocessing_time, 6),
            "acoustic_model_time_seconds": round(acoustic_time, 6),
            "vocoder_time_seconds": round(vocoder_time, 6),
            "total_inference_time_seconds": round(total_inference, 6),
            "real_time_factor": round(rtf, 4),
            "device": self._model.get_device(),
        }
        return waveform, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_output_path(self, preprocessed_text: str) -> Path:
        """Generate an automatic output path for the WAV file.

        Parameters
        ----------
        preprocessed_text:
            Clean text used to derive a descriptive filename.

        Returns
        -------
        Path
            Full path including the configured output directory.
        """
        output_dir = Path(self._config.paths.output_audio)
        output_dir.mkdir(parents=True, exist_ok=True)

        slug = _sanitize_filename(preprocessed_text)
        filename = f"{self._counter:04d}_{slug}.wav"
        return output_dir / filename
