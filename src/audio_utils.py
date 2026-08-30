"""Audio I/O and validation utilities for TTS Baseline."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

# Amplitude sanity limits for a 16-bit-normalized float32 waveform.
_AMPLITUDE_MIN = 1e-6   # Effectively silent — likely a bug
_AMPLITUDE_MAX = 2.0    # Headroom above 1.0 to tolerate minor overshoots


class AudioValidationError(ValueError):
    """Raised when an audio waveform fails a validation check."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_audio(waveform: np.ndarray, sample_rate: int) -> None:
    """Run a suite of sanity checks on a raw waveform array.

    Parameters
    ----------
    waveform:
        1-D float32 NumPy array.
    sample_rate:
        Expected sample rate in Hz.

    Raises
    ------
    AudioValidationError
        If any check fails.
    """
    if waveform is None or waveform.size == 0:
        raise AudioValidationError("Waveform is empty.")

    if np.any(np.isnan(waveform)):
        raise AudioValidationError("Waveform contains NaN values.")

    if np.any(np.isinf(waveform)):
        raise AudioValidationError("Waveform contains Inf values.")

    if sample_rate <= 0:
        raise AudioValidationError(
            f"Invalid sample rate: {sample_rate}. Must be a positive integer."
        )

    duration = waveform.shape[0] / sample_rate
    if duration <= 0:
        raise AudioValidationError(
            f"Audio has zero or negative duration ({duration:.4f} s)."
        )

    peak = float(np.max(np.abs(waveform)))
    if peak < _AMPLITUDE_MIN:
        raise AudioValidationError(
            f"Waveform peak amplitude ({peak:.2e}) is too low — "
            "the audio may be silent or corrupted."
        )
    if peak > _AMPLITUDE_MAX:
        raise AudioValidationError(
            f"Waveform peak amplitude ({peak:.4f}) exceeds the expected "
            f"maximum ({_AMPLITUDE_MAX}). The audio may be clipped or corrupted."
        )

    logger.debug(
        "Audio validation passed: duration=%.2f s, peak=%.4f, sr=%d Hz.",
        duration,
        peak,
        sample_rate,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_audio(
    waveform: np.ndarray,
    path: str | Path,
    sample_rate: int,
) -> Path:
    """Save a waveform as a WAV file.

    Parameters
    ----------
    waveform:
        1-D float32 array of audio samples.
    path:
        Destination file path (created including missing parents).
    sample_rate:
        Sample rate in Hz.

    Returns
    -------
    Path
        Absolute path to the saved file.

    Raises
    ------
    AudioValidationError
        If the waveform fails validation.
    """
    validate_audio(waveform, sample_rate)

    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_path), waveform, sample_rate, subtype="PCM_16")
    logger.info("Audio saved to %s (%.2f s).", out_path, waveform.shape[0] / sample_rate)
    return out_path


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load a WAV (or any soundfile-readable) audio file.

    Parameters
    ----------
    path:
        Path to the audio file.
    target_sr:
        If given, resample the loaded audio to this sample rate.

    Returns
    -------
    waveform : np.ndarray
        1-D float32 array.
    sample_rate : int
        Actual sample rate of the returned waveform.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=False)

    # Convert stereo → mono if necessary.
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if target_sr is not None and sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    logger.debug("Loaded audio from %s: %d samples @ %d Hz.", file_path, len(waveform), sample_rate)
    return waveform, sample_rate


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_duration(waveform: np.ndarray, sample_rate: int) -> float:
    """Return the duration of *waveform* in seconds.

    Parameters
    ----------
    waveform:
        1-D audio sample array.
    sample_rate:
        Sample rate in Hz.

    Returns
    -------
    float
        Duration in seconds.
    """
    return waveform.shape[0] / sample_rate


def normalize_audio(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize *waveform* so its absolute maximum equals *target_peak*.

    Parameters
    ----------
    waveform:
        1-D float32 audio array.
    target_peak:
        Desired peak amplitude.  Defaults to 0.95 to leave a small headroom.

    Returns
    -------
    np.ndarray
        Normalized float32 array.
    """
    peak = float(np.max(np.abs(waveform)))
    if peak < 1e-9:
        logger.warning("normalize_audio: waveform is effectively silent — returning unchanged.")
        return waveform.copy()
    return (waveform / peak * target_peak).astype(np.float32)
