"""Spectrogram and waveform visualization utilities for TTS Baseline."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in server environments.
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

logger = logging.getLogger(__name__)


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a time-domain waveform.

    Parameters
    ----------
    waveform:
        1-D float32 audio sample array.
    sample_rate:
        Sample rate in Hz.
    title:
        Figure title.
    output_path:
        If provided, save the figure to this path (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    duration = waveform.shape[0] / sample_rate
    time_axis = np.linspace(0, duration, num=waveform.shape[0])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, waveform, linewidth=0.5, color="#3d85c8")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


def plot_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Mel Spectrogram",
    output_path: str | Path | None = None,
    n_mels: int = 80,
    fmax: int = 8000,
) -> plt.Figure:
    """Plot a mel-frequency spectrogram.

    Parameters
    ----------
    waveform:
        1-D float32 audio sample array.
    sample_rate:
        Sample rate in Hz.
    title:
        Figure title.
    output_path:
        If provided, save the figure to this path (PNG).
    n_mels:
        Number of mel filter banks.
    fmax:
        Maximum frequency for the mel scale (Hz).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        fmax=fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        fmax=fmax,
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)

    return fig


def save_visualizations(
    waveform: np.ndarray,
    sample_rate: int,
    stem: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate and save both waveform and mel-spectrogram plots.

    Parameters
    ----------
    waveform:
        1-D float32 audio sample array.
    sample_rate:
        Sample rate in Hz.
    stem:
        Filename stem used for both output files (without extension).
    output_dir:
        Directory to save the PNG files.

    Returns
    -------
    dict
        Keys ``"waveform"`` and ``"spectrogram"`` mapping to saved paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform_path = out_dir / f"{stem}_waveform.png"
    spec_path = out_dir / f"{stem}_mel_spectrogram.png"

    wfig = plot_waveform(
        waveform,
        sample_rate,
        title=f"Waveform — {stem}",
        output_path=waveform_path,
    )
    plt.close(wfig)

    sfig = plot_mel_spectrogram(
        waveform,
        sample_rate,
        title=f"Mel Spectrogram — {stem}",
        output_path=spec_path,
    )
    plt.close(sfig)

    return {"waveform": waveform_path, "spectrogram": spec_path}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    logger.info("Figure saved to %s.", out)
