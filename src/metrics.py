"""Metrics computation and aggregation for TTS Baseline benchmarks."""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency & RTF statistics
# ---------------------------------------------------------------------------


def compute_statistics(values: Sequence[float]) -> dict:
    """Compute descriptive statistics for a sequence of floats.

    Parameters
    ----------
    values:
        Non-empty sequence of numeric values.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``p50``, ``p95``, ``min``, ``max``.

    Raises
    ------
    ValueError
        If *values* is empty.
    """
    if not values:
        raise ValueError("Cannot compute statistics on an empty sequence.")

    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_results(records: list[dict]) -> dict:
    """Aggregate a list of per-sample metadata records into summary statistics.

    Parameters
    ----------
    records:
        Each element is the metadata dict returned by
        :meth:`~src.synthesizer.Synthesizer.synthesize`.

    Returns
    -------
    dict
        Nested dict keyed by metric name → statistics dict.
    """
    fields = [
        "preprocessing_time_seconds",
        "acoustic_model_time_seconds",
        "vocoder_time_seconds",
        "total_inference_time_seconds",
        "real_time_factor",
        "audio_duration_seconds",
    ]

    summary: dict = {}
    for field in fields:
        vals = [r[field] for r in records if field in r]
        if vals:
            summary[field] = compute_statistics(vals)
        else:
            logger.warning("No values found for field %r — skipping.", field)

    return summary


# ---------------------------------------------------------------------------
# WER (ASR-based evaluation)
# ---------------------------------------------------------------------------


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between *reference* and *hypothesis*.

    Requires ``jiwer`` to be installed.

    Parameters
    ----------
    reference:
        The original input text.
    hypothesis:
        The transcript produced by an ASR system.

    Returns
    -------
    float
        WER in the range ``[0, ∞)``.  0 = perfect match.

    Notes
    -----
    WER measures whether synthesised speech preserves recognisable linguistic
    content / intelligibility.  It is **not** a complete measure of TTS
    naturalness or subjective speech quality.
    """
    try:
        import jiwer

        transformation = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        wer = jiwer.wer(
            reference,
            hypothesis,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )
        return float(wer)
    except ImportError as exc:
        raise ImportError(
            "jiwer is required for WER computation. "
            "Install it with: pip install jiwer"
        ) from exc


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def results_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert a list of metadata records to a :class:`pandas.DataFrame`.

    Parameters
    ----------
    records:
        Per-sample metadata dicts (output of :meth:`Synthesizer.synthesize`
        plus optional ``sample_id`` and ``wer`` keys).

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(records)


def save_report(df: pd.DataFrame, path: str) -> None:
    """Save *df* to a CSV file, creating parent directories as needed.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination CSV path.
    """
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)
    logger.info("Report saved to %s (%d rows).", out, len(df))


def print_summary(summary: dict, device: str = "unknown") -> None:
    """Print a human-readable benchmark summary to stdout.

    Parameters
    ----------
    summary:
        Output of :func:`aggregate_results`.
    device:
        Device label (e.g. ``"cpu"`` or ``"cuda:0"``).
    """
    divider = "-" * 66

    print(f"\n{divider}")
    print(f"  TTS BASELINE - BENCHMARK SUMMARY  ({device.upper()})")
    print(divider)

    labels = {
        "preprocessing_time_seconds": "Pre-processing latency (s)",
        "acoustic_model_time_seconds": "Acoustic model latency (s)",
        "vocoder_time_seconds": "Vocoder latency (s)",
        "total_inference_time_seconds": "Total inference latency (s)",
        "real_time_factor": "Real-time factor (RTF)",
        "audio_duration_seconds": "Audio duration (s)",
    }

    fmt = "{:<35}  {:>8}  {:>8}  {:>8}  {:>8}"
    print(fmt.format("Metric", "Mean", "P50", "P95", "Max"))
    print("-" * 66)

    for key, label in labels.items():
        if key not in summary:
            continue
        s = summary[key]
        print(
            fmt.format(
                label,
                f"{s['mean']:.4f}",
                f"{s['p50']:.4f}",
                f"{s['p95']:.4f}",
                f"{s['max']:.4f}",
            )
        )

    print(divider + "\n")
