"""Evaluation metrics (CER, WER, Comparative Benchmarks) and Human Evaluation Template generator."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.metrics import compute_statistics, compute_wer

logger = logging.getLogger(__name__)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between *reference* and *hypothesis*.

    Parameters
    ----------
    reference:
        The ground truth text.
    hypothesis:
        The transcript produced by an ASR system.

    Returns
    -------
    float
        CER >= 0.0 (0.0 means identical character sequences).

    Notes
    -----
    CER measures fine-grained character-level substitutions, deletions, and insertions.
    """
    try:
        import jiwer
        if hasattr(jiwer, "cer"):
            return float(jiwer.cer(reference, hypothesis))
    except Exception:
        pass

    # Levenshtein distance fallback on character level
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return float(dp[n][m] / max(len(ref), 1))


def generate_human_evaluation_template(
    sample_pairs: List[Dict[str, Any]],
    output_path: str = "outputs/reports/human_evaluation_template.csv",
) -> pd.DataFrame:
    """Generate a template CSV for double-blind or paired MOS human evaluation (1-5 scale).

    Parameters
    ----------
    sample_pairs:
        List of dicts containing sample_id, text, baseline_audio_path, and finetuned_audio_path.
    output_path:
        Destination CSV path.
    """
    rows = []
    for pair in sample_pairs:
        sample_id = pair.get("sample_id", "sample")
        text = pair.get("text", "")
        # Baseline row
        rows.append({
            "sample_id": f"{sample_id}_baseline",
            "model_type": "baseline",
            "text": text,
            "audio_file": pair.get("baseline_audio", ""),
            "naturalness (1-5)": "",
            "intelligibility (1-5)": "",
            "speaker_similarity (1-5)": "",
            "evaluator_comments": "",
        })
        # Finetuned row
        rows.append({
            "sample_id": f"{sample_id}_finetuned",
            "model_type": "finetuned",
            "text": text,
            "audio_file": pair.get("finetuned_audio", ""),
            "naturalness (1-5)": "",
            "intelligibility (1-5)": "",
            "speaker_similarity (1-5)": "",
            "evaluator_comments": "",
        })

    df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)
    logger.info("Human evaluation template saved to %s (%d rows)", out, len(df))
    return df
