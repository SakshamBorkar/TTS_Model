"""Unit tests for src.evaluation (CER, comparison metrics, and human evaluation templates)."""

import math
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.evaluation import compute_cer, generate_human_evaluation_template


class TestEvaluation:
    def test_cer_identical_strings(self):
        assert compute_cer("hello world", "hello world") == 0.0

    def test_cer_single_substitution(self):
        # "cat" vs "bat" -> 1 substitution / 3 characters = 0.3333...
        cer = compute_cer("cat", "bat")
        assert math.isclose(cer, 1 / 3, rel_tol=1e-3)

    def test_cer_single_deletion(self):
        # "cats" vs "cat" -> 1 deletion / 4 characters = 0.25
        cer = compute_cer("cats", "cat")
        assert math.isclose(cer, 0.25, rel_tol=1e-3)

    def test_cer_single_insertion(self):
        # "cat" vs "cats" -> 1 insertion / 3 characters = 0.3333...
        cer = compute_cer("cat", "cats")
        assert math.isclose(cer, 1 / 3, rel_tol=1e-3)

    def test_cer_empty_strings(self):
        assert compute_cer("", "") == 0.0
        assert compute_cer("hello", "") == 1.0

    def test_generate_human_evaluation_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "human_eval.csv"
            pairs = [
                {
                    "sample_id": "S001",
                    "text": "Hello world",
                    "baseline_audio": "audio/base1.wav",
                    "finetuned_audio": "audio/fine1.wav",
                }
            ]
            df = generate_human_evaluation_template(pairs, output_path=str(out_file))
            assert len(df) == 2  # Baseline row + Finetuned row
            assert out_file.exists()

            loaded_df = pd.read_csv(out_file)
            assert "naturalness (1-5)" in loaded_df.columns
            assert "intelligibility (1-5)" in loaded_df.columns
            assert "speaker_similarity (1-5)" in loaded_df.columns
