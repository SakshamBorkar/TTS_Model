#!/usr/bin/env python3
"""Evaluation script — optionally computes ASR-based WER.

Usage
-----
::

    # Without ASR
    python scripts/evaluate.py

    # With ASR (requires whisper)
    python scripts/evaluate.py --enable_asr

Outputs
-------
- ``outputs/reports/evaluation_report.csv``
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.metrics import compute_wer, results_to_dataframe, save_report
from src.model import TTSModel
from src.synthesizer import Synthesizer
from src.utils import load_text_file, setup_logging

logger = logging.getLogger(__name__)


def transcribe_with_whisper(audio_path: str, asr_model_name: str) -> str:
    """Transcribe an audio file using Whisper.

    Parameters
    ----------
    audio_path:
        Path to the WAV file.
    asr_model_name:
        Hugging Face model identifier for Whisper.

    Returns
    -------
    str
        Transcription text.
    """
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from src.audio_utils import load_audio

        logger.info("Loading Whisper model %s …", asr_model_name)
        processor = WhisperProcessor.from_pretrained(asr_model_name)
        asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_name)
        asr_model.eval()

        waveform, sr = load_audio(audio_path, target_sr=16000)

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = asr_model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0].strip()

    except ImportError as exc:
        raise ImportError(
            "transformers is required for ASR evaluation."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTS Baseline evaluation (latency + optional ASR WER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Override test corpus path.",
    )
    parser.add_argument(
        "--enable_asr",
        action="store_true",
        help="Enable ASR-based WER evaluation using Whisper.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = load_config(args.config)
    enable_asr: bool = args.enable_asr or config.evaluation.enable_asr

    input_file = args.input_file or Path(config.paths.input_data) / "test_sentences.txt"
    sentences = load_text_file(input_file)

    if not sentences:
        print("No sentences found. Exiting.")
        sys.exit(1)

    model = TTSModel(config)
    model.load()
    synth = Synthesizer(model, config)

    records: list[dict] = []

    for idx, sentence in enumerate(sentences, start=1):
        logger.info("[%02d/%02d] %s", idx, len(sentences), sentence)
        meta = synth.synthesize(sentence)
        meta["sample_id"] = idx

        if enable_asr:
            asr_model = config.evaluation.asr_model
            try:
                transcript = transcribe_with_whisper(meta["output_path"], asr_model)
                wer = compute_wer(sentence, transcript)
                meta["asr_transcript"] = transcript
                meta["wer"] = round(wer, 4)
                logger.info(
                    "WER=%.4f  ref=%r  hyp=%r", wer, sentence, transcript
                )
            except Exception as exc:
                logger.warning("ASR evaluation failed for sample %d: %s", idx, exc)
                meta["asr_transcript"] = ""
                meta["wer"] = None
        else:
            meta["wer"] = None

        records.append(meta)

    # Select columns for the report
    report_cols = [
        "sample_id",
        "text",
        "audio_duration_seconds",
        "total_inference_time_seconds",
        "real_time_factor",
        "wer",
        "device",
        "output_path",
    ]

    df = results_to_dataframe(records)
    # Keep only columns that exist in df
    existing = [c for c in report_cols if c in df.columns]
    report_df = df[existing]

    report_path = Path(config.paths.reports) / "evaluation_report.csv"
    save_report(report_df, str(report_path))

    print(f"\nEvaluation complete — report saved to {report_path}")
    if enable_asr:
        wer_vals = [r["wer"] for r in records if r.get("wer") is not None]
        if wer_vals:
            mean_wer = sum(wer_vals) / len(wer_vals)
            print(f"Mean WER: {mean_wer:.4f}")
            print(
                "\nNote: WER measures whether synthesised speech preserves "
                "recognisable linguistic content / intelligibility.\n"
                "WER is NOT a complete measure of TTS naturalness or "
                "subjective speech quality."
            )


if __name__ == "__main__":
    main()
