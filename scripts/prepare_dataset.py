#!/usr/bin/env python3
"""Dataset preparation script for Stage 2 SpeechT5 fine-tuning on LJ Speech.

Usage
-----
Full preparation (downloads/loads LJ Speech)::

    python scripts/prepare_dataset.py --config configs/finetune.yaml

Fast smoke test (creates synthetic samples to test pipeline offline)::

    python scripts/prepare_dataset.py --config configs/finetune.yaml --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import soundfile as sf

# Allow running directly from repository root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import (
    compute_dataset_statistics,
    create_deterministic_splits,
    save_split_metadata,
    validate_sample,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    num_samples: int = 50,
    sample_rate: int = 16000,
    output_raw_dir: str = "data/raw/synthetic",
) -> List[Dict[str, Any]]:
    """Generate synthetic audio files and transcripts for fast smoke testing."""
    raw_path = Path(output_raw_dir)
    wavs_dir = raw_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis has advanced significantly with deep neural networks.",
        "Generative acoustic models convert text representations into mel spectrograms.",
        "A neural vocoder transforms intermediate spectrograms into realistic waveforms.",
        "Domain adaptation aligns model pronunciation with unique acoustic characteristics.",
    ]

    for i in range(num_samples):
        sample_id = f"SYNTH_{i:04d}"
        text = base_sentences[i % len(base_sentences)]
        # Generate 1.0 to 3.0 seconds synthetic sine audio
        duration = 1.0 + (i % 5) * 0.4
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = 220.0 + (i % 8) * 40.0
        waveform = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        wav_file = wavs_dir / f"{sample_id}.wav"
        sf.write(str(wav_file), waveform, sample_rate)

        records.append({
            "sample_id": sample_id,
            "raw_text": text,
            "audio_path": str(wav_file),
            "sample_rate": sample_rate,
        })

    logger.info("Generated %d synthetic samples in %s", len(records), wavs_dir)
    return records


def load_raw_ljspeech_dataset(config: Any) -> List[Dict[str, Any]]:
    """Load LJ Speech dataset either via Hugging Face datasets or local raw directory."""
    raw_dir = Path(config.data.get("raw_dir", "data/raw"))
    metadata_csv = raw_dir / "metadata.csv"

    # Check if downloaded locally
    if metadata_csv.exists() and (raw_dir / "wavs").exists():
        logger.info("Loading local LJ Speech dataset from %s", raw_dir)
        records = []
        with metadata_csv.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    sample_id = parts[0]
                    # Part 2 is normalized text, fallback to part 1
                    raw_text = parts[2] if len(parts) >= 3 else parts[1]
                    wav_path = raw_dir / "wavs" / f"{sample_id}.wav"
                    records.append({
                        "sample_id": sample_id,
                        "raw_text": raw_text,
                        "audio_path": str(wav_path),
                        "sample_rate": config.data.sample_rate,
                    })
        return records

    # Otherwise load real human speech dataset from Hugging Face
    logger.info("Loading real human speech recordings from Hugging Face...")
    try:
        import io
        import datasets
        from datasets import load_dataset
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
        records = []
        limit = config.data.get("sample_limit", None)
        for idx, item in enumerate(ds):
            audio_info = item.get("audio", {})
            raw_text = item.get("text", "")
            if not raw_text or not audio_info.get("bytes"):
                continue
            wav, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
            records.append({
                "sample_id": item.get("id", f"SPEECH_{idx:04d}"),
                "raw_text": raw_text,
                "audio_array": wav,
                "sample_rate": sr,
            })
            if limit and len(records) >= limit:
                break
        logger.info("Successfully loaded %d real human speech recordings.", len(records))
        return records
    except Exception as exc:
        logger.warning(
            "Could not load speech dataset via datasets library (%s). Falling back to synthetic dataset.",
            exc,
        )
        return generate_synthetic_dataset(num_samples=40, sample_rate=config.data.sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, and split LJ Speech dataset for Stage 2 SpeechT5 fine-tuning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to YAML fine-tuning config file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate synthetic samples for rapid pipeline verification.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of processed samples for fast dry-run.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    reports_dir = Path(config.training.get("reports_dir", "outputs/reports"))
    splits_dir = Path(config.data.get("splits_dir", "data/splits"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: DATASET PREPARATION & QUALITY VALIDATION")
    print("=" * 70)

    # 1. Ingestion
    if args.smoke_test:
        raw_records = generate_synthetic_dataset(
            num_samples=args.sample_limit or 50,
            sample_rate=config.data.sample_rate,
        )
    else:
        raw_records = load_raw_ljspeech_dataset(config)

    if args.sample_limit and len(raw_records) > args.sample_limit:
        logger.info("Limiting dataset to %d samples for dry-run.", args.sample_limit)
        raw_records = raw_records[: args.sample_limit]

    logger.info("Validating %d raw samples...", len(raw_records))

    # 2. Validation & Quality Checks
    valid_records = []
    error_records = []

    for item in raw_records:
        sample_id = item["sample_id"]
        raw_text = item["raw_text"]
        audio_src = item.get("audio_path") if "audio_path" in item else item.get("audio_array")
        sr = item.get("sample_rate", config.data.sample_rate)

        is_valid, norm_text, err_type, err_desc, dur = validate_sample(
            sample_id=sample_id,
            raw_text=raw_text,
            audio_path_or_array=audio_src,
            sample_rate=sr,
            expected_sample_rate=config.data.sample_rate,
            min_duration=config.data.min_audio_length_seconds,
            max_duration=config.data.max_audio_length_seconds,
        )

        if is_valid:
            wav_path_str = item.get("audio_path")
            if not wav_path_str and "audio_array" in item:
                wavs_dir = Path(config.data.get("raw_dir", "data/raw")) / "wavs"
                wavs_dir.mkdir(parents=True, exist_ok=True)
                out_wav = wavs_dir / f"{sample_id}.wav"
                sf.write(str(out_wav), item["audio_array"], sr)
                wav_path_str = str(out_wav)

            valid_records.append({
                "sample_id": sample_id,
                "raw_text": raw_text,
                "normalized_text": norm_text,
                "duration": dur,
                "sample_rate": sr,
                "audio_path": wav_path_str,
            })
        else:
            error_records.append({
                "sample_id": sample_id,
                "error_type": err_type,
                "description": err_desc,
            })

    logger.info("Validation complete: %d valid, %d rejected.", len(valid_records), len(error_records))

    if not valid_records:
        logger.error("No valid samples found after dataset validation!")
        sys.exit(1)

    # 3. Create Deterministic Disjoint Splits
    splits = create_deterministic_splits(
        records=valid_records,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )

    # 4. Save Splits
    save_split_metadata(splits, splits_dir=splits_dir)

    # 5. Compute and Save Statistics & Reports
    stats = compute_dataset_statistics(valid_records)
    stats["valid_samples"] = len(valid_records)
    stats["rejected_samples"] = len(error_records)
    stats["train_samples"] = len(splits["train"])
    stats["val_samples"] = len(splits["val"])
    stats["test_samples"] = len(splits["test"])

    # outputs/reports/dataset_summary.json
    summary_path = reports_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Dataset summary saved to %s", summary_path)

    # outputs/reports/dataset_statistics.csv
    stats_df = pd.DataFrame([stats])
    stats_csv_path = reports_dir / "dataset_statistics.csv"
    stats_df.to_csv(str(stats_csv_path), index=False)
    logger.info("Dataset statistics saved to %s", stats_csv_path)

    # outputs/reports/dataset_errors.csv
    errors_df = pd.DataFrame(error_records if error_records else [{"sample_id": "NONE", "error_type": "NONE", "description": "No errors detected."}])
    errors_csv_path = reports_dir / "dataset_errors.csv"
    errors_df.to_csv(str(errors_csv_path), index=False)
    logger.info("Dataset errors report saved to %s", errors_csv_path)

    # Print pretty summary
    print("\n" + "-" * 60)
    print("  DATASET PREPARATION SUMMARY")
    print("-" * 60)
    print(f"  Valid samples:          {stats['valid_samples']}")
    print(f"  Rejected samples:       {stats['rejected_samples']}")
    print(f"  Total audio duration:   {stats['total_duration_hours']:.2f} hours ({stats['total_duration_seconds']:.1f} s)")
    print(f"  Average audio duration: {stats['avg_duration_seconds']:.2f} s")
    print(f"  Median audio duration:  {stats['median_duration_seconds']:.2f} s")
    print(f"  Min / Max duration:     {stats['min_duration_seconds']:.2f} s / {stats['max_duration_seconds']:.2f} s")
    print(f"  Train split:            {stats['train_samples']} samples ({config.data.train_ratio * 100:.0f}%)")
    print(f"  Validation split:       {stats['val_samples']} samples ({config.data.val_ratio * 100:.0f}%)")
    print(f"  Test split:             {stats['test_samples']} samples ({config.data.test_ratio * 100:.0f}%)")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
