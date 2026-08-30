#!/usr/bin/env python3
"""Benchmark the TTS pipeline over the test corpus.

Usage
-----
::

    python scripts/benchmark.py
    python scripts/benchmark.py --config configs/config.yaml --device cpu
    python scripts/benchmark.py --input_file data/input/test_sentences.txt

Outputs
-------
- ``outputs/reports/baseline_results.csv``   — per-sample rows
- ``outputs/reports/benchmark_summary.csv``  — aggregated statistics
- Printed human-readable summary
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import Config, load_config
from src.metrics import aggregate_results, print_summary, results_to_dataframe, save_report
from src.model import TTSModel
from src.synthesizer import Synthesizer
from src.utils import load_text_file, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTS Baseline benchmark over the test corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Override input file (default: data/input/test_sentences.txt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override device from config.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = load_config(args.config)

    # Allow device override from CLI.
    if args.device is not None:
        config._data["inference"]["device"] = args.device

    input_file = args.input_file or Path(config.paths.input_data) / "test_sentences.txt"
    sentences = load_text_file(input_file)

    if not sentences:
        print("No sentences found. Exiting.")
        sys.exit(1)

    print(f"\nLoading model ...")
    model = TTSModel(config)
    model.load()
    device_str = model.get_device()

    synth = Synthesizer(model, config)

    records: list[dict] = []
    print(f"\nBenchmarking {len(sentences)} sentences on {device_str.upper()} ...\n")

    for idx, sentence in enumerate(sentences, start=1):
        print(f"  [{idx:02d}/{len(sentences):02d}] {sentence[:60]} ...")
        meta = synth.synthesize(sentence)
        meta["sample_id"] = idx
        records.append(meta)

    # Per-sample CSV
    df = results_to_dataframe(records)
    per_sample_path = Path(config.paths.reports) / "baseline_results.csv"
    save_report(df, str(per_sample_path))

    # Aggregated summary
    summary = aggregate_results(records)
    print_summary(summary, device=device_str)

    # Build summary DataFrame
    rows = []
    for metric, stats in summary.items():
        row = {"metric": metric, **stats}
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = Path(config.paths.reports) / "benchmark_summary.csv"
    save_report(summary_df, str(summary_path))

    print(f"Per-sample results  → {per_sample_path}")
    print(f"Aggregated summary  → {summary_path}\n")


if __name__ == "__main__":
    main()
