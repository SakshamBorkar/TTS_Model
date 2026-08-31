#!/usr/bin/env python3
"""CLI training script for Stage 2 SpeechT5 fine-tuning.

Usage
-----
Standard fine-tuning::

    python scripts/train.py --config configs/finetune.yaml

Rapid smoke-test / dry-run (e.g. 5 steps)::

    python scripts/train.py --config configs/finetune.yaml --smoke-test --max-steps 5
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from repository root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import load_split_dataset
from src.trainer import TTSFineTuningTrainer
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SpeechT5 on LJ Speech dataset."
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
        help="Run a quick dry-run with limited steps on small split subset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Explicitly limit the number of training steps (useful for debugging/smoke testing).",
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

    # 1. Set reproducibility seed
    seed = config.training.get("seed", 42)
    set_seed(seed)
    logger.info("Deterministic seed set to: %d", seed)

    # 2. Hardware and environment log
    print("=" * 70)
    print("  STAGE 2: SPEECHT5 TTS FINE-TUNING")
    print("=" * 70)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:       {device_name}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Base Model:   {config.model.get('base_model', 'microsoft/speecht5_tts')}")
    print(f"  Target Split: 80% Train / 10% Val / 10% Test")
    print("=" * 70 + "\n")

    # 3. Load Splits
    splits_dir = config.data.get("splits_dir", "data/splits")
    try:
        train_records = load_split_dataset("train", splits_dir=splits_dir)
        val_records = load_split_dataset("val", splits_dir=splits_dir)
    except FileNotFoundError:
        logger.error(
            "Splits not found in %s! Please run: python scripts/prepare_dataset.py --config %s first.",
            splits_dir,
            args.config,
        )
        sys.exit(1)

    if args.smoke_test:
        logger.info("Running smoke test: slicing train/val sets to 8 samples.")
        train_records = train_records[:8]
        val_records = val_records[:4]

    # 4. Instantiate Trainer & Run
    trainer = TTSFineTuningTrainer(config=config)
    trainer.load_components()

    results = trainer.train(
        train_records=train_records,
        val_records=val_records,
        max_steps=args.max_steps or (5 if args.smoke_test else None),
    )

    print("\n" + "=" * 70)
    print("  FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"  Best Validation Loss: {results['best_validation_loss']:.4f}")
    print(f"  Total Steps:          {results['total_steps']}")
    print(f"  Duration:             {results['training_duration_seconds']:.2f} s")
    print(f"  Best Checkpoint:      {results['best_checkpoint_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
