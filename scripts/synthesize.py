#!/usr/bin/env python3
"""CLI for single-text or batch synthesis.

Usage
-----
Single text::

    python scripts/synthesize.py --text "Hello, how can I help you today?"

Batch from file::

    python scripts/synthesize.py --input_file data/input/test_sentences.txt

Optional arguments::

    --config  configs/config.yaml   (default)
    --output  outputs/audio/my.wav  (single-text only)
    --log_level INFO
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.model import TTSModel
from src.synthesizer import Synthesizer
from src.utils import load_text_file, setup_logging
from src.visualization import save_visualizations

logger = logging.getLogger(__name__)


def _print_metadata(meta: dict) -> None:
    """Pretty-print synthesis metadata."""
    print("\n" + "-" * 50)
    print("  TTS BASELINE")
    print("-" * 50)
    print(f"  Text:            {meta['text']!r}")
    print(f"  Device:          {meta['device'].upper()}")
    print(f"  Audio duration:  {meta['audio_duration_seconds']:.2f} s")
    print(f"  Inference time:  {meta['total_inference_time_seconds']:.3f} s")
    print(f"  RTF:             {meta['real_time_factor']:.3f}")
    print(f"  Output:          {meta['output_path']}")
    print("-" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTS Baseline — synthesize speech from text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text to synthesize.")
    group.add_argument(
        "--input_file",
        type=str,
        help="Path to a text file with one sentence per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "finetuned"],
        help="Choose between baseline (pretrained) or finetuned SpeechT5 model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom checkpoint directory for finetuned model (default: checkpoints/finetuned/best).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV path (single-text mode only).",
    )
    parser.add_argument(
        "--save_spectrograms",
        action="store_true",
        help="Save waveform and mel-spectrogram PNGs.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = load_config(args.config)

    # Resolve model checkpoint
    checkpoint_path = None
    if args.model == "finetuned":
        checkpoint_path = args.checkpoint or "checkpoints/finetuned/best"
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Finetuned checkpoint directory not found at: {checkpoint_path}\n"
                "Please train the model first with: python scripts/train.py --config configs/finetune.yaml"
            )
        logger.info("Loading fine-tuned model from %s", checkpoint_path)
    else:
        logger.info("Loading baseline pretrained model (%s)", config.model.name)

    model = TTSModel(config, checkpoint_path=checkpoint_path)
    model.load()
    synth = Synthesizer(model, config)

    save_specs: bool = args.save_spectrograms or config.evaluation.save_spectrograms

    if args.text:
        meta = synth.synthesize(args.text, output_path=args.output)
        _print_metadata(meta)
        if save_specs:
            from src.audio_utils import load_audio
            waveform, sr = load_audio(meta["output_path"])
            stem = Path(meta["output_path"]).stem
            save_visualizations(
                waveform, sr, stem, config.paths.output_spectrograms
            )

    else:
        sentences = load_text_file(args.input_file)
        print(f"\nSynthesising {len(sentences)} sentences ...\n")
        for i, sentence in enumerate(sentences, start=1):
            print(f"[{i:02d}/{len(sentences):02d}] {sentence}")
            meta = synth.synthesize(sentence)
            _print_metadata(meta)
            if save_specs:
                from src.audio_utils import load_audio
                waveform, sr = load_audio(meta["output_path"])
                stem = Path(meta["output_path"]).stem
                save_visualizations(
                    waveform, sr, stem, config.paths.output_spectrograms
                )

        print(f"\nBatch complete - {len(sentences)} files written.\n")


if __name__ == "__main__":
    main()
