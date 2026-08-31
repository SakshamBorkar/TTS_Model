"""Dataset handling, validation, statistics, and PyTorch Dataset for Stage 2 SpeechT5 fine-tuning."""

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.preprocessing import TextPreprocessingError, preprocess_text

logger = logging.getLogger(__name__)


@dataclass
class AudioValidationResult:
    is_valid: bool
    error_type: Optional[str] = None
    description: Optional[str] = None
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0


def validate_audio_tensor(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    expected_sample_rate: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
) -> AudioValidationResult:
    """Validate raw audio array or tensor for ML training quality.

    Checks:
      - Valid numeric values (no NaN, no Inf)
      - Non-empty waveform
      - Amplitude bounds (within [-1.0, 1.0])
      - Sample rate match
      - Duration within bounds [min_duration, max_duration]
      - Single-channel (mono)
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    if waveform is None or len(waveform) == 0:
        return AudioValidationResult(
            is_valid=False,
            error_type="EMPTY_AUDIO",
            description="Audio waveform is empty or None.",
        )

    # Check channels (should be 1D or (1, N))
    if waveform.ndim > 1:
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
            channels = 1
        elif waveform.shape[-1] == 1:
            waveform = waveform.squeeze(-1)
            channels = 1
        else:
            channels = waveform.shape[0] if waveform.ndim == 2 else waveform.shape[-1]
            return AudioValidationResult(
                is_valid=False,
                error_type="INVALID_CHANNELS",
                description=f"Expected mono audio (1 channel), got {channels} channels.",
                channels=channels,
            )
    else:
        channels = 1

    # Check for NaN / Inf
    if not np.isfinite(waveform).all():
        return AudioValidationResult(
            is_valid=False,
            error_type="NON_FINITE_VALUES",
            description="Audio array contains NaN or Inf values.",
            channels=channels,
        )

    # Check duration
    num_samples = len(waveform)
    duration = num_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if duration < min_duration:
        return AudioValidationResult(
            is_valid=False,
            error_type="TOO_SHORT",
            description=f"Audio duration {duration:.2f}s is below minimum {min_duration:.2f}s.",
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
        )

    if duration > max_duration:
        return AudioValidationResult(
            is_valid=False,
            error_type="TOO_LONG",
            description=f"Audio duration {duration:.2f}s exceeds maximum {max_duration:.2f}s.",
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
        )

    if sample_rate != expected_sample_rate:
        return AudioValidationResult(
            is_valid=False,
            error_type="SAMPLE_RATE_MISMATCH",
            description=f"Expected sample rate {expected_sample_rate} Hz, got {sample_rate} Hz.",
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
        )

    # Check amplitude silence
    max_amp = float(np.max(np.abs(waveform)))
    if max_amp < 1e-4:
        return AudioValidationResult(
            is_valid=False,
            error_type="SILENT_AUDIO",
            description="Audio amplitude is virtually silent (max amplitude < 1e-4).",
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
        )

    return AudioValidationResult(
        is_valid=True,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
    )


def validate_sample(
    sample_id: str,
    raw_text: str,
    audio_path_or_array: Union[str, Path, np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    expected_sample_rate: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str], float]:
    """Validate a single (sample_id, raw_text, audio) record.

    Returns:
      (is_valid, normalized_text, error_type, error_description, duration)
    """
    # 1. Validate and normalize text using Stage 1 preprocessing
    try:
        norm_text = preprocess_text(raw_text)
    except TextPreprocessingError as exc:
        return False, None, "INVALID_TRANSCRIPT", str(exc), 0.0
    except Exception as exc:
        return False, None, "TRANSCRIPT_ERROR", str(exc), 0.0

    if not norm_text or len(norm_text.strip()) == 0:
        return False, None, "EMPTY_TRANSCRIPT", "Normalized transcript is empty.", 0.0

    # 2. Validate audio
    if isinstance(audio_path_or_array, (str, Path)):
        audio_path = Path(audio_path_or_array)
        if not audio_path.exists():
            return False, norm_text, "FILE_NOT_FOUND", f"Audio file not found: {audio_path}", 0.0
        try:
            waveform, sr = sf.read(str(audio_path), dtype="float32")
            val_res = validate_audio_tensor(
                waveform=waveform,
                sample_rate=sr,
                expected_sample_rate=expected_sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        except Exception as exc:
            return False, norm_text, "AUDIO_DECODE_ERROR", f"Failed to decode audio: {exc}", 0.0
    else:
        val_res = validate_audio_tensor(
            waveform=audio_path_or_array,
            sample_rate=sample_rate,
            expected_sample_rate=expected_sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    if not val_res.is_valid:
        return False, norm_text, val_res.error_type, val_res.description, val_res.duration

    return True, norm_text, None, None, val_res.duration


def compute_dataset_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute descriptive statistics across verified samples."""
    if not records:
        return {
            "sample_count": 0,
            "total_duration_hours": 0.0,
            "total_duration_seconds": 0.0,
            "avg_duration_seconds": 0.0,
            "median_duration_seconds": 0.0,
            "min_duration_seconds": 0.0,
            "max_duration_seconds": 0.0,
            "avg_transcript_length": 0.0,
            "median_transcript_length": 0.0,
            "min_transcript_length": 0,
            "max_transcript_length": 0,
        }

    durations = [r["duration"] for r in records if "duration" in r]
    text_lengths = [len(r["normalized_text"]) for r in records if "normalized_text" in r]

    dur_arr = np.array(durations, dtype=np.float64) if durations else np.array([0.0])
    len_arr = np.array(text_lengths, dtype=np.float64) if text_lengths else np.array([0.0])

    return {
        "sample_count": len(records),
        "total_duration_hours": float(np.sum(dur_arr) / 3600.0),
        "total_duration_seconds": float(np.sum(dur_arr)),
        "avg_duration_seconds": float(np.mean(dur_arr)),
        "median_duration_seconds": float(np.median(dur_arr)),
        "min_duration_seconds": float(np.min(dur_arr)),
        "max_duration_seconds": float(np.max(dur_arr)),
        "avg_transcript_length": float(np.mean(len_arr)),
        "median_transcript_length": float(np.median(len_arr)),
        "min_transcript_length": int(np.min(len_arr)),
        "max_transcript_length": int(np.max(len_arr)),
    }


def create_deterministic_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create disjoint, deterministic Train/Val/Test splits with fixed seed."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={total_ratio})"
        )

    # Sort deterministically by sample_id before shuffling
    records_sorted = sorted(records, key=lambda x: str(x.get("sample_id", "")))

    rng = random.Random(seed)
    indices = list(range(len(records_sorted)))
    rng.shuffle(indices)

    n_total = len(records_sorted)
    n_train = int(round(train_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    # Test gets remaining to ensure exact partition
    n_test = n_total - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_set = [records_sorted[i] for i in train_indices]
    val_set = [records_sorted[i] for i in val_indices]
    test_set = [records_sorted[i] for i in test_indices]

    # Verify disjointness
    train_ids = {r["sample_id"] for r in train_set}
    val_ids = {r["sample_id"] for r in val_set}
    test_ids = {r["sample_id"] for r in test_set}

    overlap_tr_val = train_ids.intersection(val_ids)
    overlap_tr_te = train_ids.intersection(test_ids)
    overlap_val_te = val_ids.intersection(test_ids)

    if overlap_tr_val or overlap_tr_te or overlap_val_te:
        raise RuntimeError(
            f"Data leakage detected! Overlaps: Train-Val={len(overlap_tr_val)}, "
            f"Train-Test={len(overlap_tr_te)}, Val-Test={len(overlap_val_te)}"
        )

    logger.info(
        "Deterministic splits created: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_set),
        len(val_set),
        len(test_set),
        n_total,
    )

    return {"train": train_set, "val": val_set, "test": test_set}


def save_split_metadata(splits: Dict[str, List[Dict[str, Any]]], splits_dir: Union[str, Path]) -> None:
    """Save split records to JSON files."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        file_path = out_dir / f"{split_name}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s split (%d records) to %s", split_name, len(records), file_path)


def load_split_dataset(split_name: str, splits_dir: Union[str, Path] = "data/splits") -> List[Dict[str, Any]]:
    """Load split records from JSON file."""
    file_path = Path(splits_dir) / f"{split_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return records


class SpeechT5FineTuningDataset(Dataset):
    """PyTorch Dataset for SpeechT5 fine-tuning.

    Transforms text and raw audio into tokenizer input IDs and target log-mel spectrogram features.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor: Any,
        target_sample_rate: int = 16000,
    ) -> None:
        self.records = records
        self.processor = processor
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        text = record["normalized_text"]

        # 1. Process text into input_ids
        input_ids = self.processor(text=text).input_ids

        # 2. Extract or load audio waveform
        if "audio_array" in record and record["audio_array"] is not None:
            audio = np.array(record["audio_array"], dtype=np.float32)
        elif "audio_path" in record and record["audio_path"] is not None:
            audio, sr = sf.read(record["audio_path"], dtype="float32")
        elif "audio" in record and isinstance(record["audio"], dict) and "array" in record["audio"]:
            audio = np.array(record["audio"]["array"], dtype=np.float32)
        else:
            raise ValueError(f"Record at index {idx} ({record.get('sample_id')}) lacks audio source.")

        # If audio is stereo or multi-dimensional, convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 3. Extract log-mel spectrogram target labels via processor audio_target
        # SpeechT5 target feature extractor converts waveform -> mel-spectrogram [time_frames, 80]
        mel_labels = self.processor(
            audio_target=audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        ).input_values[0]

        # SpeechT5 reduction_factor is 2; ensure time_frames is an even number
        if mel_labels.shape[0] % 2 != 0:
            mel_labels = mel_labels[:-1]

        return {
            "input_ids": input_ids,
            "labels": mel_labels,
            "sample_id": record.get("sample_id", f"sample_{idx}"),
            "text": text,
        }
