"""Unit tests for src.collator (SpeechT5 data collation and speaker conditioning)."""

from unittest.mock import MagicMock
import pytest
import torch

from src.collator import SpeechT5DataCollatorWithPadding


class DummyProcessor:
    def pad(self, input_ids=None, labels=None, return_tensors="pt"):
        if input_ids is not None:
            max_len = max(len(x["input_ids"]) for x in input_ids)
            padded = []
            masks = []
            for item in input_ids:
                seq = item["input_ids"]
                pad_len = max_len - len(seq)
                padded.append(seq + [0] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        elif labels is not None:
            max_frames = max(len(x["input_values"]) for x in labels)
            padded = []
            masks = []
            for item in labels:
                seq = item["input_values"]
                pad_len = max_frames - len(seq)
                # Assume 80 mel channels
                pad_tensor = torch.zeros((pad_len, 80))
                padded.append(torch.cat([seq, pad_tensor], dim=0))
                mask = torch.tensor([1] * len(seq) + [0] * pad_len, dtype=torch.long)
                masks.append(mask)
            return {
                "input_values": torch.stack(padded),
                "attention_mask": torch.stack(masks),
            }


class TestCollator:
    def test_collates_batch_with_speaker_embeddings(self):
        processor = DummyProcessor()
        speaker_emb = torch.randn(512)
        collator = SpeechT5DataCollatorWithPadding(
            processor=processor,
            speaker_embeddings=speaker_emb,
        )

        features = [
            {
                "input_ids": [10, 20, 30],
                "labels": torch.randn(25, 80),
            },
            {
                "input_ids": [15, 25, 35, 45, 55],
                "labels": torch.randn(40, 80),
            },
        ]

        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "stop_labels" in batch
        assert "speaker_embeddings" in batch

        # Check shapes
        assert batch["input_ids"].shape == (2, 5)
        assert batch["attention_mask"].shape == (2, 5)
        assert batch["labels"].shape == (2, 40, 80)
        assert batch["stop_labels"].shape == (2, 40)
        assert batch["speaker_embeddings"].shape == (2, 512)

        # Check masked padding values in labels are -100
        assert (batch["labels"][0, 25:, :] == -100.0).all()

    def test_raises_on_empty_batch(self):
        processor = DummyProcessor()
        collator = SpeechT5DataCollatorWithPadding(processor=processor)
        with pytest.raises(ValueError, match="empty batch"):
            collator([])
