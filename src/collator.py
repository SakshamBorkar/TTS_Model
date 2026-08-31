"""Data collator with dynamic padding and speaker embedding conditioning for SpeechT5."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class SpeechT5DataCollatorWithPadding:
    """Collator for dynamic batching of SpeechT5 text-to-speech inputs and targets.

    SpeechT5 TTS requires:
      1. input_ids: tokenized text indices (batch_size, text_seq_len)
      2. attention_mask: text attention mask (batch_size, text_seq_len)
      3. labels: target log-mel spectrogram features (batch_size, mel_seq_len, num_mel_bins=80)
      4. stop_labels: binary stop token targets (batch_size, mel_seq_len)
      5. speaker_embeddings: fixed 512-dimensional speaker x-vector (batch_size, 512)
    """

    processor: Any
    speaker_embeddings: Optional[torch.Tensor] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch of features.")

        # Extract input_ids and labels from sample feature dicts
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        labels = [{"input_values": f["labels"]} for f in features]

        # 1. Pad text inputs
        batch = self.processor.pad(
            input_ids=input_ids,
            return_tensors="pt",
        )

        # 2. Pad target spectrograms (shape: [seq_len, num_mel_bins])
        label_tensors = [
            f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"], dtype=torch.float32)
            for f in features
        ]

        batch_size = len(features)
        max_label_len = max(lab.shape[0] for lab in label_tensors)
        if max_label_len % 2 != 0:
            max_label_len += 1
        num_mel_bins = label_tensors[0].shape[1] if label_tensors[0].ndim > 1 else 80

        # Initialize padded labels with -100 (ignored in loss computation)
        padded_labels = torch.full(
            (batch_size, max_label_len, num_mel_bins),
            -100.0,
            dtype=torch.float32,
        )
        stop_labels = torch.zeros((batch_size, max_label_len), dtype=torch.float32)

        for i, lab in enumerate(label_tensors):
            cur_len = lab.shape[0]
            if lab.ndim == 1:
                lab = lab.unsqueeze(1)
            padded_labels[i, :cur_len, : lab.shape[1]] = lab
            if cur_len > 0:
                stop_labels[i, cur_len - 1 :] = 1.0

        batch["labels"] = padded_labels
        batch["stop_labels"] = stop_labels

        # 3. Inject speaker embeddings (512-dim tensor per sample)
        if self.speaker_embeddings is not None:
            spk_emb = self.speaker_embeddings.clone().detach()
            if spk_emb.ndim == 1:
                spk_emb = spk_emb.unsqueeze(0)  # (1, 512)
            batch["speaker_embeddings"] = spk_emb.repeat(batch_size, 1)

        return batch
