"""Training engine, optimizer, validation, early stopping, and checkpointing for SpeechT5."""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    get_linear_schedule_with_warmup,
)

from src.collator import SpeechT5DataCollatorWithPadding
from src.config import Config
from src.dataset import SpeechT5FineTuningDataset, load_split_dataset

logger = logging.getLogger(__name__)


def validate_training_config(config: Config) -> None:
    """Validate fine-tuning configuration parameters.

    Raises:
      ValueError if any parameter is out of valid bounds.
    """
    lr = config.training.get("learning_rate", 1e-5)
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")

    batch_size = config.training.get("batch_size", 4)
    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}")

    epochs = config.training.get("epochs", 10)
    if epochs <= 0:
        raise ValueError(f"Epochs must be a positive integer, got {epochs}")

    weight_decay = config.training.get("weight_decay", 0.01)
    if weight_decay < 0:
        raise ValueError(f"Weight decay cannot be negative, got {weight_decay}")

    warmup_ratio = config.training.get("warmup_ratio", 0.1)
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError(f"Warmup ratio must be between 0.0 and 1.0, got {warmup_ratio}")

    grad_accum = config.training.get("gradient_accumulation_steps", 1)
    if grad_accum <= 0:
        raise ValueError(f"Gradient accumulation steps must be >= 1, got {grad_accum}")

    # Validate split ratios
    tr = config.data.get("train_ratio", 0.8)
    val = config.data.get("val_ratio", 0.1)
    test = config.data.get("test_ratio", 0.1)
    if tr <= 0 or val <= 0 or test <= 0 or not math.isclose(tr + val + test, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Split ratios must be positive and sum to 1.0 (train={tr}, val={val}, test={test})"
        )


class TTSFineTuningTrainer:
    """Encapsulates the fine-tuning lifecycle for SpeechT5."""

    def __init__(
        self,
        config: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        validate_training_config(self.config)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info("Initializing TTSFineTuningTrainer on device: %s", self.device)

        # Paths
        self.output_dir = Path(self.config.training.get("output_dir", "checkpoints/finetuned"))
        self.best_checkpoint_dir = Path(self.config.training.get("best_checkpoint_dir", "checkpoints/finetuned/best"))
        self.reports_dir = Path(self.config.training.get("reports_dir", "outputs/reports"))
        self.experiments_dir = Path(self.config.training.get("experiments_dir", "experiments/finetuning"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.processor: Optional[SpeechT5Processor] = None
        self.model: Optional[SpeechT5ForTextToSpeech] = None
        self.speaker_embeddings: Optional[torch.Tensor] = None
        self.training_history: List[Dict[str, Any]] = []

    def load_components(self) -> None:
        """Load pretrained SpeechT5 processor, model, and speaker embeddings."""
        base_model_name = self.config.model.get("base_model", "microsoft/speecht5_tts")
        logger.info("Loading SpeechT5 processor from %s", base_model_name)
        self.processor = SpeechT5Processor.from_pretrained(base_model_name)

        logger.info("Loading SpeechT5 model from %s", base_model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(base_model_name)
        self.model.to(self.device)

        # Load speaker embedding
        spk_dataset = self.config.model.get("speaker_embeddings_dataset", "Matthijs/cmu-arctic-xvectors")
        spk_index = self.config.model.get("speaker_embeddings_index", 7306)
        logger.info("Loading speaker embedding from %s (index %d)", spk_dataset, spk_index)

        try:
            from datasets import load_dataset
            spk_data = load_dataset(spk_dataset, split="validation", trust_remote_code=True)
            spk_tensor = torch.tensor(spk_data[spk_index]["xvector"]).float()
        except Exception as exc:
            logger.warning("Could not download speaker dataset (%s). Generating fallback 512-dim embedding.", exc)
            rng = torch.Generator().manual_seed(42)
            spk_tensor = torch.randn(512, generator=rng)

        self.speaker_embeddings = spk_tensor.to(self.device)

    def train(
        self,
        train_records: Optional[List[Dict[str, Any]]] = None,
        val_records: Optional[List[Dict[str, Any]]] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the fine-tuning loop with validation and early stopping."""
        if self.model is None or self.processor is None:
            self.load_components()

        # Load split records if not directly supplied
        splits_dir = self.config.data.get("splits_dir", "data/splits")
        if train_records is None:
            train_records = load_split_dataset("train", splits_dir=splits_dir)
        if val_records is None:
            val_records = load_split_dataset("val", splits_dir=splits_dir)

        logger.info("Loaded datasets: Train=%d, Validation=%d", len(train_records), len(val_records))

        # Build PyTorch datasets & dataloaders
        train_dataset = SpeechT5FineTuningDataset(
            records=train_records,
            processor=self.processor,
            target_sample_rate=self.config.data.get("sample_rate", 16000),
        )
        val_dataset = SpeechT5FineTuningDataset(
            records=val_records,
            processor=self.processor,
            target_sample_rate=self.config.data.get("sample_rate", 16000),
        )

        collator = SpeechT5DataCollatorWithPadding(
            processor=self.processor,
            speaker_embeddings=self.speaker_embeddings.cpu(),
        )

        batch_size = self.config.training.get("batch_size", 4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        # Setup Optimizer & LR Scheduler
        epochs = self.config.training.get("epochs", 10)
        grad_accum_steps = self.config.training.get("gradient_accumulation_steps", 4)
        effective_batch_size = batch_size * grad_accum_steps
        logger.info(
            "Batch size: %d, Gradient accumulation: %d -> Effective batch size: %d",
            batch_size,
            grad_accum_steps,
            effective_batch_size,
        )

        total_steps = len(train_loader) * epochs // grad_accum_steps
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        total_steps = max(total_steps, 1)

        lr = float(self.config.training.get("learning_rate", 1e-5))
        weight_decay = float(self.config.training.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        warmup_steps = int(total_steps * self.config.training.get("warmup_ratio", 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_fp16 = bool(self.config.training.get("fp16", True)) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        eval_steps = self.config.training.get("eval_steps", 50)
        save_steps = self.config.training.get("save_steps", 50)
        patience = self.config.training.get("early_stopping_patience", 3)

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0
        start_time = time.time()
        stop_training = False

        self.training_history.clear()

        logger.info("Starting fine-tuning: Total steps = %d, Warmup steps = %d", total_steps, warmup_steps)

        for epoch in range(1, epochs + 1):
            if stop_training:
                break

            self.model.train()
            accumulated_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                if stop_training:
                    break

                # Transfer tensors to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                speaker_embeddings = batch["speaker_embeddings"].to(self.device)

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        speaker_embeddings=speaker_embeddings,
                    )
                    loss = outputs.loss / grad_accum_steps

                scaler.scale(loss).backward()
                accumulated_loss += loss.item() * grad_accum_steps

                # Optimizer step on gradient accumulation boundary
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_lr = scheduler.get_last_lr()[0]
                    avg_train_loss = accumulated_loss / grad_accum_steps
                    accumulated_loss = 0.0
                    logger.info("Epoch %d | Step %d/%d | Train Loss: %.4f | LR: %.2e", epoch, global_step, total_steps, avg_train_loss, current_lr)

                    # Evaluation step
                    if global_step % eval_steps == 0 or global_step == total_steps:
                        val_loss = self.evaluate(val_loader, use_fp16=use_fp16)
                        logger.info(
                            "Epoch %d | Step %d/%d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
                            epoch,
                            global_step,
                            total_steps,
                            avg_train_loss,
                            val_loss,
                            current_lr,
                        )

                        # Checkpoint & Best model selection (based strictly on val_loss)
                        checkpoint_dir = None
                        if global_step % save_steps == 0:
                            step_ckpt = self.output_dir / f"checkpoint-{global_step}"
                            self._save_checkpoint(step_ckpt)
                            checkpoint_dir = str(step_ckpt)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            self._save_checkpoint(self.best_checkpoint_dir)
                            logger.info("New best validation loss: %.4f! Saved to %s", best_val_loss, self.best_checkpoint_dir)
                        else:
                            patience_counter += 1
                            logger.info("Validation loss did not improve. Patience: %d/%d", patience_counter, patience)

                        # Record history
                        self.training_history.append({
                            "step": global_step,
                            "epoch": epoch,
                            "training_loss": float(avg_train_loss),
                            "validation_loss": float(val_loss),
                            "learning_rate": float(current_lr),
                            "checkpoint": checkpoint_dir or "",
                            "timestamp": datetime.now().isoformat(),
                        })

                        # Early stopping check
                        if patience_counter >= patience:
                            logger.info("Early stopping triggered after %d steps without improvement.", patience_counter * eval_steps)
                            stop_training = True
                            break

                if max_steps and global_step >= max_steps:
                    stop_training = True
                    break

        total_duration = time.time() - start_time
        logger.info("Fine-tuning completed in %.2f seconds. Best Val Loss: %.4f", total_duration, best_val_loss)

        # Persist reports and training curves
        self._save_training_reports(total_duration=total_duration, best_val_loss=best_val_loss)

        return {
            "best_validation_loss": best_val_loss,
            "total_steps": global_step,
            "training_duration_seconds": total_duration,
            "best_checkpoint_path": str(self.best_checkpoint_dir),
            "history": self.training_history,
        }

    def evaluate(self, val_loader: DataLoader, use_fp16: bool = False) -> float:
        """Evaluate validation loss on held-out validation split."""
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                speaker_embeddings = batch["speaker_embeddings"].to(self.device)

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        speaker_embeddings=speaker_embeddings,
                    )
                    total_val_loss += outputs.loss.item()
                    num_batches += 1

        self.model.train()
        return float(total_val_loss / num_batches) if num_batches > 0 else float("inf")

    def _save_checkpoint(self, path: Path) -> None:
        """Save model weights and processor."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        if self.processor is not None:
            self.processor.save_pretrained(str(path))
        logger.info("Checkpoint saved to %s", path)

    def _save_training_reports(self, total_duration: float, best_val_loss: float) -> None:
        """Save training_history.csv, loss plots, and experiment markdown report."""
        if not self.training_history:
            return

        # 1. Save training_history.csv
        df = pd.DataFrame(self.training_history)
        history_csv = self.reports_dir / "training_history.csv"
        df.to_csv(str(history_csv), index=False)
        logger.info("Saved training history to %s", history_csv)

        # 2. Save loss curves plot
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df["step"], df["training_loss"], label="Training Loss", color="#1f77b4", marker="o")
            plt.plot(df["step"], df["validation_loss"], label="Validation Loss", color="#ff7f0e", marker="s")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("SpeechT5 Fine-Tuning Loss Convergence")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            loss_plot_path = self.reports_dir / "training_loss_curve.png"
            plt.savefig(str(loss_plot_path), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            logger.warning("Could not render loss curve plot: %s", exc)

        # 3. Save Experiment Markdown Report in experiments/finetuning/
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_report_path = self.experiments_dir / f"{exp_id}.md"

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        report_content = f"""# Fine-Tuning Experiment Report: {exp_id}

- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Base Model**: `{self.config.model.get("base_model", "microsoft/speecht5_tts")}`
- **Dataset**: `{self.config.data.get("dataset_name", "lj_speech")}`
- **Hardware**: {device_name} (PyTorch {torch.__version__})

## Hyperparameters
- **Epochs**: {self.config.training.get("epochs")}
- **Batch Size**: {self.config.training.get("batch_size")}
- **Gradient Accumulation Steps**: {self.config.training.get("gradient_accumulation_steps")} (Effective Batch Size: {self.config.training.get("batch_size") * self.config.training.get("gradient_accumulation_steps")})
- **Learning Rate**: {self.config.training.get("learning_rate")}
- **Weight Decay**: {self.config.training.get("weight_decay")}
- **Warmup Ratio**: {self.config.training.get("warmup_ratio")}
- **Mixed Precision (FP16)**: {self.config.training.get("fp16")}
- **Seed**: {self.config.training.get("seed")}

## Results
- **Best Validation Loss**: `{best_val_loss:.4f}`
- **Training Duration**: `{total_duration:.2f}` seconds
- **Best Checkpoint Directory**: `{self.best_checkpoint_dir}`

## Training History
| Step | Epoch | Training Loss | Validation Loss | Learning Rate |
|------|-------|---------------|-----------------|---------------|
"""
        for r in self.training_history:
            report_content += f"| {r['step']} | {r['epoch']} | {r['training_loss']:.4f} | {r['validation_loss']:.4f} | {r['learning_rate']:.2e} |\n"

        with exp_report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved experiment tracking report to %s", exp_report_path)
