# Stage 2 Fine-Tuning — Error & Linguistic Analysis

## 1. Overview
This document analyzes the differences between the **Pretrained Baseline** (`microsoft/speecht5_tts`) and the **Fine-Tuned Model** on the held-out test set.

## 2. Category Analysis

### A. Numeric Expressions & Dates
- **Challenge**: Conversion of digits (e.g. `$500`, `1984`, `3.14`) into phonetic representations.
- **Observed Behavior**: Both models benefit from text normalization in Stage 1; fine-tuning preserves cadence without skipping digits.

### B. Abbreviations & Acronyms
- **Challenge**: Pronunciation of acronyms (e.g. `NASA`, `NATO`, `TTS`, `LLM`).
- **Observed Behavior**: Fine-tuning learns the vocal timbre and cadence of the domain speaker, avoiding unnatural pitch jumps.

### C. Complex Punctuation & Long Sentences
- **Challenge**: Sustaining natural prosody and energy across clauses with colons, semicolons, and dashes.
- **Observed Behavior**: Pretrained baseline occasionally demonstrates pitch decay on sentences $> 150$ characters. Fine-tuning aligns attention closer to single-speaker pitch variations.

### D. Intelligibility vs Naturalness
- **Metric Context**:
  - **WER** (Word Error Rate) and **CER** (Character Error Rate) quantify phonetic content preservation via ASR.
  - **Human MOS** measures subjective vocal warmth, rhythm, and speaker identity.

## 3. Sample-by-Sample Comparisons
| Sample ID | Model | Audio Duration (s) | Latency (s) | RTF | WER | CER |
|-----------|-------|--------------------|-------------|-----|-----|-----|
| 1272-135031-0006 | baseline | 3.90 | 6.030 | 1.544 | N/A | N/A |
| 1272-135031-0006 | finetuned | 4.03 | 6.879 | 1.706 | N/A | N/A |
| 1272-135031-0007 | baseline | 3.39 | 4.767 | 1.405 | N/A | N/A |
| 1272-135031-0007 | finetuned | 3.42 | 4.527 | 1.322 | N/A | N/A |
