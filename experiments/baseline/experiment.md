# Experiment Log — Stage 1 Baseline

## Experiment ID
`stage1-baseline-001`

## Date
<!-- Fill in after running benchmarks -->
YYYY-MM-DD

## Model
- Acoustic: `microsoft/speecht5_tts`
- Vocoder: `microsoft/speecht5_hifigan`
- Speaker embedding: CMU Arctic x-vectors, index 7306

## Device
<!-- Fill in after running benchmarks -->
CPU / CUDA (specify)

## Sample Rate
16000 Hz

## Test Corpus
`data/input/test_sentences.txt` — 25 sentences  
Covers: greetings, verification codes, dates, currency, numbers, punctuation,
short/long sentences, customer-support language.

---

## Results

### CPU

| Metric                      | Mean | P50 | P95 | Min | Max |
|-----------------------------|------|-----|-----|-----|-----|
| Pre-processing latency (s)  | —    | —   | —   | —   | —   |
| Acoustic model latency (s)  | —    | —   | —   | —   | —   |
| Vocoder latency (s)         | —    | —   | —   | —   | —   |
| Total inference latency (s) | —    | —   | —   | —   | —   |
| RTF                         | —    | —   | —   | —   | —   |
| Audio duration (s)          | —    | —   | —   | —   | —   |

### GPU (if available)

| Metric                      | Mean | P50 | P95 | Min | Max |
|-----------------------------|------|-----|-----|-----|-----|
| Pre-processing latency (s)  | —    | —   | —   | —   | —   |
| Acoustic model latency (s)  | —    | —   | —   | —   | —   |
| Vocoder latency (s)         | —    | —   | —   | —   | —   |
| Total inference latency (s) | —    | —   | —   | —   | —   |
| RTF                         | —    | —   | —   | —   | —   |
| Audio duration (s)          | —    | —   | —   | —   | —   |

> **Note:** Fill in after running `python scripts/benchmark.py`.
> Do not invent measurements.

---

## Observations
<!-- Fill in after running benchmarks -->

- Model download size: ~
- First-run latency (cold start, including download): ~
- Subsequent latency (model in memory): ~
- Voice quality (subjective): ~
- Notable artefacts: ~

## Failure Cases
<!-- Document any sentences the model handles poorly -->

- Very long sentences (>500 chars): may be truncated by the tokenizer
- Highly abbreviated text: e.g. "ETA tmrw" — naturalness degrades

## Next Experiment
`stage2-optimization-001` — latency reduction via:

1. ONNX export + ONNX Runtime optimization
2. Dynamic INT8 quantization
3. Streaming synthesis for lower TTFB
4. Multi-speaker support
