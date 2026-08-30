# TTS Baseline

> **Stage 1** — Pretrained Transformer TTS inference baseline using  
> SpeechT5 (Microsoft) + HiFi-GAN vocoder

---

## Problem Statement

Building a production voice-agent requires a TTS engine that is fast,
reliable, and measurably correct.  Before optimizing, you need a clean
baseline: a reproducible, instrument-everything starting point that
isolates the model's raw latency and audio quality so subsequent
optimizations can be fairly compared.

## Motivation

This project establishes Stage 1 of a multi-stage TTS development
pipeline.  The goals are:

1. Produce intelligible speech from raw text using a pretrained model.
2. Measure preprocessing, acoustic-model, and vocoder latency **separately**.
3. Calculate the Real-Time Factor (RTF) for every generated sample.
4. Expose the pipeline through a FastAPI endpoint ready for integration.
5. Provide a clean codebase that a reviewer can audit in under an hour.

---

## Architecture

```mermaid
flowchart LR
    A[Raw Text] --> B[Text Preprocessing\nUnicode + whitespace]
    B --> C[SpeechT5Processor\nTokenizer]
    C --> D[SpeechT5\nAcoustic Model]
    D --> E[Speech Representation\nMel-spectrogram]
    E --> F[HiFi-GAN\nVocoder]
    F --> G[Waveform\nfloat32 array]
    G --> H[WAV File\n16 kHz, PCM-16]
```

### Acoustic model vs vocoder

| Component | Role | Model |
|-----------|------|-------|
| **SpeechT5** (acoustic) | Converts tokenized text + speaker embedding → mel-spectrogram-like representation | `microsoft/speecht5_tts` |
| **HiFi-GAN** (vocoder) | Converts the acoustic representation → raw audio waveform | `microsoft/speecht5_hifigan` |

The two stages are kept logically separate in `src/model.py` so that
either can be swapped or optimized independently.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/tts-baseline.git
cd tts-baseline

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU users:** ensure a CUDA-compatible PyTorch is installed.
> See https://pytorch.org/get-started/locally/ for the correct index URL.

Model weights (~500 MB) are downloaded automatically from Hugging Face
on first use.  Set `HF_HOME` to control the cache directory.

---

## Usage

### Single-sentence synthesis

```bash
python scripts/synthesize.py --text "Hello, how can I help you today?"
```

Sample output:

```
──────────────────────────────────────────────────
  TTS BASELINE
──────────────────────────────────────────────────
  Text:            'Hello, how can I help you today?'
  Device:          CUDA
  Audio duration:  2.14 s
  Inference time:  0.83 s
  RTF:             0.388
  Output:          outputs/audio/0001_hello_how_can_i_help.wav
──────────────────────────────────────────────────
```

### Batch synthesis (full test corpus)

```bash
python scripts/synthesize.py --input_file data/input/test_sentences.txt
```

### Save spectrograms alongside audio

```bash
python scripts/synthesize.py --text "Your order is confirmed." --save_spectrograms
```

Outputs `outputs/spectrograms/<stem>_waveform.png` and
`outputs/spectrograms/<stem>_mel_spectrogram.png`.

---

## Benchmarking

```bash
# Default device (auto-selects CUDA if available)
python scripts/benchmark.py

# Force CPU
python scripts/benchmark.py --device cpu

# Force GPU
python scripts/benchmark.py --device cuda
```

Outputs:
- `outputs/reports/baseline_results.csv` — per-sample rows
- `outputs/reports/benchmark_summary.csv` — aggregated statistics
- Human-readable summary printed to stdout

---

## Evaluation

```bash
# Latency + RTF only
python scripts/evaluate.py

# With ASR-based WER (requires additional download of whisper-base)
python scripts/evaluate.py --enable_asr
```

Output: `outputs/reports/evaluation_report.csv`

---

## API

### Launch

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The model is loaded **once** at startup.  A warm-up inference runs
automatically before the server accepts requests.

### Endpoints

#### `POST /synthesize`

```bash
curl -X POST http://localhost:8000/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Your order has been shipped."}' \
     --output synthesized.wav
```

Returns WAV audio (`audio/wav`).

#### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

#### `GET /metrics`

```bash
curl http://localhost:8000/metrics
```

```json
{
  "request_count": 10,
  "error_count": 0,
  "total_audio_seconds": 24.3,
  "total_inference_seconds": 9.1,
  "latency": {
    "mean_s": 0.91,
    "p50_s": 0.87,
    "p95_s": 1.12,
    "min_s": 0.74,
    "max_s": 1.23
  },
  "device": "cpu"
}
```

---

## Evaluation Methodology

### Latency measurement

All timings use `time.perf_counter()` (high-resolution monotonic clock).
Three stages are measured separately:

| Stage | What is measured |
|-------|-----------------|
| Preprocessing | Unicode normalization + whitespace collapse |
| Acoustic model | `generate_speech()` call (includes vocoder when fused) |
| Vocoder | Separate if the two stages are decoupled; 0.0 when fused |

**CUDA synchronization:** when the device is CUDA,
`torch.cuda.synchronize()` is called before stopping the timer so that
GPU kernel completion is included in the measurement.

Model download and initialization time are **never** included in
inference latency.

### Real-Time Factor (RTF)

```
RTF = total_inference_time (s) / generated_audio_duration (s)
```

A RTF < 1 means the model generates speech faster than real time.
Lower is better.  RTF is computed and stored for every sample.

### ASR-based WER (optional)

```
Input Text → TTS → Generated Audio → Whisper ASR → Transcript → WER
```

WER (Word Error Rate) measures whether the generated speech preserves
recognizable linguistic content and intelligibility.

**WER is NOT a complete measure of TTS naturalness or subjective speech
quality.**  A low WER means the words are intelligible; it does not
measure prosody, naturalness, or pleasantness.  Use MOS (Mean Opinion
Score) studies for perceptual quality evaluation.

---

## CPU vs GPU Baseline Benchmarking

Run the benchmark twice — once per device — and compare the resulting
CSV summaries:

```bash
python scripts/benchmark.py --device cpu
python scripts/benchmark.py --device cuda
```

The model is unmodified between runs; this is a pure baseline
measurement with no optimization applied.

---

## Spectrogram Visualization

Generated automatically when `--save_spectrograms` is passed to
`scripts/synthesize.py`, or when `evaluation.save_spectrograms: true`
in `configs/config.yaml`.

Two PNG files are saved per sample under `outputs/spectrograms/`:

- `<stem>_waveform.png` — time-domain amplitude plot
- `<stem>_mel_spectrogram.png` — 80-band mel spectrogram (magma colormap)

Visualization code lives in `src/visualization.py` and is reusable
independently of the synthesis pipeline.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests do **not** require a GPU or model download.  The `TTSModel` is
mocked in `tests/test_synthesis.py` so the full synthesis pipeline is
tested at unit-test speed.

---

## Limitations

- **Single speaker only** — Stage 1 uses one fixed speaker embedding.
- **Max input length** — SpeechT5 tokenizer handles up to ~600 chars
  reliably; longer texts require chunking (not implemented here).
- **English only** — SpeechT5 is trained on English data.
- **No streaming** — the entire waveform is generated before playback
  begins; latency scales with utterance length.
- **No quantization** — this is the unoptimized baseline; RTF > 1 on
  CPU is expected for longer sentences.

---

## Future Work (Stage 2)

1. **ONNX export** — export the acoustic model and vocoder to ONNX for
   runtime-agnostic deployment.
2. **INT8 dynamic quantization** — reduce model size and improve CPU
   throughput.
3. **Streaming synthesis** — implement chunked mel-spectrogram generation
   to reduce time-to-first-byte.
4. **Multi-speaker** — expose speaker selection via the API.
5. **Custom fine-tuning** — LoRA-based adaptation on domain-specific
   voice data.
6. **Docker + Kubernetes** — containerised, horizontally-scalable
   deployment.

---

## Project Structure

```
tts-baseline/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml           # All tunable parameters
├── src/
│   ├── config.py             # YAML config loader
│   ├── preprocessing.py      # Text normalization
│   ├── model.py              # SpeechT5 + HiFi-GAN wrapper
│   ├── synthesizer.py        # End-to-end pipeline
│   ├── audio_utils.py        # I/O, validation, normalization
│   ├── metrics.py            # Statistics & WER
│   ├── visualization.py      # Waveform & spectrogram plots
│   └── utils.py              # Logging, seeding, file helpers
├── scripts/
│   ├── synthesize.py         # CLI — single or batch synthesis
│   ├── benchmark.py          # Benchmark runner + CSV report
│   └── evaluate.py           # Evaluation report + optional WER
├── api/
│   └── main.py               # FastAPI application
├── tests/
│   ├── test_preprocessing.py
│   ├── test_audio.py
│   └── test_synthesis.py
├── data/
│   └── input/
│       └── test_sentences.txt
├── experiments/
│   └── baseline/
│       └── experiment.md
└── outputs/
    ├── audio/
    ├── spectrograms/
    └── reports/
```

---

## License

MIT
