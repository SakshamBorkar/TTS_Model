"""FastAPI application for TTS Baseline & Voice Chatbot.

Endpoints
---------
POST /api/chat      — process chat message with LLM + TTS voice synthesis
POST /api/tts       — direct text-to-speech synthesis (returns JSON + base64 audio)
POST /synthesize    — synthesise text → WAV binary response (legacy/direct download)
GET  /api/voices    — list available voice/speaker presets
GET  /health        — liveness probe
GET  /metrics       — aggregate runtime statistics
"""

import base64
import io
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import soundfile as sf

from api.llm_service import clean_text_for_tts, llm_service
from src.config import load_config
from src.model import TTSModel
from src.preprocessing import TextPreprocessingError
from src.synthesizer import Synthesizer
from src.utils import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "config": None,
    "model": None,
    "synthesizer": None,
    "loaded": False,
    "device": "unknown",
    "request_count": 0,
    "chat_count": 0,
    "error_count": 0,
    "total_audio_seconds": 0.0,
    "total_inference_seconds": 0.0,
    "latencies": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; clean up on shutdown."""
    config = load_config()
    _state["config"] = config
    model = TTSModel(config)

    logger.info("Loading TTS model at startup ...")
    model.load()

    # Warm-up inference
    logger.info("Running warm-up inference ...")
    synth = Synthesizer(model, config)
    try:
        synth.synthesize("Hello.")
        logger.info("Warm-up complete.")
    except Exception as exc:
        logger.warning("Warm-up failed (non-fatal): %s", exc)

    _state["model"] = model
    _state["synthesizer"] = synth
    _state["loaded"] = True
    _state["device"] = model.get_device()

    yield

    logger.info("Shutting down TTS model.")
    _state["loaded"] = False


app = FastAPI(
    title="TTS Baseline & Voice Chatbot API",
    description="Stage 1 SpeechT5 + HiFi-GAN Voice AI Assistant and baseline API.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite / frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be a non-empty string.")
        return v


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    provider: str = "offline"  # "offline", "openai", "groq", "ollama", "custom"
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    synthesize_voice: bool = True

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must be a non-empty string.")
        return v


class DirectTTSRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be a non-empty string.")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audio_to_base64_data_uri(waveform, sample_rate: int) -> str:
    """Encode numpy waveform into base64 WAV data URI."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint: generates LLM response and synthesizes audio voice.

    Returns
    -------
    JSON with reply text, audio base64 data URI, and latency metrics.
    """
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="TTS Model is not loaded.")

    history_dicts = [{"role": m.role, "content": m.content} for m in request.history]

    # 1. Generate text from LLM
    try:
        reply_text, llm_latency = llm_service.generate_response(
            message=request.message,
            history=history_dicts,
            provider=request.provider,
            api_key=request.api_key,
            model_name=request.model_name,
            base_url=request.base_url,
            system_prompt=request.system_prompt,
        )
    except Exception as exc:
        logger.exception("LLM generation error: %s", exc)
        reply_text = "I apologize, but I encountered an error generating a response."
        llm_latency = 0.0

    audio_uri: Optional[str] = None
    tts_metadata: dict[str, Any] = {}

    # 2. Synthesize audio if requested
    if request.synthesize_voice:
        synth: Synthesizer = _state["synthesizer"]
        cleaned_for_tts = clean_text_for_tts(reply_text)
        if not cleaned_for_tts:
            cleaned_for_tts = "Response received."

        try:
            waveform, meta = synth.synthesize_waveform(cleaned_for_tts)
            sr = _state["config"].audio.sample_rate if _state["config"] else 16000
            audio_uri = _audio_to_base64_data_uri(waveform, sr)
            tts_metadata = meta

            _state["total_audio_seconds"] += meta.get("audio_duration_seconds", 0.0)
            _state["total_inference_seconds"] += meta.get("total_inference_time_seconds", 0.0)
            _state["latencies"].append(meta.get("total_inference_time_seconds", 0.0))
        except Exception as exc:
            logger.warning("TTS synthesis error during chat: %s", exc)

    _state["chat_count"] += 1
    _state["request_count"] += 1

    return {
        "reply": reply_text,
        "audio_url": audio_uri,
        "metrics": {
            "llm_time_seconds": round(llm_latency, 3),
            "tts_time_seconds": round(tts_metadata.get("total_inference_time_seconds", 0.0), 3),
            "audio_duration_seconds": round(tts_metadata.get("audio_duration_seconds", 0.0), 3),
            "real_time_factor": round(tts_metadata.get("real_time_factor", 0.0), 3),
            "device": _state["device"],
        },
    }


@app.post("/api/tts")
async def direct_tts(request: DirectTTSRequest):
    """Synthesize text and return base64 WAV data URI."""
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="TTS Model is not loaded.")

    synth: Synthesizer = _state["synthesizer"]
    try:
        waveform, meta = synth.synthesize_waveform(request.text)
        sr = _state["config"].audio.sample_rate if _state["config"] else 16000
        audio_uri = _audio_to_base64_data_uri(waveform, sr)
    except TextPreprocessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS error: %s", exc)
        raise HTTPException(status_code=500, detail="Synthesis failed.") from exc

    _state["request_count"] += 1
    return {
        "audio_url": audio_uri,
        "metrics": {
            "total_inference_time_seconds": round(meta.get("total_inference_time_seconds", 0.0), 3),
            "audio_duration_seconds": round(meta.get("audio_duration_seconds", 0.0), 3),
            "real_time_factor": round(meta.get("real_time_factor", 0.0), 3),
            "device": _state["device"],
        },
    }


@app.post("/synthesize", response_class=FileResponse)
async def synthesize(request: SynthesizeRequest):
    """Direct WAV download endpoint."""
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    synth: Synthesizer = _state["synthesizer"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        meta = synth.synthesize(request.text, output_path=tmp_path)
    except TextPreprocessingError as exc:
        _state["error_count"] += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _state["error_count"] += 1
        logger.exception("Synthesis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal synthesis error.") from exc

    _state["request_count"] += 1
    _state["total_audio_seconds"] += meta["audio_duration_seconds"]
    _state["total_inference_seconds"] += meta["total_inference_time_seconds"]
    _state["latencies"].append(meta["total_inference_time_seconds"])

    return FileResponse(
        path=tmp_path,
        media_type="audio/wav",
        filename="synthesized.wav",
    )


@app.get("/api/voices")
async def list_voices():
    """List speaker presets and system info."""
    return {
        "current_voice": {
            "name": "Maya (Default Support Voice)",
            "dataset": "Matthijs/cmu-arctic-xvectors",
            "index": 7306,
            "speaker_id": "cmu_us_slt_arctic",
        },
        "voices": [
            {"id": "maya", "name": "Maya (Customer Support)", "index": 7306, "gender": "Female"},
            {"id": "alex", "name": "Alex (Friendly Assistant)", "index": 0, "gender": "Male"},
            {"id": "clara", "name": "Clara (Professional)", "index": 1200, "gender": "Female"},
            {"id": "david", "name": "David (Narrator)", "index": 3500, "gender": "Male"},
        ],
        "model": "microsoft/speecht5_tts",
        "vocoder": "microsoft/speecht5_hifigan",
        "sample_rate": 16000,
        "device": _state["device"],
        "status": "ready" if _state["loaded"] else "loading",
    }


@app.get("/health")
async def health():
    """Liveness probe."""
    return {
        "status": "healthy" if _state["loaded"] else "loading",
        "model_loaded": _state["loaded"],
        "device": _state["device"],
    }


@app.get("/metrics")
async def metrics():
    """Aggregate runtime statistics."""
    import numpy as np

    latencies = _state["latencies"]
    latency_stats: dict = {}
    if latencies:
        arr = np.array(latencies)
        latency_stats = {
            "mean_s": float(np.mean(arr)),
            "p50_s": float(np.percentile(arr, 50)),
            "p95_s": float(np.percentile(arr, 95)),
            "min_s": float(np.min(arr)),
            "max_s": float(np.max(arr)),
        }

    return {
        "request_count": _state["request_count"],
        "chat_count": _state["chat_count"],
        "error_count": _state["error_count"],
        "total_audio_seconds": round(_state["total_audio_seconds"], 3),
        "total_inference_seconds": round(_state["total_inference_seconds"], 3),
        "latency": latency_stats,
        "device": _state["device"],
    }


# Mount built static frontend if present
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

