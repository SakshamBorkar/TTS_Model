"""LLM Service module for Voice Chatbot.

Provides flexible LLM response generation:
1. OpenAI-compatible API providers (OpenAI, Groq, Ollama, OpenRouter, LocalAI)
2. Built-in Offline Smart Assistant (zero external API keys needed)
3. Text cleaning/formatting for high-quality speech synthesis
"""

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Offline Assistant Knowledge & Fallback Generator
# ---------------------------------------------------------------------------

OFFLINE_RESPONSES = [
    (
        r"(?i)\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))\b",
        "Hello! I am your AI Voice Assistant powered by SpeechT5. How can I help you today?",
    ),
    (
        r"(?i)\b(order|delivery|shipping|package|track)\b",
        "I can help check your order status. Your latest order has been shipped and is scheduled to arrive by tomorrow evening.",
    ),
    (
        r"(?i)\b(balance|account|statement|funds|money)\b",
        "Your current account balance is two thousand five hundred rupees. Would you like a detailed statement sent to your email?",
    ),
    (
        r"(?i)\b(refund|return|money\s*back)\b",
        "Your refund of three hundred and fifty rupees has been processed and will reflect in your account within three to five business days.",
    ),
    (
        r"(?i)\b(appointment|schedule|booking|reschedule)\b",
        "Your appointment is scheduled for September third at two thirty in the afternoon. Please let me know if you would like to reschedule.",
    ),
    (
        r"(?i)\b(password|reset|login|access)\b",
        "To reset your password, please click the secure link we have sent to your registered email address.",
    ),
    (
        r"(?i)\b(who\s*are\s*you|what\s*are\s*you|introduce\s*yourself|what\s*is\s*this)\b",
        "I am an intelligent conversational voice chatbot running a SpeechT5 acoustic model and HiFi-GAN neural vocoder for high-fidelity speech synthesis.",
    ),
    (
        r"(?i)\b(speecht5|hifigan|tts|text\s*to\s*speech|architecture|model)\b",
        "This system uses SpeechT5 for sequence-to-sequence text-to-speech encoding and HiFi-GAN as the neural vocoder to generate high-quality audio waveforms in real time.",
    ),
    (
        r"(?i)\b(support|help|agent|human|representative|contact)\b",
        "I would be glad to assist you or connect you with a customer support representative. What issue are you experiencing?",
    ),
    (
        r"(?i)\b(thank|thanks|great|awesome|good\s*job)\b",
        "You are very welcome! Is there anything else I can assist you with today?",
    ),
]


def clean_text_for_tts(text: str) -> str:
    """Strip markdown formatting, URLs, code blocks, and symbols for clean speech."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " [code snippet omitted] ", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "link", text)
    # Remove markdown headers and bold/italics markers
    text = re.sub(r"[#*_~>`]", "", text)
    # Replace multiple spaces / newlines with single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LLMService:
    """Manages LLM queries across different providers with offline fallback."""

    def __init__(self) -> None:
        pass

    def generate_response(
        self,
        message: str,
        history: Optional[list[dict]] = None,
        provider: str = "offline",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, float]:
        """Generate response text and measure latency in seconds.

        Returns
        -------
        tuple[str, float]
            (response_text, latency_seconds)
        """
        t0 = time.perf_counter()

        system_msg = system_prompt or (
            "You are a helpful, courteous AI voice assistant. "
            "Keep your responses concise, conversational, friendly, and easy to speak out loud. "
            "Avoid complex markdown tables, raw URLs, or long bulleted lists."
        )

        effective_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")

        if provider in ("openai", "groq", "ollama", "custom") and (effective_key or provider in ("ollama", "custom")):
            try:
                reply = self._query_openai_compatible(
                    message=message,
                    history=history or [],
                    provider=provider,
                    api_key=effective_key or "ollama",
                    model_name=model_name,
                    base_url=base_url,
                    system_prompt=system_msg,
                )
                latency = time.perf_counter() - t0
                return reply, latency
            except Exception as exc:
                logger.warning(
                    "LLM provider %s failed (%s). Falling back to offline assistant.",
                    provider,
                    exc,
                )

        # Fallback to Built-in Assistant
        reply = self._query_offline(message)
        latency = time.perf_counter() - t0
        return reply, latency

    def _query_openai_compatible(
        self,
        message: str,
        history: list[dict],
        provider: str,
        api_key: str,
        model_name: Optional[str],
        base_url: Optional[str],
        system_prompt: str,
    ) -> str:
        """Call an OpenAI-compatible endpoint."""
        from openai import OpenAI

        effective_base_url = base_url
        effective_model = model_name

        if provider == "groq":
            effective_base_url = effective_base_url or "https://api.groq.com/openai/v1"
            effective_model = effective_model or "llama-3.3-70b-versatile"
        elif provider == "ollama":
            effective_base_url = effective_base_url or "http://localhost:11434/v1"
            effective_model = effective_model or "llama3.2"
        else:
            effective_model = effective_model or "gpt-4o-mini"

        client = OpenAI(
            api_key=api_key,
            base_url=effective_base_url,
        )

        messages = [{"role": "system", "content": system_prompt}]
        for item in history[-6:]:  # include up to 6 recent messages for context
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=effective_model,
            messages=messages,
            max_tokens=250,
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _query_offline(self, message: str) -> str:
        """Pattern-matching offline fallback knowledge base."""
        for pattern, answer in OFFLINE_RESPONSES:
            if re.search(pattern, message):
                return answer

        return (
            f"I understand your query regarding: \"{message}\". "
            "As your assistant, I am processing this request and can help clarify or assist with any additional questions."
        )


llm_service = LLMService()
