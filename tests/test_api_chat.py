"""Tests for FastAPI Chat & Voice Endpoints."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from api.llm_service import clean_text_for_tts, llm_service


class TestChatAPI(unittest.TestCase):
    def test_clean_text_for_tts(self):
        raw = "Here is a `code` and [link](http://example.com) and **bold**.\n```python\nprint('hi')\n```"
        cleaned = clean_text_for_tts(raw)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("http://example.com", cleaned)
        self.assertIn("code", cleaned)
        self.assertIn("bold", cleaned)

    def test_llm_service_offline(self):
        reply, latency = llm_service.generate_response("Hello!", provider="offline")
        self.assertTrue("SpeechT5" in reply or "Hello" in reply)
        self.assertGreaterEqual(latency, 0.0)

    def test_api_voices(self):
        with TestClient(app) as client:
            resp = client.get("/api/voices")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("voices", data)
            self.assertGreater(len(data["voices"]), 0)
            self.assertEqual(data["model"], "microsoft/speecht5_tts")

    def test_api_health(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["model_loaded"])

    def test_api_chat_offline(self):
        with TestClient(app) as client:
            payload = {
                "message": "Hello, can you help me?",
                "provider": "offline",
                "synthesize_voice": True,
            }
            resp = client.post("/api/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("reply", data)
            self.assertIn("audio_url", data)
            self.assertTrue(data["audio_url"].startswith("data:audio/wav;base64,"))
            self.assertIn("metrics", data)
            self.assertGreater(data["metrics"]["audio_duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
