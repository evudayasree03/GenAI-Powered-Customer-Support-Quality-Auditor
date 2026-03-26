"""
SamiX API — Dependency Injection

Provides singleton instances of core managers for FastAPI endpoints.
All secrets are read from environment variables (not st.secrets).
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

logger = logging.getLogger("samix.api")


def _ensure_env() -> None:
    """Ensure critical env vars exist, mapping from st.secrets-style names."""
    mapping = {
        "GROQ_API_KEY":    lambda: os.getenv("GROQ_API_KEY", ""),
        "DEEPGRAM_API_KEY": lambda: os.getenv("DEEPGRAM_API_KEY", ""),
        "HF_TOKEN":        lambda: os.getenv("HF_TOKEN", ""),
    }
    for key, getter in mapping.items():
        if not os.getenv(key):
            val = getter()
            if val:
                os.environ[key] = val


@lru_cache(maxsize=1)
def get_groq_client():
    """Singleton GroqClient (works without Streamlit)."""
    _ensure_env()
    from src.pipeline.groq_client import GroqClient
    return GroqClient()


@lru_cache(maxsize=1)
def get_stt_processor():
    """Singleton STTProcessor."""
    _ensure_env()
    from src.pipeline.stt_processor import STTProcessor
    return STTProcessor()


@lru_cache(maxsize=1)
def get_kb_manager():
    """Singleton KBManager."""
    _ensure_env()
    from src.utils.kb_manager import KBManager
    return KBManager()


@lru_cache(maxsize=1)
def get_audio_processor():
    """Singleton AudioProcessor."""
    from src.utils.audio_processor import AudioProcessor
    return AudioProcessor()


@lru_cache(maxsize=1)
def get_cost_tracker():
    """Singleton CostTracker."""
    from src.utils.cost_tracker import CostTracker
    return CostTracker()


@lru_cache(maxsize=1)
def get_alert_engine():
    """Singleton AlertEngine."""
    from src.pipeline.alert_engine import AlertEngine
    return AlertEngine()
