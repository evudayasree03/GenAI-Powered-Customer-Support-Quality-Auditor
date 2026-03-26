"""
SamiX configuration module.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st


def _safe_console(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


def get_secret(*keys: str, default: str = "") -> str:
    """
    Retrieve a secret from environment variables or Streamlit secrets.
    Supports nested keys like get_secret("groq", "api_key").
    Falls back to env vars: GROQ_API_KEY, DEEPGRAM_API_KEY, etc.
    """
    # 1. Try flat env var: GROQ_API_KEY
    env_key = "_".join(k.upper() for k in keys)
    val = os.getenv(env_key, "")
    if val:
        return val

    # 2. Try Streamlit secrets (only when running inside Streamlit)
    try:
        import streamlit as _st
        obj = _st.secrets
        for k in keys:
            obj = obj[k]
        return str(obj)
    except Exception:
        pass

    return default



class Config:
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    API_RESPONSES_DIR = DATA_DIR / "api_responses"
    TRANSCRIPTIONS_DIR = API_RESPONSES_DIR / "transcriptions"
    LLM_SCORES_DIR = API_RESPONSES_DIR / "llm_scores"
    RAG_RESULTS_DIR = API_RESPONSES_DIR / "rag_results"
    CACHE_DIR = API_RESPONSES_DIR / "cache"
    BACKUPS_DIR = DATA_DIR / "backups"
    KB_DIR = DATA_DIR / "kb"
    AUTH_DIR = DATA_DIR / "auth"
    HISTORY_DIR = DATA_DIR / "history"
    UPLOADS_DIR = DATA_DIR / "uploads"
    EXPORTS_DIR = DATA_DIR / "exports"
    LOGS_DIR = PROJECT_ROOT / "logs"

    for directory in [
        DATA_DIR,
        API_RESPONSES_DIR,
        TRANSCRIPTIONS_DIR,
        LLM_SCORES_DIR,
        RAG_RESULTS_DIR,
        CACHE_DIR,
        BACKUPS_DIR,
        KB_DIR,
        AUTH_DIR,
        HISTORY_DIR,
        UPLOADS_DIR,
        EXPORTS_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    SQLITE_DB_PATH = str(PROJECT_ROOT / "samix.db")
    SQLITE_BACKUP_PATH = str(PROJECT_ROOT / "samix_backup.db")
    MILVUS_DB_PATH = str(PROJECT_ROOT / "milvus_lite.db")
    USERS_YAML = str(AUTH_DIR / "users.yaml")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / os.getenv("LOG_FILE", "samix.log")

    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K = 5
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    TEXT_EXTS = {".csv", ".json", ".txt"}
    CONF_THRESHOLD = 0.70

    @staticmethod
    def get_groq_api_key() -> str:
        try:
            return st.secrets["groq"]["api_key"]
        except Exception:
            return os.getenv("GROQ_API_KEY", "NOT_CONFIGURED")

    @staticmethod
    def get_deepgram_api_key() -> str:
        try:
            return st.secrets["deepgram"]["api_key"]
        except Exception:
            return os.getenv("DEEPGRAM_API_KEY", "NOT_CONFIGURED")

    @staticmethod
    def get_email_config() -> dict[str, str | int]:
        try:
            return {
                "smtp_host": st.secrets["email"]["smtp_host"],
                "smtp_port": st.secrets["email"]["smtp_port"],
                "sender_address": st.secrets["email"]["sender_address"],
                "sender_password": st.secrets["email"]["sender_password"],
            }
        except Exception:
            return {
                "smtp_host": os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("EMAIL_SMTP_PORT", "587")),
                "sender_address": os.getenv("EMAIL_SENDER_ADDRESS", ""),
                "sender_password": os.getenv("EMAIL_SENDER_PASSWORD", ""),
            }

    @staticmethod
    def validate_configuration() -> list[str]:
        errors: list[str] = []
        groq_key = Config.get_groq_api_key()
        if not groq_key or groq_key == "NOT_CONFIGURED" or "your_" in groq_key.lower():
            errors.append("WARNING: GROQ_API_KEY not properly configured")

        deepgram_key = Config.get_deepgram_api_key()
        if "your_" in deepgram_key.lower():
            _safe_console("INFO: DEEPGRAM_API_KEY not configured - local Whisper fallback will be used")
        return errors

    @classmethod
    def print_status(cls) -> None:
        print("\n" + "=" * 60)
        print("SamiX Configuration Status")
        print("=" * 60)
        print(f"Environment: {cls.APP_ENV}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"SQLite DB: {cls.SQLITE_DB_PATH}")
        print(f"Milvus DB: {cls.MILVUS_DB_PATH}")
        print(f"Users File: {cls.USERS_YAML}")
        _safe_console(
            f"Groq API: {'Configured' if 'gsk_' in cls.get_groq_api_key() else 'Not configured'}"
        )
        _safe_console(
            "Deepgram API: "
            + ("Configured" if "your_" not in cls.get_deepgram_api_key().lower() else "Using Whisper fallback")
        )
        print("=" * 60 + "\n")


errors = Config.validate_configuration()
if errors:
    for error in errors:
        _safe_console(error)
