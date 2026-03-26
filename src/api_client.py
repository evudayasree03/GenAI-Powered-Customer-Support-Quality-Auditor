"""
SamiX API Client — HTTP Client for Streamlit → FastAPI

Used by the Streamlit frontend to call the FastAPI backend.
Falls back to local processing if SAMIX_API_URL is not set.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Any

import httpx

logger = logging.getLogger("samix.client")

# Default timeout: 120s for audits (STT + LLM can be slow)
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class SamiXClient:
    """Thin HTTP wrapper around the SamiX FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("SAMIX_API_URL", "")).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=_TIMEOUT) if self.base_url else None
        self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT) if self.base_url else None

    @property
    def is_available(self) -> bool:
        """True if a backend URL is configured."""
        return bool(self.base_url)

    def health(self) -> dict[str, Any]:
        """GET /health — synchronous."""
        if not self._client:
            return {"status": "local", "note": "No API URL configured"}
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def run_audit(self, filename: str, file_bytes: bytes) -> dict[str, Any]:
        """POST /audit — async, sends audio file, returns full audit JSON."""
        if not self._async_client:
            raise RuntimeError("No API URL configured")
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        resp = await self._async_client.post("/audit", files=files)
        resp.raise_for_status()
        return resp.json()

    async def query_rag(
        self,
        question: str,
        top_k: int = 5,
        collection: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /rag/query — async."""
        if not self._async_client:
            raise RuntimeError("No API URL configured")
        payload = {"question": question, "top_k": top_k}
        if collection:
            payload["collection"] = collection
        resp = await self._async_client.post("/rag/query", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def upload_kb(
        self,
        filename: str,
        file_bytes: bytes,
        collection: str = "policies",
    ) -> dict[str, Any]:
        """POST /kb/upload — async."""
        if not self._async_client:
            raise RuntimeError("No API URL configured")
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"collection": collection}
        resp = await self._async_client.post("/kb/upload", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
