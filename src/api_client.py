"""
SamiX API Client — Optimized for Streamlit + FastAPI (2026)
Location: src/api_client.py
"""
from __future__ import annotations

import os
import httpx
import streamlit as st
from typing import Optional, Any, Dict

# High timeout (150s) because Transcription + LLM scoring takes time
_TIMEOUT = httpx.Timeout(150.0, connect=10.0)

class SamiXClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        """
        Initializes the client. 
        Priority: 1. Passed Arg, 2. Render Env Var, 3. Localhost (Port 10000)
        """
        self.base_url = (
            base_url or 
            os.getenv("BACKEND_URL") or 
            os.getenv("SAMIX_API_URL", "http://localhost:10000")
        ).rstrip("/")
        
        # Persistent clients for better performance
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_auth_header(self) -> Dict[str, str]:
        """Injects the JWT token from session state into every request."""
        token = st.session_state.get("auth_token", "")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def health(self) -> dict[str, Any]:
        """Synchronous check to see if the Render backend is awake."""
        if not self._client or self._client.is_closed:
            self._client = httpx.Client(base_url=self.base_url, timeout=10.0)
        try:
            resp = self._client.get("/health")
            return resp.json()
        except Exception:
            return {"status": "offline"}

    async def run_audit(self, filename: str, file_bytes: bytes, agent_name: str) -> dict[str, Any]:
        """Asynchronous call to process audio and get scores."""
        if not self._async_client or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT)
            
        files = {"file": (filename, file_bytes, "audio/mpeg")}
        data = {"agent_name": agent_name}
        
        resp = await self._async_client.post(
            "/api/v1/audit", 
            files=files, 
            data=data, 
            headers=self._get_auth_header()
        )
        resp.raise_for_status()
        return resp.json()

    async def query_rag(self, question: str) -> dict[str, Any]:
        """POST /rag/query — Interfaces with the Knowledge Base via FastAPI."""
        if not self._async_client or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT)
            
        resp = await self._async_client.post(
            "/api/v1/rag/query", 
            json={"question": question}, 
            headers=self._get_auth_header()
        )
        resp.raise_for_status()
        return resp.json()
