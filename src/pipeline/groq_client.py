"""
SamiX AI Inference Engine (Groq Client) - Cloud Optimized
- Client Mode: Routes analysis requests to Render Backend (Streamlit safe).
- Server Mode: Executes dual-call Llama-3 pipeline on Render.
"""
from __future__ import annotations

import json
import os
import requests
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Any, List

# Use logging for backend compatibility
logger = logging.getLogger("samix.groq")

# Attempt imports for Server Mode; fail silently for Client Mode (Streamlit)
try:
    from src.db import get_db
    from src.storage import FileStorage
    from src.utils.history_manager import (
        AuditScores, TranscriptTurn, WrongTurn,
        EngineAResult, EngineBClaim, EngineBResult, EngineCResult
    )
except ImportError:
    # These will only be used in Server Mode, so we can ignore them in Client Mode
    pass

@dataclass
class SummaryResult:
    customer_query: str
    sub_queries: list[str]
    query_category: str
    customer_expectation: str
    phases: dict
    key_moments: list[str]

@dataclass
class ScoringResult:
    scores: Any # AuditScores
    engine_a: Any # EngineAResult
    engine_b: Any # EngineBResult
    engine_c: Any # EngineCResult
    violations: list[dict]
    wrong_turns: list[Any] # list[WrongTurn]
    auto_fail: bool
    auto_fail_reason: str
    token_count: int

# --- AI System Prompt ---
_SCORING_SYSTEM_PROMPT = """
You are SamiX, a senior quality auditor. Return ONLY valid JSON.
Scoring weights: empathy 20%, professionalism 15%, compliance 25%, resolution 20%, communication 5%, integrity 15%.
Include the following JSON structure: 
{
  "empathy": float, "professionalism": float, "compliance": float, 
  "resolution": float, "communication": float, "integrity": float,
  "opening_score": float, "middle_score": float, "closing_score": float,
  "final_score": float, "verdict": string, 
  "engine_a": {"primary_query_answered": bool, ...},
  "violations": [], "auto_fail": bool, "auto_fail_reason": string
}
"""

class GroqClient:
    MODEL: str = "llama-3.3-70b-versatile"

    def __init__(self, api_base: Optional[str] = None) -> None:
        # Detect Environment: If BACKEND_URL exists, we are a Frontend Client
        self.api_url = api_base or os.environ.get("BACKEND_URL")
        self.is_client = self.api_url is not None
        
        if self.is_client:
            logger.info(f"Groq Client Mode: Routing to {self.api_url}")
            self._async_client = None
        else:
            logger.info("Groq Server Mode: Initializing AsyncGroq")
            self._async_client = self._build_async_client()
            # Database and storage are only needed on the server (Render)
            try:
                self._db = get_db()
                self._storage = FileStorage()
            except NameError:
                logger.warning("DB/Storage not imported; likely running in client-only mode.")

    def _build_async_client(self) -> Optional[object]:
        try:
            from groq import AsyncGroq
            key = os.environ.get("GROQ_API_KEY", "")
            return AsyncGroq(api_key=key) if key else None
        except Exception:
            return None

    @property
    def is_live(self) -> bool:
        if self.is_client: return True
        return self._async_client is not None

    async def summarise(self, transcript_text: str, session_id: Optional[str] = None) -> SummaryResult:
        if self.is_client:
            return await self._proxy_request("summarise", {"transcript": transcript_text})
        return await self._real_summarise(transcript_text, session_id or str(uuid.uuid4())[:8])

    async def score(self, transcript_text: str, summary: SummaryResult, rag_context: str = "", session_id: Optional[str] = None) -> ScoringResult:
        if self.is_client:
            # Flatten summary if it's a dataclass
            summary_data = summary.__dict__ if hasattr(summary, '__dict__') else summary
            return await self._proxy_request("score", {
                "transcript": transcript_text, 
                "summary": summary_data, 
                "rag": rag_context
            })
        return await self._real_score(transcript_text, summary, rag_context, session_id or str(uuid.uuid4())[:8])

    async def _proxy_request(self, endpoint: str, payload: dict):
        """Frontend Proxy: Sends data to Render and returns the result."""
        try:
            # Using a slightly longer timeout for LLM processing
            resp = requests.post(f"{self.api_url}/{endpoint}", json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if endpoint == "summarise": return SummaryResult(**data)
                return data 
            logger.error(f"Backend returned error {resp.status_code}: {resp.text}")
            return self._mock_summary() if endpoint == "summarise" else self._mock_scoring()
        except Exception as e:
            logger.error(f"Proxy connection failed: {e}")
            return self._mock_summary() if endpoint == "summarise" else self._mock_scoring()

    # --- Server Implementations (Render only) ---

    async def _real_summarise(self, transcript_text: str, session_id: str) -> SummaryResult:
        prompt = f"Summarize this transcript into a JSON object with: customer_query, sub_queries (list), query_category, customer_expectation, phases (dict), and key_moments (list).\n\nTranscript:\n{transcript_text}"
        try:
            resp = await self._async_client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            return SummaryResult(
                customer_query=data.get("customer_query", ""),
                sub_queries=data.get("sub_queries", []),
                query_category=data.get("query_category", "General"),
                customer_expectation=data.get("customer_expectation", ""),
                phases=data.get("phases", {}),
                key_moments=data.get("key_moments", [])
            )
        except Exception as e:
            logger.error(f"LLM Summary failed: {e}")
            return self._mock_summary()

    async def _real_score(self, transcript_text: str, summary: SummaryResult, rag_context: str, session_id: str) -> ScoringResult:
        user_msg = f"Transcript: {transcript_text}\nSummary Context: {summary}\nPolicy RAG Context: {rag_context}"
        try:
            resp = await self._async_client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.05,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            # Capture total tokens from the response usage metadata
            tokens = getattr(resp.usage, 'total_tokens', 0)
            return self._parse_scoring_response(data, tokens)
        except Exception as e:
            logger.error(f"LLM Scoring failed: {e}")
            return self._mock_scoring()

    def _parse_scoring_response(self, d: dict, tokens: int) -> ScoringResult:
        """Parses LLM JSON and maps it to SamiX Audit Objects."""
        cust_sent = d.get("customer_sentiment_by_turn", [5.0])
        agent_turn = d.get("agent_score_by_turn", [5.0])
        
        # Ensure we are using the correct AuditScores dataclass if available
        try:
            scores = AuditScores(
                empathy=float(d.get("empathy", 5.0)),
                professionalism=float(d.get("professionalism", 5.0)),
                compliance=float(d.get("compliance", 5.0)),
                resolution=float(d.get("resolution", 5.0)),
                communication=float(d.get("communication", 5.0)),
                integrity=float(d.get("integrity", 5.0)),
                opening=float(d.get("opening_score", 5.0)),
                middle=float(d.get("middle_score", 5.0)),
                closing=float(d.get("closing_score", 5.0)),
                phase_bonus=float(d.get("phase_bonus", 0.0)),
                final_score=float(d.get("final_score", 50.0)),
                verdict=d.get("verdict", "Neutral"),
                customer_sentiment=cust_sent,
                customer_overall=sum(cust_sent)/len(cust_sent) if cust_sent else 5.0,
                agent_by_turn=agent_turn
            )
        except NameError:
            scores = d # Fallback to dict if classes aren't imported

        return ScoringResult(
            scores=scores,
            engine_a=d.get("engine_a", {"primary_query_answered":True, "sub_queries_addressed":True, "is_fake_close":False, "resolution_state":"Closed"}),
            engine_b=d.get("engine_b", {"claims": []}),
            engine_c=d.get("engine_c", {"customer_frustrated_but_ok":False, "agent_rushed":False}),
            violations=d.get("violations", []),
            wrong_turns=d.get("wrong_turns", []),
            auto_fail=d.get("auto_fail", False),
            auto_fail_reason=d.get("auto_fail_reason", ""),
            token_count=tokens
        )

    def _mock_summary(self) -> SummaryResult:
        return SummaryResult("Query processing error", [], "Error", "Wait for retry", {}, [])

    def _mock_scoring(self) -> ScoringResult:
        return ScoringResult(None, None, None, None, [], [], False, "Inference engine timeout", 0)
