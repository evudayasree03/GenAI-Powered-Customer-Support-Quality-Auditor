"""
SamiX API — Pydantic Schemas
Updated to support the Alert Engine and Production Audit Pipeline.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    groq_live: bool = False
    deepgram_live: bool = False
    vector_enabled: bool = False
    kb_documents: int = 0


# ── Audit Components ────────────────────────────────────────────────

class WrongTurnOut(BaseModel):
    turn_number: int = 0
    speaker: str = "AGENT"
    timestamp: str = ""
    agent_said: str = ""
    what_went_wrong: str = ""
    correct_fact: str = ""
    rag_source: str = ""
    rag_confidence: float = 0.0
    score_impact: str = ""
    specific_correction: str = ""


class ScoresOut(BaseModel):
    empathy: float = 5.0
    professionalism: float = 5.0
    compliance: float = 5.0
    resolution: float = 5.0
    communication: float = 5.0
    integrity: float = 5.0
    opening: float = 5.0
    middle: float = 5.0
    closing: float = 5.0
    phase_bonus: float = 0.0
    final_score: float = 50.0
    verdict: str = "Needs work"
    customer_sentiment: List[float] = []
    customer_overall: float = 5.0
    agent_by_turn: List[float] = []


class EngineAOut(BaseModel):
    primary_query_answered: bool = False
    sub_queries_addressed: bool = False
    is_fake_close: bool = False
    resolution_state: str = "Unknown"


class EngineBClaimOut(BaseModel):
    claim: str = ""
    is_unverifiable: bool = False
    is_impossible_promise: bool = False
    is_contradiction: bool = False
    confidence_score: float = 0.0


class EngineBOut(BaseModel):
    claims: List[EngineBClaimOut] = []


class EngineCOut(BaseModel):
    customer_frustrated_but_ok: bool = False
    agent_rushed: bool = False
    resolution_confirmed_by_customer: bool = False


class SummaryOut(BaseModel):
    customer_query: str = ""
    sub_queries: List[str] = []
    query_category: str = ""
    customer_expectation: str = ""
    phases: Dict[str, Any] = {}
    key_moments: List[str] = []


class ViolationOut(BaseModel):
    type: str = ""
    phase: str = ""
    severity: str = ""


class AuditResponse(BaseModel):
    """Full audit result returned by POST /audit."""
    session_id: str = ""
    filename: str = ""
    summary: SummaryOut = Field(default_factory=SummaryOut)
    scores: ScoresOut = Field(default_factory=ScoresOut)
    engine_a: EngineAOut = Field(default_factory=EngineAOut)
    engine_b: EngineBOut = Field(default_factory=EngineBOut)
    engine_c: EngineCOut = Field(default_factory=EngineCOut)
    violations: List[ViolationOut] = []
    wrong_turns: List[WrongTurnOut] = []
    token_count: int = 0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    auto_fail: bool = False
    auto_fail_reason: str = ""
    # CRITICAL: This allows the backend to pass the triggered alert list to the UI
    alerts: List[str] = [] 


# ── RAG ─────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    collection: Optional[str] = None


class RAGChunkOut(BaseModel):
    text: str = ""
    source: str = ""
    collection: str = ""
    score: float = 0.0


class RAGQueryResponse(BaseModel):
    results: List[RAGChunkOut] = []
    groundedness: float = 0.0


# ── KB Upload ───────────────────────────────────────────────────────

class KBUploadResponse(BaseModel):
    filename: str
    collection: str
    chunks: int = 0
    size_bytes: int = 0
