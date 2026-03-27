"""
SamiX FastAPI Backend — Main Application
Updated for Production Deployment on Render.com
"""
from __future__ import annotations

import os
import sys
import asyncio
import logging
import uuid
from pathlib import Path

# Ensure project root is on sys.path so internal modules work correctly
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Audioop polyfill (needed for some Python 3.11+ environments)
try:
    import audioop
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Internal Imports
from api.schemas import (
    HealthResponse,
    AuditResponse,
    SummaryOut,
    ScoresOut,
    EngineAOut,
    EngineBClaimOut,
    EngineBOut,
    EngineCOut,
    ViolationOut,
    WrongTurnOut,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGChunkOut,
    KBUploadResponse,
)
from api.deps import (
    get_groq_client,
    get_stt_processor,
    get_kb_manager,
    get_audio_processor,
    get_cost_tracker,
)
from src.pipeline.alert_engine import AlertEngine

# ── Logging Setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("samix.api")

# Disable HF telemetry for cleaner logs
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ── App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="SamiX API",
    description="GenAI-Powered Customer Support Quality Auditor Backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Alert Engine
alert_engine = AlertEngine()

# ── Helpers ─────────────────────────────────────────────────────────
def _transcript_to_text(turns) -> str:
    """Convert list of TranscriptTurn objects to formatted plain text."""
    return "\n".join([f"{getattr(t, 'speaker', 'UNKNOWN')} [T{getattr(t, 'turn', 0)} {getattr(t, 'timestamp', '')}]: {getattr(t, 'text', '')}" for t in turns])

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "message": "SamiX API is running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health():
    groq = get_groq_client()
    stt = get_stt_processor()
    kb = get_kb_manager()
    return HealthResponse(
        status="ok",
        groq_live=groq.is_live,
        deepgram_live=stt.has_deepgram,
        vector_enabled=kb.is_vector_enabled,
        kb_documents=len(kb.files) + len(kb.generalised_kb),
    )

@app.post("/audit", response_model=AuditResponse)
async def run_audit(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"upload_{session_id}.wav"
    file_bytes = await file.read()

    # Get injected dependencies
    groq = get_groq_client()
    stt = get_stt_processor()
    kb = get_kb_manager()
    audio = get_audio_processor()
    cost = get_cost_tracker()

    try:
        # 1. Process and Transcribe
        logger.info(f"[{session_id}] Processing: {filename}")
        turns = await stt.process(file_bytes, filename, session_id=session_id)
        tx_text = _transcript_to_text(turns)

        # 2. Get Audio Metadata for cost tracking
        _, meta = await asyncio.to_thread(audio.convert_to_wav, file_bytes, filename)
        duration = meta.get("duration_sec", 0)

        # 3. Analyze with LLM
        summary = await groq.summarise(tx_text, session_id=session_id)
        
        # 4. RAG Retrieval
        rag_results = await kb.query(
            question=f"{summary.customer_query} {' '.join(summary.sub_queries)}",
            top_k=5
        )
        rag_context = "\n\n".join([f"[{r.source}]\n{r.text}" for r in rag_results])

        # 5. Full Scoring Audit
        scoring = await groq.score(tx_text, summary, rag_context=rag_context, session_id=session_id)

        # 6. TRIGGER ALERTS
        alerts_triggered = await alert_engine.check_and_fire(
            filename=filename,
            agent_name="Agent", # Future: extract from turns if possible
            final_score=scoring.scores.final_score,
            violations=scoring.violations,
            auto_fail=scoring.auto_fail,
            auto_fail_reason=scoring.auto_fail_reason,
            recipient_email=os.getenv("ADMIN_EMAIL", "")
        )

        # 7. Calculate Costs
        cost_obj = cost.calculate_session_cost(
            token_count=scoring.token_count,
            audio_duration_sec=duration
        )

        # 8. Construct Response
        s = scoring.scores
        return AuditResponse(
            session_id=session_id,
            filename=filename,
            summary=SummaryOut(**summary.__dict__),
            scores=ScoresOut(
                empathy=s.empathy, professionalism=s.professionalism, compliance=s.compliance,
                resolution=s.resolution, communication=s.communication, integrity=s.integrity,
                opening=s.opening, middle=s.middle, closing=s.closing,
                phase_bonus=s.phase_bonus, final_score=s.final_score, verdict=s.verdict,
                customer_sentiment=s.customer_sentiment, customer_overall=s.customer_overall,
                agent_by_turn=s.agent_by_turn
            ),
            engine_a=EngineAOut(**scoring.engine_a.__dict__),
            engine_b=EngineBOut(claims=[EngineBClaimOut(**c.__dict__) for c in scoring.engine_b.claims]),
            engine_c=EngineCOut(**scoring.engine_c.__dict__),
            violations=[ViolationOut(**v) for v in scoring.violations],
            wrong_turns=[WrongTurnOut(**wt.__dict__) for wt in scoring.wrong_turns],
            token_count=scoring.token_count,
            cost_usd=cost_obj.total_cost_usd,
            duration_sec=duration,
            auto_fail=scoring.auto_fail,
            auto_fail_reason=scoring.auto_fail_reason,
            alerts=alerts_triggered # Added for UI visibility
        )

    except Exception as exc:
        logger.exception(f"[{session_id}] Audit critical failure")
        raise HTTPException(status_code=500, detail=str(exc))

# ── RAG & KB Management ─────────────────────────────────────────────

@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    kb = get_kb_manager()
    results = await kb.query(question=req.question, top_k=req.top_k, collection=req.collection)
    groundedness = round(sum(r.score for r in results) / len(results), 3) if results else 0.0
    return RAGQueryResponse(
        results=[RAGChunkOut(text=r.text, source=r.source, collection=r.collection, score=r.score) for r in results],
        groundedness=groundedness
    )

@app.post("/kb/upload", response_model=KBUploadResponse)
async def kb_upload(file: UploadFile = File(...), collection: str = Form(default="policies")):
    kb = get_kb_manager()
    file_bytes = await file.read()
    kbf = kb.add_file(file_bytes, file.filename or "doc.txt", collection)
    return KBUploadResponse(filename=kbf.filename, collection=kbf.collection, chunks=kbf.chunks, size_bytes=kbf.size_bytes)

# ── Execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Render assigns a port dynamically via the PORT env var
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
