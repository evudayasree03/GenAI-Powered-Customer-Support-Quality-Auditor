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
# ... other imports ...
from src.pipeline.alert_engine import AlertEngine

alert_engine = AlertEngine()

@app.post("/audit")
async def run_audit(file: UploadFile = File(...)):
    # 1. Process Audio/Text
    turns = await stt_processor.process(await file.read(), file.filename)
    transcript_text = transcript_to_text(turns)
    
    # 2. Run Groq Scoring (your existing logic)
    summary = await groq_client.summarise(transcript_text)
    audit_results = await groq_client.score(transcript_text, summary)

    # 3. TRIGGER THE NEW ALERT ENGINE
    # This catches the list of alert strings we just wrote in alert.py
    alerts_triggered = await alert_engine.check_and_fire(
        filename=file.filename,
        agent_name="Agent", # Or extract from results
        final_score=audit_results.scores.final_score,
        violations=audit_results.violations,
        auto_fail=audit_results.auto_fail,
        auto_fail_reason=audit_results.auto_fail_reason,
        recipient_email=os.getenv("ADMIN_EMAIL", "") 
    )

    return {
        "status": "success",
        "score": audit_results.scores.final_score,
        "alerts": alerts_triggered, # CRITICAL: Sends alerts to Streamlit
        "data": audit_results
    }
# Ensure project root is on sys.path so `api.*` imports work correctly on Render
# If file is in /api/main.py, parent is /api, parent.parent is / (root)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# Disable HF telemetry and warnings to keep logs clean
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Audioop polyfill (needed for some Python 3.11+ environments)
import audioop
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("samix.api")

# ── App Setup ───────────────────────────────────────────────────────

app = FastAPI(
    title="SamiX API",
    description="GenAI-Powered Customer Support Quality Auditor — Backend",
    version="1.0.0",
)

# Allow Streamlit Cloud and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your specific streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ─────────────────────────────────────────────────────────

def _transcript_to_text(turns) -> str:
    """Convert list of TranscriptTurn objects to plain text."""
    lines = []
    for t in turns:
        speaker = getattr(t, "speaker", "UNKNOWN")
        text = getattr(t, "text", "")
        timestamp = getattr(t, "timestamp", "")
        turn_num = getattr(t, "turn", 0)
        lines.append(f"{speaker} [T{turn_num} {timestamp}]: {text}")
    return "\n".join(lines)

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint to pass Render's default health check."""
    return {
        "status": "online",
        "message": "SamiX API is running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check — reports status of all services."""
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
async def run_audit(
    file: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())[:8]
    filename = file.filename or f"upload_{session_id}.wav"
    file_bytes = await file.read()

    groq = get_groq_client()
    stt = get_stt_processor()
    kb = get_kb_manager()
    audio = get_audio_processor()
    cost = get_cost_tracker()

    try:
        # 1. Convert audio
        logger.info(f"[{session_id}] Converting audio: {filename}")
        wav_bytes, meta = await asyncio.to_thread(
            audio.convert_to_wav, file_bytes, filename
        )
        duration = meta.get("duration_sec", 0)

        # 2. Transcribe
        logger.info(f"[{session_id}] Transcribing...")
        turns = await stt.process(file_bytes, filename, session_id=session_id)
        tx_text = _transcript_to_text(turns)

        # 3. Summarize
        logger.info(f"[{session_id}] Summarizing...")
        summary = await groq.summarise(tx_text, session_id=session_id)

        # 4. RAG Retrieve
        logger.info(f"[{session_id}] Retrieving policy context...")
        rag_results = await kb.query(
            question=summary.customer_query + " " + " ".join(summary.sub_queries),
            top_k=6,
        )
        rag_context_text = "\n\n".join(
            f"[{r.source} | {r.collection} | conf {r.score:.2f}]\n{r.text}"
            for r in rag_results
        )

        # 5. Score with RAG grounding
        logger.info(f"[{session_id}] Scoring with RAG grounding...")
        scoring = await groq.score(
            tx_text, summary,
            rag_context=rag_context_text,
            session_id=session_id,
        )

        # 6. Post-audit RAG verification on wrong turns
        for wt in scoring.wrong_turns:
            try:
                audit = await kb.audit_chain(wt.agent_said, wt.what_went_wrong)
                if audit and isinstance(audit, dict) and audit.get("top_source"):
                    wt.rag_source = audit["top_source"]
                    wt.rag_confidence = float(audit.get("top_score", wt.rag_confidence))
            except Exception:
                pass

        # Cost calculation
        cost_obj = cost.calculate_session_cost(
            token_count=scoring.token_count,
            audio_duration_sec=duration,
        )

        # Build response (Scores mapping)
        s = scoring.scores
        return AuditResponse(
            session_id=session_id,
            filename=filename,
            summary=SummaryOut(
                customer_query=summary.customer_query,
                sub_queries=summary.sub_queries,
                query_category=summary.query_category,
                customer_expectation=summary.customer_expectation,
                phases=summary.phases,
                key_moments=summary.key_moments,
            ),
            scores=ScoresOut(
                empathy=s.empathy,
                professionalism=s.professionalism,
                compliance=s.compliance,
                resolution=s.resolution,
                communication=s.communication,
                integrity=s.integrity,
                opening=s.opening,
                middle=s.middle,
                closing=s.closing,
                phase_bonus=s.phase_bonus,
                final_score=s.final_score,
                verdict=s.verdict,
                customer_sentiment=s.customer_sentiment,
                customer_overall=s.customer_overall,
                agent_by_turn=s.agent_by_turn,
            ),
            engine_a=EngineAOut(
                primary_query_answered=scoring.engine_a.primary_query_answered,
                sub_queries_addressed=scoring.engine_a.sub_queries_addressed,
                is_fake_close=scoring.engine_a.is_fake_close,
                resolution_state=scoring.engine_a.resolution_state,
            ),
            engine_b=EngineBOut(
                claims=[
                    EngineBClaimOut(
                        claim=c.claim,
                        is_unverifiable=c.is_unverifiable,
                        is_impossible_promise=c.is_impossible_promise,
                        is_contradiction=c.is_contradiction,
                        confidence_score=c.confidence_score,
                    )
                    for c in scoring.engine_b.claims
                ]
            ),
            engine_c=EngineCOut(
                customer_frustrated_but_ok=scoring.engine_c.customer_frustrated_but_ok,
                agent_rushed=scoring.engine_c.agent_rushed,
                resolution_confirmed_by_customer=scoring.engine_c.resolution_confirmed_by_customer,
            ),
            violations=[
                ViolationOut(**v) for v in scoring.violations
            ],
            wrong_turns=[
                WrongTurnOut(
                    turn_number=wt.turn_number,
                    speaker=wt.speaker,
                    timestamp=wt.timestamp,
                    agent_said=wt.agent_said,
                    what_went_wrong=wt.what_went_wrong,
                    correct_fact=wt.correct_fact,
                    rag_source=wt.rag_source,
                    rag_confidence=wt.rag_confidence,
                    score_impact=wt.score_impact,
                    specific_correction=wt.specific_correction,
                )
                for wt in scoring.wrong_turns
            ],
            token_count=scoring.token_count,
            cost_usd=cost_obj.total_cost_usd,
            duration_sec=duration,
            auto_fail=scoring.auto_fail,
            auto_fail_reason=scoring.auto_fail_reason,
        )

    except Exception as exc:
        logger.exception(f"[{session_id}] Audit failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Query the RAG knowledge base."""
    kb = get_kb_manager()
    results = await kb.query(
        question=req.question,
        top_k=req.top_k,
        collection=req.collection,
    )
    groundedness = (
        round(sum(r.score for r in results) / len(results), 3)
        if results else 0.0
    )
    return RAGQueryResponse(
        results=[
            RAGChunkOut(
                text=r.text,
                source=r.source,
                collection=r.collection,
                score=r.score,
            )
            for r in results
        ],
        groundedness=groundedness,
    )

@app.post("/kb/upload", response_model=KBUploadResponse)
async def kb_upload(
    file: UploadFile = File(...),
    collection: str = Form(default="policies"),
):
    """Upload and index a document into the knowledge base."""
    kb = get_kb_manager()
    file_bytes = await file.read()
    filename = file.filename or "uploaded.txt"

    kbf = kb.add_file(file_bytes, filename, collection)
    return KBUploadResponse(
        filename=kbf.filename,
        collection=kbf.collection,
        chunks=kbf.chunks,
        size_bytes=kbf.size_bytes,
    )

# ── Run directly ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # CRITICAL: Render provides the PORT variable. 
    # Default to 10000 for standard Render blueprint compliance.
    port = int(os.environ.get("PORT", 10000))
    
    # CRITICAL: Set reload=False for production to avoid worker crashes.
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
