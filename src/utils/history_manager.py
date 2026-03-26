"""
SamiX Audit Persistence & Data Schema

This module defines the core data structures that represent an audit session.
It provides a robust, file-based persistence layer using JSON, ensuring that
all audit history is preserved across application restarts.

Key Guarantees:
- Filename Integrity: The original uploaded filename is preserved exactly.
- Session Identity: Each audit is assigned a unique, immutable short ID (8 chars).
- Data Richness: Tracks transcripts, multi-engine analysis, and supervisor feedback.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from src.db import get_db


# Audit Detail Schemas 

@dataclass
class WrongTurn:
    """
    Represents a specific point of failure in a customer interaction.
    Maps an agent's statement to a RAG-verified policy or factual error.
    """
    turn_number: int
    speaker: str
    timestamp: str
    agent_said: str                    # The exact verbatim quote from the transcript.
    what_went_wrong: str                # A concise explanation of the error.
    correct_fact: str                   # The verified fact the agent should have known.
    rag_source: str                     # Pointer to the knowledge base document (e.g. "Policy_A §2").
    rag_confidence: float               # The retrieval confidence score from Milvus.
    score_impact: str                   # Human-readable deduction (e.g. "Compliance -10pts").
    specific_correction: str            # Tailored advice for the agent.


@dataclass
class EngineAResult:
    """ Analysis results from the Query Resolution Engine (A). """
    primary_query_answered: bool = False
    sub_queries_addressed: bool = False
    is_fake_close: bool = False
    resolution_state: str = "Unresolved"

@dataclass
class EngineBClaim:
    """ A single factual claim extracted and validated by Engine B. """
    claim: str = ""
    is_unverifiable: bool = False
    is_impossible_promise: bool = False
    is_contradiction: bool = False
    confidence_score: float = 0.0

@dataclass
class EngineBResult:
    """ Analysis results from the Factual Integrity Engine (B). """
    claims: list[EngineBClaim] = field(default_factory=list)

@dataclass
class EngineCResult:
    """ Analysis results from the Interaction Quality Engine (C). """
    customer_frustrated_but_ok: bool = False
    agent_rushed: bool = False
    resolution_confirmed_by_customer: bool = False


@dataclass
class AuditScores:
    """
    The mathematical representation of agent performance.
    Aggregates multi-dimensional scores (0-10) and calculates the final grade.
    """
    # Core Quality Dimensions
    empathy: float         = 0.0
    professionalism: float = 0.0
    compliance: float      = 0.0
    resolution: float      = 0.0
    communication: float   = 0.0
    integrity: float       = 0.0
    
    # Structural Phase Performance
    opening: float         = 0.0
    middle: float          = 0.0
    closing: float         = 0.0
    phase_bonus: float     = 0.0
    
    # Final Result
    final_score: float     = 0.0
    verdict: str           = "Pending"
    
    # Real-time Sentiment & Quality Tracking (Turn-by-Turn)
    customer_sentiment: list[float] = field(default_factory=list)
    customer_overall: float = 0.0
    agent_by_turn: list[float] = field(default_factory=list)


@dataclass
class TranscriptTurn:
    """ A single conversational node in the transcript. """
    turn:      int
    speaker:   str          # "AGENT" or "CUSTOMER"
    text:      str
    timestamp: str
    sentiment: float = 0.0


@dataclass
class AuditSession:
    """
    The monolithic record of a complete quality audit.
    Contains everything from the raw transcript to the final financial cost.
    """
    session_id:   str
    filename:     str          # The immutable filename provided by the user.
    upload_time:  str
    mode:         str          # "upload" (pre-recorded) or "live" (streaming)
    agent_name:   str
    duration_sec: int          = 0
    
    # Audit Payload
    scores:       AuditScores = field(default_factory=AuditScores)
    transcript:   list[TranscriptTurn] = field(default_factory=list)
    violations:   int          = 0
    wrong_turns:  list[WrongTurn] = field(default_factory=list)
    
    # Structured AI Summarization
    summary_customer_query: str = ""
    summary_sub_queries: list[str] = field(default_factory=list)
    summary_query_category: str = "General"
    summary_customer_expectation: str = ""
    summary_phases: dict = field(default_factory=dict)
    summary_key_moments: list[str] = field(default_factory=list)
    summary:      str          = ""  # Backward compatibility
    
    # Engine Analytics
    engine_a:     EngineAResult = field(default_factory=EngineAResult)
    engine_b:     EngineBResult = field(default_factory=EngineBResult)
    engine_c:     EngineCResult = field(default_factory=EngineCResult)

    # Operational Metadata
    feedback:     list[dict]   = field(default_factory=list)
    token_count:  int          = 0
    cost_usd:     float        = 0.0

    @property
    def stored_name(self) -> str:
        """ Returns the filename as it appears in the archival records. """
        return self.filename

    @classmethod
    def new(cls, filename: str, mode: str = "upload",
            agent_name: str = "Agent") -> "AuditSession":
        """ Factory method for initializing a new session with a unique ID and timestamp. """
        return cls(
            session_id=str(uuid.uuid4())[:8],
            filename=filename,
            upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode=mode,
            agent_name=agent_name,
        )


# Persistence Logic 

class HistoryManager:
    """
    Manages the local database of audit sessions.
    
    Implementation: Every session is serialized as a JSON file in `data/history/`.
    The manager maintains an in-memory cache for fast UI searching and filtering.
    """

    DIR: str = "data/history"

    def __init__(self) -> None:
        """ Ensures storage directories exist and pre-loads history into cache. """
        os.makedirs(self.DIR, exist_ok=True)
        self._db = get_db()
        self._cache: dict[str, AuditSession] = {}
        self._load_all()

    def _session_path(self, session_id: str) -> str:
        """ Computes the absolute filesystem path for a session record. """
        return os.path.join(self.DIR, f"{session_id}.json")

    def _load_all(self) -> None:
        """ Hydrates the in-memory cache from disk. """
        for fname in os.listdir(self.DIR):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.DIR, fname)) as fh:
                    raw = json.load(fh)
                sess = self._from_dict(raw)
                self._cache[sess.session_id] = sess
            except Exception:
                pass  # Tolerate corrupted JSON files.

    @staticmethod
    def _transcript_to_text(turns: list[TranscriptTurn]) -> str:
        return "\n".join(f"{t.speaker} [T{t.turn} {t.timestamp}]: {t.text}" for t in turns)

    @staticmethod
    def _from_dict(d: dict) -> AuditSession:
        """ 
        Safely reconstructs an AuditSession and its nested dataclasses 
        from a raw dictionary. Includes robust field-level mapping.
        """
        scores_raw = d.pop("scores", {})
        scores = AuditScores(**{
            k: v for k, v in scores_raw.items()
            if k in AuditScores.__dataclass_fields__
        })

        tx_raw = d.pop("transcript", [])
        transcript = [TranscriptTurn(**t) for t in tx_raw]

        wt_raw = d.pop("wrong_turns", [])
        wrong_turns = [WrongTurn(**w) for w in wt_raw]

        ea_raw = d.pop("engine_a", {})
        engine_a = EngineAResult(**ea_raw) if isinstance(ea_raw, dict) else EngineAResult()

        eb_raw = d.pop("engine_b", {})
        eb_claims = eb_raw.get("claims", []) if isinstance(eb_raw, dict) else []
        engine_b = EngineBResult(claims=[EngineBClaim(**c) for c in eb_claims])

        ec_raw = d.pop("engine_c", {})
        engine_c = EngineCResult(**ec_raw) if isinstance(ec_raw, dict) else EngineCResult()

        return AuditSession(
            scores=scores,
            transcript=transcript,
            wrong_turns=wrong_turns,
            engine_a=engine_a,
            engine_b=engine_b,
            engine_c=engine_c,
            **{k: v for k, v in d.items()
               if k in AuditSession.__dataclass_fields__
               and k not in ("scores", "transcript", "wrong_turns", "engine_a", "engine_b", "engine_c")}
        )

    def save(self, session: AuditSession) -> None:
        """ Serializes a session to disk and updates the local cache. """
        self._cache[session.session_id] = session
        data = asdict(session)
        with open(self._session_path(session.session_id), "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        self._db.save_audit_session(
            session_id=session.session_id,
            filename=session.filename,
            agent_name=session.agent_name,
            upload_time=session.upload_time,
            mode=session.mode,
            transcript_text=self._transcript_to_text(session.transcript),
            empathy_score=session.scores.empathy,
            compliance_score=session.scores.compliance,
            resolution_score=session.scores.resolution,
            overall_score=session.scores.final_score,
            summary=session.summary or session.summary_customer_query,
            violations=session.violations,
            key_moments=session.summary_key_moments,
            token_count=session.token_count,
            cost_usd=session.cost_usd,
            is_flagged=session.scores.final_score < 60,
            flag_reason=session.scores.verdict,
        )

    def get_all(self) -> list[AuditSession]:
        """ Returns all sessions, sorted by ingestion time (newest first). """
        return sorted(
            self._cache.values(),
            key=lambda s: s.upload_time,
            reverse=True,
        )

    def get_by_id(self, session_id: str) -> Optional[AuditSession]:
        """ Retrieves a single session from the cache by ID. """
        return self._cache.get(session_id)

    def search(self, query: str) -> list[AuditSession]:
        """ Filters the audit history by filename or agent name. """
        q = query.lower()
        return [
            s for s in self.get_all()
            if q in s.filename.lower() or q in s.agent_name.lower()
        ]

    def delete(self, session_id: str) -> None:
        """ Purges a session from the cache and deletes its physical file. """
        self._cache.pop(session_id, None)
        path = self._session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        self._db.execute("DELETE FROM audit_sessions WHERE session_id = ?", (session_id,))

    def migrate_json_to_sqlite(self) -> None:
        """ Backfills legacy JSON sessions into SQLite. """
        for session in self.get_all():
            self.save(session)
