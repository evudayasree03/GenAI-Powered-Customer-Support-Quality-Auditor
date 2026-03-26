"""
SamiX AI Inference Engine (Groq Client)

This module implements a sophisticated dual-call LLM pipeline using Groq Llama-3.
The pipeline follows these stages:
1. Contextual Summary: Distills the transcript into core queries and key moments.
2. Multidimensional Audit: Evaluates 6 quality dimensions, identifies violations,
   and verifies facts using the RAG knowledge base.
"""
from __future__ import annotations

import json
import os
import re
import random
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Any

import streamlit as st

from src.db import get_db
from src.storage import FileStorage
from src.utils.history_manager import (
    AuditScores, TranscriptTurn, WrongTurn,
    EngineAResult, EngineBClaim, EngineBResult, EngineCResult
)


# Internal Data Schemas 

@dataclass
class SummaryResult:
    """ High-level overview extracted from the raw transcript. """
    customer_query: str
    sub_queries: list[str]
    query_category: str
    customer_expectation: str
    phases: dict                    # E.g., {"opening": [1,2], "middle": [3,9], "closing": [10,11]}
    key_moments: list[str]


@dataclass
class ScoringResult:
    """ Comprehensive audit package containing scores, validations, and token usage. """
    scores: AuditScores
    engine_a: EngineAResult
    engine_b: EngineBResult
    engine_c: EngineCResult
    violations: list[dict]
    wrong_turns: list[WrongTurn]
    auto_fail: bool
    auto_fail_reason: str
    token_count: int


# Mock Data Generator 
# Used when the environment is not configured with a valid Groq API key.

_MOCK_TRANSCRIPT_TEXT = """
CUSTOMER [T1 00:00:08]: Hi, I've been charged twice for my subscription and need this fixed immediately.
AGENT    [T2 00:00:19]: Thank you for calling. Let me pull up your account. Can I verify your email?
CUSTOMER [T3 00:00:31]: It's priya@example.com. This is the second time this happened. I'm frustrated.
AGENT    [T4 00:00:48]: I can see the duplicate charge. I'll process the refund -- it should be back in about 2 days.
CUSTOMER [T5 00:01:05]: Only 2 days? Last time it took 2 weeks. That doesn't seem right.
AGENT    [T6 00:01:18]: Yes, 2 days maximum. Is there anything else I can help you with?
CUSTOMER [T7 00:01:29]: Why does this keep happening? This is the second charge in two months.
AGENT    [T8 00:01:42]: It must have been a system error. I've noted it on your account.
CUSTOMER [T9 00:01:55]: That's not an answer. I want to speak to a supervisor.
AGENT    [T10 00:02:08]: I'll transfer you. Thank you for calling.
"""

def _mock_summary() -> SummaryResult:
    """ Returns a predefined summary for testing the UI flow. """
    return SummaryResult(
        customer_query="Duplicate subscription charge -- refund request",
        sub_queries=["Why did I get charged twice?", "Why did it take 2 weeks last time?"],
        query_category="Billing / Refund",
        customer_expectation="Immediate refund and an explanation of the root cause.",
        phases={"opening": [1, 3], "middle": [4, 8], "closing": [9, 10]},
        key_moments=[
            "T4: Agent quoted 2-day refund -- policy breach",
            "T6: False close attempted while customer unresolved",
            "T8: Vague 'system error' response dismissed root cause query",
        ],
    )


def _mock_scoring() -> ScoringResult:
    """ Returns a predefined audit scoring result for testing. """
    agent_by_turn = [7.8, 7.5, 6.2, 5.8, 5.1, 4.5, 4.9, 4.2, 3.2, 2.9]
    cust_by_turn  = [7.0, 6.5, 4.0, 3.0, 2.0, 2.5, 1.5, 1.0, 1.0, 1.5]

    scores = AuditScores(
        empathy=6.0,
        professionalism=7.2,
        compliance=4.0,
        resolution=3.5,
        communication=7.8,
        integrity=3.2,
        opening=7.8,
        middle=5.1,
        closing=2.9,
        phase_bonus=-5.0,
        final_score=50.0,
        verdict="Critical / Fail",
        customer_sentiment=cust_by_turn,
        customer_overall=round(sum(cust_by_turn) / len(cust_by_turn), 1),
        agent_by_turn=agent_by_turn,
    )

    ea = EngineAResult(
        primary_query_answered=True,
        sub_queries_addressed=False,
        is_fake_close=True,
        resolution_state="Escalated due to frustration"
    )
    eb = EngineBResult(
        claims=[
            EngineBClaim("Refund in 2 days", False, False, True, 0.94),
            EngineBClaim("System error caused recurrence", True, False, False, 0.65)
        ]
    )
    ec = EngineCResult(
        customer_frustrated_but_ok=False,
        agent_rushed=True,
        resolution_confirmed_by_customer=False
    )

    wrong_turns = [
        WrongTurn(
            turn_number=4,
            speaker="AGENT",
            timestamp="00:00:48",
            agent_said="It should be back in about 2 days.",
            what_went_wrong=(
                "Refund Policy §2.3 mandates 7–10 business days. "
                "The agent stated 2 days -- a material policy breach."
            ),
            correct_fact="Refunds are processed within 7–10 business days from date of approval (§2.3).",
            rag_source="Refund_Policy_v3.pdf · §2.3",
            rag_confidence=0.94,
            score_impact="Integrity −4.2 pts",
            specific_correction=(
                '"Your refund will be processed within 7 to 10 business days from today. "'
            ),
        ),
    ]

    return ScoringResult(
        scores=scores,
        engine_a=ea,
        engine_b=eb,
        engine_c=ec,
        violations=[
            {"type": "Wrong policy info", "phase": "Middle", "severity": "Critical"},
            {"type": "False close",        "phase": "Closing","severity": "Critical"},
        ],
        wrong_turns=wrong_turns,
        auto_fail=False,
        auto_fail_reason="",
        token_count=4218,
    )


# AI Prompt Engineering 

_SCORING_SYSTEM_PROMPT = """
You are SamiX, a senior quality auditor for customer support calls.

Return ONLY valid JSON matching this exact schema -- no markdown:
{
  "empathy":         <float 0-10>,
  "professionalism": <float 0-10>,
  "compliance":      <float 0-10>,
  "resolution":      <float 0-10>,
  "communication":   <float 0-10>,
  "integrity":       <float 0-10>,
  "opening_score":   <float 0-10>,
  "middle_score":    <float 0-10>,
  "closing_score":   <float 0-10>,
  "phase_bonus":     <float -5 to +5>,
  "customer_sentiment_by_turn": [<float 0-10>, ...],
  "agent_score_by_turn":        [<float 0-10>, ...],
  
  "engine_a": {
    "primary_query_answered": <bool>,
    "sub_queries_addressed": <bool>,
    "is_fake_close": <bool>,
    "resolution_state": "<str, e.g. Escalatated, Closed, Abandoned>"
  },
  "engine_b": {
    "claims": [
      {
        "claim": "<str>",
        "is_unverifiable": <bool>,
        "is_impossible_promise": <bool>,
        "is_contradiction": <bool>,
        "confidence_score": <float 0-1>
      }
    ]
  },
  "engine_c": {
    "customer_frustrated_but_ok": <bool>,
    "agent_rushed": <bool>,
    "resolution_confirmed_by_customer": <bool>
  },

  "violations": [
    {"type": "<str>", "phase": "<str>", "severity": "Critical|High|Medium"}
  ],
  "wrong_turns": [
    {
      "turn_number":   <int>,
      "speaker":       "AGENT",
      "timestamp":     "<str>",
      "agent_said":    "<exact quote>",
      "what_went_wrong": "<specific factual error, cite policy>",
      "correct_fact":  "<policy/KB sourced fact>",
      "rag_source":    "<document · section · page>",
      "rag_confidence": <float 0-1>,
      "score_impact":  "<dim name> −<pts>",
      "specific_correction": "<exact alternative phrase>"
    }
  ],
  "auto_fail": <bool>,
  "auto_fail_reason": "<str or empty>"
}

Scoring weights:
  empathy 20%, professionalism 15%, compliance 25%,
  resolution 20%, communication 5%, integrity 15%.

When POLICY CONTEXT is provided, you MUST cite it when populating
"rag_source" and "correct_fact" fields. Prioritize policy facts over
general knowledge. If no policy context is provided, use best knowledge.
"""


class GroqClient:
    """
    Primary interface for AI-powered audit analysis.
    
    This client manages connections to the Groq API and coordinates the two-step
    analysis process. It ensures graceful degradation using mock data if the API
    is unavailable.
    """

    # We use Llama-3.3-70B for its high speed and strong reasoning capabilities.
    MODEL: str = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        """ Establishes the connection to the Groq API if secrets are available. """
        self._client = self._build_client()
        self._async_client = self._build_async_client()
        self._db = get_db()
        self._storage = FileStorage()

    def _build_client(self) -> Optional[object]:
        """ Initializes the synchronous Groq SDK client. """
        try:
            api_key = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("groq", {}).get("api_key", "")
            if api_key.startswith("gsk_REPLACE"): return None
            from groq import Groq
            return Groq(api_key=api_key)
        except Exception:
            return None

    def _build_async_client(self) -> Optional[object]:
        """ Initializes the asynchronous Groq SDK client. """
        try:
            api_key = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("groq", {}).get("api_key", "")
            if api_key.startswith("gsk_REPLACE"): return None
            from groq import AsyncGroq
            return AsyncGroq(api_key=api_key)
        except Exception:
            return None

    @property
    def is_live(self) -> bool:
        """ Returns True if valid Groq API connections are established. """
        return self._client is not None and self._async_client is not None

    async def summarise(self, transcript_text: str, session_id: Optional[str] = None) -> SummaryResult:
        """
        [STEP 1] Generates a structured summary and phase breakdown of the transcript.
        """
        storage_session_id = session_id or str(uuid.uuid4())[:8]
        if not self.is_live:
            result = _mock_summary()
            self._store_exchange(
                session_id=storage_session_id,
                api_name="groq_summary_mock",
                endpoint="chat.completions",
                request_payload={"transcript_text": transcript_text},
                response_payload=result.__dict__,
                processing_time_ms=0.0,
            )
            return result
        return await self._real_summarise(transcript_text, storage_session_id)

    async def score(self, transcript_text: str, summary: SummaryResult, rag_context: str = "", session_id: Optional[str] = None) -> ScoringResult:
        """
        [STEP 2] Performs a deep-dive audit, scoring behavior and detecting anomalies.
        """
        storage_session_id = session_id or str(uuid.uuid4())[:8]
        if not self.is_live:
            result = _mock_scoring()
            self._store_exchange(
                session_id=storage_session_id,
                api_name="groq_score_mock",
                endpoint="chat.completions",
                request_payload={"transcript_text": transcript_text, "summary": summary.__dict__},
                response_payload={
                    "scores": result.scores.__dict__,
                    "violations": result.violations,
                    "wrong_turns": [wt.__dict__ for wt in result.wrong_turns],
                    "token_count": result.token_count,
                },
                processing_time_ms=0.0,
                tokens_used=result.token_count,
            )
            return result
        return await self._real_score(transcript_text, summary, rag_context, storage_session_id)

    def _store_exchange(
        self,
        *,
        session_id: str,
        api_name: str,
        endpoint: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        processing_time_ms: float,
        status_code: int = 200,
        tokens_used: Optional[int] = None,
    ) -> None:
        file_path, request_hash = self._storage.save_json(
            "llm_scores",
            session_id,
            {
                "api_name": api_name,
                "endpoint": endpoint,
                "request": request_payload,
                "response": response_payload,
            },
            filename_prefix=api_name,
        )
        self._db.save_api_response(
            session_id=session_id,
            api_name=api_name,
            endpoint=endpoint,
            request_hash=request_hash,
            response_json=response_payload,
            status_code=status_code,
            processing_time_ms=processing_time_ms,
            tokens_used=tokens_used,
            file_path=file_path,
        )
        if tokens_used is not None:
            self._db.record_api_cost(session_id, "groq", None, tokens_used)

    # Implementation Details 

    async def _real_summarise(self, transcript_text: str, session_id: str) -> SummaryResult:
        """ Executes the Step 1 AI call to summarize context using AsyncGroq. """
        prompt = (
            "Analyse this customer support transcript and return a JSON object with:\n"
            "customer_query (str), sub_queries ([str]), query_category (str),\n"
            "customer_expectation (str), phases ({opening:[start,end], ...}),\n"
            "key_moments ([str])\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            "Return ONLY valid JSON."
        )
        try:
            started = time.perf_counter()
            resp = await self._async_client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            result = SummaryResult(
                customer_query=data.get("customer_query", ""),
                sub_queries=data.get("sub_queries", []),
                query_category=data.get("query_category", "General"),
                customer_expectation=data.get("customer_expectation", ""),
                phases=data.get("phases", {}),
                key_moments=data.get("key_moments", []),
            )
            self._store_exchange(
                session_id=session_id,
                api_name="groq_summary",
                endpoint="chat.completions",
                request_payload={"transcript_text": transcript_text},
                response_payload=data,
                processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
                tokens_used=getattr(resp.usage, "total_tokens", None),
            )
            return result
        except Exception as exc:
            st.warning(f"Groq Call 1 failed ({exc}). Using mock.")
            return _mock_summary()

    async def _real_score(self, transcript_text: str, summary: SummaryResult, rag_context: str, session_id: str) -> ScoringResult:
        """ Executes the Step 2 AI call to perform a comprehensive audit using AsyncGroq. """
        rag_section = (
            f"\n\nPOLICY CONTEXT (RAG-retrieved — use for wrong turn verification):\n"
            f"{rag_context[:3000]}"
        ) if rag_context else ""
        user_msg = (
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"CUSTOMER QUERY:\n{summary.customer_query}\n\n"
            f"CUSTOMER EXPECTATIONS:\n{summary.customer_expectation}\n\n"
            f"KEY MOMENTS:\n{chr(10).join(summary.key_moments)}"
            f"{rag_section}"
        )
        try:
            started = time.perf_counter()
            resp = await self._async_client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.05,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            raw   = resp.choices[0].message.content.strip()
            usage = resp.usage
            data  = json.loads(raw)
            result = self._parse_scoring_response(data, usage.total_tokens if usage else 4000)
            self._store_exchange(
                session_id=session_id,
                api_name="groq_score",
                endpoint="chat.completions",
                request_payload={"transcript_text": transcript_text, "summary": summary.__dict__},
                response_payload=data,
                processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
                tokens_used=usage.total_tokens if usage else None,
            )
            return result
        except Exception as exc:
            st.warning(f"Groq Call 2 failed ({exc}). Using mock.")
            return _mock_scoring()

    @staticmethod
    def _parse_scoring_response(d: dict, tokens: int) -> ScoringResult:
        """ Transforms raw AI JSON into a structured ScoringResult package. """
        # 1. Normalize Turn Analysis
        n_turns = max(
            len(d.get("customer_sentiment_by_turn", [])),
            len(d.get("agent_score_by_turn", [])),
        )
        cust_sent  = d.get("customer_sentiment_by_turn", [5.0] * n_turns)
        agent_turn = d.get("agent_score_by_turn",        [5.0] * n_turns)

        # 2. Calculate Final Weighted Score
        w = {"empathy": 0.20, "professionalism": 0.15, "compliance": 0.25,
             "resolution": 0.20, "communication": 0.05, "integrity": 0.15}
        dims = {k: float(d.get(k, 5.0)) for k in w}
        raw_score = sum(v * w[k] * 10 for k, v in dims.items())
        phase_bonus = float(d.get("phase_bonus", 0.0))
        final = min(100.0, max(0.0, raw_score + phase_bonus))

        # 3. Determine Quality Verdict
        if   final >= 80: verdict = "Excellent"
        elif final >= 70: verdict = "Good"
        elif final >= 60: verdict = "Needs work"
        else:             verdict = "Critical / Fail"

        # 4. Construct Result Components
        scores = AuditScores(
            empathy=dims["empathy"], professionalism=dims["professionalism"],
            compliance=dims["compliance"], resolution=dims["resolution"],
            communication=dims["communication"], integrity=dims["integrity"],
            opening=float(d.get("opening_score", 7.0)),
            middle=float(d.get("middle_score",   5.0)),
            closing=float(d.get("closing_score", 3.0)),
            phase_bonus=phase_bonus,
            final_score=round(final, 1),
            verdict=verdict,
            customer_sentiment=cust_sent,
            customer_overall=round(sum(cust_sent)/len(cust_sent), 1) if cust_sent else 5.0,
            agent_by_turn=agent_turn,
        )

        ea_data = d.get("engine_a", {})
        ea = EngineAResult(
            primary_query_answered=ea_data.get("primary_query_answered", False),
            sub_queries_addressed=ea_data.get("sub_queries_addressed", False),
            is_fake_close=ea_data.get("is_fake_close", False),
            resolution_state=ea_data.get("resolution_state", "Unknown")
        )

        eb_data = d.get("engine_b", {})
        eb_claims = eb_data.get("claims", [])
        eb = EngineBResult(claims=[
            EngineBClaim(
                claim=c.get("claim", ""),
                is_unverifiable=c.get("is_unverifiable", False),
                is_impossible_promise=c.get("is_impossible_promise", False),
                is_contradiction=c.get("is_contradiction", False),
                confidence_score=c.get("confidence_score", 0.0)
            ) for c in eb_claims
        ])

        ec_data = d.get("engine_c", {})
        ec = EngineCResult(
            customer_frustrated_but_ok=ec_data.get("customer_frustrated_but_ok", False),
            agent_rushed=ec_data.get("agent_rushed", False),
            resolution_confirmed_by_customer=ec_data.get("resolution_confirmed_by_customer", False)
        )

        wrong_turns = [
            WrongTurn(
                turn_number=wt.get("turn_number", 0),
                speaker=wt.get("speaker", "AGENT"),
                timestamp=wt.get("timestamp", ""),
                agent_said=wt.get("agent_said", ""),
                what_went_wrong=wt.get("what_went_wrong", ""),
                correct_fact=wt.get("correct_fact", ""),
                rag_source=wt.get("rag_source", ""),
                rag_confidence=float(wt.get("rag_confidence", 0.8)),
                score_impact=wt.get("score_impact", ""),
                specific_correction=wt.get("specific_correction", ""),
            )
            for wt in d.get("wrong_turns", [])
        ]

        return ScoringResult(
            scores=scores,
            engine_a=ea,
            engine_b=eb,
            engine_c=ec,
            violations=d.get("violations", []),
            wrong_turns=wrong_turns,
            auto_fail=bool(d.get("auto_fail", False)),
            auto_fail_reason=d.get("auto_fail_reason", ""),
            token_count=tokens,
        )
