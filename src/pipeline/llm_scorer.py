from __future__ import annotations

from .groq_client import GroqClient, ScoringResult, SummaryResult


class LLMScorer:
    """Thin wrapper so the scoring engine can evolve independently from the Groq client."""

    def __init__(self, groq: GroqClient | None = None) -> None:
        self._groq = groq or GroqClient()

    async def score(self, transcript_text: str, summary: SummaryResult) -> ScoringResult:
        return await self._groq.score(transcript_text, summary)
