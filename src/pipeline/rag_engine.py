from __future__ import annotations

from typing import Any

from src.utils.kb_manager import KBManager


class RAGEngine:
    """Adapter around the current KB manager while the retrieval stack evolves."""

    def __init__(self, kb: KBManager | None = None) -> None:
        self._kb = kb or KBManager()

    async def query(self, question: str, top_k: int = 5) -> list[Any]:
        return await self._kb.query(question, top_k=top_k)

    async def audit(self, agent_statement: str, context_question: str) -> dict:
        return await self._kb.audit_chain(agent_statement, context_question)
