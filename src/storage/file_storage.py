from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import Config


class FileStorage:
    """Persists API responses and derived artifacts to the filesystem."""

    BASE_DIR = Path(Config.API_RESPONSES_DIR)
    SUBDIRS = {
        "transcriptions": Path(Config.TRANSCRIPTIONS_DIR),
        "llm_scores": Path(Config.LLM_SCORES_DIR),
        "rag_results": Path(Config.RAG_RESULTS_DIR),
        "cache": Path(Config.CACHE_DIR),
    }

    def __init__(self) -> None:
        for path in self.SUBDIRS.values():
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _request_hash(payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def save_json(
        self,
        category: str,
        session_id: str,
        payload: Any,
        *,
        filename_prefix: Optional[str] = None,
    ) -> tuple[str, str]:
        directory = self.SUBDIRS.get(category, self.BASE_DIR)
        request_hash = self._request_hash(payload)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        prefix = filename_prefix or category
        path = directory / f"{prefix}_{session_id}_{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path), request_hash

    def load_json(self, path: str) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))
