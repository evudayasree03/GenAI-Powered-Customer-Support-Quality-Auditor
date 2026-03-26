from __future__ import annotations

from typing import Any, Optional

from src.db import get_db
from .file_storage import FileStorage


class CacheManager:
    """Simple request-hash cache for API responses."""

    def __init__(self) -> None:
        self._db = get_db()
        self._storage = FileStorage()

    def get(self, session_id: str, api_name: str, request_hash: str) -> Optional[dict[str, Any]]:
        row = self._db.fetch_one(
            """
            SELECT response_json, file_path FROM api_responses
            WHERE session_id = ? AND api_name = ? AND request_hash = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, api_name, request_hash),
        )
        if not row:
            return None
        if row["file_path"]:
            return self._storage.load_json(row["file_path"])
        return {"response_json": row["response_json"]}
