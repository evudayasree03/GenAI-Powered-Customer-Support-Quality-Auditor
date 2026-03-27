from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from config import Config


class DBManager:
    """Thin SQLite data-access layer for SamiX runtime persistence."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(Path(db_path or Config.SQLITE_DB_PATH))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        self.db_path = os.path.join(os.getcwd(), "samix.db")
        schema = schema_path.read_text(encoding="utf-8")
        with self._lock, self.connect() as conn:
            conn.executescript(schema)

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        with self._lock, self.connect() as conn:
            return conn.execute(query, params).fetchone()

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock, self.connect() as conn:
            return conn.execute(query, params).fetchall()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        with self._lock, self.connect() as conn:
            cur = conn.execute(query, params)
            return int(cur.lastrowid or 0)

    def upsert_user(
        self,
        email: str,
        name: str,
        password_hash: str,
        role: str = "agent",
        is_active: bool = True,
    ) -> int:
        email = email.lower().strip()
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE users
                    SET name = ?, password_hash = ?, role = ?, is_active = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE email = ?
                    """,
                    (name, password_hash, role, int(is_active), email),
                )
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO users (email, name, password_hash, role, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email, name, password_hash, role, int(is_active)),
            )
            return int(cur.lastrowid or 0)

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        row = self.fetch_one("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
        return dict(row) if row else None

    def update_last_login(self, email: str) -> None:
        self.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE email = ?",
            (email.lower().strip(),),
        )

    def save_transcription(
        self,
        session_id: str,
        filename: str,
        transcription_text: str,
        *,
        audio_path: Optional[str] = None,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        confidence_score: Optional[float] = None,
        processor_used: Optional[str] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO transcriptions (
                session_id, filename, audio_path, transcription_text, language,
                duration_seconds, confidence_score, processor_used, status, error_message, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                filename = excluded.filename,
                audio_path = excluded.audio_path,
                transcription_text = excluded.transcription_text,
                language = excluded.language,
                duration_seconds = excluded.duration_seconds,
                confidence_score = excluded.confidence_score,
                processor_used = excluded.processor_used,
                status = excluded.status,
                error_message = excluded.error_message,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                filename,
                audio_path,
                transcription_text,
                language,
                duration_seconds,
                confidence_score,
                processor_used,
                status,
                error_message,
            ),
        )

    def save_api_response(
        self,
        *,
        session_id: str,
        api_name: str,
        endpoint: Optional[str] = None,
        request_hash: Optional[str] = None,
        response_json: Optional[dict[str, Any] | str] = None,
        status_code: Optional[int] = None,
        processing_time_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        file_path: Optional[str] = None,
    ) -> int:
        payload = response_json if isinstance(response_json, str) else json.dumps(response_json or {}, default=str)
        return self.execute(
            """
            INSERT INTO api_responses (
                session_id, api_name, endpoint, request_hash, response_json, status_code,
                processing_time_ms, tokens_used, cost_usd, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                api_name,
                endpoint,
                request_hash,
                payload,
                status_code,
                processing_time_ms,
                tokens_used,
                cost_usd,
                file_path,
            ),
        )

    def save_audit_session(
        self,
        *,
        session_id: str,
        filename: str,
        agent_name: str,
        upload_time: str,
        mode: str,
        transcript_text: str,
        empathy_score: float,
        compliance_score: float,
        resolution_score: float,
        overall_score: float,
        summary: str,
        violations: int,
        key_moments: list[str] | str,
        token_count: int = 0,
        cost_usd: float = 0.0,
        processing_time_seconds: Optional[float] = None,
        is_flagged: bool = False,
        flag_reason: Optional[str] = None,
    ) -> None:
        key_moments_payload = key_moments if isinstance(key_moments, str) else json.dumps(key_moments, default=str)
        self.execute(
            """
            INSERT INTO audit_sessions (
                session_id, filename, agent_name, upload_time, mode, transcript_text,
                transcript_word_count, empathy_score, compliance_score, resolution_score,
                overall_score, summary, violations, key_moments, token_count, cost_usd,
                processing_time_seconds, is_flagged, flag_reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                filename = excluded.filename,
                agent_name = excluded.agent_name,
                upload_time = excluded.upload_time,
                mode = excluded.mode,
                transcript_text = excluded.transcript_text,
                transcript_word_count = excluded.transcript_word_count,
                empathy_score = excluded.empathy_score,
                compliance_score = excluded.compliance_score,
                resolution_score = excluded.resolution_score,
                overall_score = excluded.overall_score,
                summary = excluded.summary,
                violations = excluded.violations,
                key_moments = excluded.key_moments,
                token_count = excluded.token_count,
                cost_usd = excluded.cost_usd,
                processing_time_seconds = excluded.processing_time_seconds,
                is_flagged = excluded.is_flagged,
                flag_reason = excluded.flag_reason,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                filename,
                agent_name,
                upload_time,
                mode,
                transcript_text,
                len(transcript_text.split()),
                empathy_score,
                compliance_score,
                resolution_score,
                overall_score,
                summary,
                violations,
                key_moments_payload,
                token_count,
                cost_usd,
                processing_time_seconds,
                int(is_flagged),
                flag_reason,
            ),
        )

    def add_compliance_alert(self, session_id: str, alert_type: str, message: str, severity: str = "medium") -> int:
        return self.execute(
            """
            INSERT INTO compliance_alerts (session_id, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, alert_type, severity, message),
        )

    def record_api_cost(
        self,
        session_id: Optional[str],
        service: str,
        cost_usd: Optional[float],
        tokens_used: Optional[int] = None,
    ) -> int:
        return self.execute(
            "INSERT INTO api_costs (session_id, service, cost_usd, tokens_used) VALUES (?, ?, ?, ?)",
            (session_id, service, cost_usd, tokens_used),
        )

    def log_action(self, action: str, resource: str, details: str = "", user_id: Optional[int] = None) -> int:
        return self.execute(
            "INSERT INTO audit_logs (user_id, action, resource, details) VALUES (?, ?, ?, ?)",
            (user_id, action, resource, details),
        )

    def list_audit_rows(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.fetch_all("SELECT * FROM audit_sessions ORDER BY upload_time DESC")]


_db_singleton: Optional[DBManager] = None


def get_db() -> DBManager:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = DBManager()
    return _db_singleton
