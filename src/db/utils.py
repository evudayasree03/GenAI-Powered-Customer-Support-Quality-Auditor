from __future__ import annotations

from typing import Any

from .db_manager import DBManager


def table_counts(db: DBManager) -> dict[str, int]:
    tables = [
        "users",
        "transcriptions",
        "api_responses",
        "audit_sessions",
        "compliance_alerts",
        "api_costs",
        "audit_logs",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        row = db.fetch_one(f"SELECT COUNT(*) AS count FROM {table}")
        counts[table] = int(row["count"]) if row else 0
    return counts


def sqlite_healthcheck(db: DBManager) -> dict[str, Any]:
    row = db.fetch_one("SELECT sqlite_version() AS version")
    return {
        "db_path": db.db_path,
        "sqlite_version": row["version"] if row else "unknown",
        "counts": table_counts(db),
    }
