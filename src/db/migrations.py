from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .db_manager import DBManager


@dataclass
class Migration:
    version: int
    name: str
    apply: Callable[[DBManager], None]


def run_migrations(db: DBManager) -> None:
    """Placeholder migration runner. Schema-first for now, extensible later."""
    db.initialize()
