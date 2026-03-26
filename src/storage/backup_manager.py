from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from config import Config


class BackupManager:
    """Creates timestamped backups of the SQLite database."""

    def __init__(self) -> None:
        self._backup_dir = Path(Config.BACKUPS_DIR)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_database(self, source_path: str = Config.SQLITE_DB_PATH) -> str:
        src = Path(source_path)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        dst = self._backup_dir / f"samix_{timestamp}.db"
        if src.exists():
            shutil.copy2(src, dst)
        return str(dst)
