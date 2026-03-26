from __future__ import annotations

from _bootstrap import setup_project_root

setup_project_root()

from src.auth.authenticator import AuthManager
from src.utils.history_manager import HistoryManager


def main() -> None:
    AuthManager().migrate_yaml_to_sqlite()
    HistoryManager().migrate_json_to_sqlite()
    print("Migrated legacy YAML/JSON records into SQLite.")


if __name__ == "__main__":
    main()
