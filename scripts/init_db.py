from _bootstrap import setup_project_root

setup_project_root()

from src.db import get_db


def main() -> None:
    db = get_db()
    db.initialize()
    print(f"Initialized SQLite database at {db.db_path}")


if __name__ == "__main__":
    main()
