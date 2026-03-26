from __future__ import annotations

import shutil
from pathlib import Path

from _bootstrap import setup_project_root

setup_project_root()

from config import Config


def main() -> None:
    for path in [Path(Config.CACHE_DIR), Path(Config.EXPORTS_DIR)]:
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    print("Cleared cache and export directories.")


if __name__ == "__main__":
    main()
