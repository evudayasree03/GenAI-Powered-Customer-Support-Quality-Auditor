from __future__ import annotations

import sys
from pathlib import Path


def setup_project_root() -> Path:
    """Ensure direct script execution can import project packages like `src`."""
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
