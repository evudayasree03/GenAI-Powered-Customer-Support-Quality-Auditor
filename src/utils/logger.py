from __future__ import annotations

import logging
from pathlib import Path

from config import Config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    Path(Config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
