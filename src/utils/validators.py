from __future__ import annotations

import re


def is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


def safe_text(value: str, default: str = "") -> str:
    text = (value or "").strip()
    return text if text else default
