"""
SamiX Utility Validators & Text Helpers
Location: src/utils/validators.py
"""
from __future__ import annotations
import re

def is_valid_email(email: str) -> bool:
    """
    Validates email format using a robust regex pattern.
    Matches: user@domain.com, user.name@company.org, etc.
    """
    if not email:
        return False
    # More comprehensive pattern than the basic one
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.fullmatch(pattern, email.strip()))

def safe_text(value: str, default: str = "") -> str:
    """Cleans input text by removing whitespace and providing a fallback."""
    text = (value or "").strip()
    return text if text else default

def is_strong_password(password: str) -> bool:
    """
    Basic security check: Ensures password is at least 8 characters.
    You can expand this to check for symbols/numbers later.
    """
    return len(password) >= 8

def sanitize_filename(filename: str) -> str:
    """
    Removes special characters from filenames to prevent 
    errors during audio processing/saving.
    """
    # Keep alphanumeric, dots, and underscores only
    return re.sub(r'[^a-zA-Z0-9._]', '_', filename)
