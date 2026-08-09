"""Small .env reader for semantic-pipeline CLIs."""

from __future__ import annotations

import os
from pathlib import Path


def get_env_value(name: str) -> str | None:
    """Return an exported value first, then the repository `.env` value."""
    if value := os.getenv(name):
        return value

    dotenv_path = Path(__file__).resolve().parents[3] / ".env"
    if not dotenv_path.is_file():
        return None

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None
