"""Persistent daily request budgeting and Gemini quota-error classification."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PACIFIC_TIMEZONE = ZoneInfo("America/Los_Angeles")


def pacific_day() -> str:
    """Gemini request-per-day quota resets at midnight Pacific time."""
    return datetime.now(PACIFIC_TIMEZONE).date().isoformat()


@dataclass
class DailyRequestBudget:
    path: Path
    limit: int
    day: str
    used: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        limit: int,
        initial_used: int = 0,
    ) -> "DailyRequestBudget":
        if limit < 1:
            raise ValueError("daily request limit must be at least 1")
        if not 0 <= initial_used <= limit:
            raise ValueError("initial daily request usage must be within the daily limit")
        state_path = Path(path)
        current_day = pacific_day()
        if not state_path.exists():
            return cls(state_path, limit, current_day, initial_used)
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Request budget state must be a JSON object: {state_path}")
        used = int(raw.get("used", 0)) if raw.get("day") == current_day else 0
        if used < 0:
            raise ValueError(f"Request budget state has negative usage: {state_path}")
        return cls(state_path, limit, current_day, used)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self, request_count: int) -> None:
        if request_count < 0 or request_count > self.remaining:
            raise ValueError("request budget reservation exceeds remaining daily quota")
        self.used += request_count
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(
                    {
                        "day": self.day,
                        "timezone": "America/Los_Angeles",
                        "used": self.used,
                        "limit": self.limit,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def is_daily_quota_error(error: Exception) -> bool:
    """Recognize Gemini's explicit daily-quota response without stopping on RPM."""
    message = str(error).casefold().replace("-", "_").replace(" ", "_")
    daily_markers = (
        "requestsperday",
        "requests_per_day",
        "request_per_day",
        "per_day",
        "daily_quota",
        "rpd",
    )
    return any(marker in message for marker in daily_markers)


def is_rate_limit_error(error: Exception) -> bool:
    """Recognize per-minute/per-second quota responses that are safe to retry."""
    if is_daily_quota_error(error):
        return False
    message = str(error).casefold().replace("-", "_").replace(" ", "_")
    rate_markers = (
        "rate_limit_exceeded",
        "requestsperminute",
        "requests_per_minute",
        "request_per_minute",
        "perminute",
        "per_second",
        "retryinfo",
    )
    return any(marker in message for marker in rate_markers)


def is_transient_service_error(error: Exception) -> bool:
    """Recognize temporary model unavailability that is safe to retry."""
    message = str(error).casefold().replace("-", "_").replace(" ", "_")
    return (
        ("503" in message and "unavailable" in message)
        or "service_unavailable" in message
        or "temporarily_unavailable" in message
        or "internal_server_error" in message
        or "empty_response" in message
        or "empty_ocr_response" in message
    )
