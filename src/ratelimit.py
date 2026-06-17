"""Lightweight rate limiting to protect your free OpenRouter credits.

Two guards, both configurable in config.py / .env:
  1. MIN_SECONDS_BETWEEN_CALLS - throttle: wait between consecutive LLM calls.
  2. MAX_CALLS_PER_DAY - hard daily cap on LLM calls (persisted to disk so it
     survives restarts). Raises DailyLimitReached when hit.

Each user question makes 2 calls (answer + verification).
"""
import json
import threading
import time
from datetime import date

import config

_lock = threading.Lock()
_last_call = 0.0
_USAGE_FILE = config.DATA_DIR / "usage.json"


class DailyLimitReached(RuntimeError):
    """Raised when the configured daily call cap has been reached."""


def _load() -> dict:
    try:
        data = json.loads(_USAGE_FILE.read_text())
    except Exception:
        data = {}
    # Reset the counter when the day rolls over.
    if data.get("date") != date.today().isoformat():
        data = {"date": date.today().isoformat(), "count": 0}
    return data


def _save(data: dict) -> None:
    try:
        _USAGE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def usage_today():
    """Return (calls_used_today, daily_cap) for display in the UI."""
    return _load()["count"], config.MAX_CALLS_PER_DAY


def guard() -> None:
    """Call this immediately before every LLM request.

    Enforces the daily cap (raises DailyLimitReached) and the min-interval
    throttle (sleeps if needed), then records the call.
    """
    global _last_call
    with _lock:
        data = _load()
        if data["count"] >= config.MAX_CALLS_PER_DAY:
            raise DailyLimitReached(
                f"Daily limit reached ({config.MAX_CALLS_PER_DAY} LLM calls). "
                "This protects your free credits. Raise MAX_CALLS_PER_DAY in .env "
                "or wait until tomorrow."
            )

        wait = config.MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)

        _last_call = time.monotonic()
        data["count"] += 1
        _save(data)
