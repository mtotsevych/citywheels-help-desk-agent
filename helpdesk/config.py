"""Settings read from environment variables (and from .env, loaded by the CLI)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

SHOP_TIMEZONE = ZoneInfo("Europe/Warsaw")
# The data README fixes "now" so that answers are the same every time.
DEFAULT_NOW = "2026-06-10T09:00:00+02:00"

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class ConfigError(Exception):
    """A setting is missing or has a wrong value."""


def parse_datetime(value: str, *, default_time_from: datetime | None = None) -> datetime:
    """Parse an ISO 8601 date/time. Times without an offset are shop time (Europe/Warsaw).

    A bare date such as "2026-06-12" gets the time of day from `default_time_from`
    (the current shop time), so "cancel on Friday" means Friday at the current hour.
    Raises ValueError for anything that is not a date.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):  # Python 3.10 does not understand the Z suffix
        text = text[:-1] + "+00:00"
    if len(text) == 10 and default_time_from is not None:
        day = date.fromisoformat(text)
        parsed = datetime.combine(day, default_time_from.timetz().replace(tzinfo=None))
    else:
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHOP_TIMEZONE)
    return parsed


def shop_now() -> datetime:
    """The shop's "current" date and time: CITYWHEELS_NOW or the fixed default."""
    value = os.getenv("CITYWHEELS_NOW") or DEFAULT_NOW
    try:
        return parse_datetime(value)
    except ValueError as exc:
        raise ConfigError(f"CITYWHEELS_NOW is not a valid ISO date and time: {value!r}") from exc


def data_dir() -> Path:
    return Path(os.getenv("CITYWHEELS_DATA_DIR") or DEFAULT_DATA_DIR)


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float | None = None

    @classmethod
    def from_env(cls, model_override: str | None = None) -> LLMSettings:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and put your API key there.")
        temperature_text = os.getenv("LLM_TEMPERATURE", "").strip()
        try:
            temperature = float(temperature_text) if temperature_text else None
        except ValueError as exc:
            raise ConfigError(f"LLM_TEMPERATURE must be a number, got {temperature_text!r}") from exc
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            model=model_override or os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL,
            temperature=temperature,
        )
