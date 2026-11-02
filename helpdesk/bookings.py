"""Loading bookings and finding one by its number."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Accepts "CW-1002", "cw1002", "CW 1002", "#1002" and plain "1002".
_BOOKING_ID = re.compile(r"^\s*#?\s*(?:CW)?[\s\-_]*(\d{1,6})\s*$", re.IGNORECASE)


def normalize_booking_id(raw: object) -> str | None:
    """Return the canonical form "CW-1234", or None if this is not a booking number."""
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        return None
    match = _BOOKING_ID.match(str(raw))
    return f"CW-{match.group(1)}" if match else None


class BookingStore:
    def __init__(self, bookings: list[dict]):
        self._by_id = {booking["id"].upper(): booking for booking in bookings}

    @classmethod
    def from_file(cls, path: Path) -> BookingStore:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def get(self, booking_id: object) -> dict | None:
        """The booking as a dict (a copy), or None when it does not exist."""
        normalized = normalize_booking_id(booking_id)
        if normalized is None:
            return None
        booking = self._by_id.get(normalized)
        return dict(booking) if booking is not None else None
