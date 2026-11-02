"""The tools the model can call, their JSON schemas, and a dispatcher that runs them.

Every tool returns a JSON-serialisable dict. Problems (unknown booking, bad
arguments, a bug in a tool) come back as a result the model can read, never as
an exception that would stop the program.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .bookings import BookingStore, normalize_booking_id
from .config import parse_datetime
from .policies import PolicyIndex
from .refunds import compute_refund

log = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": (
                "Search the CityWheels documents: prices, add-ons, deposits, late return fees, "
                "cancellation and refund rules, opening hours, location, age limits, payment, "
                "helmets, damage, theft and other FAQ. Returns the most relevant sections. "
                "Use it for every question about the shop and its rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords in English, e.g. 'opening hours Saturday' or 'e-bike price per day'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": (
                "Get one booking by its number, e.g. CW-1002: status, bike type, number of bikes, "
                "start and end time, add-ons, rental price, booking fee, deposit and payment method. "
                "Returns found=false when there is no such booking."
            ),
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string", "description": "Booking number, e.g. CW-1002."}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": (
                "Calculate how much money the customer gets back if they cancel a booking, using "
                "the CityWheels cancellation rules. Cancels nothing, only calculates. Always use it "
                "for refund amounts instead of calculating them yourself. The 'summary' field states "
                "every amount and the rule that applies: base your answer on it. Returns found=false "
                "when there is no such booking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string", "description": "Booking number, e.g. CW-1002."},
                    "cancel_at": {
                        "type": "string",
                        "description": (
                            "Only for 'what if I cancel later' questions: when the customer would "
                            "cancel, as an ISO 8601 date and time in shop time, e.g. "
                            "2026-06-11T09:00:00+02:00. Leave it out to cancel now."
                        ),
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
]


def _not_found(booking_id: object) -> dict:
    normalized = normalize_booking_id(booking_id)
    if normalized is None:
        return {
            "found": False,
            "booking_id": str(booking_id),
            "message": f"'{booking_id}' is not a valid booking number. Booking numbers look like CW-1001.",
        }
    return {
        "found": False,
        "booking_id": normalized,
        "message": f"There is no booking with the number {normalized}.",
    }


class HelpDeskTools:
    def __init__(self, bookings: BookingStore, policies: PolicyIndex, now: datetime):
        self.bookings = bookings
        self.policies = policies
        self.now = now
        self._functions: dict[str, Callable[..., dict]] = {
            "search_policies": self.search_policies,
            "get_booking": self.get_booking,
            "calculate_refund": self.calculate_refund,
        }

    @classmethod
    def from_data_dir(cls, data_dir: Path, now: datetime) -> HelpDeskTools:
        return cls(
            bookings=BookingStore.from_file(data_dir / "bookings.json"),
            policies=PolicyIndex.from_dir(data_dir / "policies"),
            now=now,
        )

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    # --- the tools ----------------------------------------------------------

    def search_policies(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip():
            return {"query": query, "results": [], "message": "The query is empty."}
        hits = self.policies.search(query, limit=3)
        if not hits:
            return {
                "query": query,
                "results": [],
                "message": "Nothing in the CityWheels documents matches this query.",
            }
        return {
            "query": query,
            "results": [
                {
                    "source": f"data/policies/{section.source}",
                    "section": section.heading,
                    "text": section.text,
                }
                for section, _score in hits
            ],
        }

    def get_booking(self, booking_id: str) -> dict:
        booking = self.bookings.get(booking_id)
        if booking is None:
            return _not_found(booking_id)
        return {"found": True, "booking": booking}

    def calculate_refund(self, booking_id: str, cancel_at: str | None = None) -> dict:
        booking = self.bookings.get(booking_id)
        if booking is None:
            return _not_found(booking_id)

        when = self.now
        if cancel_at:
            try:
                when = parse_datetime(str(cancel_at), default_time_from=self.now)
            except ValueError:
                return {"error": f"cancel_at is not a valid ISO 8601 date and time: {cancel_at!r}"}
            if when < self.now:
                return {
                    "error": (
                        f"cancel_at {when.isoformat()} is in the past (now is {self.now.isoformat()}). "
                        "A booking can only be cancelled now or later."
                    )
                }
        return {"found": True, **compute_refund(booking, when)}

    # --- dispatcher ---------------------------------------------------------

    def run(self, name: str, arguments: str | dict | None) -> dict:
        """Run a tool call from the model. Never raises: errors become a result."""
        function = self._functions.get(name)
        if function is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(self._functions)}."}

        if isinstance(arguments, dict):
            args: Any = arguments
        else:
            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                return {"error": f"The arguments for {name} are not valid JSON: {arguments!r}"}
        if not isinstance(args, dict):
            return {"error": f"The arguments for {name} must be a JSON object."}

        try:
            return function(**args)
        except TypeError as exc:  # missing or unexpected arguments
            return {"error": f"Wrong arguments for {name}: {exc}"}
        except Exception as exc:  # a bug or bad data must not crash the agent
            log.exception("Tool %s failed", name)
            return {"error": f"The tool {name} failed: {exc}"}
