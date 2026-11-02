"""Refund rules from data/policies/cancellations.md, implemented in code.

The model never calculates money: it calls the calculate_refund tool, which uses
compute_refund() below. The result includes a summary sentence written by this
code, so the model only has to pass it on, not interpret the numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

FULL_REFUND_NOTICE = timedelta(hours=48)  # 48 hours or more before the start: 100%
HALF_REFUND_NOTICE = timedelta(hours=24)  # 24 hours or more (but less than 48): 50%
CANCELLABLE_STATUSES = {"confirmed"}

REFUND_METHOD = {
    "card": "Refunded to the same card within 5 working days.",
    "voucher": (
        "Refunded as a new CityWheels voucher of the same value, never as money. "
        "The deposit is refunded the same way it was paid."
    ),
}
UNKNOWN_METHOD = "Unknown payment method: the staff will explain how the money comes back."


def _money(value: object) -> Decimal:
    # str() first, so 0.1 stays 0.1 instead of 0.1000000000000000055...
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _eur(amount: Decimal) -> str:
    return f"{amount:.2f} EUR"


def rental_refund_share(notice: timedelta) -> tuple[Decimal, str]:
    """Share of the rental price refunded for a cancellation `notice` before the start, and the rule's condition."""
    if notice >= FULL_REFUND_NOTICE:
        return Decimal("1"), "48 hours or more"
    if notice >= HALF_REFUND_NOTICE:
        return Decimal("0.5"), "less than 48 hours but 24 hours or more"
    return Decimal("0"), "less than 24 hours"


def compute_refund(booking: dict, cancel_at: datetime) -> dict:
    """How much comes back if `booking` is cancelled at `cancel_at` (timezone-aware)."""
    booking_id = booking["id"]
    status = booking["status"]
    start = datetime.fromisoformat(booking["start"])
    times = {"cancel_at": cancel_at.isoformat(), "rental_start": start.isoformat()}
    not_cancellable = {"booking_id": booking_id, "status": status, "can_cancel": False, "total_refund_eur": 0.0}

    if status not in CANCELLABLE_STATUSES:
        summary = (
            f"Booking {booking_id} cannot be cancelled: bookings with the status '{status}' "
            "cannot be cancelled, and nothing is refunded for them."
        )
        if status == "in_progress":
            summary += (
                " The deposit is not part of a cancellation: it is returned in full when the bike "
                "comes back on time and without damage."
            )
        return {**not_cancellable, "summary": summary, **times}

    if cancel_at >= start:
        summary = (
            f"Booking {booking_id} cannot be cancelled at that time: the rental starts at "
            f"{start:%Y-%m-%d %H:%M}, so it has already started and nothing is refunded."
        )
        return {**not_cancellable, "summary": summary, **times}

    notice = start - cancel_at
    hours = round(notice.total_seconds() / 3600, 2)
    share, condition = rental_refund_share(notice)
    percent = int(share * 100)
    rental_price = _money(booking["rental_price_eur"])
    rental_refund = _money(rental_price * share)
    deposit_refund = _money(booking["deposit_eur"])  # always refunded in full on cancellation
    booking_fee = _money(booking["booking_fee_eur"])  # never refunded
    total = rental_refund + deposit_refund
    paid_with = booking.get("paid_with")
    method = REFUND_METHOD.get(paid_with, UNKNOWN_METHOD)

    summary = " ".join(
        [
            f"Total refund: {_eur(total)}.",
            f"Rental price: {_eur(rental_refund)} of {_eur(rental_price)} back ({percent}%: cancelled "
            f"{hours:g} hours before the start, which is {condition}).",
            f"Deposit: {_eur(deposit_refund)} back in full.",
            f"Booking fee: {_eur(booking_fee)}, not refunded." if booking_fee else "No booking fee was paid.",
            method,
        ]
    )
    return {
        "booking_id": booking_id,
        "status": status,
        "can_cancel": True,
        "total_refund_eur": float(total),
        "summary": summary,
        **times,
        "hours_before_start": hours,
        "rental_price_eur": float(rental_price),
        "rental_refund_percent": percent,
        "rental_refund_eur": float(rental_refund),
        "deposit_refund_eur": float(deposit_refund),
        "booking_fee_eur": float(booking_fee),
        "booking_fee_refunded": False,
        "paid_with": paid_with,
        "refund_method": method,
    }
