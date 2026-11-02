"""Refund rules from data/policies/cancellations.md, implemented in code.

The model never calculates money: it calls the calculate_refund tool, which uses
compute_refund() below, and repeats the result.
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


def _money(value: object) -> Decimal:
    # str() first, so 0.1 stays 0.1 instead of 0.1000000000000000055...
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def rental_refund_share(notice: timedelta) -> tuple[Decimal, str]:
    """Share of the rental price refunded for a cancellation `notice` before the start."""
    if notice >= FULL_REFUND_NOTICE:
        return Decimal("1"), "Cancelled 48 hours or more before the start: 100% of the rental price is refunded."
    if notice >= HALF_REFUND_NOTICE:
        return Decimal("0.5"), (
            "Cancelled less than 48 hours but 24 hours or more before the start: 50% of the rental price is refunded."
        )
    return Decimal("0"), "Cancelled less than 24 hours before the start: the rental price is not refunded."


def compute_refund(booking: dict, cancel_at: datetime) -> dict:
    """How much comes back if `booking` is cancelled at `cancel_at` (timezone-aware)."""
    status = booking["status"]
    start = datetime.fromisoformat(booking["start"])
    result: dict = {
        "booking_id": booking["id"],
        "status": status,
        "cancel_at": cancel_at.isoformat(),
        "rental_start": start.isoformat(),
    }

    if status not in CANCELLABLE_STATUSES:
        result.update(
            can_cancel=False,
            total_refund_eur=0.0,
            reason=f"Bookings with the status '{status}' cannot be cancelled, and nothing is refunded for them.",
        )
        if status == "in_progress":
            result["deposit_note"] = (
                "The deposit is not part of a cancellation: it is returned in full when the bike "
                "comes back on time and without damage."
            )
        return result

    if cancel_at >= start:
        result.update(
            can_cancel=False,
            total_refund_eur=0.0,
            reason="At that time the rental has already started, so the booking can no longer be cancelled.",
        )
        return result

    notice = start - cancel_at
    share, rule = rental_refund_share(notice)
    rental_price = _money(booking["rental_price_eur"])
    rental_refund = _money(rental_price * share)
    deposit_refund = _money(booking["deposit_eur"])  # always refunded in full on cancellation
    booking_fee = _money(booking["booking_fee_eur"])  # never refunded
    paid_with = booking.get("paid_with")

    result.update(
        can_cancel=True,
        hours_before_start=round(notice.total_seconds() / 3600, 2),
        rule_applied=rule,
        rental_price_eur=float(rental_price),
        rental_refund_percent=int(share * 100),
        rental_refund_eur=float(rental_refund),
        deposit_refund_eur=float(deposit_refund),
        booking_fee_eur=float(booking_fee),
        booking_fee_refund_eur=0.0,
        total_refund_eur=float(rental_refund + deposit_refund),
        not_refunded_eur=float(rental_price - rental_refund + booking_fee),
        paid_with=paid_with,
        refund_method=REFUND_METHOD.get(
            paid_with, "Unknown payment method: the staff will explain how the money comes back."
        ),
    )
    return result
