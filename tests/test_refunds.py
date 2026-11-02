"""Every rule of data/policies/cancellations.md, plus the example bookings."""

from datetime import datetime, timedelta

import pytest

from helpdesk.refunds import compute_refund

START = datetime.fromisoformat("2026-06-20T10:00:00+02:00")


def booking(**changes) -> dict:
    data = {
        "id": "CW-TEST",
        "status": "confirmed",
        "bike_type": "city",
        "bikes": 1,
        "start": START.isoformat(),
        "end": "2026-06-21T10:00:00+02:00",
        "add_ons": [],
        "rental_price_eur": 40.0,
        "booking_fee_eur": 2.0,
        "deposit_eur": 50.0,
        "paid_with": "card",
    }
    data.update(changes)
    return data


# Rule: how much of the rental price comes back depends on the notice before the start.
@pytest.mark.parametrize(
    "notice, percent, rental_refund",
    [
        (timedelta(days=5), 100, 40.0),
        (timedelta(hours=48), 100, 40.0),  # exactly 48 h is "48 hours or more"
        (timedelta(hours=47, minutes=59), 50, 20.0),
        (timedelta(hours=30), 50, 20.0),
        (timedelta(hours=24), 50, 20.0),  # exactly 24 h is "24 hours or more"
        (timedelta(hours=23, minutes=59), 0, 0.0),
        (timedelta(minutes=1), 0, 0.0),
    ],
)
def test_rental_refund_depends_on_notice(notice, percent, rental_refund):
    result = compute_refund(booking(), START - notice)
    assert result["can_cancel"] is True
    assert result["rental_refund_percent"] == percent
    assert result["rental_refund_eur"] == rental_refund
    assert result["total_refund_eur"] == rental_refund + 50.0


# Rule: the booking fee is never refunded.
@pytest.mark.parametrize("notice", [timedelta(days=3), timedelta(hours=30), timedelta(hours=1)])
def test_booking_fee_is_never_refunded(notice):
    result = compute_refund(booking(), START - notice)
    assert result["booking_fee_refund_eur"] == 0.0
    assert result["total_refund_eur"] == result["rental_refund_eur"] + result["deposit_refund_eur"]


# Rule: the deposit is always refunded in full when a booking is cancelled.
@pytest.mark.parametrize("notice", [timedelta(days=3), timedelta(hours=30), timedelta(hours=1)])
def test_deposit_is_always_refunded_in_full(notice):
    result = compute_refund(booking(deposit_eur=300.0), START - notice)
    assert result["deposit_refund_eur"] == 300.0


# Rule: in_progress, completed and cancelled bookings cannot be cancelled; nothing is refunded.
@pytest.mark.parametrize("status", ["in_progress", "completed", "cancelled"])
def test_only_confirmed_bookings_can_be_cancelled(status):
    result = compute_refund(booking(status=status), START - timedelta(days=5))
    assert result["can_cancel"] is False
    assert result["total_refund_eur"] == 0.0
    assert "cannot be cancelled" in result["reason"]
    assert "rental_refund_eur" not in result


def test_confirmed_booking_cannot_be_cancelled_once_the_rental_has_started():
    for cancel_at in (START, START + timedelta(hours=1)):
        result = compute_refund(booking(), cancel_at)
        assert result["can_cancel"] is False
        assert result["total_refund_eur"] == 0.0


# Rule: card payments go back to the card within 5 working days.
def test_card_refund_goes_back_to_the_card():
    result = compute_refund(booking(paid_with="card"), START - timedelta(days=3))
    assert "same card within 5 working days" in result["refund_method"]


# Rule: voucher payments come back as a new voucher, never as money.
def test_voucher_refund_comes_back_as_a_voucher():
    result = compute_refund(booking(paid_with="voucher"), START - timedelta(days=3))
    assert "new CityWheels voucher" in result["refund_method"]
    assert "never as money" in result["refund_method"]


def test_half_refund_is_rounded_to_cents():
    result = compute_refund(booking(rental_price_eur=12.35), START - timedelta(hours=30))
    assert result["rental_refund_eur"] == 6.18  # 6.175 rounded half up


def test_not_refunded_part_is_the_rest_of_the_rental_plus_the_fee():
    result = compute_refund(booking(), START - timedelta(hours=30))
    assert result["not_refunded_eur"] == 20.0 + 2.0


# The example bookings in data/bookings.json, cancelled at the fixed "now" (2026-06-10 09:00).
@pytest.mark.parametrize(
    "booking_id, total",
    [
        ("CW-1001", 90.0),  # 73 h before: 40 + 50 deposit
        ("CW-1002", 340.0),  # 33 h before: 50% of 80 + 300 deposit
        ("CW-1003", 50.0),  # 6 h before: only the 50 deposit
        ("CW-1004", 73.0),  # 10 days before: 23 + 50 deposit, as a voucher
        ("CW-1005", 0.0),  # already cancelled
        ("CW-1006", 0.0),  # completed
        ("CW-1007", 186.0),  # exactly 48 h before: 36 + 150 deposit
        ("CW-1008", 0.0),  # in progress
        ("CW-1009", 169.0),  # 5 days before: 119 + 50 deposit
        ("CW-1010", 170.0),  # exactly 24 h before: 50% of 40 + 150 deposit
    ],
)
def test_example_bookings(tools, booking_id, total):
    result = tools.calculate_refund(booking_id)
    assert result["found"] is True
    assert result["total_refund_eur"] == total


def test_unknown_booking_is_not_found(tools):
    result = tools.calculate_refund("CW-9999")
    assert result["found"] is False
    assert "CW-9999" in result["message"]


def test_cancel_later_uses_cancel_at(tools):
    # Friday 12 June 09:00 is 25 h before CW-1001 starts: 50% of 40 + 50 deposit.
    result = tools.calculate_refund("CW-1001", cancel_at="2026-06-12T09:00:00+02:00")
    assert result["rental_refund_percent"] == 50
    assert result["total_refund_eur"] == 70.0


def test_cancel_at_without_offset_is_shop_time(tools):
    result = tools.calculate_refund("CW-1001", cancel_at="2026-06-12T09:00:00")
    assert result["cancel_at"] == "2026-06-12T09:00:00+02:00"


def test_cancel_at_with_only_a_date_uses_the_current_time_of_day(tools):
    result = tools.calculate_refund("CW-1001", cancel_at="2026-06-12")
    assert result["cancel_at"] == "2026-06-12T09:00:00+02:00"


def test_cancel_at_in_utc_is_converted(tools):
    # 07:00 UTC is 09:00 in Warsaw: the same 25 h notice as above.
    result = tools.calculate_refund("CW-1001", cancel_at="2026-06-12T07:00:00Z")
    assert result["hours_before_start"] == 25.0


def test_cancel_at_in_the_past_is_rejected(tools):
    result = tools.calculate_refund("CW-1001", cancel_at="2026-06-09T09:00:00+02:00")
    assert "error" in result


def test_cancel_at_that_is_not_a_date_is_rejected(tools):
    result = tools.calculate_refund("CW-1001", cancel_at="next Friday")
    assert "error" in result
