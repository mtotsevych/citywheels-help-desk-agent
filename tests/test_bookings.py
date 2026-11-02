import json

import pytest

from helpdesk.config import DEFAULT_DATA_DIR


@pytest.mark.parametrize("raw", ["CW-1002", "cw-1002", "CW1002", "cw 1002", "  CW-1002  ", "1002", "#1002", 1002])
def test_booking_number_formats_are_accepted(tools, raw):
    result = tools.get_booking(raw)
    assert result["found"] is True
    assert result["booking"]["id"] == "CW-1002"


def test_booking_is_returned_exactly_as_stored(tools):
    stored = json.loads((DEFAULT_DATA_DIR / "bookings.json").read_text(encoding="utf-8"))
    expected = next(b for b in stored if b["id"] == "CW-1004")
    assert tools.get_booking("CW-1004") == {"found": True, "booking": expected}


@pytest.mark.parametrize("raw", ["CW-9999", "9999", "cw-0"])
def test_unknown_booking_number_is_not_found(tools, raw):
    result = tools.get_booking(raw)
    assert result["found"] is False
    assert "no booking" in result["message"].lower()


@pytest.mark.parametrize("raw", ["", "hello", "CW-", "AB-1002", "1002; DROP TABLE", None, 12.5, True, ["CW-1002"]])
def test_invalid_booking_number_is_not_found_and_does_not_crash(tools, raw):
    result = tools.get_booking(raw)
    assert result["found"] is False
    assert "not a valid booking number" in result["message"]


def test_changing_a_returned_booking_does_not_change_the_data(tools):
    tools.get_booking("CW-1001")["booking"]["status"] = "cancelled"
    assert tools.get_booking("CW-1001")["booking"]["status"] == "confirmed"
