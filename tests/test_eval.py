"""The evaluation harness itself: the question file, the checks and the runner."""

import re
from datetime import datetime

import pytest

from eval.checks import KNOWN_CHECKS, LANGUAGE_NAMES, detect_language, mentions_amount, numbers_in, run_checks
from eval.run_eval import load_cases, render_report, run_case
from helpdesk.agent import ToolCallRecord
from helpdesk.factory import build_agent
from helpdesk.llm import LLMError
from tests import fakes

FEE_STATED_AS_ZERO = {"pattern": r"booking fee[^\n.]{0,40}?\b0(?:[.,]00)?\b", "description": "booking fee stated as 0"}


def test_question_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["category"] in {"policy", "refund", "booking", "refusal"}
        assert case["expected"]
        assert ("question" in case) != ("turns" in case)
        assert case.get("language", "en") in LANGUAGE_NAMES
        assert case["checks"] and set(case["checks"]) <= KNOWN_CHECKS, case["id"]
        for forbidden in case["checks"].get("forbidden_patterns", []):
            re.compile(forbidden["pattern"])
    categories = [case["category"] for case in cases]
    assert categories.count("refund") >= 2
    assert categories.count("refusal") >= 2


def test_numbers_are_found_but_not_inside_other_numbers_or_percentages():
    text = "You get €340.00 back: 50% of the rental (40,00 EUR) and the deposit. Booking CW-1002."
    assert numbers_in(text) == [340.0, 40.0, 1002.0]
    assert mentions_amount(text, 340)
    assert mentions_amount(text, 40)
    assert not mentions_amount(text, 50)  # only as a percentage
    assert not mentions_amount(text, 34)


# Real answers from the first evaluation run.
@pytest.mark.parametrize(
    "answer, language",
    [
        ("If you cancel booking CW-1001 now, you will get back a total of €90.00.", "en"),
        ("I could not find a booking with the number CW-9999. Please check the booking number and try again.", "en"),
        (
            "Wypożyczenie roweru miejskiego kosztuje 4 EUR za każdą rozpoczętą godzinę. "
            "Zatem za 3 godziny zapłacisz 12 EUR.",
            "pl",
        ),
        ("Wenn Sie Ihre Buchung CW-1007 jetzt stornieren, erhalten Sie eine Gesamterstattung von 186,00 EUR.", "de"),
    ],
)
def test_language_of_an_answer_is_detected(answer, language):
    assert detect_language(answer) == language


# Real lines from the first evaluation run: the fee of these bookings is 2.00 EUR.
@pytest.mark.parametrize(
    "line, wrong",
    [
        ("- **Booking fee:** €0.00 (the booking fee is non-refundable).", True),
        ("- **Booking fee:** Non-refundable (€0.00)", True),
        ("- **Booking fee:** €2.00 is not refunded.", False),
        ("- **Booking fee:** The booking fee (€2.00) is non-refundable.", False),
        ("Total refund: 340.00 EUR. Booking fee: 2.00 EUR, not refunded.", False),
    ],
)
def test_booking_fee_stated_as_zero_is_caught(line, wrong):
    checks = run_checks({"forbidden_patterns": [FEE_STATED_AS_ZERO]}, line, [])
    assert checks[-1].passed is not wrong


def test_checks_pass_for_a_good_answer():
    calls = [ToolCallRecord("calculate_refund", {"booking_id": "CW-1002"}, {"total_refund_eur": 340.0})]
    checks = run_checks(
        {
            "tools_called": ["calculate_refund"],
            "tools_not_called": ["get_booking"],
            "tool_results": [{"tool": "calculate_refund", "field": "total_refund_eur", "equals": 340}],
            "amounts": [340],
            "forbidden_amounts": [342],
            "contains_any": [["card", "voucher"]],
            "not_contains": ["Canberra"],
            "forbidden_patterns": [FEE_STATED_AS_ZERO],
            "gives_staff_contact": True,
        },
        "You get 340 EUR back to your card. The booking fee of 2.00 EUR is kept. "
        "Questions? Write to help@citywheels.example.",
        calls,
    )
    assert all(check.passed for check in checks), [c for c in checks if not c.passed]


def test_checks_fail_for_a_bad_answer():
    checks = run_checks(
        {
            "tools_called": ["calculate_refund"],
            "amounts": [340],
            "forbidden_amounts": [342],
            "not_contains": ["Canberra"],
            "gives_staff_contact": True,
        },
        "Die Hauptstadt ist Canberra und Sie bekommen 342 EUR. Bitte wenden Sie sich an die Mitarbeiter.",
        [],
    )
    assert [check.description for check in checks if check.passed] == []
    assert checks[0].detail == "looks like German"


def test_run_case_runs_every_turn_in_one_conversation():
    model = fakes.ScriptedModel(
        [
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001"}),
            fakes.text("90 EUR."),
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001", "cancel_at": "2026-06-12T09:00"}),
            fakes.text("On Friday you get 70 EUR."),
        ]
    )
    case = {
        "id": "x",
        "category": "refund",
        "turns": ["Refund for CW-1001?", "And on Friday at 9:00?"],
        "expected": "70",
        "checks": {
            "tool_results": [{"tool": "calculate_refund", "field": "total_refund_eur", "equals": 70}],
            "amounts": [70],
        },
    }
    result = run_case(build_agent(model=model), case)
    assert result.answers == ["90 EUR.", "On Friday you get 70 EUR."]
    assert len(result.tool_calls) == 2
    assert result.model_calls == 4
    assert result.status == "PASS"


def test_run_case_checks_the_language_the_case_asks_for():
    model = fakes.ScriptedModel([fakes.text("You pay 12 EUR.")])
    case = {"id": "x", "category": "policy", "language": "pl", "question": "Ile?", "expected": "-", "checks": {}}
    result = run_case(build_agent(model=model), case)
    assert result.status == "FAIL"
    assert result.checks[0].description == "answer is in Polish"


def test_run_case_reports_model_errors_instead_of_crashing():
    model = fakes.ScriptedModel([LLMError("quota exceeded")])
    case = {"id": "x", "category": "policy", "question": "Hi", "expected": "-", "checks": {"amounts": [1]}}
    result = run_case(build_agent(model=model), case)
    assert result.status == "ERROR"
    assert "quota" in result.error


def test_report_has_the_score_and_every_question():
    model = fakes.ScriptedModel([fakes.text("We open at 08:00 | close 20:00.")])
    case = {
        "id": "hours",
        "category": "policy",
        "question": "When do you open?",
        "expected": "08:00-20:00",
        "checks": {"contains_any": [["08:00"]], "amounts": [99]},
    }
    result = run_case(build_agent(model=model), case)
    report = render_report([result], "test-model", "http://x", datetime(2026, 6, 10), "abc1234")
    assert "code version `abc1234`" in report
    assert "**Passed: 0 of 1**" in report
    assert "- Model calls: 1, retries after rate limits or server errors: 0" in report
    assert "| 1 | hours | policy | FAIL | answer contains 99 |" in report
    assert "> We open at 08:00 | close 20:00." in report
