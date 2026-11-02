"""The evaluation harness itself: the question file, the checks and the runner."""

from datetime import datetime

from eval.checks import KNOWN_CHECKS, mentions_amount, numbers_in, run_checks
from eval.run_eval import load_cases, render_report, run_case
from helpdesk.agent import ToolCallRecord
from helpdesk.factory import build_agent
from helpdesk.llm import LLMError
from tests import fakes


def test_question_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["category"] in {"policy", "refund", "booking", "refusal"}
        assert case["expected"]
        assert ("question" in case) != ("turns" in case)
        assert case["checks"] and set(case["checks"]) <= KNOWN_CHECKS, case["id"]
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
            "refers_to_staff": True,
        },
        "You get 340 EUR back to your card. Questions? Ask our staff.",
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
            "refers_to_staff": True,
        },
        "The capital is Canberra. You get 342 EUR.",
        [],
    )
    assert [check.passed for check in checks] == [False, False, False, False, False]


def test_run_case_runs_every_turn_in_one_conversation():
    model = fakes.ScriptedModel(
        [
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001"}),
            fakes.text("90 EUR."),
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001", "cancel_at": "2026-06-12T09:00"}),
            fakes.text("On Friday: 70 EUR."),
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
    assert result.answers == ["90 EUR.", "On Friday: 70 EUR."]
    assert len(result.tool_calls) == 2
    assert result.status == "PASS"


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
    report = render_report([run_case(build_agent(model=model), case)], "test-model", "http://x", datetime(2026, 6, 10))
    assert "**Passed: 0 of 1**" in report
    assert "| 1 | hours | policy | FAIL | answer contains 99 |" in report
    assert "> We open at 08:00 | close 20:00." in report
