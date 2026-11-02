"""Automatic checks of one agent answer against the expectations in questions.json.

They are deliberately simple (tool names, numbers, phrases), so a failed check
is easy to understand. The price is that they can be fooled: read the answers
in results.md too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from helpdesk.agent import ToolCallRecord

STAFF_HINTS = ("help@citywheels.example", "+48 000 000 000", "staff")
KNOWN_CHECKS = {
    "tools_called",
    "tools_called_any",
    "tools_not_called",
    "tool_results",
    "amounts",
    "forbidden_amounts",
    "contains_any",
    "not_contains",
    "refers_to_staff",
}

# A number such as 340, 340.00 or 340,00 that is not part of a longer number and not a percentage.
_NUMBER = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)?(?![\d%])(?!\s*%)")


@dataclass
class Check:
    description: str
    passed: bool
    detail: str = ""


def numbers_in(text: str) -> list[float]:
    return [float(match.group().replace(",", ".")) for match in _NUMBER.finditer(text)]


def mentions_amount(text: str, amount: float) -> bool:
    return any(abs(number - amount) < 0.005 for number in numbers_in(text))


def run_checks(checks: dict, answer: str, tool_calls: list[ToolCallRecord]) -> list[Check]:
    called = [call.name for call in tool_calls]
    called_text = ", ".join(called) or "no tools"
    lower = answer.lower()
    results: list[Check] = []

    for tool in checks.get("tools_called", []):
        results.append(Check(f"calls {tool}", tool in called, f"called: {called_text}"))

    if "tools_called_any" in checks:
        options = checks["tools_called_any"]
        results.append(
            Check(f"calls one of {', '.join(options)}", any(t in called for t in options), f"called: {called_text}")
        )

    for tool in checks.get("tools_not_called", []):
        results.append(Check(f"does not call {tool}", tool not in called, f"called: {called_text}"))

    for expected in checks.get("tool_results", []):
        values = [call.result.get(expected["field"]) for call in tool_calls if call.name == expected["tool"]]
        results.append(
            Check(
                f"{expected['tool']} returns {expected['field']} = {expected['equals']}",
                expected["equals"] in values,
                f"returned: {values}",
            )
        )

    for amount in checks.get("amounts", []):
        results.append(Check(f"answer contains {amount}", mentions_amount(answer, amount)))

    for amount in checks.get("forbidden_amounts", []):
        results.append(Check(f"answer does not contain {amount}", not mentions_amount(answer, amount)))

    for group in checks.get("contains_any", []):
        found = [phrase for phrase in group if phrase.lower() in lower]
        label = group[0] if len(group) == 1 else "one of: " + ", ".join(group)
        results.append(Check(f"answer contains {label}", bool(found)))

    for phrase in checks.get("not_contains", []):
        results.append(Check(f"answer does not contain {phrase}", phrase.lower() not in lower))

    if checks.get("refers_to_staff"):
        results.append(Check("answer refers to the staff", any(hint in lower for hint in STAFF_HINTS)))

    return results
