"""The agent loop, with a fake model: no network and no API key needed."""

import json

import pytest

from helpdesk.factory import build_agent
from helpdesk.llm import LLMError
from tests import fakes


def test_agent_runs_the_tool_the_model_asks_for_and_returns_the_final_answer():
    model = fakes.ScriptedModel(
        [
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1002"}, call_id="call_7"),
            fakes.text("You get 340 EUR back."),
        ]
    )
    agent = build_agent(model=model)

    reply = agent.ask("I want to cancel booking CW-1002. How much will I get back?")

    assert reply.answer == "You get 340 EUR back."
    assert reply.model_calls == 2
    assert [call.name for call in reply.tool_calls] == ["calculate_refund"]
    assert reply.tool_calls[0].arguments == {"booking_id": "CW-1002"}
    assert reply.tool_calls[0].result["total_refund_eur"] == 340.0
    # The second model call got the tool result, linked to the tool call.
    tool_message = model.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_7"
    assert json.loads(tool_message["content"])["total_refund_eur"] == 340.0


def test_model_gets_the_instructions_and_the_tool_schemas():
    model = fakes.ScriptedModel([fakes.text("Hello!")])
    build_agent(model=model).ask("Hi")

    first = model.requests[0]
    system = first["messages"][0]
    assert system["role"] == "system"
    assert "CityWheels" in system["content"]
    assert "Wednesday, 10 June 2026, 09:00 (shop time, Europe/Warsaw, UTC+02:00)." in system["content"]
    assert "help@citywheels.example" in system["content"]  # staff contact, read from the policies
    assert first["messages"][1] == {"role": "user", "content": "Hi"}
    assert {t["function"]["name"] for t in first["tools"]} == {"search_policies", "get_booking", "calculate_refund"}


def test_unknown_booking_does_not_crash_and_the_model_is_told_it_was_not_found():
    model = fakes.ScriptedModel(
        [
            fakes.tool_call("get_booking", {"booking_id": "CW-9999"}),
            fakes.text("I could not find booking CW-9999. Please check the number."),
        ]
    )
    reply = build_agent(model=model).ask("What is the status of CW-9999?")

    assert reply.tool_calls[0].result["found"] is False
    assert json.loads(model.requests[1]["messages"][-1]["content"])["found"] is False
    assert "could not find" in reply.answer


def test_broken_tool_calls_are_reported_back_and_the_loop_continues():
    model = fakes.ScriptedModel(
        [
            fakes.tool_calls(
                [
                    ("cancel_booking", {"booking_id": "CW-1001"}, "a"),
                    ("get_booking", '{"booking_id": ', "b"),
                ]
            ),
            fakes.text("Sorry, please contact the staff."),
        ]
    )
    reply = build_agent(model=model).ask("Cancel CW-1001")

    assert "Unknown tool" in reply.tool_calls[0].result["error"]
    assert "not valid JSON" in reply.tool_calls[1].result["error"]
    assert reply.tool_calls[1].arguments == '{"booking_id": '  # kept as text when it is not JSON
    assert reply.answer == "Sorry, please contact the staff."


def test_several_tool_calls_in_one_reply_are_all_answered_in_order():
    model = fakes.ScriptedModel(
        [
            fakes.tool_calls(
                [
                    ("get_booking", {"booking_id": "CW-1001"}, "first"),
                    ("calculate_refund", {"booking_id": "CW-1001"}, "second"),
                ]
            ),
            fakes.text("90 EUR."),
        ]
    )
    build_agent(model=model).ask("Refund for CW-1001?")

    tool_messages = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["first", "second"]


def test_agent_stops_when_the_model_keeps_calling_tools():
    model = fakes.ScriptedModel([fakes.tool_call("search_policies", {"query": "x"}) for _ in range(10)])
    agent = build_agent(model=model)
    agent.max_steps = 3

    reply = agent.ask("Loop forever")

    assert reply.hit_step_limit is True
    assert len(model.requests) == 3
    assert "contact the CityWheels staff" in reply.answer
    assert "help@citywheels.example" in reply.answer


def test_empty_answer_is_replaced_by_the_fallback():
    model = fakes.ScriptedModel([fakes.text("")])
    reply = build_agent(model=model).ask("Hello?")
    assert "contact the CityWheels staff" in reply.answer


def test_follow_up_question_sees_the_earlier_conversation():
    model = fakes.ScriptedModel(
        [
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001"}),
            fakes.text("You get 90 EUR back."),
            fakes.tool_call("calculate_refund", {"booking_id": "CW-1001", "cancel_at": "2026-06-12T09:00:00+02:00"}),
            fakes.text("On Friday you would get 70 EUR back."),
        ]
    )
    agent = build_agent(model=model)

    agent.ask("How much do I get back if I cancel CW-1001?")
    reply = agent.ask("And what if I cancel on Friday at 9:00 instead?")

    second_turn = model.requests[2]["messages"]
    assert {"role": "user", "content": "How much do I get back if I cancel CW-1001?"} in second_turn
    assert {"role": "assistant", "content": "You get 90 EUR back."} in second_turn
    assert reply.tool_calls[0].result["total_refund_eur"] == 70.0
    assert reply.answer == "On Friday you would get 70 EUR back."


def test_a_model_error_leaves_the_conversation_as_it_was():
    model = fakes.ScriptedModel([fakes.tool_call("get_booking", {"booking_id": "CW-1001"}), LLMError("rate limit")])
    agent = build_agent(model=model)
    before = list(agent.messages)

    with pytest.raises(LLMError):
        agent.ask("Status of CW-1001?")
    assert agent.messages == before
