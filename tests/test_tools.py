"""The dispatcher that runs the model's tool calls."""

import inspect
import json

from helpdesk.tools import TOOL_SCHEMAS


def test_every_schema_matches_a_tool_function(tools):
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        method = getattr(tools, function["name"])
        parameters = inspect.signature(method).parameters
        assert set(function["parameters"]["properties"]) == set(parameters)
        for name in function["parameters"]["required"]:
            assert parameters[name].default is inspect.Parameter.empty


def test_run_decodes_json_arguments(tools):
    result = tools.run("get_booking", json.dumps({"booking_id": "CW-1001"}))
    assert result["found"] is True


def test_run_accepts_already_decoded_arguments(tools):
    assert tools.run("calculate_refund", {"booking_id": "CW-1002"})["total_refund_eur"] == 340.0


def test_unknown_tool_is_reported_not_raised(tools):
    result = tools.run("delete_booking", '{"booking_id": "CW-1001"}')
    assert "Unknown tool" in result["error"]


def test_invalid_json_is_reported_not_raised(tools):
    assert "not valid JSON" in tools.run("get_booking", '{"booking_id": ')["error"]


def test_arguments_that_are_not_an_object_are_reported(tools):
    assert "JSON object" in tools.run("get_booking", '["CW-1001"]')["error"]


def test_wrong_argument_names_are_reported(tools):
    assert "Wrong arguments" in tools.run("get_booking", '{"id": "CW-1001"}')["error"]
    assert "Wrong arguments" in tools.run("get_booking", "{}")["error"]


def test_a_failing_tool_does_not_crash_the_agent(tools, monkeypatch):
    def broken(booking_id):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(tools._functions, "get_booking", broken)
    assert "disk on fire" in tools.run("get_booking", '{"booking_id": "CW-1001"}')["error"]
