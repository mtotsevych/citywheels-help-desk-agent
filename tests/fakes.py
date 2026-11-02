"""A fake model for testing the agent loop without network or API key."""

from __future__ import annotations

import copy
import json

from helpdesk.llm import ModelReply, ToolCall


class ScriptedModel:
    """Returns prepared replies in order and records every request it gets."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self.replies:
            raise AssertionError("The agent called the model more often than the test expected.")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> ModelReply:
    return tool_calls([(name, arguments, call_id)])


def tool_calls(calls: list[tuple[str, dict | str, str]]) -> ModelReply:
    parsed = [
        ToolCall(id=call_id, name=name, arguments=args if isinstance(args, str) else json.dumps(args))
        for name, args, call_id in calls
    ]
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in parsed
        ],
    }
    return ModelReply(content=None, tool_calls=parsed, message=message)


def text(answer: str) -> ModelReply:
    return ModelReply(content=answer, tool_calls=[], message={"role": "assistant", "content": answer})
