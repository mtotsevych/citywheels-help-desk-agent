"""The agent loop.

1. Send the conversation and the tool schemas to the model.
2. If the model asks for tools, run them and add their results to the conversation.
3. Repeat until the model answers with text, or the step limit is reached.

The conversation is kept between ask() calls, so follow-up questions work.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import ChatModel
from .tools import HelpDeskTools

FALLBACK_ANSWER = "Sorry, I could not answer this. Please contact the CityWheels staff."


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict | str
    result: dict


@dataclass
class AgentReply:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_calls: int = 0
    hit_step_limit: bool = False


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: HelpDeskTools,
        system_prompt: str,
        max_steps: int = 6,
        on_tool_call: Callable[[ToolCallRecord], None] | None = None,
        fallback_answer: str = FALLBACK_ANSWER,
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.on_tool_call = on_tool_call
        self.fallback_answer = fallback_answer
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def ask(self, question: str) -> AgentReply:
        history_length = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        try:
            return self._loop()
        except Exception:
            # Forget the half-finished turn, so the next question starts from a valid history.
            del self.messages[history_length:]
            raise

    def _loop(self) -> AgentReply:
        records: list[ToolCallRecord] = []
        for step in range(1, self.max_steps + 1):
            reply = self.model.complete(self.messages, self.tools.schemas)
            self.messages.append(reply.message)

            if not reply.tool_calls:
                answer = (reply.content or "").strip()
                if not answer:
                    answer = self.fallback_answer
                    self.messages[-1] = {"role": "assistant", "content": answer}
                return AgentReply(answer=answer, tool_calls=records, model_calls=step)

            for call in reply.tool_calls:
                result = self.tools.run(call.name, call.arguments)
                record = ToolCallRecord(call.name, _decode(call.arguments), result)
                records.append(record)
                if self.on_tool_call:
                    self.on_tool_call(record)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # The model kept calling tools: stop and give a safe answer instead of looping forever.
        self.messages.append({"role": "assistant", "content": self.fallback_answer})
        return AgentReply(
            answer=self.fallback_answer, tool_calls=records, model_calls=self.max_steps, hit_step_limit=True
        )


def _decode(arguments: str) -> dict | str:
    try:
        decoded = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return arguments
    return decoded if isinstance(decoded, dict) else arguments
