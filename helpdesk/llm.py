"""Talking to the language model.

The agent depends only on the small ChatModel interface below. That keeps it
testable with a fake model, and lets any OpenAI-compatible Chat Completions API
be used: the Gemini API (default), a local Ollama, OpenAI itself, and others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import openai


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model wrote it


@dataclass
class ModelReply:
    content: str | None
    tool_calls: list[ToolCall]
    # The assistant message to append to the conversation history. It is kept as
    # the provider sent it: Gemini 3 models attach "thought signatures" to tool
    # calls and reject the next request with HTTP 400 if they are missing.
    message: dict


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...


class LLMError(Exception):
    """The model could not be reached, or the provider rejected the request."""


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float | None = None,
        timeout: float = 60.0,
        rate_limit_retries: int = 3,
        rate_limit_wait: float = 10.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_wait = rate_limit_wait
        # The SDK itself retries short network errors and 429/5xx responses twice.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        request: dict = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = tools
        if self.temperature is not None:
            request["temperature"] = self.temperature

        response = self._create(request)
        if not response.choices:
            raise LLMError("The model returned an empty response.")
        message = response.choices[0].message

        raw = message.model_dump(exclude_unset=True)
        raw["role"] = "assistant"
        tool_calls = []
        for i, call in enumerate(message.tool_calls or []):
            function = getattr(call, "function", None)
            if function is None:  # not a function call (e.g. a "custom" tool): ignore it
                continue
            call_id = call.id or f"call_{i}"
            raw["tool_calls"][i]["id"] = call_id  # so the tool result can refer to it
            tool_calls.append(ToolCall(id=call_id, name=function.name, arguments=function.arguments or "{}"))
        return ModelReply(content=message.content, tool_calls=tool_calls, message=raw)

    def _create(self, request: dict):
        for attempt in range(self.rate_limit_retries + 1):
            try:
                return self.client.chat.completions.create(**request)
            except openai.RateLimitError as exc:
                # Per-minute limits pass after a short wait; a used-up daily quota does not.
                if "PerDay" in str(exc) or attempt == self.rate_limit_retries:
                    raise LLMError(f"Rate limit or quota exceeded for {self.model}: {_short(exc)}") from exc
                time.sleep(self.rate_limit_wait * (attempt + 1))
            except openai.AuthenticationError as exc:
                raise LLMError("The API key was rejected. Check LLM_API_KEY in your .env file.") from exc
            except openai.NotFoundError as exc:
                raise LLMError(
                    f"Model '{self.model}' or URL '{self.base_url}' not found. Check LLM_MODEL and LLM_BASE_URL."
                ) from exc
            except openai.APIConnectionError as exc:
                raise LLMError(f"Could not connect to {self.base_url}: {exc}") from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"The model API returned HTTP {exc.status_code}: {_short(exc)}") from exc
        raise AssertionError("unreachable")


def _short(exc: Exception, limit: int = 300) -> str:
    text = str(exc)
    return text if len(text) <= limit else text[:limit] + "..."
