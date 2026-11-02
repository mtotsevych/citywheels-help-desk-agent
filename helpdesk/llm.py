"""Talking to the language model.

The agent depends only on the small ChatModel interface below. That keeps it
testable with a fake model, and lets any OpenAI-compatible Chat Completions API
be used: the Gemini API (default), a local Ollama, OpenAI itself, and others.
"""

from __future__ import annotations

import json
import re
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


@dataclass
class ModelStats:
    calls: int = 0  # requests the agent made
    retries: int = 0  # extra attempts after a rate limit, server error or network error
    seconds: float = 0.0  # total time spent waiting for the model, retries included


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        retry_wait: float = 5.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.stats = ModelStats()
        # Retries are done in _create(), not by the SDK, so they can be counted.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        request: dict = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = tools
        if self.temperature is not None:
            request["temperature"] = self.temperature

        started = time.monotonic()
        try:
            response = self._create(request)
        finally:
            self.stats.calls += 1
            self.stats.seconds += time.monotonic() - started
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
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**request)
            except openai.RateLimitError as exc:
                # A per-minute limit passes after a short wait; a used-up daily quota does not.
                if "PerDay" in error_text(exc):
                    raise LLMError(f"Daily quota used up for {self.model}: {_short(exc)}") from exc
                error: Exception = exc
            except (openai.InternalServerError, openai.APIConnectionError) as exc:
                error = exc  # e.g. HTTP 503 "model overloaded", or a dropped connection
            except openai.AuthenticationError as exc:
                raise LLMError("The API key was rejected. Check LLM_API_KEY in your .env file.") from exc
            except openai.NotFoundError as exc:
                raise LLMError(
                    f"Model '{self.model}' or URL '{self.base_url}' not found. Check LLM_MODEL and LLM_BASE_URL."
                ) from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"The model API returned HTTP {exc.status_code}: {_short(exc)}") from exc

            if attempt == self.max_retries:
                raise LLMError(f"The model API still failed after {attempt + 1} attempts: {_short(error)}") from error
            self.stats.retries += 1
            time.sleep(retry_delay(error) or self.retry_wait * 2**attempt)
        raise AssertionError("unreachable")


def retry_delay(error: Exception) -> float | None:
    """The wait the server asks for: a Retry-After header, or Gemini's "retryDelay": "23s" in the body."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return min(float(header), 60.0)
        except ValueError:
            pass
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", error_text(error))
    return min(float(match.group(1)), 60.0) if match else None


def error_text(error: Exception) -> str:
    """The error message plus the response body.

    Gemini wraps its errors in a list ([{"error": {...}}]). The SDK then puts only
    "Error code: 429" into the message, and the details stay in `.body`.
    """
    body = getattr(error, "body", None)
    if body is None:
        return str(error)
    return f"{error} {json.dumps(body, ensure_ascii=False)}"


def _short(exc: Exception, limit: int = 300) -> str:
    text = error_text(exc)
    return text if len(text) <= limit else text[:limit] + "..."
