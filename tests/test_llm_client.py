"""The OpenAI-compatible client against a local HTTP server that imitates the API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from helpdesk.llm import LLMError, OpenAICompatibleModel


class FakeAPI:
    """Answers each POST with the next prepared (status, body) and records the request bodies."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                api.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, body = api.responses.pop(0)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1beta/openai/"


@pytest.fixture
def make_model():
    servers = []

    def make(responses, **kwargs):
        api = FakeAPI(responses)
        servers.append(api.server)
        model = OpenAICompatibleModel(model="test-model", api_key="test-key", base_url=api.url, **kwargs)
        model.client = model.client.with_options(max_retries=0)  # keep tests fast
        return model, api

    yield make
    for server in servers:
        server.shutdown()


def completion(message: dict) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


GEMINI_TOOL_CALL = completion(
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_booking", "arguments": '{"booking_id": "CW-1001"}'},
                "extra_content": {"google": {"thought_signature": "c2lnbmF0dXJl"}},
            }
        ],
    }
)


def test_tool_calls_are_parsed(make_model):
    model, api = make_model([(200, GEMINI_TOOL_CALL)])
    reply = model.complete([{"role": "user", "content": "Status of CW-1001?"}], tools=[])

    assert reply.content is None
    assert [(c.id, c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [
        ("call_1", "get_booking", {"booking_id": "CW-1001"})
    ]
    assert api.requests[0]["model"] == "test-model"


def test_gemini_thought_signature_is_sent_back_with_the_tool_call(make_model):
    # Gemini 3 rejects a request whose earlier tool calls lost their thought signature.
    model, api = make_model([(200, GEMINI_TOOL_CALL), (200, completion({"role": "assistant", "content": "Done."}))])
    messages = [{"role": "user", "content": "Status of CW-1001?"}]
    first = model.complete(messages, tools=[])
    messages += [first.message, {"role": "tool", "tool_call_id": "call_1", "content": "{}"}]

    second = model.complete(messages, tools=[])

    replayed = api.requests[1]["messages"][1]["tool_calls"][0]
    assert replayed["extra_content"] == {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    assert second.content == "Done."


def test_missing_tool_call_id_gets_one_that_the_tool_result_can_use(make_model):
    body = completion(
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "get_booking", "arguments": "{}"}}],
        }
    )
    model, _ = make_model([(200, body)])
    reply = model.complete([{"role": "user", "content": "x"}], tools=[])
    assert reply.tool_calls[0].id == "call_0"
    assert reply.message["tool_calls"][0]["id"] == "call_0"


def test_rejected_api_key_gives_a_clear_error(make_model):
    model, _ = make_model([(401, {"error": {"message": "API key not valid"}})])
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        model.complete([{"role": "user", "content": "x"}], tools=[])


def test_per_minute_rate_limit_is_retried(make_model):
    limited = (429, {"error": {"message": "Too many requests per minute"}})
    model, api = make_model([limited, (200, completion({"role": "assistant", "content": "OK"}))], rate_limit_wait=0)
    assert model.complete([{"role": "user", "content": "x"}], tools=[]).content == "OK"
    assert len(api.requests) == 2


def test_used_up_daily_quota_is_not_retried(make_model):
    daily = (429, {"error": {"message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
    model, api = make_model([daily], rate_limit_wait=0)
    with pytest.raises(LLMError, match="quota"):
        model.complete([{"role": "user", "content": "x"}], tools=[])
    assert len(api.requests) == 1
