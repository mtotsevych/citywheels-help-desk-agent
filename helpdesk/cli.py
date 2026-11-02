"""Command line: ask one question, or chat with follow-up questions."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .agent import Agent, ToolCallRecord
from .config import PROJECT_ROOT, ConfigError
from .factory import build_agent
from .llm import LLMError


def _print_tool_call(record: ToolCallRecord) -> None:
    args = json.dumps(record.arguments, ensure_ascii=False)
    result = json.dumps(record.result, ensure_ascii=False)
    if len(result) > 300:
        result = result[:300] + "..."
    print(f"  [tool] {record.name}({args}) -> {result}", file=sys.stderr)


def _answer(agent: Agent, question: str, prefix: str = "") -> bool:
    try:
        reply = agent.ask(question)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False
    print(prefix + reply.answer)
    return True


def _chat(agent: Agent) -> int:
    print("CityWheels help desk. Ask a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            _answer(agent, question, prefix="Agent: ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m helpdesk",
        description="Ask the CityWheels help desk agent. Without a question it starts a chat.",
    )
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the tool calls")
    parser.add_argument("--model", help="model name, overrides LLM_MODEL")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        agent = build_agent(model_name=args.model, on_tool_call=_print_tool_call if args.verbose else None)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.question:
        return 0 if _answer(agent, " ".join(args.question)) else 1
    return _chat(agent)
