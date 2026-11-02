"""Wiring: builds an agent from the data files and the environment settings."""

from __future__ import annotations

from collections.abc import Callable

from .agent import FALLBACK_ANSWER, Agent, ToolCallRecord
from .config import LLMSettings, data_dir, shop_now
from .llm import ChatModel, OpenAICompatibleModel
from .prompts import build_system_prompt, staff_contact
from .tools import HelpDeskTools


def build_agent(
    model: ChatModel | None = None,
    model_name: str | None = None,
    on_tool_call: Callable[[ToolCallRecord], None] | None = None,
) -> Agent:
    """An agent with the real tools and instructions.

    Pass `model` to use another model (tests pass a fake one); otherwise the
    OpenAI-compatible model configured by the LLM_* environment variables is used.
    """
    tools = HelpDeskTools.from_data_dir(data_dir(), shop_now())
    if model is None:
        settings = LLMSettings.from_env(model_override=model_name)
        model = OpenAICompatibleModel(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=settings.temperature,
        )
    contact = staff_contact(tools.policies)
    return Agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(tools.now, contact),
        on_tool_call=on_tool_call,
        fallback_answer=f"{FALLBACK_ANSWER}\n{contact}",
    )
