from datetime import datetime

import pytest

from helpdesk.config import DEFAULT_DATA_DIR
from helpdesk.tools import HelpDeskTools

NOW = datetime.fromisoformat("2026-06-10T09:00:00+02:00")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Tests must not depend on the developer's .env or shell settings."""
    for name in (
        "CITYWHEELS_NOW",
        "CITYWHEELS_DATA_DIR",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tools() -> HelpDeskTools:
    return HelpDeskTools.from_data_dir(DEFAULT_DATA_DIR, NOW)
