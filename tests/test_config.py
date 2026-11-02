from datetime import datetime

import pytest

from helpdesk.config import DEFAULT_MODEL, ConfigError, LLMSettings, shop_now


def test_current_time_defaults_to_the_date_from_the_data_readme():
    assert shop_now() == datetime.fromisoformat("2026-06-10T09:00:00+02:00")


def test_current_time_can_be_changed(monkeypatch):
    monkeypatch.setenv("CITYWHEELS_NOW", "2026-06-12T09:00")
    assert shop_now().isoformat() == "2026-06-12T09:00:00+02:00"


def test_invalid_current_time_is_a_config_error(monkeypatch):
    monkeypatch.setenv("CITYWHEELS_NOW", "tomorrow")
    with pytest.raises(ConfigError):
        shop_now()


def test_missing_api_key_is_a_config_error():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        LLMSettings.from_env()


def test_llm_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    settings = LLMSettings.from_env()
    assert settings.api_key == "secret"
    assert settings.model == DEFAULT_MODEL
    assert settings.temperature == 0.0
    assert LLMSettings.from_env(model_override="other-model").model == "other-model"
