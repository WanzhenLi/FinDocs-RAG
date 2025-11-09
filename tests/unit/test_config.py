import importlib
import os

import pytest
import dotenv

import config


@pytest.fixture(autouse=True)
def reload_config(monkeypatch):
    """Reload the config module after each test to pick up env changes."""
    yield
    importlib.reload(config)


def test_langchain_tracing_disabled_when_missing(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    importlib.reload(config)
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_langchain_tracing_respects_existing_setting(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "custom")
    importlib.reload(config)
    assert os.environ["LANGCHAIN_TRACING_V2"] == "custom"
