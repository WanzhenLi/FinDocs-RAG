from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def patch_streamlit(monkeypatch) -> Callable[[Any], SimpleNamespace]:
    """Patch a module-level `st` attribute with a lightweight fake Streamlit shim."""

    def _patch(module: Any) -> SimpleNamespace:
        fake_st = SimpleNamespace(session_state={})
        monkeypatch.setattr(module, "st", fake_st)
        return fake_st

    return _patch
