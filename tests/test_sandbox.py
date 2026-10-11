"""AI Sandbox (T66–T72): data, explanations, diagrams, builder, queue, launcher, and the
Streamlit pages rendered headless with ``streamlit.testing`` (no browser)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_KEYS = ["overview", "explorer", "compare", "build", "launch", "glossary"]


def _run_page(key: str) -> None:            # executed by AppTest as a script
    import importlib
    importlib.import_module(f"racer.sandbox.pages.{key}").render()


# ---------------------------------------------------------------- launcher (T66)

def test_server_settings_are_local_and_private():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        sandbox = importlib.import_module("sandbox")
    finally:
        sys.path.pop(0)
    s = sandbox.server_settings(8501)
    opts = dict(zip(s[::2], s[1::2]))
    assert opts["--server.address"] == "localhost"
    assert opts["--browser.gatherUsageStats"] == "false"
    assert opts["--server.headless"] == "true"          # no first-run e-mail prompt, no ~/.streamlit


def test_every_page_is_routed():
    from racer.sandbox import pages
    assert [k for k, *_ in pages.PAGES] == PAGE_KEYS


def _run_app() -> None:                     # executed by AppTest as a script
    from racer.sandbox.pages import run_app
    run_app()


def test_app_with_navigation_renders_home():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and at.title[0].value == "Overview"


# ---------------------------------------------------------------- pages (headless)

@pytest.mark.parametrize("key", PAGE_KEYS)
def test_page_renders(key):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=(key,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
