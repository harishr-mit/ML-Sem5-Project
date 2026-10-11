"""Streamlit pages of the AI Sandbox. ``run_app`` is called by ``scripts/sandbox.py``."""
from __future__ import annotations

import streamlit as st

from racer.sandbox.pages import build, compare, explorer, glossary, launch, overview

# (key, module, title, material icon); the first page is the home page
PAGES = [
    ("overview", overview, "Overview", "dashboard"),
    ("explorer", explorer, "Model Explorer", "psychology"),
    ("compare", compare, "Compare", "compare_arrows"),
    ("build", build, "Build & Queue", "build"),
    ("launch", launch, "Launch", "sports_score"),
    ("glossary", glossary, "Glossary", "menu_book"),
]
_pages: dict[str, st.Page] = {}


def goto(key: str) -> None:
    """Switch to another page (state to hand over goes in ``st.session_state`` first)."""
    st.switch_page(_pages[key])


def run_app() -> None:
    st.set_page_config(page_title="AI Sandbox · RL Racing Line", page_icon=":material/sports_motorsports:",
                       layout="wide")
    _pages.clear()
    for i, (key, module, title, icon) in enumerate(PAGES):
        _pages[key] = st.Page(module.render, title=title, icon=f":material/{icon}:", url_path=key, default=i == 0)
    st.navigation(list(_pages.values())).run()
