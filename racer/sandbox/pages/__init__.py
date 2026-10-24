"""Streamlit pages of the AI Sandbox. ``run_app`` is called by ``scripts/sandbox.py``."""
from __future__ import annotations

import streamlit as st

from racer.sandbox.pages import build, compare, draw, explorer, glossary, launch, overview, presets, workbench

# (key, module, title, material icon); the first page is the home page
PAGES = [
    ("overview", overview, "Overview", "dashboard"),
    ("explorer", explorer, "Model Explorer", "psychology"),
    ("compare", compare, "Compare", "compare_arrows"),
    ("workbench", workbench, "Workbench", "monitoring"),
    ("build", build, "Build & Queue", "build"),
    ("presets", presets, "Presets", "tune"),
    ("draw", draw, "Draw a Track", "draw"),
    ("launch", launch, "Launch", "sports_score"),
    ("glossary", glossary, "Glossary", "menu_book"),
]
_pages: dict[str, st.Page] = {}


def goto(key: str) -> None:
    """Open another page on the next run (safe inside button callbacks); state to hand over
    goes in ``st.session_state`` first."""
    st.session_state["_goto"] = key


def run_app() -> None:
    st.set_page_config(page_title="AI Sandbox · RL Racing Line", page_icon=":material/sports_motorsports:",
                       layout="wide")
    _pages.clear()
    for i, (key, module, title, icon) in enumerate(PAGES):
        _pages[key] = st.Page(module.render, title=title, icon=f":material/{icon}:", url_path=key, default=i == 0)
    nav = st.navigation(list(_pages.values()))
    target = st.session_state.pop("_goto", None)
    if target in _pages and _pages[target] is not nav:
        st.switch_page(_pages[target])
    nav.run()
