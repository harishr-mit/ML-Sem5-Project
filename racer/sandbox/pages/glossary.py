"""Glossary page: every parameter, what it is and what it affects (T67)."""
from __future__ import annotations

import streamlit as st

from racer.sandbox import explain
from racer.sandbox.pages._ui import md_table


def _match(q: str, *texts) -> bool:
    return not q or any(q in str(t).lower() for t in texts if t)


def render() -> None:
    from racer.agents.algorithms import ALGORITHMS
    from racer.env.actions import LAYOUTS
    from racer.env.features import describe_features

    st.title("Glossary")
    st.caption("What every setting means and what changing it does. The same notes appear next to each value in "
               "the Model Explorer and in Compare. Typical ranges are rules of thumb for this project.")
    q = st.text_input("Search", placeholder="e.g. grip, gamma, rays, crash", key="glossary_q").strip().lower()

    feats = [f for f in describe_features() if _match(q, f["feature"], f["description"])]
    if feats:
        with st.expander(f"Inputs the car can sense ({len(feats)} features)", expanded=bool(q)):
            st.markdown(md_table(["Feature", "What it senses", "Parameters (default)"],
                                 [(f"`{f['feature']}`", f["description"],
                                   ", ".join(f"{k} = {'required' if v is None else v}" for k, v in f["params"].items())
                                   or "—") for f in feats]))
    lays = [c for c in LAYOUTS.values() if _match(q, c.name, c.description)]
    if lays:
        with st.expander(f"Output layouts ({len(lays)})", expanded=bool(q)):
            st.markdown(md_table(["Layout", "Outputs", "Meaning"],
                                 [(f"`{c.name}`", ", ".join(c.labels), c.description) for c in lays]))
    algs = [a for a in ALGORITHMS.values() if _match(q, a.name, a.description)]
    if algs:
        with st.expander(f"Learning algorithms ({len(algs)})", expanded=bool(q)):
            st.markdown(md_table(["Algorithm", "How it learns"], [(f"`{a.name}`", a.description) for a in algs]))

    g = explain.glossary()
    shown = 0
    for group, title, intro in explain.groups():
        entries = [(k, e) for k, e in g.items() if e["group"] == group
                   and _match(q, k, e.get("short"), e.get("affects"), e.get("related"))]
        if not entries:
            continue
        shown += len(entries)
        with st.expander(f"{title} ({len(entries)})", expanded=bool(q)):
            if intro:
                st.caption(intro)
            st.markdown(md_table(["Setting", "What it is", "What it affects", "Typical", "See"],
                                 [(f"`{k}`", e.get("short", ""), e.get("affects", ""), e.get("range", ""),
                                   e.get("related", "")) for k, e in entries]))
    if q and not (shown or feats or lays or algs):
        st.info("Nothing matches. Try a shorter word.")
