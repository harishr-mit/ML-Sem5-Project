"""Draw a Track (D27, T119): draw a closed loop, see it as a track, save it, race on it."""
from __future__ import annotations

import streamlit as st

from racer.config import load_preset
from racer.sandbox import diagrams
from racer.sandbox.draw import canvas_points, save_user_track, to_metres, track_name
from racer.sandbox.pages._ui import track_names
from racer.track.sketch import sketch_track


def _canvas(**kw):
    from streamlit_drawable_canvas import st_canvas
    return st_canvas(**kw)


def _save(track, name: str) -> None:
    try:
        path = save_user_track(track, name)
        track_names.clear()
        st.session_state["dr_saved"] = path.stem
        st.session_state["dr_msg"] = ("success", f"Saved as tracks/user/{path.name}.")
    except (ValueError, OSError) as e:
        st.session_state["dr_msg"] = ("error", f"Could not save: {e}")


def _race(name: str) -> None:
    from racer.sandbox.pages import goto
    st.session_state["launch_track"] = name
    goto("launch")


def render() -> None:
    cfg = load_preset("demo", "editor")
    gen = load_preset("track", "default")["generator"]
    cv = cfg["canvas"]
    st.title("Draw a Track")
    st.caption("Draw one closed loop: where you start is the start/finish line, and the cars drive the way you "
               "drew it. The drawing is smoothed and resized to a valid track (800–4000 m, no corner tighter "
               "than 15 m). Crossings and sections too close together can't be fixed: they are marked in red. "
               "Saved tracks go to tracks/user/ (never trained on by the current track sets) and appear in Launch.")
    c1, c2 = st.columns([3, 1])
    mode = c2.radio("Draw by", ["Freehand", "Clicking points"], key="dr_mode",
                    help="Clicking points: click the corners, then click the first point again to close the loop.")
    width = c2.slider("Road width (m)", 8.0, 16.0, float(gen["width_m"]), 0.5, key="dr_width")
    c2.caption("The canvas toolbar (above it) has undo, redo and clear. The last stroke is the one used.")
    with c1:
        res = _canvas(stroke_width=int(cv["stroke_px"]), stroke_color="#333333", background_color="#cfe3c0",
                      height=int(cv["height_px"]), width=int(cv["width_px"]),
                      drawing_mode="freedraw" if mode == "Freehand" else "polygon", fill_color="rgba(0,0,0,0)",
                      key=f"dr_canvas_{mode}")
    data = getattr(res, "json_data", None)
    px = canvas_points(data)
    if px is None:
        kinds = sorted({str(o.get("type")) for o in (data or {}).get("objects", [])})
        st.info("Draw a loop on the canvas to see it as a track."
                + (f" (The canvas holds {', '.join(kinds)}: use freehand or clicked points.)" if kinds else ""))
        return
    pts = to_metres(px, float(cv["height_px"]), float(cv["width_m"]) / float(cv["width_px"]))
    track, problems = sketch_track(pts, "drawn", cfg["sketch"], width_m=width)
    if track is None:
        st.warning(problems[0])
        return
    if problems:
        st.error("This drawing can't be a track yet: " + "; ".join(problems))
    else:
        st.success(f"Valid track: {track.length:.0f} m, {len(track.spec.control_points)} control points.")
    st.image(diagrams.png(diagrams.track_figure(track, problems)), width=560)
    if problems:
        return
    n1, n2, n3 = st.columns([3, 1, 1.4])
    name = n1.text_input("Name", key="dr_name", placeholder="e.g. faculty_loop")
    n2.button("Save", type="primary", key="dr_save", disabled=not track_name(name or ""),
              on_click=_save, args=(track, name or ""))
    saved = st.session_state.get("dr_saved")
    if saved:
        n3.button(f"Race on '{saved}'", key="dr_race", on_click=_race, args=(saved,),
                  help="Opens Launch with this map chosen; pick the models there.")
    msg = st.session_state.pop("dr_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
