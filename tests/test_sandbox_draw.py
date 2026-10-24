"""Draw a Track in the sandbox (D27, T119): canvas JSON → track → tracks/user/, and the page."""
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from racer.sandbox.draw import canvas_points, save_user_track, to_metres, track_name
from racer.track.io import load_track_spec
from racer.track.track import Track


def circle_path(cx=360, cy=240, r=200, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = cx + r * np.cos(a), cy + r * np.sin(a)
    path = [["M", xs[0], ys[0]]] + [["Q", xs[i - 1], ys[i - 1], xs[i], ys[i]] for i in range(1, n)]
    return {"objects": [{"type": "rect"}, {"type": "path", "path": path + [["L", xs[-1], ys[-1]]]}]}


@pytest.fixture(autouse=True)
def _restore_main_module():
    main = sys.modules.get("__main__")
    yield
    if main is not None:
        sys.modules["__main__"] = main


def test_canvas_points_reads_the_last_stroke_or_polygon():
    pts = canvas_points(circle_path())
    assert pts.shape == (61, 2) and np.allclose(pts[0], [560, 240])
    poly = {"objects": [{"type": "polygon", "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 0}]}]}
    assert canvas_points(poly).tolist() == [[1, 2], [3, 4], [5, 0]]
    assert canvas_points({"objects": [{"type": "Polygon", "points": poly["objects"][0]["points"]}]}).shape == (3, 2)
    assert canvas_points(None) is None and canvas_points({"objects": [{"type": "rect"}]}) is None


def test_to_metres_flips_y_so_drawing_direction_is_kept():
    m = to_metres(np.array([[0.0, 0.0], [10.0, 480.0]]), 480, 2.0)
    assert m.tolist() == [[0.0, 960.0], [20.0, 0.0]]


def test_track_name_is_file_safe():
    assert track_name("  Faculty Loop #1! ") == "faculty_loop_1"


def test_saved_track_loads_validates_and_is_never_overwritten(tmp_path):
    from racer.config import load_preset
    from racer.track.sketch import sketch_track
    track, problems = sketch_track(to_metres(canvas_points(circle_path()), 480, 2.0), "drawn",
                                   load_preset("demo", "editor")["sketch"])
    assert problems == []
    (tmp_path / "showcase").mkdir()
    (tmp_path / "showcase" / "taken.json").write_text("{}")
    path = save_user_track(track, "My Loop", tmp_path, tmp_path / "png")
    assert path == tmp_path / "user" / "my_loop.json" and (tmp_path / "png" / "my_loop.png").is_file()
    spec = load_track_spec(path)
    assert spec.name == "my_loop" and spec.meta["drawn"]["source"] == "sandbox"
    assert Track(spec, build_field=False).validate() == []
    for name in ("my loop", "taken", "  "):
        with pytest.raises(ValueError):
            save_user_track(track, name, tmp_path, tmp_path / "png")


def _run_page(key: str) -> None:            # executed by AppTest as a script
    import importlib
    importlib.import_module(f"racer.sandbox.pages.{key}").render()


def test_draw_page_builds_saves_and_hands_the_track_to_launch(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.pages import draw
    monkeypatch.setattr(draw, "_canvas", lambda **kw: SimpleNamespace(json_data=circle_path()))
    monkeypatch.setattr(draw, "save_user_track", partial(save_user_track, tracks_dir=tmp_path,
                                                         previews_dir=tmp_path / "png"))
    monkeypatch.setattr(draw.track_names, "clear", lambda: None)
    at = AppTest.from_function(_run_page, args=("draw",), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Valid track" in s.value for s in at.success)
    at.text_input(key="dr_name").input("Page Test").run()
    at.button(key="dr_save").click().run()
    assert (tmp_path / "user" / "page_test.json").is_file()
    assert any("Saved as tracks/user/page_test.json" in s.value for s in at.success)
    at.button(key="dr_race").click().run()
    assert at.session_state["launch_track"] == "page_test" and at.session_state["_goto"] == "launch"


def test_draw_page_without_a_drawing_explains_what_to_do(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.pages import draw
    monkeypatch.setattr(draw, "_canvas", lambda **kw: SimpleNamespace(json_data=None))
    at = AppTest.from_function(_run_page, args=("draw",), default_timeout=120)
    at.run()
    assert not at.exception and "Draw a loop" in at.info[0].value


def test_launch_takes_the_handed_over_map(monkeypatch):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=("launch",), default_timeout=120)
    at.session_state["launch_track"] = "fast_flow"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="ln_track").value == "fast_flow"
