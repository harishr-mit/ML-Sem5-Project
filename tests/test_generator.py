"""Generator: valid, deterministic, and held-out seeds never reachable from the training stream."""
import numpy as np

from racer.config import load_preset
from racer.track.generator import TRAIN_STREAM, generate
from racer.track.io import TRACKS_DIR, load_track_spec


def test_generated_tracks_are_valid_and_deterministic():
    a, _ = generate(3, build_field=False)
    b, _ = generate(3, build_field=False)
    assert a.validate() == []
    assert np.allclose(a.spec.control_points, b.spec.control_points)


def test_heldout_stream_differs_from_training():
    cfg = load_preset("track", "default")
    stream = cfg["heldout"]["stream"]
    assert stream != TRAIN_STREAM
    for k in range(cfg["heldout"]["count"]):
        ho = load_track_spec(TRACKS_DIR / "heldout" / f"heldout_{k:02d}.json")
        assert ho.meta["generator"]["stream"] == stream
        regen, _ = generate(k, stream, build_field=False)
        assert np.allclose(regen.spec.control_points, ho.control_points, atol=1e-3)
        train, _ = generate(k, TRAIN_STREAM, build_field=False)
        assert not np.allclose(np.asarray(train.spec.control_points).shape, 0) and (
            np.shape(train.spec.control_points) != np.shape(ho.control_points)
            or not np.allclose(train.spec.control_points, ho.control_points))
