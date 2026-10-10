"""Track sets (D7): legacy track lists, generated pools, determinism, held-out protection."""
import numpy as np
import pytest

from racer.config import ConfigError, load_preset
from racer.env import trackset as TS
from racer.env.racing_env import RacingEnv, make_env_config
from racer.track.generator import generate
from racer.track.io import TRACKS_DIR, load_track_spec


def test_legacy_track_list_still_works():
    env = RacingEnv(make_env_config(["training_ground", "technical"]))
    seen = set()
    for s in range(12):
        env.reset(seed=s)
        seen.add(env.track.key)
    assert seen == {"training_ground", "technical"}
    assert env.track_names == ["training_ground", "technical"]


def test_trackset_expands_folders_into_names():
    cfg = make_env_config(trackset="final_mix")
    fixed = [f["track"] for f in cfg["tracks"]["fixed"]]
    assert {"training_ground", "technical", "fast_flow"} <= set(fixed)
    assert cfg["presets"]["trackset"] == "final_mix"
    assert cfg["tracks"]["generated"]["pool_size"] >= 1


def test_same_seed_same_track_sequence():
    def sequence(seed: int) -> list[str]:
        env = RacingEnv(make_env_config(trackset="final_mix"))
        out = [env.reset(seed=seed)[1]["track"]]
        out += [env.reset()[1]["track"] for _ in range(40)]
        return out

    a, b = sequence(3), sequence(3)
    assert a == b
    assert any(n.startswith("gen_") for n in a) and any(not n.startswith("gen_") for n in a)


def test_generated_pool_rotates_and_stays_small():
    env = RacingEnv(make_env_config(trackset="gen_wide"))
    env.reset(seed=0)
    names = set()
    for _ in range(120):
        names.add(env.reset()[1]["track"])
        assert len(env.sampler.pool) == env.sampler.gen["pool_size"]
    assert len([n for n in names if n.startswith("gen_")]) > env.sampler.gen["pool_size"]   # it rotated
    widths = [t.spec.width_m for t in env.sampler.pool]
    assert all(14.0 <= w <= 18.0 for w in widths)


def test_heldout_tracks_regenerate_identically():
    """Generator changes must never change the held-out set (scalar width keeps the RNG sequence)."""
    cfg = load_preset("track", "default")
    for path in sorted((TRACKS_DIR / "heldout").glob("*.json"))[:4]:
        spec = load_track_spec(path)
        g = spec.meta["generator"]
        track, attempt = generate(g["seed"], g["stream"], name=spec.name, track_cfg=cfg, build_field=False)
        assert attempt == g["attempt"]
        assert np.allclose(track.spec.control_points, spec.control_points, atol=1e-9)
        assert track.spec.width_m == spec.width_m and track.spec.direction == spec.direction


def test_heldout_tracks_are_refused(monkeypatch):
    monkeypatch.setattr(TS, "load_preset", lambda kind, name: {"fixed": [{"folder": "heldout", "weight": 1.0}]})
    with pytest.raises(ConfigError, match="held-out"):
        TS.resolve_trackset("bad")


def test_test_maps_are_refused(monkeypatch):
    monkeypatch.setattr(TS, "load_preset", lambda kind, name: {"fixed": [{"track": "alpine", "weight": 1.0}]})
    with pytest.raises(ConfigError, match="test track"):
        TS.resolve_trackset("bad")


def test_map_library_split():
    """D19: 7 training maps in maps7, 2 test maps, every hand-designed map valid."""
    fixed = [f["track"] for f in make_env_config(trackset="maps7")["tracks"]["fixed"]]
    assert len(fixed) == 7 and not any(TS.is_heldout(n) for n in fixed)
    test_maps = TS.folder_tracks("test")
    assert len(test_maps) == 2 and all(TS.is_heldout(n) for n in test_maps)
    from racer.track.track import Track
    for name in fixed + test_maps:
        assert Track.load(name, load_preset("track", "default")).validate() == [], name


def test_track_cache_is_bounded():
    base = load_preset("track", "default")
    for i in range(TS._CACHE_MAX + 5):
        cfg = dict(base, _variant=i)    # distinct config → distinct cache key
        TS.get_track("training_ground", cfg)
    assert len(TS._TRACK_CACHE) <= TS._CACHE_MAX
