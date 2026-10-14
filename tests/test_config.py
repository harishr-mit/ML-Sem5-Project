"""Config presets load, and typos are rejected."""
import dataclasses

import pytest

from racer.config import CONFIG_DIR, ConfigError, deep_merge, from_dict, load_preset
from racer.physics.car import PhysicsParams

# Every preset in configs/<kind>/ must load (extends chains, strict overrides).
PRESETS = sorted((p.parent.name, p.stem) for p in CONFIG_DIR.glob("*/*.yaml"))


@pytest.mark.parametrize("kind,name", PRESETS)
def test_presets_load(kind, name):
    cfg = load_preset(kind, name)
    assert isinstance(cfg, dict) and cfg


def test_physics_dataclass_strict():
    cfg = load_preset("physics", "default")
    PhysicsParams.from_config(cfg)
    with pytest.raises(ConfigError):
        PhysicsParams.from_config({**cfg, "mass_kgg": 1.0})
    missing = dict(cfg)
    del missing["mass_kg"]
    with pytest.raises(ConfigError):
        PhysicsParams.from_config(missing)


def test_override_typo_raises():
    with pytest.raises(ConfigError):
        load_preset("physics", "default", overrides={"masss_kg": 900})
    cfg = load_preset("physics", "default", overrides={"mass_kg": 900, "surfaces": {"road": {"mu_static": 1.0}}})
    assert cfg["mass_kg"] == 900
    assert cfg["surfaces"]["road"]["mu_static"] == 1.0
    assert cfg["surfaces"]["road"]["mu_kinetic"] == 0.90  # untouched sibling kept


def test_from_dict_fills_only_declared_defaults():
    """A field added later with a default (e.g. env.episode.track_limits) must not break
    snapshots saved before it existed; fields without a default stay required."""
    @dataclasses.dataclass(frozen=True)
    class P:
        a: float
        added_later: bool = False

    assert from_dict(P, {"a": 1.0}) == P(1.0, False)
    assert from_dict(P, {"a": 1.0, "added_later": True}).added_later is True
    with pytest.raises(ConfigError):
        from_dict(P, {"added_later": True})


def test_deep_merge_non_strict_adds_keys():
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}, strict=False) == {"a": {"b": 1, "c": 2}}
