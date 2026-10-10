"""Config presets load, and typos are rejected."""
import pytest

from racer.config import ConfigError, deep_merge, load_preset
from racer.physics.car import PhysicsParams

PRESETS = [
    ("physics", "default"),
    ("env", "rays5"),
    ("reward", "progress"),
    ("train", "ppo_default"),
    ("track", "default"),
]


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


def test_deep_merge_non_strict_adds_keys():
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}, strict=False) == {"a": {"b": 1, "c": 2}}
