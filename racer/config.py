"""YAML preset loading for ``configs/<kind>/<name>.yaml``.

Presets are plain dicts. A preset may start with ``extends: <other preset of the same
kind>`` to inherit from it. Overrides are deep-merged strictly: an override key that the
base doesn't have is an error, so a typo can't silently be ignored. Dataclasses built with
:func:`from_dict` reject unknown *and* missing keys, so every tunable number has to be
written in a config file (no hidden defaults in code).
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

T = TypeVar("T")


class ConfigError(ValueError):
    """A config file or override is malformed."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file that must contain a mapping."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def preset_path(kind: str, name: str) -> Path:
    """Path of ``configs/<kind>/<name>.yaml``."""
    return CONFIG_DIR / kind / f"{name}.yaml"


def load_preset(kind: str, name: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a preset, resolve ``extends`` chains and apply strict overrides."""
    seen: list[str] = []
    data = _load_with_extends(kind, name, seen)
    if overrides:
        data = deep_merge(data, overrides, strict=True)
    return data


def _load_with_extends(kind: str, name: str, seen: list[str]) -> dict[str, Any]:
    if name in seen:
        raise ConfigError(f"circular 'extends' in {kind}: {' -> '.join(seen + [name])}")
    seen.append(name)
    path = preset_path(kind, name)
    if not path.is_file():
        raise ConfigError(f"no such preset: {path}")
    data = load_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(_load_with_extends(kind, parent, seen), data, strict=True)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any], strict: bool = True) -> dict[str, Any]:
    """Return ``base`` updated with ``override``; nested mappings are merged recursively.

    With ``strict``, every key in ``override`` must already exist in ``base``.
    """
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key not in out:
            if strict:
                raise ConfigError(f"unknown config key {key!r} (known: {sorted(out)})")
            out[key] = copy.deepcopy(value)
        elif isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value, strict=strict)
        else:
            out[key] = copy.deepcopy(value)
    return out


def from_dict(cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    """Build dataclass ``cls`` from ``data``, rejecting unknown and missing keys."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    names = {f.name for f in dataclasses.fields(cls)}
    label = where or cls.__name__
    unknown = sorted(set(data) - names)
    missing = sorted(names - set(data))
    if unknown:
        raise ConfigError(f"{label}: unknown keys {unknown}")
    if missing:
        raise ConfigError(f"{label}: missing keys {missing}")
    return cls(**dict(data))
