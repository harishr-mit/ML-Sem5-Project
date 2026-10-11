"""Explanations: glossary lookup, snapshot flattening and config diffs (T67).

Every config value the sandbox shows has a glossary key in one namespace:

    physics.*  reward.*  env.*  tracks.*  track.*  presets.*   from the snapshot's env_config
    train.*                                                    train preset (hyperparameters)
    manifest.*                                                 derived model shape (D23)
    run.*                                                      identity: ids, dates, notes, versions

``glossary.yaml`` explains each key (wildcards such as ``physics.surfaces.*.mu_static``
allowed); feature, action-layout and algorithm descriptions come from their code registries.
A test fails if any key of any preset or saved snapshot has no explanation.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

GLOSSARY_PATH = Path(__file__).with_name("glossary.yaml")

# top-level snapshot keys that describe the run rather than configure it
IDENTITY_TOP = {"run_id", "tag", "seed", "created_at", "git_commit", "notes", "curriculum", "command",
                "model_id", "steps", "saved_at", "versions", "obs_spec_version"}
# config keys that differ between seeds / copies of the same experiment, hidden by default in diffs
IDENTITY_KEYS = {"train.seed"}
ENV_SECTIONS = ("physics", "reward", "env", "tracks", "track", "presets")

_glossary: dict[str, dict[str, Any]] | None = None


# ------------------------------------------------------------------ glossary
def glossary() -> dict[str, dict[str, Any]]:
    """Flat glossary: key → {short, affects, range?, related?, group}."""
    global _glossary
    if _glossary is None:
        raw = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
        flat: dict[str, dict[str, Any]] = {}
        for group, body in raw.items():
            for key, entry in (body.get("keys") or {}).items():
                flat[key] = {"group": group, "group_title": body.get("title", group), **(entry or {})}
        _glossary = flat
    return _glossary


def groups() -> list[tuple[str, str, str]]:
    """(group, title, intro) in file order, for the Glossary page."""
    raw = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
    return [(g, b.get("title", g), b.get("intro", "")) for g, b in raw.items()]


def lookup(key: str) -> dict[str, Any] | None:
    """Glossary entry for a key: exact match first, then the most specific wildcard."""
    g = glossary()
    if key in g:
        return g[key]
    matches = [p for p in g if any(c in p for c in "*?[") and fnmatch.fnmatchcase(key, p)]
    if not matches:
        return None
    return g[max(matches, key=lambda p: len(p.replace("*", "")))]


def note(key: str) -> str:
    """One-line explanation (what it is; what it affects) or ''."""
    e = lookup(key)
    if not e:
        return ""
    return " ".join(x for x in (e.get("short", ""), e.get("affects", "")) if x)


# ------------------------------------------------------------------ flattening
def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} → {'a.b': 1}. Lists are kept whole (a ray list is one value)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def snapshot_rows(snapshot: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A snapshot as flat glossary-namespace keys (``env_config.physics.mass_kg`` → ``physics.mass_kg``).

    ``manifest`` replaces the stored one (pass ``manifest_of(snapshot)`` so models from before
    D23 get one too)."""
    rows: dict[str, Any] = {}
    for k, v in snapshot.items():
        if k == "env_config":
            for sec, body in v.items():
                if sec in ENV_SECTIONS:
                    rows.update(flatten(body, sec) if isinstance(body, Mapping) else {sec: body})
                else:
                    rows[f"env_config.{sec}"] = body
        elif k == "train":
            rows.update(flatten(v, "train"))
        elif k == "manifest":
            continue
        else:
            rows.update(flatten(v, f"run.{k}") if isinstance(v, Mapping) else {f"run.{k}": v})
    env = (snapshot.get("env_config") or {}).get("env")
    if isinstance(env, Mapping) and "env.action_spec.layout" not in rows:
        from racer.env.actions import action_spec_of
        rows["env.action_spec.layout"] = action_spec_of(env).get("layout")   # no action_spec = steer_gas (D22)
    man = manifest if manifest is not None else snapshot.get("manifest")
    if man:
        rows.update(flatten(dict(man), "manifest"))
    return rows


def preset_rows(kind: str, preset: Mapping[str, Any]) -> dict[str, Any]:
    """A config preset file (``configs/<kind>/*.yaml``) as glossary-namespace keys."""
    section = {"trackset": "tracks"}.get(kind, kind)
    return flatten(preset, section)


def row_kind(key: str) -> str:
    """'identity' (ids, dates, seeds), 'derived' (manifest) or 'config'."""
    if key in IDENTITY_KEYS or (key.startswith("run.") and key.split(".")[1] in IDENTITY_TOP):
        return "identity"
    if key.startswith("manifest."):
        return "derived"
    return "config"


# ------------------------------------------------------------------ diff
_MISSING = object()


def diff(snapshot_rows_list: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows of several flattened snapshots side by side.

    Each row: key, kind, values (``None`` where a snapshot lacks the key), differs, note.
    Keys in order of first appearance."""
    keys: list[str] = []
    seen = set()
    for rows in snapshot_rows_list:
        for k in rows:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    out = []
    for k in keys:
        vals = [rows.get(k, _MISSING) for rows in snapshot_rows_list]
        canon = [_canon(v) for v in vals]
        out.append({"key": k, "kind": row_kind(k), "values": [None if v is _MISSING else v for v in vals],
                    "missing": [v is _MISSING for v in vals], "differs": len(set(canon)) > 1, "note": note(k)})
    return out


def _canon(v: Any) -> str:
    if v is _MISSING:
        return "<missing>"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return yaml.safe_dump(_plain(v), sort_keys=True, default_flow_style=True).strip()


def _plain(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


# ------------------------------------------------------------------ sections
def section(rows: Mapping[str, Any], prefix: str, reference: Mapping[str, Any] | None = None,
            exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Rows under ``prefix.`` with their value, the reference (default preset) value, whether
    it differs, and the glossary entry."""
    skip = set(exclude)
    out = []
    for k, v in rows.items():
        if not k.startswith(prefix + ".") or k in skip:
            continue
        ref = reference.get(k, _MISSING) if reference is not None else _MISSING
        e = lookup(k) or {}
        out.append({"key": k, "name": k[len(prefix) + 1:], "value": v, "default": None if ref is _MISSING else ref,
                    "changed": reference is not None and _canon(v) != _canon(ref),
                    "short": e.get("short", ""), "affects": e.get("affects", ""), "range": e.get("range", ""),
                    "related": e.get("related", "")})
    return out


def reference_rows(kind: str, name: str) -> dict[str, Any]:
    """A preset's values as glossary-namespace rows ({} if it doesn't exist)."""
    from racer.config import ConfigError, load_preset
    try:
        return preset_rows(kind, load_preset(kind, name))
    except (ConfigError, FileNotFoundError):
        return {}


# ------------------------------------------------------------------ display
def fmt(v: Any) -> str:
    """Compact human-readable value (ray lists as angles and ranges)."""
    if v is None or v is _MISSING:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list) and v and all(isinstance(x, Mapping) and "angle_deg" in x for x in v):
        return ", ".join(f"{float(x['angle_deg']):+g}°/{float(x['range_m']):g} m" for x in v)
    if isinstance(v, list) and v and all(isinstance(x, Mapping) and "feature" in x for x in v):
        return " · ".join(_feature_label(x) for x in v)
    if isinstance(v, list) and v and all(isinstance(x, Mapping) and "track" in x for x in v):
        return ", ".join(f"{x['track']}×{float(x.get('weight', 1)):g}" for x in v)
    if isinstance(v, (list, dict)):
        return yaml.safe_dump(_plain(v), default_flow_style=True, sort_keys=False, width=10_000).strip()
    return str(v)


def _feature_label(f: Mapping[str, Any]) -> str:
    name = f["feature"]
    if name == "rays":
        return f"rays({len(f.get('rays') or f.get('params', {}).get('rays') or [])})"
    params = {k: v for k, v in f.items() if k not in ("feature", "dims", "labels", "params")}
    params.update(f.get("params") or {})
    return f"{name}({', '.join(f'{k}={fmt(v)}' for k, v in params.items())})" if params else name


def unexplained(keys: Iterable[str]) -> list[str]:
    """Keys without a glossary entry (the coverage test uses this)."""
    return sorted({k for k in keys if lookup(k) is None})
