"""Quick-pick presets and suggested names for the Build & Queue form (D30).

A quick-pick is a real preset file (``configs/{rays,readings,network,learning,budget}/``) that
only *fills the form*: the run still stores its full resolved settings, so nothing in a model
depends on a quick-pick's name. Pure Python (no Streamlit), unit-tested.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from racer.config import deep_merge, load_preset
from racer.sandbox.builder import (BUDGET_KEYS, SEED_SUFFIX, Plan, _int_or_none, _num, _same, budget_label, ordered,
                                   preset_names, run_tags, train_preset_name)

PICK_KINDS = ("rays", "readings", "memory", "network", "learning", "budget")


def _rays_of(features: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    rays = next((f["rays"] for f in features if f["feature"] == "rays"), [])
    return sorted(({"angle_deg": float(r["angle_deg"]), "range_m": float(r["range_m"])} for r in rays),
                  key=lambda r: r["angle_deg"])


def _readings_of(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in ordered(features) if f["feature"] not in ("rays", "history")]


def history_of(features: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    return next((f for f in features if f["feature"] == "history"), None)


def fix_history(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Features with the memory trimmed to inputs that still exist (dropped when none are left)."""
    have = [f["feature"] for f in features if f["feature"] != "history"]
    out = []
    for f in features:
        if f["feature"] == "history":
            of = [n for n in (f.get("of") or [n for n in have if n != "rays"]) if n in have]
            if not of:
                continue
            f = {**f, "of": of} if f.get("of") else dict(f)
        out.append(dict(f))
    return ordered(out)


def _learning_values(algorithm: str, name: str) -> dict[str, Any]:
    """The algorithm's settings with a learning style applied (``default`` = the train preset itself)."""
    values = copy.deepcopy(load_preset("train", train_preset_name(algorithm))[algorithm])
    if name != "default":
        values = deep_merge(values, load_preset("learning", name).get(algorithm, {}), strict=True)
    return values


def pick_names(kind: str, algorithm: str = "ppo") -> list[str]:
    """The quick-picks of a kind; learning styles are ``default`` plus those of the algorithm."""
    names = preset_names(kind)
    if kind == "learning":
        return ["default"] + [n for n in names if load_preset("learning", n)["algorithm"] == algorithm]
    return names


def pick_help(kind: str, name: str) -> str:
    """The preset's own description (the comment block at the top of its file)."""
    if name == "default":
        return "The algorithm's standard settings (configs/train)."
    from racer.sandbox.explain import preset_header
    return preset_header(kind, name)


def apply_pick(plan: Plan, kind: str, name: str) -> None:
    """Fill the plan's fields from a quick-pick preset."""
    if kind == "learning":
        keep = {k: v for k, v in plan.hparams.items() if k in BUDGET_KEYS}
        plan.hparams = {**_learning_values(plan.algorithm, name), **keep}
        return
    cfg = load_preset(kind, name)
    if kind == "rays":
        plan.features = fix_history(_readings_of(plan.features) + [{"feature": "rays", "rays": cfg["rays"]}]
                                    + [f for f in plan.features if f["feature"] == "history"])
    elif kind == "readings":
        keep = [f for f in plan.features if f["feature"] in ("rays", "history")]
        plan.features = fix_history(keep + cfg["features"])
    elif kind == "memory":
        rest = [f for f in plan.features if f["feature"] != "history"]
        old = history_of(plan.features) or {}
        mem = [] if int(cfg["steps"]) == 0 else [{**({"of": old["of"]} if old.get("of") else {}),
                                                   "feature": "history", "steps": int(cfg["steps"]),
                                                   "stride": int(cfg["stride"])}]
        plan.features = fix_history(rest + mem)
    elif kind == "network":
        plan.net_arch, plan.activation = [int(n) for n in cfg["net_arch"]], cfg["activation"]
    elif kind == "budget":
        if plan.algorithm == "ga":
            plan.generations = _int_or_none(cfg["generations"])
        else:
            plan.steps = _int_or_none(cfg["steps"])
        plan.seeds, plan.trial = [int(s) for s in cfg["seeds"]], bool(cfg["trial"])
    else:
        raise ValueError(f"no quick-pick kind {kind!r}")


def pick_matches(plan: Plan, kind: str, name: str) -> bool:
    """Does the plan currently hold exactly this quick-pick's values?"""
    if kind == "learning":
        want = _learning_values(plan.algorithm, name)
        return all(_same(_num(plan.hparams.get(k)), _num(v)) for k, v in want.items() if k not in BUDGET_KEYS)
    cfg = load_preset(kind, name)
    if kind == "rays":
        return _same(_num(_rays_of(plan.features)), _num(sorted(cfg["rays"], key=lambda r: r["angle_deg"])))
    if kind == "readings":
        return _same(_num(_readings_of(plan.features)), _num(ordered(cfg["features"])))
    if kind == "memory":
        h = history_of(plan.features)
        return (int(h["steps"]) if h else 0) == int(cfg["steps"]) and (int(h["stride"]) if h else 1) == int(cfg["stride"])
    if kind == "network":
        return ([int(n) for n in plan.net_arch] == [int(n) for n in cfg["net_arch"]]
                and plan.activation == cfg["activation"])
    ga = plan.algorithm == "ga"
    return ((plan.generations if ga else plan.steps) == _int_or_none(cfg["generations" if ga else "steps"])
            and [int(s) for s in plan.seeds] == [int(s) for s in cfg["seeds"]]
            and bool(plan.trial) == bool(cfg["trial"]))


# ------------------------------------------------------------------ suggested names
def _budget_short(plan: Plan) -> str:
    if plan.algorithm == "ga":
        return f"{int(plan.generations)}g" if plan.generations else "plateau"
    return f"{(plan.steps or 0) / 1e6:g}m".replace(".", "p")


def suggest_tag(plan: Plan, taken: set[str] | frozenset[str] = frozenset()) -> str:
    """A tag built from what the run is (``ppo_r7_256x256_5m``), made unique against ``taken``."""
    base = SEED_SUFFIX.sub("", "_".join([plan.algorithm, f"r{len(_rays_of(plan.features))}",
                                         "x".join(str(int(n)) for n in plan.net_arch), _budget_short(plan)]))
    tag, n = base, 1
    while True:
        probe = plan.copy()
        probe.tag = tag
        if not any(t in taken for t, _ in run_tags(probe)):
            return tag
        n += 1
        tag = f"{base}_v{n}"


def describe_changes(ref: Plan | None, plan: Plan) -> list[str]:
    """What the plan changes against a reference plan, in words (empty without a reference)."""
    if ref is None:
        return []
    out = []
    if plan.algorithm != ref.algorithm:
        out.append(f"algorithm {ref.algorithm.upper()} → {plan.algorithm.upper()}")
    a, b = _rays_of(ref.features), _rays_of(plan.features)
    if a != b:
        out.append(f"rays {len(a)} → {len(b)}" if len(a) != len(b) else "ray angles or ranges")
    ra, rb = [f["feature"] for f in _readings_of(ref.features)], [f["feature"] for f in _readings_of(plan.features)]
    if ra != rb:
        out.append("car readings: " + (", ".join(rb) or "none"))
    if [int(n) for n in ref.net_arch] != [int(n) for n in plan.net_arch]:
        out.append("network " + "×".join(str(int(n)) for n in plan.net_arch))
    if ref.activation != plan.activation:
        out.append(f"activation {plan.activation}")
    if plan.algorithm == ref.algorithm:
        out += [f"{k} {ref.hparams.get(k)} → {v}" for k, v in plan.hparams.items()
                if k not in BUDGET_KEYS and not isinstance(v, Mapping) and not _same(_num(v), _num(ref.hparams.get(k)))]
    for label, x, y in (("car", ref.physics, plan.physics), ("reward", ref.reward, plan.reward),
                        ("tracks", ref.trackset, plan.trackset)):
        if x != y:
            out.append(f"{label} {y}")
    return out


def suggest_why(plan: Plan, ref: Plan | None) -> str:
    """A starting sentence for the 'why': what the run is, and what it changes against its start."""
    text = (f"{plan.algorithm.upper()} with {len(_rays_of(plan.features))} rays and a "
            f"{'×'.join(str(int(n)) for n in plan.net_arch)} network, "
            f"{budget_label(plan.steps, plan.generations)}, on {plan.trackset}.")
    if ref is not None and ref.started_from:
        changes = describe_changes(ref, plan)
        text += f" Compared with {ref.started_from}: " + ("; ".join(changes) if changes else "the same setup (a repeat).")
    return text
