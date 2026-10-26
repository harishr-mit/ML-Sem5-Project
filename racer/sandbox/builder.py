"""Build a new training run from the sandbox form (T71).

A :class:`Plan` holds what the form edits: inputs (v2 features), outputs (action layout),
network, algorithm and hyperparameters, car / reward / track-set presets, steps, seeds and
the "why". From it this module derives exactly what ``scripts/train.py`` needs:

* the env preset: an existing one if the inputs, outputs, ray marching and episode rules
  match it, else a new standalone v2 preset ``configs/env/sb_<tag>.yaml`` (untracked, so
  real runs still start (D12); the snapshot stores the full spec anyway);
* ``--set`` overrides for every hyperparameter or network change against the algorithm's
  default preset (no train preset file is written);
* one queue entry per seed, with args as a list (no YAML quoting problems).

Validation uses the same code as train.py (``make_env_config`` + ``build_manifest``).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from racer.config import PROJECT_ROOT, ConfigError, deep_merge, load_preset

# Order features appear in a new spec (rays first, previous action last, like every preset).
FEATURE_ORDER = ["rays", "heading_sincos", "v_long", "v_lat", "yaw_rate", "steer_angle", "onroad_frac",
                 "road_position", "wheel_surface", "accel_g", "grip_use", "curvature_ahead", "prev_action", "history"]
TAG_RE = re.compile(r"[a-z0-9_]+")
SEED_SUFFIX = re.compile(r"_s\d+$")
STEP_PRESETS = [1_000_000, 2_000_000, 3_000_000, 5_000_000]


# ------------------------------------------------------------------ obs specs
def v1_to_v2(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The v2 feature list that computes exactly what a v1 spec computes (same order)."""
    sc = spec["scales"]
    feats: list[dict[str, Any]] = [{"feature": "rays", "rays": [{"angle_deg": float(r["angle_deg"]),
                                                                  "range_m": float(r["range_m"])}
                                                                 for r in spec["rays"]]}]
    for e in spec["extras"]:
        if e in ("v_long", "v_lat", "yaw_rate"):
            feats.append({"feature": e, "scale": float(sc[e])})
        else:
            feats.append({"feature": e})
    return feats


def features_of(env_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An env config's inputs as a v2 feature list (v1 specs converted)."""
    spec = env_cfg["obs_spec"]
    if int(spec.get("version", 1)) == 1:
        return v1_to_v2(spec)
    return [_canon_feature(f) for f in spec["features"]]


def _canon_feature(f: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"feature": f["feature"]}
    for k, v in f.items():
        if k == "feature":
            continue
        if k == "rays":
            out[k] = [{"angle_deg": float(r["angle_deg"]), "range_m": float(r["range_m"])} for r in v]
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]    # names (history.of) stay text
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def ordered(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Features in the standard order (unknown names keep their place at the end)."""
    rank = {n: i for i, n in enumerate(FEATURE_ORDER)}
    return sorted((_canon_feature(f) for f in features), key=lambda f: rank.get(f["feature"], len(rank)))


# ------------------------------------------------------------------ the plan
@dataclass
class Plan:
    tag: str = ""
    why: str = ""
    features: list[dict[str, Any]] = field(default_factory=list)
    layout: str = "steer_gas"
    algorithm: str = "ppo"
    hparams: dict[str, Any] = field(default_factory=dict)      # the algorithm's section (ppo: / sac: / ga:)
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    physics: str = "default"
    reward: str = "progress"
    trackset: str = "maps7"
    steps: int | None = 2_000_000       # PPO/SAC budget
    generations: int | None = None      # neuroevolution budget; None = until the validation plateau
    seeds: list[int] = field(default_factory=lambda: [1])
    trial: bool = False
    raycast: dict[str, Any] = field(default_factory=dict)
    episode: dict[str, Any] = field(default_factory=dict)
    started_from: str | None = None

    def copy(self) -> "Plan":
        return copy.deepcopy(self)


BUILDABLE = ("ppo", "sac", "ga")     # algorithms the Build page can configure
GA_TRACKSET = "train7"               # neuroevolution drives named maps, never a generated pool


def train_preset_name(algorithm: str) -> str:
    return f"{algorithm}_default"


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _generations(train: Mapping[str, Any]) -> int | None:
    return _int_or_none((train.get("ga") or {}).get("max_generations"))


def budget_label(steps: int | None, generations: int | None) -> str:
    """A run's budget in words (GA runs count generations, PPO/SAC steps)."""
    if generations:
        return f"{generations} generations"
    return f"{steps / 1e6:g}M steps" if steps else "until it stops improving"


BUDGET_KEYS = {"max_generations"}      # set by the budget, not as a hyperparameter


def plan_from_presets(env: str = "rays7_narrow", algorithm: str = "ppo", trackset: str | None = None) -> Plan:
    env_cfg = load_preset("env", env)
    train = load_preset("train", train_preset_name(algorithm))
    return Plan(features=features_of(env_cfg), layout=(env_cfg.get("action_spec") or {}).get("layout", "steer_gas"),
                algorithm=algorithm, hparams=dict(train[algorithm]), net_arch=list(train["policy"]["net_arch"]),
                activation=train["policy"]["activation"],
                trackset=trackset or (GA_TRACKSET if algorithm == "ga" else "maps7"),
                steps=_int_or_none(train.get("total_steps")), generations=_generations(train),
                raycast=dict(env_cfg["raycast"]), episode=dict(env_cfg["episode"]))


def plan_from_snapshot(snap: Mapping[str, Any], model_id: str | None = None) -> Plan:
    """A plan that repeats a saved model's setup (edit it before queueing)."""
    env = snap["env_config"]
    train = snap["train"]
    algo = snap.get("algorithm", train.get("algorithm", "ppo"))
    presets = env.get("presets", {})
    tracks = env.get("tracks")
    trackset = presets.get("trackset") or (tracks.get("trackset") if isinstance(tracks, Mapping) else None)
    if not trackset:          # old snapshots list track names: Training Ground alone is the 'tg' set
        trackset = "tg" if tracks == ["training_ground"] else "maps7"
    return Plan(features=features_of(env["env"]), layout=(env["env"].get("action_spec") or {}).get("layout", "steer_gas"),
                algorithm=algo, hparams=dict(train.get(algo, {})), net_arch=[int(n) for n in train["policy"]["net_arch"]],
                activation=train["policy"]["activation"], physics=presets.get("physics", "default"),
                reward=presets.get("reward", "progress"), trackset=trackset or "maps7",
                steps=_int_or_none(train.get("total_steps", 2_000_000)), generations=_generations(train),
                seeds=[int(snap.get("seed", 1))],
                raycast=dict(env["env"]["raycast"]), episode=dict(env["env"]["episode"]),
                started_from=model_id or snap.get("model_id"))


# ------------------------------------------------------------------ derived configs
def env_section(plan: Plan, name: str) -> dict[str, Any]:
    """A standalone v2 env preset for the plan."""
    return {"obs_spec": {"version": 2, "name": name, "features": ordered(plan.features)},
            "action_spec": {"layout": plan.layout}, "raycast": dict(plan.raycast), "episode": dict(plan.episode)}


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def _num(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_num(v) for v in x]
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    return x


def preset_names(kind: str, root: Path = PROJECT_ROOT) -> list[str]:
    return sorted(p.stem for p in (root / "configs" / kind).glob("*.yaml"))


def matching_env_preset(plan: Plan, root: Path = PROJECT_ROOT) -> str | None:
    """An existing env preset with the same inputs (in the same order), outputs, ray marching
    and episode rules, or None."""
    want = _num(ordered(plan.features))
    for name in preset_names("env", root):
        try:
            cfg = load_preset("env", name)
        except (ConfigError, KeyError, ValueError):
            continue
        layout = (cfg.get("action_spec") or {}).get("layout", "steer_gas")
        if (layout == plan.layout and _same(_num(features_of(cfg)), want)
                and _same(_num(cfg.get("raycast")), _num(plan.raycast))
                and _same(_num(cfg.get("episode")), _num(plan.episode))):
            return name
    return None


def train_sets(plan: Plan) -> list[str]:
    """``--set`` overrides: every hyperparameter / network value that differs from the
    algorithm's default preset."""
    base = load_preset("train", train_preset_name(plan.algorithm))
    out = []
    for k, v in plan.hparams.items():
        if k in BUDGET_KEYS:
            continue
        if k not in base[plan.algorithm]:
            raise ValueError(f"unknown {plan.algorithm} hyperparameter {k!r}")
        if not _same(_num(v), _num(base[plan.algorithm][k])):
            out.append(f"{plan.algorithm}.{k}={_yaml_value(v)}")
    if [int(n) for n in plan.net_arch] != [int(n) for n in base["policy"]["net_arch"]]:
        out.append(f"policy.net_arch={_yaml_value([int(n) for n in plan.net_arch])}")
    if plan.activation != base["policy"]["activation"]:
        out.append(f"policy.activation={plan.activation}")
    return out


def _yaml_value(v: Any) -> str:
    """A value as train.py's ``--set`` reads it back (``yaml.safe_load``)."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = repr(v)
        mant, e, exp = s.partition("e")
        if e and "." not in mant:          # YAML 1.1 reads '3e-05' as text: write '3.0e-05'
            s = f"{mant}.0e{exp}"
        return s
    return json.dumps(v).replace(", ", ",")


def train_config(plan: Plan, seed: int | None = None) -> dict[str, Any]:
    """The train config train.py will build (preset + --steps/--seed + --set)."""
    cfg = load_preset("train", train_preset_name(plan.algorithm))
    if plan.algorithm == "ga":
        cfg["ga"]["max_generations"] = plan.generations
    else:
        cfg["total_steps"] = _int_or_none(plan.steps)
    if seed is not None:
        cfg["seed"] = int(seed)
    override: dict[str, Any] = {plan.algorithm: {}, "policy": {}}
    for item in train_sets(plan):
        key, _, value = item.partition("=")
        sec, _, k = key.partition(".")
        override[sec][k] = yaml.safe_load(value)
    return deep_merge(cfg, {k: v for k, v in override.items() if v}, strict=True)


def env_config(plan: Plan) -> dict[str, Any]:
    """The env config train.py will build (with the plan's env section in place)."""
    from racer.env.racing_env import make_env_config
    cfg = make_env_config(trackset=plan.trackset, physics=plan.physics, env="rays5", reward=plan.reward)
    cfg["env"] = env_section(plan, "sb_preview")
    return cfg


def preview(plan: Plan) -> dict[str, Any]:
    """The model's manifest (inputs, outputs, parameter counts) — raises ValueError if invalid."""
    from racer.agents.compat import build_manifest
    return build_manifest(env_config(plan), train_config(plan))


# ------------------------------------------------------------------ validation
def run_tags(plan: Plan) -> list[tuple[str, int]]:
    """(tag, seed) of each run: the seed suffix ``_s<seed>`` is added unless the tag has one
    and there is a single seed."""
    if len(plan.seeds) == 1 and SEED_SUFFIX.search(plan.tag):
        return [(plan.tag, int(plan.seeds[0]))]
    base = SEED_SUFFIX.sub("", plan.tag)
    return [(f"{base}_s{int(s)}", int(s)) for s in plan.seeds]


def validate(plan: Plan, taken_tags: set[str] = frozenset(), queued: set[str] = frozenset()) -> list[str]:
    """Problems that stop the plan from being queued (empty = OK)."""
    errs = []
    if not plan.tag or not TAG_RE.fullmatch(plan.tag):
        errs.append("Tag: lowercase letters, digits and underscores only (e.g. narrow9_lr1e4).")
    if len(plan.why.strip()) < 10:
        errs.append("Why: write a sentence on what this run should show (it is stored with the model).")
    if not plan.seeds or len(set(plan.seeds)) != len(plan.seeds):
        errs.append("Seeds: give one or more different whole numbers.")
    for tag, _ in run_tags(plan) if plan.tag and TAG_RE.fullmatch(plan.tag) else []:
        if tag in taken_tags:
            errs.append(f"Tag {tag!r} is already used by a run; choose another.")
        if tag in queued:
            errs.append(f"{tag!r} is already in the queue.")
    if plan.algorithm == "ga":
        if plan.generations is not None and plan.generations < 1:
            errs.append("Generations: at least 1.")
        if plan.trackset in preset_names("trackset"):
            from racer.agents.evolve import training_maps
            from racer.env.trackset import resolve_trackset
            try:
                training_maps({"tracks": resolve_trackset(plan.trackset)})
            except (ValueError, ConfigError) as e:
                errs.append(f"Tracks: {e}")
    elif plan.steps is None:
        errs.append("Steps: PPO and SAC need a number of steps.")
    else:
        ckpt = int(load_preset("train", train_preset_name(plan.algorithm))["checkpoint_every"])
        if plan.steps < ckpt:
            errs.append(f"Steps: at least {ckpt:,} (one checkpoint).")
    if not plan.net_arch or any(int(n) < 1 for n in plan.net_arch):
        errs.append("Network: every hidden layer needs at least 1 neuron.")
    for kind, name in (("physics", plan.physics), ("reward", plan.reward), ("trackset", plan.trackset)):
        if name not in preset_names(kind):
            errs.append(f"No {kind} preset {name!r}.")
    if not plan.features:
        errs.append("Inputs: choose at least one.")
    if not errs:
        try:
            preview(plan)
        except (ValueError, KeyError, TypeError, ConfigError) as e:
            errs.append(f"Configuration: {e}")
    return errs


def warnings(plan: Plan) -> list[str]:
    """Things that are allowed but probably unintended."""
    out = []
    if plan.algorithm == "ppo" and plan.hparams:
        n_envs = int(load_preset("train", "ppo_default")["n_envs"])
        rollout = int(plan.hparams.get("n_steps", 1)) * n_envs
        if rollout % max(int(plan.hparams.get("batch_size", 1)), 1):
            out.append(f"batch_size doesn't divide the rollout ({rollout:,} = n_steps × {n_envs} envs): "
                       "the last minibatch is partial.")
    if not any(f["feature"] == "rays" for f in plan.features):
        out.append("No rays: the car can't see the road ahead.")
    return out


# ------------------------------------------------------------------ output
def env_preset_name(plan: Plan) -> str:
    return f"sb_{SEED_SUFFIX.sub('', plan.tag)}"


def write_env_preset(plan: Plan, root: Path = PROJECT_ROOT) -> Path:
    """Write ``configs/env/sb_<tag>.yaml`` (never overwrites a different preset)."""
    name = env_preset_name(plan)
    path = root / "configs" / "env" / f"{name}.yaml"
    body = env_section(plan, name)
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if _same(_num(existing), _num(body)):
            return path
        raise FileExistsError(f"{path.name} already exists with other settings; choose another tag")
    header = (f"# Environment preset {name!r}, written by the AI Sandbox (T71) for run tag {plan.tag!r}.\n"
              "# obs_spec v2 (D21), standalone. Untracked files don't block real runs (D12); commit it to keep\n"
              "# the preset with the code (the model's snapshot stores the full spec anyway).\n")
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train_argv(plan: Plan, env_name: str, tag: str, seed: int) -> list[str]:
    curriculum = "AI Sandbox" + (f" (from {plan.started_from})" if plan.started_from else "")
    argv = ["--tag", tag, "--trackset", plan.trackset, "--env", env_name, "--train", train_preset_name(plan.algorithm),
            "--physics", plan.physics, "--reward", plan.reward]
    if plan.algorithm == "ga":
        if plan.generations is not None:
            argv += ["--generations", str(int(plan.generations))]
    elif plan.steps is not None:
        argv += ["--steps", str(int(plan.steps))]
    argv += ["--seed", str(seed)]
    for s in train_sets(plan):
        argv += ["--set", s]
    argv += ["--curriculum", curriculum, "--notes", plan.why.strip()]
    if plan.trial:
        argv.append("--scratch")
    return argv


def queue_entries(plan: Plan, env_name: str) -> list[dict[str, Any]]:
    return [{"name": tag, "args": train_argv(plan, env_name, tag, seed)} for tag, seed in run_tags(plan)]


EXTEND_KEEP = ("--tag", "--train", "--trackset", "--tracks", "--env", "--physics", "--reward", "--seed")


def extend_entry(snapshot: Mapping[str, Any], patience: int, generations: int | None, workers: int | None,
                 taken: set[str]) -> dict[str, Any]:
    """A queue entry that continues a neuroevolution run (D29): ``train.py --extend`` with the flags of
    the run's own command that identify it (the run keeps its saved settings, so they only have to
    be valid), the new patience and optionally a generation budget and worker count."""
    toks = str(snapshot.get("command") or "").split()
    if "--tag" not in toks:
        raise ValueError("this run's snapshot has no command to continue it from")
    args: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i] in EXTEND_KEEP:
            args.append(toks[i])
            i += 1
            while i < len(toks) and not toks[i].startswith("--"):
                args.append(toks[i])
                i += 1
        else:
            i += 1
    args += ["--extend", "--set", f"ga.validation.patience={int(patience)}"]
    if workers:
        args += ["--set", f"ga.workers={int(workers)}"]
    if generations:
        args += ["--generations", str(int(generations))]
    run_id = str(snapshot["run_id"])
    if run_id.startswith("S"):
        args.append("--scratch")
    n = 1
    while f"extend_{run_id}_{n}" in taken:
        n += 1
    return {"name": f"extend_{run_id}_{n}", "args": args}


def estimate_minutes(steps: int | None, runs_ahead: int, parallel: int, cfg: Mapping[str, Any] | None = None,
                     algorithm: str = "ppo", generations: int | None = None) -> dict[str, float] | None:
    """Rough time for one run and for the queue: ``runs_ahead`` includes this one.

    PPO/SAC from ``steps``, neuroevolution from ``generations``; None without a budget (a
    neuroevolution run then ends at its validation plateau)."""
    q = (cfg or load_preset("sandbox", "default"))["queue"]
    if algorithm == "ga":
        if generations is None:
            return None
        run = generations * float(q["ga_minutes_per_generation"]) + float(q["ga_eval_minutes"])
        return {"run": run, "queue": run * max(runs_ahead, 1) / max(parallel, 1)}
    if steps is None:
        return None
    per_m = float(q["minutes_per_million_parallel" if parallel > 1 else "minutes_per_million_alone"])
    run = steps / 1e6 * per_m + steps / 100_000 * float(q["eval_minutes_per_checkpoint"])
    return {"run": run, "queue": run * max(runs_ahead, 1) / max(parallel, 1)}


def drop_marked(rows: list[dict[str, Any]], marked: list[bool]) -> list[dict[str, Any]]:
    """``rows`` without the ones whose flag is true (the Build page's *Remove ticked rays*, T134)."""
    return [r for r, m in zip(rows, marked) if not m]
