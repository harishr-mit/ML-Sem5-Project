"""Build & Queue quick-picks, suggested names and the history (memory) input (D30)."""
import numpy as np
import pytest

from racer.env.racing_env import RacingEnv
from racer.sandbox import builder as B
from racer.sandbox import picks as P


def _plan(algo="ppo"):
    return B.plan_from_presets("rays7_lean" if algo == "ga" else "rays7_narrow", algo)


@pytest.mark.parametrize("kind", P.PICK_KINDS)
def test_every_quick_pick_applies_matches_and_validates(kind):
    for algo in ("ppo", "sac", "ga"):
        for name in P.pick_names(kind, algo):
            plan = _plan(algo)
            P.apply_pick(plan, kind, name)
            assert P.pick_matches(plan, kind, name), (kind, name)
            plan.tag, plan.why = "check", "every quick pick must give a valid plan"
            assert B.validate(plan) == [], (kind, name, B.validate(plan))


def test_a_pick_changes_only_its_own_part():
    plan = _plan()
    P.apply_pick(plan, "network", "large")
    P.apply_pick(plan, "learning", "ppo_cautious")
    P.apply_pick(plan, "rays", "wide9")
    P.apply_pick(plan, "memory", "last5")
    assert plan.net_arch == [512, 512] and plan.hparams["learning_rate"] == 0.0001
    assert [f["feature"] for f in plan.features][-1] == "history"
    assert P.pick_matches(plan, "network", "large") and not P.pick_matches(plan, "network", "medium")
    assert P.pick_matches(plan, "rays", "wide9") and P.pick_matches(plan, "learning", "ppo_cautious")
    P.apply_pick(plan, "readings", "lean")                       # the memory keeps only inputs that still exist
    hist = P.history_of(plan.features)
    assert hist.get("of") in (None, ["heading_sincos", "v_long"])
    assert B.validate(plan_with(plan)) == []


def plan_with(plan):
    plan.tag, plan.why = "check", "a plan built from several quick picks"
    return plan


def test_memory_off_removes_the_history_feature():
    plan = _plan()
    P.apply_pick(plan, "memory", "last10")
    assert P.history_of(plan.features)["steps"] == 10
    P.apply_pick(plan, "memory", "off")
    assert P.history_of(plan.features) is None and P.pick_matches(plan, "memory", "off")


def test_suggested_tag_is_unique_and_the_why_names_the_changes():
    plan = _plan()
    base = P.suggest_tag(plan)
    assert base == "ppo_r7_256x256_2m"
    assert P.suggest_tag(plan, {"ppo_r7_256x256_2m_s1"}) == "ppo_r7_256x256_2m_v2"
    ref = plan.copy()
    ref.started_from = "R015"
    P.apply_pick(plan, "network", "large")
    assert "network 512×512" in P.suggest_why(plan, ref)


def test_history_feature_remembers_earlier_states_and_forgets_at_reset():
    plan = _plan()
    plan.features = plan.features + [{"feature": "history", "steps": 3, "stride": 2, "of": ["v_long"]}]
    env = RacingEnv(B.env_config(plan))
    assert env.observer.labels[-3:] == ["v_long (-2)", "v_long (-4)", "v_long (-6)"]
    obs, _ = env.reset(seed=1, options={"speed": 10.0})
    col = env.observer.labels.index("v_long")
    assert np.allclose(obs[-3:], obs[col])                       # no past yet: the present
    seen = [obs[col]]
    for _ in range(6):
        obs, *_ = env.step(np.array([0.0, 1.0]))
        seen.append(obs[col])
    assert np.allclose(obs[-3:], [seen[-3], seen[-5], seen[-7]])
    obs, _ = env.reset(seed=2, options={"speed": 5.0})
    assert np.allclose(obs[-3:], obs[col])


def test_history_must_be_last_and_name_earlier_features():
    plan = _plan()
    with pytest.raises(ValueError, match="last feature"):
        env = B.env_config(plan)
        env["env"]["obs_spec"]["features"].insert(1, {"feature": "history", "steps": 2, "of": ["rays"]})
        RacingEnv(env)
    plan.features = plan.features + [{"feature": "history", "steps": 2, "of": ["curvature_ahead"]}]
    with pytest.raises(ValueError, match="don't come before"):
        RacingEnv(B.env_config(plan))


@pytest.mark.parametrize("layout", ["steer_gas", "steer_throttle_brake", "steer_rate", "steer_throttle_brake_abs"])
def test_every_new_input_and_layout_drives_within_bounds(layout):
    plan = _plan()
    plan.layout = layout
    plan.features = B.ordered(plan.features + [{"feature": n} for n in ("road_position", "wheel_surface", "accel_g",
                                                                         "grip_use")])
    env = RacingEnv(B.env_config(plan))
    obs, _ = env.reset(seed=3, options={"speed": 25.0})
    low, high = env.observer.bounds()
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, *_, done = (*env.step(rng.uniform(-1, 1, env.action_layout.dim)),)[:4] + (False,)
        assert np.all(obs >= low - 1e-6) and np.all(obs <= high + 1e-6)
    assert plan.layout == layout and env.observer.dim == len(env.observer.labels)


def test_steer_rate_turns_the_wheel_from_where_it_is_and_abs_never_locks():
    from racer.env.actions import SteerRate
    lay = SteerRate()
    lay.neutral()
    assert [round(lay.to_controls(np.array([1.0, 0.0]))[0], 2) for _ in range(5)] == [0.3, 0.6, 0.9, 1.0, 1.0]
    lay.neutral()
    assert lay.to_controls(np.array([0.0, 0.5])) == (0.0, 0.5)
    plan = _plan()
    plan.layout = "steer_throttle_brake_abs"
    env = RacingEnv(B.env_config(plan))
    env.reset(seed=1, options={"speed": 30.0, "lateral": 0.0, "heading_offset": 0.0})
    for _ in range(12):
        env.step(np.array([0.0, -1.0, 1.0]))                      # full brake pedal
        assert env.car.grip_scale == 1.0 and env.car.speed < 30.0     # braking hard, wheels never locked
