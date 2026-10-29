"""GA input normalisation (D33): a frozen standardiser, measured once from the baseline and carried by every exported car."""
import numpy as np
import pytest
import yaml

from racer.agents import neuroevolution as ne
from racer.agents.compat import build_manifest
from racer.agents.evolve import Evolution, State, export_model, settings_key
from racer.agents.loader import LoadedModel
from racer.agents.obsnorm import calibrate_obs, standardise
from racer.config import load_preset
from racer.env.racing_env import RacingEnv, make_env_config

ENV = make_env_config(["training_ground"], env="rays7_pen")


@pytest.fixture(scope="module")
def stats():
    return calibrate_obs(ENV, ["training_ground"], seed=5, min_std=0.05, clip=10.0, steps_per_map=400)


def test_calibration_is_deterministic_and_has_a_floor(stats):
    again = calibrate_obs(ENV, ["training_ground"], seed=5, min_std=0.05, clip=10.0, steps_per_map=400)
    assert again == stats
    assert len(stats["mean"]) == len(stats["var"]) == 17 and stats["clip"] == 10.0
    assert min(stats["var"]) >= 0.05 ** 2 - 1e-12
    other = calibrate_obs(ENV, ["training_ground"], seed=6, min_std=0.05, clip=10.0, steps_per_map=400)
    assert other != stats


def test_standardise_is_the_vecnormalize_formula(stats):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from racer.agents.evolve import _Spaces
    env = RacingEnv(ENV)
    vn = VecNormalize(DummyVecEnv([lambda: _Spaces(env.observation_space, env.action_space)]), training=False,
                      norm_obs=True, norm_reward=False, clip_obs=stats["clip"])
    vn.obs_rms.mean, vn.obs_rms.var = np.array(stats["mean"]), np.array(stats["var"])
    obs = np.random.default_rng(0).uniform(-2, 3, (20, 17)).astype(np.float32)
    np.testing.assert_allclose(standardise(obs, stats), vn.normalize_obs(obs), atol=1e-5)


def make_train(stats):
    train = load_preset("train", "ga_default")
    train["ga"].update(normalise=True, obs_norm=stats)
    return train


def test_an_exported_car_standardises_like_the_run_did(tmp_path, stats):
    train = make_train(stats)
    env = RacingEnv(ENV)
    sizes = ne.layer_sizes(17, train["policy"]["net_arch"], 2)
    flat = ne.init_weights(sizes, 7)
    folder = tmp_path / "S001_GA_norm@1k"
    export_model(folder, flat, train, env.observation_space, env.action_space)
    snap = {"model_id": folder.name, "algorithm": "ga", "train": train, "env_config": ENV,
            "manifest": build_manifest(ENV, train)}
    (folder / "config.yaml").write_text(yaml.safe_dump(snap, sort_keys=False), encoding="utf-8")
    obs = np.stack([env.reset(seed=s)[0] for s in range(24)])
    expected = ne.forward(ne.unflatten(flat, sizes), standardise(obs, stats))
    np.testing.assert_allclose(LoadedModel(folder).act_batch(obs), expected, atol=1e-5)


def test_a_normalised_run_resumes_identically_and_the_stats_are_part_of_its_settings(tmp_path, stats):
    train = make_train(stats)
    train["ga"].update(population=6, elite_frac=0.34, workers=1, state_every=3,
                       validation={"every": 3, "tracks": 1, "stream": 2, "patience": 1000, "min_gain": 0.01})

    def ev(d):
        return Evolution(d, ENV, train, obs_dim=17, act_dim=2, log=lambda s: None)

    assert ev(tmp_path / "a").run(pause_after=6) == "paused"
    assert ev(tmp_path / "b").run(pause_after=3) == "paused"
    assert ev(tmp_path / "b").run(pause_after=6) == "paused"
    a, b = State.load(tmp_path / "a" / "state.npz"), State.load(tmp_path / "b" / "state.npz")
    assert np.array_equal(a.weights, b.weights)
    plain = load_preset("train", "ga_default")
    assert settings_key(ENV, train) != settings_key(ENV, plain)


def test_a_run_without_normalisation_is_unchanged():
    from racer.agents.evolve import GAParams
    from racer.config import from_dict
    p = from_dict(GAParams, load_preset("train", "ga_default")["ga"], where="t")
    assert p.normalise is False and p.obs_norm is None
