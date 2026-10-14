"""Neuroevolution runs (D26): a GA car saved as a model folder drives exactly like the GA's network."""
import numpy as np
import pytest
import yaml

from racer.agents import neuroevolution as ne
from racer.agents.compat import build_manifest, check
from racer.agents.evolve import export_model
from racer.agents.loader import LoadedModel
from racer.config import load_preset
from racer.env.racing_env import RacingEnv, make_env_config


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    env_config = make_env_config(["training_ground"], env="rays7_lean")
    train = load_preset("train", "ga_default")
    env = RacingEnv(env_config)
    sizes = ne.layer_sizes(env.observation_space.shape[0], train["policy"]["net_arch"], env.action_space.shape[0])
    flat = ne.init_weights(sizes, 123)
    folder = tmp_path_factory.mktemp("models") / "S001_GA_test@1k"
    export_model(folder, flat, train, env.observation_space, env.action_space)
    snap = {"model_id": folder.name, "algorithm": "ga", "train": train, "env_config": env_config,
            "manifest": build_manifest(env_config, train)}
    (folder / "config.yaml").write_text(yaml.safe_dump(snap, sort_keys=False), encoding="utf-8")
    return folder, flat, sizes, env


def test_loaded_model_drives_like_the_numpy_network(exported):
    folder, flat, sizes, env = exported
    m = LoadedModel(folder)
    obs = np.stack([env.reset(seed=s)[0] for s in range(32)])
    obs[::4] = np.random.default_rng(0).uniform(-3, 3, obs[::4].shape)   # also outside the input range: clipping
    expected = ne.forward(ne.unflatten(flat, sizes), obs)
    np.testing.assert_allclose(m.act_batch(obs), expected, atol=1e-5)
    np.testing.assert_allclose(np.stack([m.act(o) for o in obs]), expected, atol=1e-5)


def test_folder_is_a_complete_model(exported):
    folder, *_ = exported
    from test_integrity import MODEL_FILES
    for f in MODEL_FILES:
        assert (folder / f).is_file()
    snap = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    assert check(snap) == []
    assert snap["manifest"]["n_params"] == 69_122 and snap["manifest"]["obs_dim"] == 10
    assert (folder / "model.zip").stat().st_size < 2_000_000


def test_inputs_are_not_normalised(exported):
    folder, *_ = exported
    m = LoadedModel(folder)
    x = np.linspace(-1, 1, 10, dtype=np.float32)[None]
    np.testing.assert_array_equal(m.normalise(x), x)


def test_folders_are_never_overwritten(exported):
    folder, flat, sizes, env = exported
    with pytest.raises(FileExistsError):
        export_model(folder, flat, load_preset("train", "ga_default"), env.observation_space, env.action_space)

