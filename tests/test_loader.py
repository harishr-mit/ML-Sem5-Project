"""Loader v2 (T43): model lookup and fast batched inference identical to SB3 predict."""
import numpy as np
import pytest

from racer.agents.loader import LoadedModel, resolve_model
from racer.config import PROJECT_ROOT
from racer.env.racing_env import RacingEnv

MODEL_ID = "R001_ppo_rays5_tg@1200k"
HAVE = (PROJECT_ROOT / "models" / MODEL_ID / "model.zip").is_file()


@pytest.mark.skipif(not HAVE, reason="R001 not on this machine")
def test_resolve_by_id_and_batched_actions_match_predict():
    assert resolve_model(MODEL_ID) == PROJECT_ROOT / "models" / MODEL_ID
    m = LoadedModel(MODEL_ID)
    env = RacingEnv(m.env_config)
    obs = [env.reset(seed=s)[0] for s in range(16)]
    batch = np.stack(obs)
    fast = m.act_batch(batch)
    ref = np.stack([m.policy.predict(m.normalise(o), deterministic=True)[0] for o in obs])
    assert fast.shape == (16, 2)
    # One row at a time is bit-identical to SB3; a batch only differs by BLAS summation order.
    single = np.stack([m.act(o) for o in obs])
    np.testing.assert_array_equal(single, ref.astype(np.float32))
    np.testing.assert_allclose(fast, single, atol=1e-5)


def test_unknown_model_raises():
    with pytest.raises(FileNotFoundError):
        resolve_model("R999_nope@1k")
