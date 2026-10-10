"""Golden regression: the M1 model must keep driving exactly as it did when it was saved.

Models are local only (git-ignored), so this skips when the folder isn't there.
Week 1 numbers (T28): R001@1200k completes 6/6 evenly spaced standing starts on
Training Ground, best lap 42.70 s.
"""
import pytest

from racer.config import PROJECT_ROOT

FOLDER = PROJECT_ROOT / "models" / "R001_ppo_rays5_tg@1200k"


@pytest.mark.skipif(not (FOLDER / "model.zip").is_file(), reason="R001@1200k not on this machine")
def test_r001_still_laps_like_week1():
    from racer.agents.evaluation import evaluate
    from racer.agents.loader import LoadedModel

    m = LoadedModel(FOLDER)
    summary = evaluate(m.env_config, lambda obs, env: m.act(obs), ["training_ground"], n_starts=6)
    assert summary["completion_rate"] == 1.0
    assert summary["best_lap"] == pytest.approx(42.70, abs=0.01)
