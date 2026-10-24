"""Checkpoint selection for the Evolution Race (T126)."""
import pytest

from racer.agents.checkpoints import STRATEGIES, choose, run_checkpoints

ROWS = [{"model_id": f"R001_x@{s}k", "run_id": "R001", "steps": s * 1000, "path": f"models/{s}",
         "score": sc, "overall_completion": sc / 10}
        for s, sc in ((300, 5.0), (100, 1.0), (500, 9.0), (200, 3.0), (400, 2.0))]
ROWS += [{"model_id": "R002_y@100k", "run_id": "R002", "steps": 100000, "path": "models/y"},
         {"model_id": "R001_x@600k", "run_id": "R001", "steps": 600000, "path": None}]   # no folder → not raceable


def ids(strategy, n):
    return [r["model_id"] for r in choose(run_checkpoints(ROWS, "R001"), strategy, n)]


def test_run_checkpoints_are_sorted_and_need_a_folder():
    assert [r["steps"] for r in run_checkpoints(ROWS, "R001")] == [100000, 200000, 300000, 400000, 500000]
    assert run_checkpoints(ROWS, "R999") == []


def test_every_nth_starts_at_the_first_checkpoint():
    assert ids("every", 2) == ["R001_x@100k", "R001_x@300k", "R001_x@500k"]
    assert ids("every", 1)[0] == "R001_x@100k" and len(ids("every", 1)) == 5


def test_spaced_keeps_first_and_last_and_never_repeats():
    assert ids("spaced", 3) == ["R001_x@100k", "R001_x@300k", "R001_x@500k"]
    assert ids("spaced", 2) == ["R001_x@100k", "R001_x@500k"]
    assert ids("spaced", 99) == ids("every", 1)               # more than exist: all of them
    assert ids("spaced", 1) == ["R001_x@500k"]                # one checkpoint: the final one


def test_top_picks_the_best_scores_in_training_order():
    assert ids("top", 2) == ["R001_x@300k", "R001_x@500k"]


def test_unknown_strategy_is_refused():
    assert set(STRATEGIES) == {"every", "spaced", "top"}
    with pytest.raises(ValueError):
        choose(run_checkpoints(ROWS, "R001"), "random", 3)
