"""Evaluation protocol v2 (D20): per-group metrics, lap tracks, newest protocol wins in the registry."""
from racer.agents.evaluation import compute_metrics, lap_tracks, load_protocol, protocol_version
from racer.registry.registry import Registry, metrics_row, newer_or_same_protocol


def _ep(group, track, completed, laps=()):
    return {"group": group, "track": track, "completed": completed, "termination": None if completed else "barrier",
            "lap_times": list(laps), "mean_speed": 20.0, "offroad_pct": 0.0, "mean_lat_g": 0.5}


def test_v2_protocol_shape():
    p = load_protocol("v2")
    assert protocol_version(p) == 2 and protocol_version("v2") == 2 and protocol_version("v1") == 1
    assert len(p["groups"]["train"]["tracks"]) == 7 and len(p["groups"]["test"]["tracks"]) == 2
    assert lap_tracks(p) == p["groups"]["train"]["tracks"] + p["groups"]["test"]["tracks"]
    assert lap_tracks(load_protocol("v1")) == ["training_ground", "technical", "fast_flow"]


def test_v2_metrics_per_group():
    p = load_protocol("v2")
    eps = [_ep("train", "esses", True, [60.0]), _ep("train", "esses", False),
           _ep("test", "alpine", True, [80.0]), _ep("test", "alpine", False),
           _ep("heldout", "heldout_00", True), _ep("heldout", "heldout_01", True)]
    m = compute_metrics(eps, p["groups"], {"esses": 50.0, "alpine": 80.0}, p)
    assert m["completion_rate"] == 0.5 and m["test_completion"] == 0.5 and m["heldout_completion"] == 1.0
    assert m["unseen_completion"] == 0.75 and abs(m["overall_completion"] - 4 / 6) < 1e-4
    assert m["best_lap_esses"] == 60.0 and m["best_lap_coastline"] is None
    assert m["lap_ratio"] == 1.1


def test_newest_protocol_wins(tmp_path):
    reg = Registry(tmp_path)
    reg.upsert_run({"run_id": "R001"})
    reg.upsert_model({"model_id": "R001_x@100k", "run_id": "R001", "steps": 100000})
    v2 = {"model_id": "R001_x@100k", "protocol": "v2", "evaluated_at": "t2",
          "metrics": {"overall_completion": 0.5, "unseen_completion": 0.4}}
    v1 = {"model_id": "R001_x@100k", "protocol": "v1", "evaluated_at": "t1", "metrics": {"overall_completion": 0.9}}
    reg.upsert_model(metrics_row(v2))
    assert not newer_or_same_protocol(reg.model("R001_x@100k"), v1)
    assert newer_or_same_protocol(reg.model("R001_x@100k"), v2)
    row = reg.model("R001_x@100k")
    assert row["eval_protocol"] == "v2" and row["unseen_completion"] == "0.4" and row["completion_rate"] == ""
    assert reg.set_best("R001") == "R001_x@100k"
