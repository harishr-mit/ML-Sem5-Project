"""Parallel overnight queue (T64): slots, state, shared registry and run-id reservation."""
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from racer.config import PROJECT_ROOT
from racer.registry.registry import Registry

DUMMY = '''
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1]); name = sys.argv[2]
with out.open("a") as f: f.write(json.dumps({"name": name, "t": time.time(), "ev": "start", "argv": sys.argv[3:]}) + "\\n")
time.sleep(1.0)
with out.open("a") as f: f.write(json.dumps({"name": name, "t": time.time(), "ev": "end"}) + "\\n")
sys.exit(1 if name == "bad" else 0)
'''


def test_parallel_queue_runs_two_at_a_time(tmp_path):
    stem = f"pytest_queue_{uuid.uuid4().hex[:8]}"
    script = tmp_path / "dummy.py"
    script.write_text(DUMMY)
    events = tmp_path / "events.jsonl"
    queue = tmp_path / f"{stem}.yaml"
    queue.write_text("parallel: 2\nruns:\n" + "".join(
        f"  - name: {n}\n    args: ['{events.as_posix()}', '{n}']\n" for n in ("a", "bad", "c")))
    log_dir = PROJECT_ROOT / "outputs" / "queue_logs" / stem
    try:
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "queue_overnight.py"), str(queue),
                            "--script", str(script), "--poll", "0.1"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 1, r.stdout + r.stderr          # one entry failed
        state = json.loads((log_dir / "state.json").read_text())
        assert {k: v["status"] for k, v in state.items()} == {"a": "done", "bad": "failed", "c": "done"}
        ev = [json.loads(line) for line in events.read_text().splitlines()]
        live, peak = 0, 0
        for e in sorted(ev, key=lambda e: e["t"]):
            live += 1 if e["ev"] == "start" else -1
            peak = max(peak, live)
        assert peak == 2                                          # two at a time, never three
        starts = [e for e in ev if e["ev"] == "start"]
        assert all(e["argv"][-2:] == ["--eval-workers", "3"] for e in starts)   # 6 workers shared by 2 slots
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def _upsert_many(args):
    root, k = args
    reg = Registry(root)
    for i in range(25):
        reg.upsert_model({"model_id": f"P{k}_m{i}", "run_id": f"P{k}", "steps": i})
    return k


def test_registry_writes_from_parallel_processes_are_not_lost(tmp_path):
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_upsert_many, [(str(tmp_path), k) for k in range(4)]))
    ids = {r["model_id"] for r in Registry(tmp_path).models()}
    assert len(ids) == 100
    assert not (tmp_path / "registry.lock").exists()


def _reserve(args):
    out_dir, _ = args
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reserve_run_id("S", Path(out_dir), Path(out_dir))


def test_run_ids_are_unique_when_runs_start_together(tmp_path):
    with ProcessPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(_reserve, [(str(tmp_path), k) for k in range(8)]))
    assert sorted(ids) == [f"S{i:03d}" for i in range(1, 9)]
