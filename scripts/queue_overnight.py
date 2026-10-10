"""Run a queue of training runs one after another, unattended (PROJECT_CONTEXT §8, D18).

    .venv\\Scripts\\python scripts\\queue_overnight.py configs\\queue\\night1.yaml
                                                   [--dry-run] [--retry-failed]

Start it yourself from a normal terminal: it needs no Claude session (so it uses no tokens).
Each entry runs ``scripts/train.py`` with the entry's arguments; train.py registers and
evaluates every checkpoint itself. The queue

* writes one log per run to ``outputs/queue_logs/<queue>/<name>.log`` and its progress to
  ``state.json`` next to them, so a stopped or crashed queue resumes where it left off
  (finished entries are skipped; failed ones are re-run only with ``--retry-failed``);
* keeps going when a run fails;
* keeps Windows awake only while it runs (``SetThreadExecutionState``; no power-plan
  changes). Keep the laptop plugged in; closing the lid may still sleep it.

Queue file::

    runs:
      - name: e2_nogrip_s1
        args: --tag e2_nogrip_s1 --trackset final_mix --physics no_grip_limit --seed 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "outputs" / "queue_logs"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this process runs (reset when it ends)."""
    if sys.platform != "win32":
        return
    import ctypes
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def load_queue(path: Path) -> list[dict]:
    import yaml
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    runs = data.get("runs") or []
    names = [r.get("name") for r in runs]
    if not runs or any(not n for n in names) or len(set(names)) != len(names):
        raise SystemExit(f"{path}: needs a 'runs' list whose entries have unique 'name's")
    for r in runs:
        r["argv"] = r["args"] if isinstance(r["args"], list) else shlex.split(str(r["args"]))
        r["argv"] = [str(a) for a in r["argv"]]
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queue", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--retry-failed", action="store_true", help="re-run entries that failed before")
    ap.add_argument("--script", default="scripts/train.py", help=argparse.SUPPRESS)   # tests only
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs = load_queue(args.queue)
    log_dir = LOG_ROOT / args.queue.stem
    state_path = log_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    def save_state() -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    todo = [r for r in runs if state.get(r["name"], {}).get("status") != "done"
            and (args.retry_failed or state.get(r["name"], {}).get("status") != "failed")]
    print(f"[queue] {args.queue.name}: {len(runs)} runs, {len(todo)} to do; logs in {log_dir}", flush=True)
    if args.dry_run:
        for r in todo:
            print(f"  {r['name']}: train.py {' '.join(r['argv'])}")
        return 0

    keep_awake(True)
    t_queue = time.perf_counter()
    try:
        for k, r in enumerate(todo, 1):
            log_path = log_dir / f"{r['name']}.log"
            log_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(PROJECT_ROOT / args.script), *r["argv"]]
            state[r["name"]] = {"status": "running", "started": now(), "log": str(log_path)}
            save_state()
            print(f"[queue] {now()} ({k}/{len(todo)}) {r['name']} ...", flush=True)
            t0 = time.perf_counter()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"=== {now()} {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    proc.wait()
                    state[r["name"]] = {"status": "interrupted", "finished": now()}
                    save_state()
                    print(f"[queue] interrupted during {r['name']}; run the queue again to resume", flush=True)
                    return 130
            minutes = round((time.perf_counter() - t0) / 60, 1)
            status = "done" if code == 0 else "failed"
            state[r["name"]].update(status=status, returncode=code, finished=now(), minutes=minutes)
            save_state()
            print(f"[queue] {now()} {r['name']}: {status.upper()} (exit {code}) after {minutes} min", flush=True)
    finally:
        keep_awake(False)

    lines = [f"queue {args.queue.name} finished {now()} after {(time.perf_counter() - t_queue) / 60:.0f} min"]
    for r in runs:
        s = state.get(r["name"], {})
        lines.append(f"  {r['name']:<28} {s.get('status', 'not run'):<12} {s.get('minutes', '-')} min")
    summary = "\n".join(lines)
    (log_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary, flush=True)
    return 0 if all(state.get(r["name"], {}).get("status") == "done" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
