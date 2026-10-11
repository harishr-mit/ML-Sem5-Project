"""Run a queue of training runs unattended, one or several at a time (PROJECT_CONTEXT §8, D18, T64).

    .venv\\Scripts\\python scripts\\queue_overnight.py configs\\queue\\night1.yaml
                                                   [--parallel 2] [--dry-run] [--retry-failed]

Start it yourself from a normal terminal: it needs no Claude session (so it uses no tokens).
Each entry runs ``scripts/train.py`` with the entry's arguments; train.py registers and
evaluates every checkpoint itself. The queue

* runs up to ``--parallel`` entries at once (or the queue file's ``parallel``; default 1).
  Measured (T62): 2 runs with 6 envs each give 1.62× the total throughput of one run;
  each run takes ~20 % longer on the clock. Runs in parallel share the evaluation workers;
* re-reads the queue file whenever a slot is free, so entries added while it runs are picked up;
* writes one log per run to ``outputs/queue_logs/<queue>/<name>.log`` and its progress to
  ``state.json`` next to them, so a stopped or crashed queue resumes where it left off
  (finished entries are skipped; failed ones are re-run only with ``--retry-failed``);
* keeps going when a run fails;
* keeps Windows awake only while it runs (``SetThreadExecutionState``; no power-plan
  changes). Keep the laptop plugged in; closing the lid may still sleep it.

Queue file::

    parallel: 2                 # optional
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "outputs" / "queue_logs"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
EVAL_WORKERS = 6    # evaluation processes shared by the runs in parallel (one run alone: train.py's default, n_envs)


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this process runs (reset when it ends)."""
    if sys.platform != "win32":
        return
    import ctypes
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def load_queue(path: Path) -> tuple[list[dict], int | None]:
    """(runs, parallel) from a queue file; ``parallel`` is optional (None = not set)."""
    import yaml
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - {"runs", "parallel"}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)} (allowed: runs, parallel)")
    runs = data.get("runs") or []
    names = [r.get("name") for r in runs]
    if not runs or any(not n for n in names) or len(set(names)) != len(names):
        raise SystemExit(f"{path}: needs a 'runs' list whose entries have unique 'name's")
    for r in runs:
        r["argv"] = r["args"] if isinstance(r["args"], list) else shlex.split(str(r["args"]))
        r["argv"] = [str(a) for a in r["argv"]]
    return runs, data.get("parallel")


def launch_argv(argv: list[str], n_slots: int) -> list[str]:
    """train.py arguments for one entry. With several runs at once, each run evaluates its
    checkpoints with a share of the evaluation workers (unless the entry sets its own)."""
    if n_slots > 1 and "--eval-workers" not in argv:
        return [*argv, "--eval-workers", str(max(EVAL_WORKERS // n_slots, 1))]
    return list(argv)


def pending(runs: list[dict], state: dict, retry_failed: bool, started: set[str]) -> list[dict]:
    """Entries still to run: not done, not failed (unless retrying), not started by this queue run."""
    return [r for r in runs if r["name"] not in started and state.get(r["name"], {}).get("status") != "done"
            and (retry_failed or state.get(r["name"], {}).get("status") != "failed")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queue", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--retry-failed", action="store_true", help="re-run entries that failed before")
    ap.add_argument("--parallel", type=int, default=None,
                    help="runs at the same time (default: the queue file's 'parallel', else 1; T62: 2 is fastest)")
    ap.add_argument("--script", default="scripts/train.py", help=argparse.SUPPRESS)   # tests only
    ap.add_argument("--poll", type=float, default=5.0, help=argparse.SUPPRESS)        # tests only
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    runs, parallel_file = load_queue(args.queue)
    n_slots = max(int(args.parallel or parallel_file or 1), 1)
    log_dir = LOG_ROOT / args.queue.stem
    state_path = log_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    def save_state() -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(state_path)

    started: set[str] = set()
    todo = pending(runs, state, args.retry_failed, started)
    print(f"[queue] {args.queue.name}: {len(runs)} runs, {len(todo)} to do, {n_slots} at a time; logs in {log_dir}",
          flush=True)
    if args.dry_run:
        for r in todo:
            print(f"  {r['name']}: train.py {' '.join(launch_argv(r['argv'], n_slots))}")
        return 0

    keep_awake(True)
    t_queue = time.perf_counter()
    running: dict[str, tuple[subprocess.Popen, float, Any]] = {}   # name → (process, start time, log file)

    def start(r: dict) -> None:
        log_path = log_dir / f"{r['name']}.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(PROJECT_ROOT / args.script), *launch_argv(r["argv"], n_slots)]
        state[r["name"]] = {"status": "running", "started": now(), "log": str(log_path)}
        save_state()
        print(f"[queue] {now()} start {r['name']} ({len(running) + 1}/{n_slots} slots)", flush=True)
        log = log_path.open("a", encoding="utf-8")
        log.write(f"=== {now()} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        running[r["name"]] = (proc, time.perf_counter(), log)
        started.add(r["name"])

    def finish(name: str, code: int) -> None:
        proc, t0, log = running.pop(name)
        log.close()
        minutes = round((time.perf_counter() - t0) / 60, 1)
        status = "done" if code == 0 else "failed"
        state[name].update(status=status, returncode=code, finished=now(), minutes=minutes)
        save_state()
        print(f"[queue] {now()} {name}: {status.upper()} (exit {code}) after {minutes} min", flush=True)

    try:
        while True:
            # Re-read the queue file whenever a slot is free, so entries added while the
            # queue runs (e.g. from the AI Sandbox) are picked up; removed entries are skipped.
            if len(running) < n_slots:
                try:
                    runs, _ = load_queue(args.queue)
                except SystemExit as e:          # a half-saved file: keep the last good list
                    print(f"[queue] could not re-read the queue file ({e}); using the previous list", flush=True)
                for r in pending(runs, state, args.retry_failed, started)[: n_slots - len(running)]:
                    start(r)
                    if len(running) < n_slots:
                        time.sleep(min(args.poll, 5.0))   # stagger start-up (imports, env workers)
            if not running:
                break
            time.sleep(args.poll)
            for name, (proc, _, _) in list(running.items()):
                code = proc.poll()
                if code is not None:
                    finish(name, code)
    except KeyboardInterrupt:
        for name, (proc, _, log) in list(running.items()):
            proc.terminate()
            proc.wait()
            log.close()
            state[name] = {"status": "interrupted", "finished": now()}
        save_state()
        print(f"[queue] interrupted ({', '.join(running) or 'nothing running'}); run the queue again to resume",
              flush=True)
        return 130
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
