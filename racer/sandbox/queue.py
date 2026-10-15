"""The sandbox's training queue (T71): ``configs/queue/sandbox.yaml`` + ``queue_overnight.py``.

The queue file is git-ignored (it changes all the time and must not make the tree dirty,
D12). Entries are written with their arguments as a YAML list, atomically; the runner
re-reads the file whenever a slot frees (T64), so entries can be added or removed while it
runs. *Start queue* opens the runner in its own console window (owner, T65), so it keeps
running after the sandbox closes; *Stop queue* asks it to stop like Ctrl+C (owner, T66).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from racer.config import PROJECT_ROOT, load_preset
from racer.sandbox import procs

CREATE_NEW_CONSOLE = 0x00000010
HEADER = """\
# Training queue written by the AI Sandbox (Build & Queue page). Git-ignored.
# Run it with:  .venv\\Scripts\\python scripts\\queue_overnight.py configs\\queue\\sandbox.yaml
# (or press "Start queue" in the sandbox). Entries can be added while it runs.
"""


class QueueControl:
    def __init__(self, root: Path | str = PROJECT_ROOT, file: str | None = None,
                 script: str = "scripts/queue_overnight.py", python: str | None = None) -> None:
        self.root = Path(root)
        cfg = load_preset("sandbox", "default")["queue"]
        self.path = self.root / (file or cfg["file"])
        self.default_parallel = int(cfg["parallel"])
        self.log_dir = self.root / "outputs" / "queue_logs" / self.path.stem
        self.script = script
        self.python = python or sys.executable

    # ----------------------------------------------------------- the file
    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"parallel": self.default_parallel, "runs": []}
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return {"parallel": int(data.get("parallel") or self.default_parallel), "runs": list(data.get("runs") or [])}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump({"parallel": int(data["parallel"]),
                               "runs": [{"name": r["name"], "args": [str(a) for a in r["args"]]} for r in data["runs"]]},
                              sort_keys=False, allow_unicode=True, width=10_000)
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(HEADER + body, encoding="utf-8")
        os.replace(tmp, self.path)

    def names(self) -> set[str]:
        return {r["name"] for r in self.load()["runs"]}

    def add(self, entries: Sequence[dict[str, Any]]) -> None:
        data = self.load()
        clash = self.names() & {e["name"] for e in entries}
        if clash:
            raise ValueError(f"already in the queue: {sorted(clash)}")
        data["runs"] += [{"name": e["name"], "args": list(e["args"])} for e in entries]
        self._save(data)

    def remove(self, name: str) -> None:
        """Remove an entry that hasn't started (a running or finished one stays)."""
        st = self.state().get(name, {})
        if st.get("status") in ("running", "done"):
            raise ValueError(f"{name} is {st['status']}; only waiting entries can be removed")
        data = self.load()
        data["runs"] = [r for r in data["runs"] if r["name"] != name]
        self._save(data)

    def set_parallel(self, n: int) -> None:
        data = self.load()
        data["parallel"] = max(int(n), 1)
        self._save(data)

    # ----------------------------------------------------------- the runner
    def state(self) -> dict[str, dict[str, Any]]:
        try:
            return _read_json(self.log_dir / "state.json")
        except (FileNotFoundError, ValueError):
            return {}

    def runner(self) -> dict[str, Any] | None:
        """The runner working on this queue file, if it is alive."""
        try:
            info = _read_json(self.log_dir / "runner.json")
        except (FileNotFoundError, ValueError):
            return None
        try:
            import psutil
            p = psutil.Process(int(info["pid"]))
            cmd = " ".join(p.cmdline())
            if "queue_overnight.py" in cmd and p.is_running():
                return info
        except Exception:      # gone, recycled or not readable
            return None
        return None

    def entries(self) -> list[dict[str, Any]]:
        """Each entry with its status: waiting / running / done / failed / interrupted."""
        st = self.state()
        live = self.runner() is not None
        out = []
        for r in self.load()["runs"]:
            s = dict(st.get(r["name"], {}))
            status = s.get("status", "waiting")
            if status == "running" and not live:
                status = "interrupted"                 # the runner died without updating state.json
            out.append({"name": r["name"], "args": [str(a) for a in r["args"]], "status": status,
                        "steps": steps_of(r["args"]), "started": s.get("started"), "finished": s.get("finished"),
                        "minutes": s.get("minutes"), "log": s.get("log"), "scratch": "--scratch" in r["args"]})
        return out

    def busy(self) -> list[str]:
        """Why a new queue shouldn't start now (trainings or queues already running)."""
        reasons = []
        for p in procs.find_scripts(procs.QUEUE_SCRIPTS):
            q = next((a for a in p["cmdline"][2:] if str(a).endswith((".yaml", ".yml"))), "?")
            reasons.append(f"a queue is running ({Path(q).name}, PID {p['pid']})")
        for p in procs.find_scripts(procs.TRAINING_SCRIPTS):
            tag = _arg(p["cmdline"], "--tag") or "?"
            reasons.append(f"training {tag} is running (PID {p['pid']})")
        return reasons

    def start(self) -> int:
        """Open the runner on this queue file in its own console window; returns its PID."""
        if not any(e["status"] in ("waiting", "interrupted") for e in self.entries()):
            raise RuntimeError("nothing to run: add an entry first")
        reasons = self.busy()
        if reasons:
            raise RuntimeError("; ".join(reasons))
        (self.log_dir / "stop.request").unlink(missing_ok=True)
        flags = CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        proc = subprocess.Popen([self.python, str(self.root / self.script), str(self.path)], cwd=self.root,
                                creationflags=flags)
        return proc.pid

    def request_stop(self) -> bool:
        """Ask the running runner to stop (like Ctrl+C); False if it isn't running."""
        if self.runner() is None:
            return False
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "stop.request").write_text("stop requested by the AI Sandbox\n", encoding="utf-8")
        return True

    def log_tail(self, name: str, lines: int = 15) -> str:
        path = self.log_dir / f"{name}.log"
        try:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def _read_json(path: Path, attempts: int = 10) -> Any:
    """Read a JSON file the queue runner replaces atomically; on Windows a read that meets the
    replace fails with PermissionError for a moment, so retry briefly."""
    import time
    for i in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05)


def steps_of(args: Sequence[Any]) -> int | None:
    v = _arg([str(a) for a in args], "--steps")
    return int(v) if v and re.fullmatch(r"\d+", v) else None


def _arg(argv: Sequence[Any], flag: str) -> str | None:
    argv = [str(a) for a in argv]
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def command_line(args: Sequence[Any]) -> str:
    """The train.py command an entry runs, for display."""
    return ".venv\\Scripts\\python scripts\\train.py " + subprocess.list2cmdline([str(a) for a in args])
