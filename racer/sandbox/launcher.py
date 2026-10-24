"""Start, list and stop game windows (``scripts/demo.py``) from the sandbox (T70).

Each Launch starts a separate process, so several windows can race different models at
once, and closing the sandbox never closes a game. The process has no console; its output
goes to ``outputs/sandbox/logs/<instance>.log``. Instances are remembered in
``outputs/sandbox/instances.json`` (PID + creation time), so the list survives a sandbox
restart; stopped and finished ones are pruned.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from racer.config import PROJECT_ROOT, load_preset
from racer.sandbox import procs

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NEW_CONSOLE = 0x00000010


def settings() -> dict[str, Any]:
    return load_preset("sandbox", "default")


def demo_argv(track: str, models: Sequence[str] = (), run: str | None = None, every: int = 1, laps: int | None = None,
              baseline: bool = False, fair: bool = False, title: str | None = None,
              extra: Sequence[str] = ()) -> list[str]:
    """``demo.py`` arguments for one race (``models`` or a ``run``'s checkpoints)."""
    if models and run:
        raise ValueError("give models or a run, not both")
    if not (models or run or baseline):
        raise ValueError("nothing to race: choose models, a run or the baseline")
    argv = ["--track", track]
    if models:
        argv += ["--models", *models]
    if run:
        argv += ["--run", run, "--every", str(max(int(every), 1))]
    if laps:
        argv += ["--laps", str(int(laps))]
    if baseline:
        argv.append("--baseline")
    if fair:
        argv.append("--fair")
    if title:
        argv += ["--title", title]
    return argv + list(extra)


class Launcher:
    def __init__(self, root: Path | str = PROJECT_ROOT, python: str | None = None,
                 script: str = "scripts/demo.py") -> None:
        self.root = Path(root)
        self.dir = self.root / "outputs" / "sandbox"
        self.path = self.dir / "instances.json"
        self.python = python or sys.executable
        self.script = script

    # ----------------------------------------------------------- state file
    def _load(self) -> list[dict[str, Any]]:
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def instances(self) -> list[dict[str, Any]]:
        """Every remembered instance with ``running`` refreshed; keeps the newest finished ones."""
        items = self._load()
        for it in items:
            was = it.get("running", True)
            it["running"] = procs.alive(it["pid"], it.get("created"))
            if was and not it["running"] and not it.get("ended"):
                it["ended"] = _now()
        keep = int(settings()["launch"]["keep_finished"])
        done = [it for it in items if not it["running"]][-keep:]
        items = [it for it in items if it["running"] or it in done]
        self._save(items)
        return items

    def running(self) -> list[dict[str, Any]]:
        return [it for it in self.instances() if it["running"]]

    # ----------------------------------------------------------- start / stop
    def launch(self, track: str, models: Sequence[str] = (), run: str | None = None, every: int = 1,
               laps: int | None = None, baseline: bool = False, fair: bool = False,
               label: str | None = None, extra: Sequence[str] = ()) -> dict[str, Any]:
        """Open a game window racing ``models`` (or a run's checkpoints) on ``track``."""
        cfg = settings()["launch"]
        items = self.instances()
        n = max([int(it["id"]) for it in items], default=0) + 1
        title = f"AI Sandbox #{n} · {track} · {label or _describe(models, run, baseline)}"
        argv = demo_argv(track, models, run, every, laps, baseline, fair, title, extra)
        log = self.dir / "logs" / f"game_{n:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        slot = len([it for it in items if it["running"]]) % int(cfg["cascade_slots"])
        x0, y0 = (int(v) for v in cfg["window_origin_px"])
        step = int(cfg["window_cascade_px"])
        env = dict(os.environ, SDL_VIDEO_WINDOW_POS=f"{x0 + slot * step},{y0 + slot * step}")
        cmd = [self.python, str(self.root / self.script), *argv]
        flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
        with log.open("w", encoding="utf-8") as f:
            f.write(f"=== {_now()} {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=self.root, stdout=f, stderr=subprocess.STDOUT, env=env,
                                    creationflags=flags, close_fds=True)
        item = {"id": n, "pid": proc.pid, "created": procs.created(proc.pid), "title": title, "track": track,
                "models": list(models), "run": run, "every": every, "laps": laps, "baseline": baseline,
                "fair": fair, "started": _now(), "log": str(log), "argv": argv, "running": True}
        self._save(self._load() + [item])
        return item

    def stop(self, instance_id: int) -> str:
        items = self._load()
        it = next((i for i in items if int(i["id"]) == int(instance_id)), None)
        if it is None:
            return "unknown"
        result = procs.stop(int(it["pid"]), it.get("created"), float(settings()["launch"]["stop_grace_s"]))
        it["running"] = False
        it["ended"] = _now()
        self._save(items)
        return result

    def stop_all(self) -> int:
        n = 0
        for it in self.running():
            self.stop(it["id"])
            n += 1
        return n

    def forget_finished(self) -> None:
        self._save([it for it in self.instances() if it["running"]])

    @staticmethod
    def log_tail(item: dict[str, Any], lines: int = 20) -> str:
        try:
            return "\n".join(Path(item["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


def _describe(models: Sequence[str], run: str | None, baseline: bool) -> str:
    if run:
        return f"evolution of {run}"
    shown = [m.split("_", 1)[0] + "@" + m.rpartition("@")[2] if "@" in m else m for m in models[:3]]
    extra = f" +{len(models) - 3}" if len(models) > 3 else ""
    return ", ".join(shown) + extra + (" + baseline" if baseline else "")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
