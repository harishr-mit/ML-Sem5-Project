"""Process helpers for the sandbox (T70): is a process still the one we started, which
trainings and queues are running, and stopping a game window gracefully.

A process is identified by its PID **and** creation time, so a PID that Windows has
reused for another program is never mistaken for ours. On Windows a venv's ``python.exe``
starts the real interpreter as a child process, so stopping covers the whole tree.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Iterable

import psutil

TRAINING_SCRIPTS = ("train.py",)
QUEUE_SCRIPTS = ("queue_overnight.py",)


def created(pid: int) -> float | None:
    """Creation time of a process (seconds since the epoch), or None if it doesn't exist."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def alive(pid: int | None, created_at: float | None) -> bool:
    """True if ``pid`` is running and is the same process (same creation time)."""
    if not pid:
        return False
    try:
        p = psutil.Process(int(pid))
        if created_at is not None and abs(p.create_time() - float(created_at)) > 0.01:
            return False
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _tree(pid: int) -> list[psutil.Process]:
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def find_scripts(names: Iterable[str]) -> list[dict[str, Any]]:
    """Running Python processes whose command line runs one of the scripts ``names``.

    A venv launcher and the interpreter it starts have the same command line; only the
    outermost one is listed."""
    names = tuple(names)
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time", "ppid"]):
        try:
            if not str(p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = p.info["cmdline"] or []
            script = next((a for a in cmd[1:3] if str(a).replace("\\", "/").rsplit("/", 1)[-1] in names), None)
            if script is None:
                continue
            found.append({"pid": p.info["pid"], "ppid": p.info["ppid"], "created": p.info["create_time"],
                          "cmdline": cmd, "script": str(script)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    by_pid = {f["pid"]: f for f in found}
    return [f for f in found if not (f["ppid"] in by_pid and by_pid[f["ppid"]]["cmdline"][1:] == f["cmdline"][1:])]


def _close_windows(pids: set[int]) -> int:
    """Post WM_CLOSE to every top-level window of these processes (Windows); returns how many."""
    if sys.platform != "win32" or not pids:
        return 0
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    wm_close = 0x0010
    hits = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            hits.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    for hwnd in hits:
        user32.PostMessageW(hwnd, wm_close, 0, 0)
    return len(hits)


def stop(pid: int, created_at: float | None, grace_s: float = 3.0) -> str:
    """Stop a process we started: close its windows, wait ``grace_s``, then kill the tree.

    Returns 'not running', 'closed' or 'killed'."""
    if not alive(pid, created_at):
        return "not running"
    procs = _tree(int(pid))
    _close_windows({p.pid for p in procs})
    _, still = psutil.wait_procs(procs, timeout=grace_s)
    if not still:
        return "closed"
    for p in still:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(still, timeout=5)
    return "killed"


def wait_gone(pid: int, created_at: float | None, timeout_s: float = 10.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not alive(pid, created_at):
            return True
        time.sleep(0.1)
    return False
