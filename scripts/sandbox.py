"""AI Sandbox: understand, compare, configure, queue and launch models (PROJECT_CONTEXT §10.5, D24).

    .venv\Scripts\python scripts\sandbox.py [--port 8501] [--browser | --no-browser]

Run with plain Python, this file starts itself under ``streamlit run`` with every setting on
the command line (localhost only, headless, no usage statistics), so nothing is written to
the user profile, then shows it in its own window (D25: pywebview on Edge WebView2, with its
data folder in the git-ignored ``outputs/sandbox/webview/``). Closing the window stops the
server. ``--browser`` opens a browser tab instead, ``--no-browser`` only runs the server; then
stop it with Ctrl+C in its console. Under Streamlit it is the app: it only routes to the pages
in ``racer/sandbox/pages/``. Game windows and queues it started keep running either way.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8501


def under_streamlit() -> bool:
    """True when this file is being executed by a Streamlit server."""
    try:
        from streamlit.runtime import exists
    except ImportError:
        return False
    return exists()


def server_settings(port: int) -> list[str]:
    """Streamlit options, all on the command line (owner: localhost only, no telemetry)."""
    return ["--server.address", "localhost", "--server.port", str(port), "--server.headless", "true",
            "--browser.gatherUsageStats", "false", "--browser.serverAddress", "localhost",
            "--global.developmentMode", "false", "--server.fileWatcherType", "none"]


def is_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def window_settings() -> dict[str, Any]:
    from racer.sandbox.launcher import settings
    return settings()["window"]


def open_window(url: str, cfg: dict[str, Any]) -> bool:
    """Show ``url`` in a native window and block until it is closed (D25).

    WebView2 runs in private mode with its data folder inside the project, so nothing goes to
    the user profile. Returns False if no window could be opened (the caller falls back to
    the browser)."""
    try:
        import webview
    except ImportError as e:
        print(f"[sandbox] no window ({e}); opening the browser instead", flush=True)
        return False
    storage = PROJECT_ROOT / cfg["storage_dir"]
    storage.mkdir(parents=True, exist_ok=True)
    width, height = cfg["size_px"]
    try:
        webview.create_window(cfg["title"], url, width=width, height=height, min_size=tuple(cfg["min_size_px"]))
        webview.start(private_mode=True, storage_path=str(storage))
    except Exception as e:                            # e.g. WebView2 runtime missing
        print(f"[sandbox] the window failed ({e}); opening the browser instead", flush=True)
        return False
    return True


def show(url: str, view: str, cfg: dict[str, Any]) -> bool:
    """Show the app; True if this blocked until the user closed a window."""
    if view == "window" and open_window(url, cfg):
        return True
    if view != "none":
        webbrowser.open(url)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Stop the server's whole tree: the venv's python.exe is a launcher with the real server as its child."""
    from racer.sandbox import procs
    procs.stop(proc.pid, procs.created(proc.pid), grace_s=0)


def launch(port: int, view: str = "window") -> int:
    url = f"http://localhost:{port}"
    cfg = window_settings()
    if is_up(port):                                   # already running: just show it
        print(f"[sandbox] already running at {url}")
        show(url, view, cfg)
        return 0
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *server_settings(port)]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    try:
        deadline = time.monotonic() + float(cfg["server_wait_s"])
        while not is_up(port):                        # wait for the server
            if proc.poll() is not None:
                return proc.returncode
            if time.monotonic() > deadline:
                print(f"[sandbox] the server didn't start within {cfg['server_wait_s']} s", flush=True)
                stop_server(proc)
                return 1
            time.sleep(0.5)
        print(f"[sandbox] running at {url}", flush=True)
        if show(url, view, cfg):                      # the window was closed: stop the server
            print("[sandbox] window closed, stopping the server", flush=True)
            stop_server(proc)
            return 0
        print("[sandbox] Ctrl+C to stop", flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        stop_server(proc)
        return 0


def app() -> None:
    from racer.sandbox.pages import run_app
    run_app()


if __name__ == "__main__":
    if under_streamlit():
        app()
    else:
        ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        ap.add_argument("--port", type=int, default=DEFAULT_PORT)
        view = ap.add_mutually_exclusive_group()
        view.add_argument("--browser", action="store_const", const="browser", dest="view",
                          help="open a browser tab instead of the sandbox window")
        view.add_argument("--no-browser", action="store_const", const="none", dest="view",
                          help="only run the server (stop it with Ctrl+C)")
        a = ap.parse_args()
        sys.exit(launch(a.port, a.view or "window"))
