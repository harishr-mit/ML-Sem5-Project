"""AI Sandbox: understand, compare, configure, queue and launch models (PROJECT_CONTEXT §10.5, D24).

    .venv\\Scripts\\python scripts\\sandbox.py [--port 8501] [--no-browser]

Run with plain Python, this file starts itself under ``streamlit run`` with every setting on
the command line (localhost only, headless, no usage statistics), so nothing is written to
the user profile, then opens the browser. Under Streamlit it is the app: it only routes to
the pages in ``racer/sandbox/pages/``. Stop it with Ctrl+C in its console; game windows and
queues it started keep running.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

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


def launch(port: int, open_browser: bool) -> int:
    url = f"http://localhost:{port}"
    if is_up(port):                                   # already running: just show it
        print(f"[sandbox] already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return 0
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *server_settings(port)]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    try:
        for _ in range(120):                          # wait up to ~60 s for the server
            if proc.poll() is not None:
                return proc.returncode
            if is_up(port):
                print(f"[sandbox] running at {url} (Ctrl+C to stop)", flush=True)
                if open_browser:
                    webbrowser.open(url)
                break
            time.sleep(0.5)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
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
        ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
        a = ap.parse_args()
        sys.exit(launch(a.port, not a.no_browser))
