"""The simulation never imports pygame, and the debug window runs headless."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_simulation_is_pygame_free():
    code = ("import sys, racer.physics.car, racer.track.track, racer.track.generator, "
            "racer.track.progress, racer.env.racing_env; sys.exit(1 if 'pygame' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_drive_headless_screenshot(tmp_path):
    out = tmp_path / "shot.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "drive.py"), "--mode", "baseline",
                        "--screenshot", str(out), "--frames", "90"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000


def test_drive_human_mode_runs_past_terminations(tmp_path):
    """Human mode with no input: the car would be 'stuck' after 3 s; free drive must keep running."""
    out = tmp_path / "human.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "drive.py"), "--mode", "human",
                        "--screenshot", str(out), "--frames", "600"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
