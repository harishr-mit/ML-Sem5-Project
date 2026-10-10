"""Physics sanity tests (PROJECT_CONTEXT §5.5). Run after every physics change.

    .venv\\Scripts\\python scripts\\physics_sanity.py [--preset default] [--no-plots]

Prints a PASS/FAIL table, saves plots to outputs/plots/sanity/ and exits non-zero on failure.
Expected values are derived from the physics config (e.g. μ·g), so the tests follow presets.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from racer.config import PROJECT_ROOT, load_preset
from racer.physics.car import Car, PhysicsParams
from racer.physics.surfaces import GRASS_ID, ROAD_ID

OUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "sanity"

# Test tolerances (test criteria, not physics tunables).
SKIDPAD_RADIUS_M = 50.0
SKIDPAD_TOL = 0.10            # peak lateral acceleration within ±10 % of μ_static·g
SKIDPAD_LOST_M = 2.0          # radial error that counts as losing the line
BRAKE_V0 = 30.0
BRAKE_THRESHOLD_FRACTION = 0.98   # brake at 98 % of the static grip limit
BRAKE_TOL = 0.08              # stopping distance within ±8 % of v²/(2μg)
GRASS_TOPSPEED_MAX_RATIO = 0.90
GRASS_GRIP_MAX_RATIO = 0.55


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    series: dict = field(default_factory=dict)


def run_skidpad(p: PhysicsParams, surface: int) -> tuple[float, float | None, dict]:
    """Follow a circle (centre (0, R), counter-clockwise) while speed ramps up.

    Returns (peak smoothed lateral acceleration before losing the line, speed when lost, series).
    """
    car = Car(p)
    R = SKIDPAD_RADIUS_M
    car.reset(0.0, 0.0, 0.0, 5.0)
    dt = p.dt
    ramp = 0.25                       # m/s² target-speed ramp (slow, so drive force stays small)
    v_target = 5.0
    window = int(0.5 / dt)
    a_hist: list[float] = []
    speeds, alat_s = [], []
    peak, lost_speed = 0.0, None
    for _ in range(int(240 / dt)):
        dx, dy = car.x - 0.0, car.y - R
        dist = math.hypot(dx, dy)
        radial_err = dist - R
        if abs(radial_err) > SKIDPAD_LOST_M:
            lost_speed = car.speed
            break
        tangent = math.atan2(dy, dx) + math.pi / 2.0
        head_err = (car.theta - tangent + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan(p.wheelbase_m / R) + 0.08 * radial_err - 1.0 * head_err
        steer_cmd = delta / p.max_steer(car.speed)
        v_target += ramp * dt
        gas = max(min(0.5 * (v_target - car.speed), 1.0), -1.0)
        car.step(steer_cmd, gas, (surface,) * 4)
        a_hist.append(car.a_lat)
        if len(a_hist) >= window:
            a_mean = sum(a_hist[-window:]) / window
            peak = max(peak, a_mean)
            speeds.append(car.speed)
            alat_s.append(a_mean)
    return peak, lost_speed, {"speed": speeds, "a_lat": alat_s}


def test_skidpad(p: PhysicsParams) -> Result:
    sf = Car(p).surfaces
    expected = sf.road.mu_static * p.gravity
    peak, lost, series = run_skidpad(p, ROAD_ID)
    ok = lost is not None and abs(peak - expected) <= SKIDPAD_TOL * expected
    lost_txt = f"{lost:.1f} m/s" if lost is not None else "never"
    return Result("1 Skidpad (road)", ok,
                  f"peak a_lat {peak:.2f} m/s² vs μg {expected:.2f} (±{SKIDPAD_TOL:.0%}); "
                  f"lost the line at {lost_txt} (R = {SKIDPAD_RADIUS_M:.0f} m)", series)


def run_braking(p: PhysicsParams, gas: float) -> tuple[float, dict]:
    car = Car(p)
    car.reset(0.0, 0.0, 0.0, BRAKE_V0)
    xs, vs = [0.0], [BRAKE_V0]
    for _ in range(int(30 / p.dt)):
        car.step(0.0, gas, (ROAD_ID,) * 4)
        xs.append(car.x)
        vs.append(car.speed)
        if car.speed == 0.0:
            break
    return car.x, {"x": xs, "v": vs}


def test_braking(p: PhysicsParams) -> Result:
    sf = Car(p).surfaces
    mu_s, g = sf.road.mu_static, p.gravity
    expected = BRAKE_V0 ** 2 / (2 * mu_s * g)
    threshold_gas = -BRAKE_THRESHOLD_FRACTION * mu_s * p.mass_kg * g / p.brake_force_max
    d_thr, s_thr = run_braking(p, threshold_gas)
    d_full, s_full = run_braking(p, -1.0)
    locks = p.brake_force_max > mu_s * p.mass_kg * g
    ok = abs(d_thr - expected) <= BRAKE_TOL * expected
    ok = ok and ((d_full > d_thr) if locks else (d_full <= d_thr + 0.5))
    return Result("2 Braking 30→0 m/s", ok,
                  f"threshold ({-threshold_gas:.2f} brake) {d_thr:.1f} m vs v²/2μg {expected:.1f} m "
                  f"(±{BRAKE_TOL:.0%}); full brake {d_full:.1f} m "
                  f"({'locks: must be longer' if locks else 'no lock-up'})",
                  {"threshold": s_thr, "full": s_full})


def test_coastdown(p: PhysicsParams) -> Result:
    car = Car(p)
    car.reset(0.0, 0.0, 0.0, BRAKE_V0)
    ts, vs = [0.0], [BRAKE_V0]
    monotonic, reversed_ = True, False
    t = 0.0
    for _ in range(int(600 / p.dt)):
        prev = car.speed
        car.step(0.0, 0.0, (ROAD_ID,) * 4)
        t += p.dt
        ts.append(t)
        vs.append(car.speed)
        if car.speed > prev + 1e-9:
            monotonic = False
        if car.v_long < -1e-9:
            reversed_ = True
        if car.speed == 0.0:
            break
    stopped = car.speed == 0.0
    ok = monotonic and not reversed_ and stopped
    return Result("3 Coast-down", ok,
                  f"stopped after {t:.0f} s, {car.x:.0f} m; monotonic={monotonic}, reversed={reversed_}",
                  {"t": ts, "v": vs})


def test_grass(p: PhysicsParams) -> Result:
    def top_speed(surface: int) -> tuple[float, list[float]]:
        car = Car(p)
        vs = []
        for _ in range(int(90 / p.dt)):
            car.step(0.0, 1.0, (surface,) * 4)
            vs.append(car.speed)
        return car.speed, vs

    v_road, vs_road = top_speed(ROAD_ID)
    v_grass, vs_grass = top_speed(GRASS_ID)
    grip_road, _, _ = run_skidpad(p, ROAD_ID)
    grip_grass, _, _ = run_skidpad(p, GRASS_ID)
    ok = (v_grass <= GRASS_TOPSPEED_MAX_RATIO * v_road) and (grip_grass <= GRASS_GRIP_MAX_RATIO * grip_road)
    return Result("4 Grass vs road", ok,
                  f"top speed {v_grass:.1f} vs {v_road:.1f} m/s (≤{GRASS_TOPSPEED_MAX_RATIO:.0%}); "
                  f"cornering {grip_grass:.2f} vs {grip_road:.2f} m/s² (≤{GRASS_GRIP_MAX_RATIO:.0%})",
                  {"road": vs_road, "grass": vs_grass})


def test_stability(p: PhysicsParams) -> Result:
    rng = np.random.default_rng(0)
    problems = []
    # (a) at rest, full lock, gas flickering around zero: must stay put without NaN or jitter
    car = Car(p)
    car.reset(0.0, 0.0, 0.0, 0.0)
    for i in range(int(20 / p.dt)):
        gas = 0.02 * math.sin(i * 0.7) - 0.01
        steer = 1.0 if (i // 30) % 2 == 0 else -1.0
        car.step(steer, gas, (ROAD_ID,) * 4)
    vals = (car.x, car.y, car.theta, car.vx, car.vy)
    if not all(math.isfinite(v) for v in vals):
        problems.append("NaN at rest")
    if car.speed > 1.0 or math.hypot(car.x, car.y) > 20.0:
        problems.append(f"drifted at rest (speed {car.speed:.2f}, moved {math.hypot(car.x, car.y):.1f} m)")
    # (b) full lock from standstill, full throttle: tight circles, no NaN, no wild yaw jumps
    car.reset(0.0, 0.0, 0.0, 0.0)
    max_dtheta = 0.0
    for _ in range(int(20 / p.dt)):
        th = car.theta
        car.step(1.0, 1.0, (ROAD_ID,) * 4)
        d = abs((car.theta - th + math.pi) % (2 * math.pi) - math.pi)
        max_dtheta = max(max_dtheta, d)
    if not all(math.isfinite(v) for v in (car.x, car.y, car.theta, car.vx, car.vy)):
        problems.append("NaN at full lock")
    if max_dtheta > 0.2:
        problems.append(f"yaw jump {max_dtheta:.2f} rad/step")
    # (c) random action fuzz on random surfaces
    car.reset(0.0, 0.0, 0.0, 10.0)
    for _ in range(20000):
        a = rng.uniform(-1, 1, 2)
        car.step(float(a[0]), float(a[1]), tuple(int(s) for s in rng.integers(-1, 3, 4)))
    if not all(math.isfinite(v) for v in (car.x, car.y, car.theta, car.vx, car.vy)):
        problems.append("NaN in fuzz")
    return Result("5 Low-speed stability", not problems,
                  "; ".join(problems) if problems else
                  f"no NaN, rest drift ok, max yaw step at full lock {max_dtheta:.3f} rad")


def save_plots(results: list[Result], p: PhysicsParams) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by = {r.name[0]: r for r in results}
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    s = by["1"].series
    axs[0, 0].plot(s["speed"], s["a_lat"])
    axs[0, 0].axhline(Car(p).surfaces.road.mu_static * p.gravity, ls="--", c="grey", label="μ_static·g")
    axs[0, 0].set(title="Skidpad R=50 m (road)", xlabel="speed (m/s)", ylabel="lateral acc (m/s²)")
    axs[0, 0].legend()
    s = by["2"].series
    axs[0, 1].plot(s["threshold"]["x"], s["threshold"]["v"], label="threshold braking")
    axs[0, 1].plot(s["full"]["x"], s["full"]["v"], label="full brake (locked)")
    axs[0, 1].set(title="Braking from 30 m/s", xlabel="distance (m)", ylabel="speed (m/s)")
    axs[0, 1].legend()
    s = by["3"].series
    axs[1, 0].plot(s["t"], s["v"])
    axs[1, 0].set(title="Coast-down (road)", xlabel="time (s)", ylabel="speed (m/s)")
    s = by["4"].series
    t = np.arange(len(s["road"])) * p.dt
    axs[1, 1].plot(t, s["road"], label="road")
    axs[1, 1].plot(t, s["grass"], label="grass")
    axs[1, 1].set(title="Full throttle", xlabel="time (s)", ylabel="speed (m/s)")
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "physics_sanity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="default", help="configs/physics/<preset>.yaml")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = PhysicsParams.from_config(load_preset("physics", args.preset))

    results = [test_skidpad(p), test_braking(p), test_coastdown(p), test_grass(p), test_stability(p)]
    width = max(len(r.name) for r in results)
    print(f"Physics sanity — preset '{args.preset}'")
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  {r.detail}")
    if not args.no_plots:
        print(f"  plots: {save_plots(results, p).relative_to(PROJECT_ROOT)}")
    n_fail = sum(not r.passed for r in results)
    print("ALL PASSED" if n_fail == 0 else f"{n_fail} FAILED")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
