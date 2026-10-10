"""Debug driving window: drive with the keyboard, watch the baseline, or play a trained model.

    .venv\\Scripts\\python scripts\\drive.py [--track training_ground] [--mode human|baseline|model]
                                           [--model models/<model_id>] [--seed 0]
                                           [--screenshot out.png --frames 120]   # headless capture

Keys: arrows / WASD drive · 1 human · 2 baseline · 3 model · R rays · T trail · F follow/overview
      +/− or wheel zoom · [ ] sim speed · Space pause · Enter reset at start · Backspace random reset
      N next track · H help · Esc quit

The window runs the same RacingEnv as training (physics, terminations, reward), so what you
see is what the agent experiences. With --model the env uses that model's config snapshot.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

from racer.agents.baseline import BaselineController
from racer.config import PROJECT_ROOT, load_preset
from racer.env.racing_env import RacingEnv, make_env_config
from racer.env.sensors import cast_rays
from racer.track.io import TRACKS_DIR

HELP = ["arrows/WASD drive", "1 human", "2 baseline", "3 model", "R rays", "T trail", "F overview",
        "+/− or wheel zoom", "[ ] sim speed", "Space pause", "Enter reset at start", "Backspace random start",
        "N next track", "H hide help", "Esc quit"]


def showcase_names() -> list[str]:
    return [p.stem for p in sorted((TRACKS_DIR / "showcase").glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", default="training_ground")
    ap.add_argument("--mode", choices=["human", "baseline", "model"], default=None)
    ap.add_argument("--model", default=None, help="model folder (models/<model_id>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screenshot", default=None, help="run headless and save a PNG after --frames frames")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sim-speed", type=float, default=1.0, help="initial sim speed (one of render.yaml sim_speeds)")
    ap.add_argument("--trace", default=None, help="write one CSV row per env step (debugging determinism)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame   # after SDL_VIDEODRIVER is set
    from racer.render.debug_view import SURFACE_NAMES, DebugView

    model = None
    if args.model:
        from racer.agents.loader import LoadedModel
        model = LoadedModel(Path(args.model))
        config = dict(model.env_config)
        config["tracks"] = [args.track]
    else:
        config = make_env_config([args.track])
    mode = args.mode or ("model" if model else "human")
    if mode == "model" and model is None:
        ap.error("--mode model needs --model")

    rcfg = load_preset("demo", "render")["debug"]
    pygame.init()
    size = tuple(rcfg["window"])
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("RL Racing Line — debug drive")
    clock = pygame.time.Clock()
    view = DebugView(size, rcfg)

    env = RacingEnv(config)
    env.record_substeps = True
    env.free_drive = mode == "human"   # human: solid walls, no terminations; AI modes: as in training
    baseline = BaselineController(load_preset("train", "baseline"))
    tracks = showcase_names()
    if args.track not in tracks:
        tracks.insert(0, args.track)
    track_i = tracks.index(args.track)
    rng = np.random.default_rng(args.seed)
    speeds = list(rcfg["sim_speeds"])
    speed_i = speeds.index(args.sim_speed)
    trace = open(args.trace, "w", encoding="utf-8") if args.trace else None
    if trace:
        trace.write("episode,step,x,y,theta,speed,steer,gas\n")
    episode = [0]

    state = {"obs": None, "ep_reward": 0.0, "last_r": 0.0, "info": {}, "action": np.zeros(2, np.float32)}
    poses: deque = deque()
    trail: deque = deque(maxlen=int(rcfg["trail_length"]))
    show_rays, show_trail, show_help, paused = True, True, True, False   # help shown until H
    end_reason = None          # termination waiting to be shown
    banner_until = 0.0
    acc = 0.0

    def reset(**opts) -> None:
        nonlocal end_reason
        obs, info = env.reset(seed=int(rng.integers(1 << 31)), options={"track": tracks[track_i], **opts})
        episode[0] += 1
        state.update(obs=obs, ep_reward=0.0, last_r=0.0, info=info, action=np.zeros(2, np.float32))
        poses.clear()
        poses.append((env.car.x, env.car.y, env.car.theta))
        trail.append(None)
        end_reason = None
        if view.track is not env.track:
            view.set_track(env.track)
        view.snap_camera(env.car.x, env.car.y)

    def reset_start() -> None:
        reset(start_s=0.0, lateral=0.0, heading_offset=0.0, speed=0.0)

    def respawn_here() -> None:
        """Human mode: put the car back on the centreline where it left the track."""
        s = env.tracker.s if env.tracker else 0.0
        reset(start_s=s, lateral=0.0, heading_offset=0.0, speed=0.0)

    reset_start()

    def choose_action() -> np.ndarray:
        if mode == "baseline":
            return baseline.act(env)
        if mode == "model":
            return model.act(state["obs"])
        keys = pygame.key.get_pressed()
        steer = (1.0 if keys[pygame.K_LEFT] or keys[pygame.K_a] else 0.0) - (1.0 if keys[pygame.K_RIGHT] or keys[pygame.K_d] else 0.0)
        gas = (1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0)
        return np.array([steer, gas], dtype=np.float32)

    frame = 0
    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEWHEEL:
                view.zoom_by(rcfg["zoom_step"] ** ev.y)
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_1:
                    mode = "human"
                elif k == pygame.K_2:
                    mode = "baseline"
                elif k == pygame.K_3 and model is not None:
                    mode = "model"
                elif k == pygame.K_r:
                    show_rays = not show_rays
                elif k == pygame.K_t:
                    show_trail = not show_trail
                elif k == pygame.K_f:
                    view.overview = not view.overview
                elif k == pygame.K_h:
                    show_help = not show_help
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    view.zoom_by(rcfg["zoom_step"])
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view.zoom_by(1.0 / rcfg["zoom_step"])
                elif k == pygame.K_RIGHTBRACKET:
                    speed_i = min(speed_i + 1, len(speeds) - 1)
                elif k == pygame.K_LEFTBRACKET:
                    speed_i = max(speed_i - 1, 0)
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RETURN:
                    reset_start()
                elif k == pygame.K_BACKSPACE:
                    reset()
                elif k == pygame.K_n:
                    track_i = (track_i + 1) % len(tracks)
                    trail.clear()
                    reset_start()

        env.free_drive = mode == "human"   # follows mode switches (1 / 2 / 3)

        # ---- advance the simulation by whole physics substeps (60 Hz × sim speed)
        if not paused and now >= banner_until:
            if end_reason is not None and len(poses) <= 1:
                if mode == "human":
                    respawn_here()
                else:
                    reset()
            acc += speeds[speed_i]
            n = int(acc)
            acc -= n
            for _ in range(n):
                if len(poses) <= 1 and end_reason is None:
                    if len(poses) == 1:
                        poses.popleft()
                    action = choose_action()
                    obs, r, term, trunc, info = env.step(action)
                    state.update(obs=obs, last_r=r, info=info, action=action)
                    state["ep_reward"] += r
                    poses.extend(env.substep_poses)
                    if trace:
                        trace.write(f"{episode[0]},{env.steps},{env.car.x:.6f},{env.car.y:.6f},{env.car.theta:.6f},"
                                    f"{env.car.speed:.6f},{action[0]:.6f},{action[1]:.6f}\n")
                    if term or trunc:
                        end_reason = info["termination"] or ("lap limit" if info["laps"] >= env.ep.max_laps else "step limit")
                if len(poses) > 1 or (end_reason is not None and poses):
                    p = poses.popleft()
                    trail.append((p[0], p[1], env.car.speed))
                    if not poses and end_reason is not None:
                        poses.append(p)
                        banner_until = now + rcfg["banner_s"]
                        break

        # ---- draw
        x, y, th = poses[0] if poses else (env.car.x, env.car.y, env.car.theta)
        car = env.car
        view.update_camera(x, y, car.vx, car.vy, 0.0 if paused else speeds[speed_i])
        view.draw_world(screen)
        if show_trail:
            view.draw_trail(screen, trail)
        if show_rays:
            ob = env.observer
            dists = np.empty(ob.n_rays)
            f = env.track.field
            cast_rays(f.dist, f.idx, f.geom, x, y, th, ob.angles, ob.ranges, ob.raycast_params, dists)
            view.draw_rays(screen, x, y, th, ob.angles, dists, ob.ranges)
        view.draw_car(screen, car, x, y, th, braking=car.brake > 0.05)

        info = state["info"]
        c = view.c
        cur_lap = env.time - env._last_cross_t if env._last_cross_t is not None else None
        last = env.lap_times[-1] if env.lap_times else None
        best = min(env.lap_times) if env.lap_times else None
        fmt = lambda v: f"{v:6.2f}s" if v is not None else "   -   "
        wheels = " ".join(SURFACE_NAMES[int(s)] for s in env._wheels)
        mode_txt = mode if mode != "model" else f"model {model.model_id}"
        lines = [
            (f"{env.track.name}  ·  {mode_txt}  ·  sim ×{speeds[speed_i]:g}{'  PAUSED' if paused else ''}", None),
            (f"speed {car.speed * 3.6:6.1f} km/h   lat {car.a_lat / 9.81:+5.2f} g   long {car.a_long / 9.81:+5.2f} g", None),
            (f"lap {env.laps}   s {env.tracker.s:6.0f}/{env.track.length:.0f} m   offset {env.tracker.lateral:+5.1f} m", None),
            (f"lap time {fmt(cur_lap)}   last {fmt(last)}   best {fmt(best)}", None),
            (f"wheels {wheels}", c["hud_bad"] if "grass" in wheels or "out" in wheels else None),
            (f"grip {car.grip_scale:4.2f}  μ {car.mu_eff:4.2f}  {'SLIDING' if car.sliding else ''}", c["hud_bad"] if car.sliding else None),
            (f"reward {state['last_r']:+6.2f}   episode {state['ep_reward']:8.1f}", None),
            (f"warning: {info['warning'].replace('_', ' ')} (free drive: no reset)" if info.get("warning") else
             ("free drive: walls are solid, nothing resets" if env.free_drive else "AI mode: episode ends as in training"),
             c["hud_bad"] if info.get("warning") else c["hud_dim"]),
            (f"fps {clock.get_fps():4.0f}", c["hud_dim"]),
        ]
        view.draw_lines(screen, lines, (16, 14), 520)
        a = state["action"]
        bw = 220
        view.bar(screen, "steer (action)", float(a[0]), pygame.Rect(24, size[1] - 70, bw, 18), c["steer_bar"], True)
        view.bar(screen, "gas", float(max(a[1], 0)), pygame.Rect(24 + bw + 30, size[1] - 70, bw // 2, 18), c["gas_bar"], False)
        view.bar(screen, "brake", float(max(-a[1], 0)), pygame.Rect(24 + bw + 30 + bw // 2 + 30, size[1] - 70, bw // 2, 18), c["brake_bar"], False)
        if show_help:
            help_lines = view.wrap(HELP, size[0] - 60)
            lh = view.font.get_linesize()
            top = size[1] - 110 - lh * len(help_lines)
            view.draw_lines(screen, [(s, c["hud_text"]) for s in help_lines], (24, top), size[0] - 32)
        else:
            view.text(screen, "H help", (size[0] - 80, size[1] - 28), c["hud_dim"])
        if end_reason is not None and len(poses) <= 1 and now < banner_until:
            view.banner(screen, f"{end_reason.replace('_', ' ').upper()}", c["hud_bad"] if end_reason not in ("lap limit",) else c["hud_good"])

        pygame.display.flip()
        frame += 1
        if args.screenshot and frame >= args.frames:
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            print(f"saved {out}")
            running = False
        clock.tick(rcfg["fps"] if not args.screenshot else 0)

    if trace:
        trace.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
