"""RL Racing Line demo (PROJECT_CONTEXT §10). Week 2: the Model Arena (up to 20 ghost cars).

    .venv\\Scripts\\python scripts\\demo.py                       model picker (registry browser)
    .venv\\Scripts\\python scripts\\demo.py --track technical --models R001_ppo_rays5_tg@1200k R002_…
                                         [--baseline] [--laps 3] [--fair]
    .venv\\Scripts\\python scripts\\demo.py --track fast_flow --run R002 [--every 5]   a run's checkpoints
    .venv\\Scripts\\python scripts\\demo.py --track technical --best                   every run's best checkpoint
        [--sim-speed 1] [--screenshot out.png --frames 600]                         headless capture
        [--title "window caption"]                                                  (AI Sandbox)

Race keys: Tab / Shift+Tab or click a leaderboard row: follow a car · F overview · R rays ·
T trails of all cars · L labels · Space pause · +/− sim speed ×0.25–×4 · mouse wheel zoom ·
Esc back to the picker (or quit).

Each car runs its own model's snapshot (physics, sensors, episode rules); --fair puts every
car on the same physics preset. Crashed cars respawn after a short pause and the crash is
counted (D14).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

from racer.config import load_preset
from racer.registry.registry import Registry, typed
from racer.track.io import track_folders


def list_tracks() -> list[tuple[str, str]]:
    return [(d.name, p.stem) for d in track_folders() for p in sorted(d.glob("*.json"))]


def blocked_models(rows: list[dict]) -> dict[str, str]:
    """Model id → why this engine can't run it (D23). One snapshot is checked per run."""
    from racer.agents.compat import check
    from racer.config import PROJECT_ROOT, load_yaml

    reasons: dict[str, str | None] = {}
    out: dict[str, str] = {}
    for r in rows:
        if not r.get("path"):
            continue
        run = r["run_id"]
        if run not in reasons:
            cfg = PROJECT_ROOT / r["path"] / "config.yaml"
            reasons[run] = ("; ".join(check(load_yaml(cfg))) or None) if cfg.is_file() else "model folder missing"
        if reasons[run]:
            out[r["model_id"]] = reasons[run]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", default="training_ground")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--models", nargs="+", help="model ids or folders (skips the picker)")
    sel.add_argument("--run", help="race the checkpoints of this run")
    sel.add_argument("--best", action="store_true", help="race every run's best checkpoint")
    ap.add_argument("--every", type=int, default=1, help="--run: take every N-th checkpoint")
    ap.add_argument("--baseline", action="store_true", help="add the hand-coded PID baseline car")
    ap.add_argument("--laps", type=int, default=None)
    ap.add_argument("--fair", action="store_true", help="every car on the same physics (arena.yaml fair_physics)")
    ap.add_argument("--sim-speed", type=float, default=1.0)
    ap.add_argument("--screenshot", default=None, help="run headless and save a PNG after --frames frames")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--title", default=None, help="window caption (the AI Sandbox names each game window)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame   # after SDL_VIDEODRIVER is set
    from racer.render.debug_view import load_font
    from racer.render.ui import ModelPicker, Selection

    acfg = load_preset("demo", "arena")
    dcfg = load_preset("demo", "render")["debug"]
    colours = {k: tuple(int(v) for v in c) for k, c in dcfg["colours"].items()}
    laps = args.laps or int(acfg["race"]["laps"])

    reg_rows = [{k: typed(v) for k, v in r.items()} for r in Registry().models()]
    preset: Selection | None = None
    if args.models or args.run or args.best or args.baseline:
        if args.models:
            ids = args.models
        elif not (args.run or args.best):
            ids = []                     # --baseline alone
        elif args.run:
            ids = [r["model_id"] for r in sorted((r for r in reg_rows if r["run_id"] == args.run),
                                                 key=lambda r: r.get("steps") or 0)][::max(args.every, 1)]
        else:
            ids = [r["model_id"] for r in reg_rows if r.get("is_best") and r.get("path")]
        ids = ids[: int(acfg["max_cars"]) - int(args.baseline)]
        if not ids and not args.baseline:
            ap.error("no models selected")
        preset = Selection(args.track, ids, args.baseline, laps, args.fair)

    pygame.init()
    size = tuple(dcfg["window"])
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(args.title or "RL Racing Line — Model Arena")
    clock = pygame.time.Clock()
    font = load_font(dcfg["font"], int(dcfg["font_size"]))
    font_big = load_font(dcfg["font"], int(dcfg["font_size"] * 3))
    font_small = load_font(dcfg["font"], int(acfg["label_font_size"]))
    models: dict[str, object] = {}     # loaded once, reused across races

    while True:
        selection = preset
        if selection is None:
            picker = ModelPicker(reg_rows, list_tracks(), size, font, colours, acfg["palette"],
                                 max_cars=int(acfg["max_cars"]), laps=laps, track=args.track,
                                 blocked=blocked_models(reg_rows))
            while picker.done is None and not picker.quit:
                for ev in pygame.event.get():
                    picker.handle(ev)
                picker.draw(screen)
                pygame.display.flip()
                clock.tick(60)
            if picker.quit:
                break
            selection = picker.done
            args.track, laps = selection.track, selection.laps
        result = run_race(screen, clock, selection, models, acfg, dcfg, colours, (font, font_big, font_small), args)
        if result == "quit" or preset is not None:
            break
    pygame.quit()
    return 0


def run_race(screen, clock, sel, models, acfg, dcfg, colours, fonts, args) -> str:
    """Run one race until Esc ("back") or the window closes ("quit")."""
    import pygame

    from racer.agents.baseline import BaselineController
    from racer.agents.loader import LoadedModel
    from racer.env.race import Entry, Race
    from racer.env.racing_env import make_env_config
    from racer.env.sensors import cast_rays
    from racer.render.camera import Camera
    from racer.render.cars import CarSprites
    from racer.render.debug_view import speed_colours
    from racer.render.hud import RaceHud, TextCache, short_label
    from racer.render.world import TrackWorld

    font, font_big, font_small = fonts
    loading = font.render(f"loading {len(sel.models)} model(s)…", True, colours["hud_text"])
    screen.fill(colours["outside"])
    screen.blit(loading, (20, 20))
    pygame.display.flip()

    from racer.agents.compat import IncompatibleModel

    entries = []
    for mid in sel.models:
        if mid not in models:
            try:
                models[mid] = LoadedModel(mid)
            except (IncompatibleModel, FileNotFoundError) as e:
                print(f"skipping {mid}: {e}", flush=True)     # a model the engine can't run never crashes the race
                continue
        m = models[mid]
        entries.append(Entry(m.model_id, m.env_config, lambda obs, envs, m=m: m.act_batch(obs), key=m))
    if sel.baseline:
        ctrl = BaselineController(load_preset("train", "baseline"))
        entries.append(Entry("PID baseline", make_env_config([sel.track]),
                             lambda obs, envs: np.stack([ctrl.act(e) for e in envs]), key="baseline"))
    if not entries:
        print("no model in the selection can run on this engine", flush=True)
        return "back"
    rules = dict(acfg["race"], laps=sel.laps)
    fair = load_preset("physics", acfg["fair_physics"]) if sel.fair else None
    race = Race(sel.track, entries, rules, physics_override=fair)

    size = screen.get_size()
    cam = Camera(size, dcfg)
    world = TrackWorld(dcfg, colours)
    world.set_track(race.track)
    p0 = race.cars[0].env.params
    sprites = CarSprites(acfg["palette"], p0.car_length_m, p0.car_width_m)
    hud = RaceHud(font, font_big, colours, int(acfg["leaderboard_width"]))
    labels = TextCache(font_small)
    lut = speed_colours()
    vmax = float(dcfg["trail_speed_max"])
    n_trail, n_sel_trail = int(acfg["trail_length"]), int(acfg["selected_trail_length"])
    trails = [deque(maxlen=n_sel_trail) for _ in race.cars]
    speeds = list(dcfg["sim_speeds"])
    speed_i = speeds.index(args.sim_speed) if args.sim_speed in speeds else speeds.index(1.0)
    selected = 0
    show_rays, all_trails, show_labels, paused = True, False, True, False
    rows: list = []
    acc = 0.0
    frame = 0
    frame_ms: list[float] = []
    x, y, _ = race.cars[0].pose
    cam.snap(x, y)
    fit_w = size[0] - int(acfg["leaderboard_width"]) - 24
    result = None

    while result is None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                result = "quit"
            elif ev.type == pygame.MOUSEWHEEL:
                cam.zoom_by(dcfg["zoom_step"] ** ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for rect, idx in rows:
                    if rect.collidepoint(ev.pos):
                        selected = idx
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    result = "back"
                elif k == pygame.K_TAB:
                    step = -1 if ev.mod & pygame.KMOD_SHIFT else 1
                    order = [c.index for c in race.standings()]
                    selected = order[(order.index(selected) + step) % len(order)]
                elif k == pygame.K_f:
                    cam.overview = not cam.overview
                elif k == pygame.K_r:
                    show_rays = not show_rays
                elif k == pygame.K_t:
                    all_trails = not all_trails
                elif k == pygame.K_l:
                    show_labels = not show_labels
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_i = min(speed_i + 1, len(speeds) - 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_i = max(speed_i - 1, 0)

        # ---- simulate whole physics substeps (60 Hz × sim speed)
        if not paused:
            acc += speeds[speed_i]
            n = int(acc)
            acc -= n
            for _ in range(n):
                race.tick()
                if race.started and not race.over:
                    for car, tr in zip(race.cars, trails):
                        tr.append((car.pose[0], car.pose[1], car.env.car.speed) if car.state != "respawn" else None)

        # ---- draw
        sc = race.cars[selected]
        x, y, th = sc.pose
        cam.update(x, y, sc.env.car.vx, sc.env.car.vy, 0.0 if paused else speeds[speed_i],
                   race.track.field.extent, fit_w)
        world.draw(screen, cam)
        for car, tr in zip(race.cars, trails):
            if car.index == selected:
                prev = None
                for p in tr:
                    if p is None:
                        prev = None
                        continue
                    q = cam.to_screen(p[0], p[1])
                    if prev is not None:
                        pygame.draw.line(screen, lut[min(int(p[2] / vmax * (len(lut) - 1)), len(lut) - 1)], prev, q, 2)
                    prev = q
            elif all_trails and len(tr) > 1:
                pts = [cam.to_screen(p[0], p[1]) for p in list(tr)[-n_trail:] if p is not None]
                if len(pts) > 1:
                    pygame.draw.lines(screen, sprites.colour(car.index), False, pts, 1)
        if show_rays and sc.state != "respawn":
            ob = sc.env.observer
            d = np.empty(ob.n_rays)
            f = sc.track.field
            cast_rays(f.dist, f.idx, f.geom, x, y, th, ob.angles, ob.ranges, ob.raycast_params, d)
            o = cam.to_screen(x, y)
            for a, dist, rng in zip(ob.angles, d, ob.ranges):
                frac = float(dist / rng)
                col = (min(255, int(510 * (1 - frac))), min(255, int(510 * frac)), 60)
                hpt = cam.to_screen(x + dist * np.cos(th + a), y + dist * np.sin(th + a))
                pygame.draw.line(screen, col, o, hpt, 1)
        for car in sorted(race.cars, key=lambda c: c.index == selected):    # followed car on top
            px, py, pth = car.pose
            sprites.draw(screen, cam, car.index, px, py, pth, braking=car.env.car.brake > 0.05,
                         ghost=car.state == "respawn")
            if show_labels:
                img = labels.get(short_label(car.label), sprites.colour(car.index))
                lx, ly = cam.to_screen(px, py + float(acfg["label_offset_m"]))
                if 0 <= lx < size[0] and 0 <= ly < size[1]:
                    screen.blit(img, img.get_rect(midbottom=(int(lx), int(ly))))
        title = f"{race.track.name} · {race.rules.laps} laps · ×{speeds[speed_i]:g}" + (" · PAUSED" if paused else "") \
            + (" · fair physics" if sel.fair else "")
        if race.over:
            hud.results(screen, race, sprites.colour)
            rows = []
        else:
            rows = hud.leaderboard(screen, race, sprites.colour, selected, title)
            hud.countdown(screen, race)
        hud.car_readout(screen, race, sc, sprites.colour(selected))
        fps = labels.get(f"fps {clock.get_fps():3.0f}", colours["hud_dim"])
        screen.blit(fps, (12, 12))

        pygame.display.flip()
        frame += 1
        if args.screenshot and frame >= args.frames:
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            ms = np.asarray(frame_ms[1:] or [0.0])
            print(f"saved {out}  (race t = {race.t:.1f} s, {len(race.cars)} cars, fps {clock.get_fps():.0f}; "
                  f"racing frames {len(ms)}: mean {1000 / max(ms.mean(), 1e-9):.0f} fps, "
                  f"1% low {1000 / max(np.percentile(ms, 99), 1e-9):.0f} fps, worst frame {ms.max():.1f} ms)")
            result = "quit"
        dt_ms = clock.tick(dcfg["fps"] if not args.screenshot else 0)
        if args.screenshot and race.started and not paused:
            frame_ms.append(dt_ms)       # headless: uncapped frame times while racing (T51)
    return result


if __name__ == "__main__":
    sys.exit(main())
