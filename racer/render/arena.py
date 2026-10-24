"""Model Arena (PROJECT_CONTEXT §10.1, D14): the model picker and the multi-car race window.

Race keys: Tab / Shift+Tab or click a leaderboard row: follow a car · F overview · R rays ·
T trails of all cars · L labels · Space pause · +/− sim speed ×0.25–×4 · mouse wheel zoom ·
H key list · Esc back to the picker (or quit).

Each car runs its own model's snapshot (physics, sensors, episode rules); ``fair`` puts every
car on the same physics preset. Crashed cars respawn after a short pause and the crash is
counted (D14).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pygame

from racer.config import load_preset
from racer.render.app import Context
from racer.agents.activations import ActivationProbe
from racer.agents.loader import LoadedModel
from racer.render.audio import CarAudio
from racer.render.brain_panel import BrainPanel, ModelInfo, model_info
from racer.render.ui import ModelPicker, Selection
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


def run_arena(ctx: Context, preset: Selection | None) -> str:
    """The arena mode: picker → race → back to the picker, until Esc / window close."""
    models_blocked = None
    while True:
        selection = preset
        if selection is None:
            if models_blocked is None:
                models_blocked = blocked_models(ctx.reg_rows)
            picker = ModelPicker(ctx.reg_rows, list_tracks(), ctx.screen.get_size(), ctx.fonts[0], ctx.colours,
                                 ctx.acfg["palette"], max_cars=int(ctx.acfg["max_cars"]), laps=ctx.laps,
                                 track=ctx.track, blocked=models_blocked)
            while picker.done is None and not picker.quit:
                for ev in pygame.event.get():
                    picker.handle(ev)
                picker.draw(ctx.screen)
                pygame.display.flip()
                ctx.clock.tick(60)
            if picker.quit:
                return "back"
            selection = picker.done
            ctx.track, ctx.laps = selection.track, selection.laps
        result = run_race(ctx, selection)
        if result == "quit" or preset is not None:
            return result


def arena_keys(quit_on_esc: bool, human: bool = False) -> list[tuple[str, str]]:
    """The race's key list (T82); must match the key handling in :func:`run_race`."""
    drive = [("arrows / WASD", "drive your car"), ("K", "ABS on / off")] if human else []
    return drive + [
        ("Tab / Shift+Tab", "follow next / previous car"),
        ("click a row", "follow that car"),
        ("F", "overview / follow"),
        ("R", "rays of the followed car"),
        ("B", "brain panel (network activity)"),
        ("T", "trails of all cars"),
        ("L", "car labels"),
        ("Space", "pause"),
        ("+ / −", "sim speed"),
        ("mouse wheel", "zoom"),
        ("M", "sound on / off"),
        ("H", "hide / show this list"),
        ("Esc", "quit" if quit_on_esc else "back to the picker"),
    ]


def run_race(ctx: Context, sel: Selection) -> str:
    """Run one race until Esc ("back") or the window closes ("quit")."""
    screen, clock, models = ctx.screen, ctx.clock, ctx.models
    acfg, dcfg, colours, fonts = ctx.acfg, ctx.dcfg, ctx.colours, ctx.fonts

    from racer.agents.baseline import BaselineController
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
    human = None
    if sel.human:
        from racer.render.human import HumanDriver, human_entry, poll_pad
        human = HumanDriver(acfg["human"])
        entries.insert(0, human_entry(sel.track, sel.human, human))
        pad = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None
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
    palette = acfg["palette"]
    if sel.gradient:
        from racer.render.evorace import gradient_palette
        palette = gradient_palette(len(race.cars))
    sprites = CarSprites(palette, p0.car_length_m, p0.car_width_m)
    hud = RaceHud(font, font_big, colours, int(acfg["leaderboard_width"]))
    labels = TextCache(font_small)
    lut = speed_colours()
    vmax = float(dcfg["trail_speed_max"])
    n_trail, n_sel_trail = int(acfg["trail_length"]), int(acfg["selected_trail_length"])
    trails = [deque(maxlen=n_sel_trail) for _ in race.cars]
    speeds = list(dcfg["sim_speeds"])
    speed_i = speeds.index(ctx.sim_speed) if ctx.sim_speed in speeds else speeds.index(1.0)
    selected = 0
    show_rays, all_trails, show_labels, paused = True, False, True, False
    brain = BrainPanel(ctx.fonts, colours, (int(acfg["brain"]["width"]), int(acfg["brain"]["height"])),
                       float(acfg["brain"]["smoothing"]))
    show_brain, brain_car = bool(acfg["brain"]["default_on"]), -1
    show_keys = bool(acfg["show_keys_at_start"])
    key_items = arena_keys(quit_on_esc=ctx.direct, human=human is not None)
    rows: list = []
    acc = 0.0
    audio = ctx.audio
    was_paused = False
    if audio is not None:
        audio.reset()
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
                elif k == pygame.K_b:
                    show_brain = not show_brain
                elif k == pygame.K_l:
                    show_labels = not show_labels
                elif k == pygame.K_h:
                    show_keys = not show_keys
                elif k == pygame.K_m and ctx.audio is not None:
                    ctx.audio.toggle()
                elif k == pygame.K_k and human is not None:
                    human.toggle_abs()
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_i = min(speed_i + 1, len(speeds) - 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_i = max(speed_i - 1, 0)

        if human is not None:
            from racer.render.human import read_input
            human.set_input(*read_input(pygame.key.get_pressed(), poll_pad(pad, acfg["human"]), acfg["human"]))
        # ---- simulate whole physics substeps (60 Hz × sim speed)
        n = 0
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
        if audio is not None:
            audio.follow(selected)
            if paused != was_paused:
                audio.pause(paused)
            was_paused = paused
            if n:
                c = sc.env.car
                audio.update(n / 60.0, CarAudio(c.speed, c.throttle, c.brake, c.v_lat, c.sliding))
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
        if show_brain:
            if brain_car != selected:
                brain.reset()
                brain_car = selected
            m = sc.entry.key
            if isinstance(m, LoadedModel):
                probe = ctx.probes.get(m.model_id)
                if probe is None:
                    probe = ctx.probes[m.model_id] = ActivationProbe(m.policy.policy, m.normalise)
                brain.draw(screen, (12, 56), model_info(m), probe.capture(sc.obs), sc.obs)
            else:
                brain.draw(screen, (12, 56), ModelInfo(sc.label, [], [], ""), None)
        hud.keys(screen, key_items, show_keys)
        fps = labels.get(f"fps {clock.get_fps():3.0f}", colours["hud_dim"])
        screen.blit(fps, (12, 12))
        if audio is not None:
            screen.blit(labels.get(audio.label + (f" · gear {audio.gear}" if audio.enabled else ""),
                                   colours["hud_dim"]), (82, 12))
        if human is not None:
            abs_txt = "ABS off (K)" if not human.abs_on else ("ABS ACTIVE" if human.abs_active else "ABS on (K)")
            screen.blit(labels.get(abs_txt, colours["abs_active"] if human.abs_active else colours["hud_dim"]),
                        (12, 34))

        pygame.display.flip()
        frame += 1
        if ctx.screenshot and frame >= ctx.frames:
            out = Path(ctx.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(out))
            ms = np.asarray(frame_ms[1:] or [0.0])
            print(f"saved {out}  (race t = {race.t:.1f} s, {len(race.cars)} cars, fps {clock.get_fps():.0f}; "
                  f"racing frames {len(ms)}: mean {1000 / max(ms.mean(), 1e-9):.0f} fps, "
                  f"1% low {1000 / max(np.percentile(ms, 99), 1e-9):.0f} fps, worst frame {ms.max():.1f} ms)")
            result = "quit"
        dt_ms = clock.tick(dcfg["fps"] if not ctx.screenshot else 0)
        if ctx.screenshot and race.started and not paused:
            frame_ms.append(dt_ms)       # headless: uncapped frame times while racing (T51)
    if audio is not None:
        audio.stop()
    return result
