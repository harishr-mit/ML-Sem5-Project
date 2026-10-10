Shared context for every agent working in this directory. Read this file **before** starting work and update it **as** you work.

## Status Legend

|Marker|Meaning|
|-|-|
|`[ ]` TODO|Planned, not started|
|`[~]` WORKING|In progress (set *before* starting the task)|
|`[x]` DONE|Completed and verified|
|`[!]` FAILED|Attempted but failed or interrupted; add a note on why|

## Rules for Agents

1. **Read first.** Check "Current Status" and the latest log entries to learn the context and what is in progress.
2. **Plan as TODOs.** Add your tasks under "Task List" as `[ ]` before touching any files.
3. **Mark `[~]` before starting** a task, and `[x]` immediately after finishing it. Never batch updates at the end.
4. **Log every change** (file created, edited, deleted or moved) in the "Change Log", with the timestamp, agent/model name and files touched.
5. **On failure, don't delete.** Mark the task `[!]` and record the error, the last successful step and any partial changes, so the next agent can diagnose and resume.
6. **Recovering from an interrupted run:** any task still `[~]` when you start means the previous agent stopped midway. Inspect the files listed in its log entry (`git status` / `git diff`), then either finish it or mark it `[!]` with notes.
7. Keep entries append-only, with the newest first in the Change Log. Don't rewrite history.

## Current Status

* **Last updated:** 2026-10-10 (Claude Opus 5.5): **M1 reached** (T06 done, T17–T30). R001 (PPO, Training Ground, 2M steps, 20 min) laps reliably; the best checkpoint `models/R001_ppo_rays5_tg@1200k` completes 6/6 starts, best lap 42.70 s vs the baseline's 46.6 s. 43 tests pass, physics sanity passes. Open: owner's drive-feel check (`scripts\drive.py`, T26); Week 1 code + R001 are **not committed yet**. Next: Week 2 (T07): registry, full evaluate.py with best-checkpoint selection, multi-track curriculum, arena.
* **Project:** RL Racing Line. A 2D top-down racing game written from scratch, in which a car learns a racing line with PPO (Stable-Baselines3). Grip-limited physics, ray sensors, a model registry (CSV → xlsx), and a multi-car demo (10–20 ghost cars, evolution race, draw-a-track, human vs AI).
* **Plan:** `PROJECT_CONTEXT.md` is the single source of truth: design, physics, env, training, registry, demo, timeline (§12) and setup (§14). `CLAUDE.md` holds the agent rules.
* **Architecture:** a headless simulation (`racer/physics`, `racer/track`, `racer/env`, with no pygame) used by the training scripts (`scripts/train.py`, `evaluate.py`) and by the pygame-ce demo (`scripts/demo.py`, `racer/render`).
* **Environment:** Windows, Python 3.11, project-local `.venv`, CPU training (i5-10200H, 16 GB; the GTX 1650 isn't used).
* **Commands:** setup and the full command table are in `README.md`. Run everything with `.venv\Scripts\python`: `scripts\drive.py` (debug window), `scripts\train.py --tag … --tracks …`, `-m pytest -q`, `scripts\physics_sanity.py`, `scripts\preview_tracks.py`, `scripts\generate_tracks.py`, `scripts\bench_env.py`, `-m tensorboard.main --logdir runs`.
* **Blockers / Notes:** assets are in place. `assets/LICENSES.md` is the manifest (files, licences, cut points, usage). The engine sound is **CC BY 4.0: credit required** in the README and the game. The code must still fall back to placeholders if an asset is missing.

## Task List

* [x] T01: Project context file (`PROJECT_CONTEXT.md`)
* [x] T02: Agent rules (`CLAUDE.md`): context tracking, local-only dependencies, run hygiene
* [x] T03: Asset folder skeleton (`assets/sprites/cars`, `sprites/props`, `textures/ground`, `fonts`, `sounds`)
* [x] T04: Initialise `log.md` and `.gitignore`
* [x] T05: Environment setup: `.venv`, `requirements.txt` (§14.1), CPU torch, verify imports, `git init`
* [x] T06: **M1 (Week 1):** physics, surfaces, tracks, sensors, Gymnasium env, sanity tests, debug renderer; first PPO run completes laps
* [ ] T07: **M2 (Week 2):** configs, train/evaluate, checkpoints, registry and xlsx export, loader, curriculum, basic 20-car Model Arena
* [ ] T08: **M3 (Week 3):** textures, effects, HUD, brain panel, minimap, camera, Evolution Race, Draw-a-Track, Human vs AI, video recorder
* [ ] T09: **M4 (Week 4):** ablations, plots, analysis view, highlight video, slides, README, rehearsal
* [x] T10: Asset audit: rename, prune unused, cut and loop the engine and tyre sounds, write `assets/LICENSES.md`, update PROJECT_CONTEXT §10/§14
* [x] T11: Add the rule "goal/requirement changes must be reflected in the relevant documents" to `CLAUDE.md`
* [x] T12: Fix `.gitignore` (inline `#` comments broke `runs/` and `outputs/frames/`; `assets/_source/` missing) and untrack `assets/_source/`
* [x] T13: Repo skeleton only (no milestone code): §4 folder tree, empty `racer` package stubs, `pyproject.toml` for an editable install, `README.md`
* [x] T14: Restore T10's PROJECT_CONTEXT edits that never landed in the file: §3 Audio row, §10.2 Car + Scenery rows, §10.3 `S` key, §10.4 Audio, §12 MoSCoW (the licence facts in Budget/§14.2/§16 were fixed in T13). Needs the owner's audio design, or the T10 session notes.
* [x] T15: Commit the setup; rewrite history so `assets/_source/` was never in the repo; force-push `main` (owner-approved). The processed game assets stay tracked.
* [x] T16: Plan T06 (Week 1 → M1): split it into subtasks with "done when" checks; list the design decisions that need the owner's OK
* *T06 is split into T17–T29 below (in build order). Decisions D1–D6 (T16 Change Log entry) were approved by the owner and recorded in PROJECT_CONTEXT (T30).*
* [x] T17: Configs + loader: `configs/physics/default.yaml`, `configs/env/rays5.yaml`, `configs/reward/progress.yaml`, `configs/train/ppo_default.yaml`, `configs/track/default.yaml` (geometry + generator, D5); `racer/config.py` (load preset by name, deep-merge overrides, unknown keys raise). Add `pytest` (D6). **Done when:** every preset loads and a typo'd key raises.
* [x] T18: Physics: `racer/physics/surfaces.py` (`Surface` → `Road` → `Kerb`), `racer/physics/car.py` (§5.4 step, per-wheel μ and Crr, speed-dependent steer lock, steer-rate limit, low-speed clamp; surfaces come in through a callback, so no track import). **Done when:** unit tests pass and nothing in `racer/physics` imports pygame.
* [x] T19: `scripts/physics_sanity.py`: skidpad, braking (D4), coast-down, grass, low-speed stability. Prints a PASS/FAIL table, saves plots to `outputs/plots/sanity/`, exits non-zero on failure. **Done when:** all 5 pass.
* [x] T20: Track geometry: `racer/track/spline.py` (closed centripetal Catmull-Rom, 1 m arc-length resampling with tangent, normal, curvature, `s`), `io.py` (track JSON), `track.py` (kerb mask by curvature; a numba-built **field** of 0.5 m cells holding distance to the centreline + nearest-sample index; `surface_at(points)`; barrier check, D3). **Done when:** a circle track gives length 2πR and κ = 1/R, and the field agrees with brute force within 0.05 m.
* [x] T21: `racer/track/progress.py`: nearest sample from the field + sub-sample refinement, signed Δs with lap wrap-around, lap count, start/finish crossing times; a jump to another track section = barrier (D3). **Done when:** tests for forwards, backwards, wrap-around and lap counting pass.
* [x] T22: Showcase tracks `tracks/showcase/{training_ground,technical,fast_flow}.json` + `scripts/preview_tracks.py` (PNG previews in `outputs/plots/tracks/`). **Done when:** all 3 pass validation and the previews look right.
* [x] T23: `racer/track/generator.py`: §6 recipe + validation (section clearance / self-intersection, corner radius ≥ 15 m, length range, random direction and mirroring); 10 held-out tracks with fixed seeds in `tracks/heldout/`, from a seed stream the training sampler never uses. **Done when:** the acceptance rate is measured and the held-out previews look right.
* [x] T24: `racer/env/sensors.py`: numba sphere-traced rays on the field (5 rays: 60 m sides, 200 m forward; off-road behaviour per D2). **Done when:** rays match an exact ray–edge intersection within 0.1 m, and 5 rays cost ≤ ~10 µs per car.
* [x] T25: Env: `racer/env/obs_spec.py` (v1 = `rays5`, 13 dims, fixed scales, D2), `reward.py` (§7.3), `racing_env.py` (§7.4 random starts, 3 physics substeps per action, the 4 terminations, truncation at 2 laps / 4000 steps, rich `info`); `racer/agents/baseline.py` (centreline-following PID at a safe speed); `scripts/bench_env.py`. **Done when:** Gymnasium and SB3 `check_env` pass, the baseline completes laps on all 3 showcase tracks, and one core does ≥ 3k env steps/s.
* [x] T26: Debug renderer: `racer/render/debug_view.py` (world rasterised from the same field the physics uses, car rectangle, rays, HUD text) + `scripts/drive.py` (keyboard driving, `--baseline`, `--model`). **Done when:** the owner drives a lap and the handling feels right (tune in YAML only). *Built and smoke-tested headless; the owner's drive-feel check is still pending.*
* [x] T27: Minimal `scripts/train.py`: SubprocVecEnv(6) + VecNormalize + PPO (§8 hyperparameters, linear LR decay), TensorBoard in `runs/`, checkpoints every 100k, §9 model folders (`model.zip`, `vecnormalize.pkl`, `config.yaml` snapshot with git hash and seed); `--scratch` writes to a git-ignored `outputs/scratch/`. **Done when:** a 50k-step smoke run finishes end to end and the saved model plays in `drive.py`.
* [x] T28: First real PPO run, `R001` on Training Ground (~2–3M steps). **Done when (M1):** the deterministic policy completes laps from 3 different starts without crashing; lap times and the baseline's lap time are logged here.
* [x] T29: Docs: README commands, PROJECT_CONTEXT (§5.5, §6, §7.1 frozen obs details, §16 decisions), `log.md` Current Status.
* [x] T30: Owner approved D1–D6 (2026-10-10). Record them in PROJECT_CONTEXT first (§4 layout, §5.1, §5.5, §6, §7.1, §16, open questions), before any code.

*Split milestones into smaller tasks as you start them. Add new tasks at the bottom with the next free ID. Never reuse or renumber IDs.*


## Change Log

<!-- Newest first. Copy the template below for each entry.

### YYYY-MM-DD HH:MM — Agent (Model) — Txx: short title

* **Task:** what was asked
* **Status:** DONE / WORKING / FAILED, plus how it was verified
* **Files:** created / edited / deleted / moved
* **Details:** key decisions, measurements
* **Notes / Errors:** anything the next agent needs to know
-->

### 2026-10-10 — Claude Code (Opus 5.5) — T28, T29: R001 (M1 reached), docs

* **Status:** DONE. **M1 reached.** `pytest`: 43 passed; `physics_sanity.py`: ALL PASSED.
* **Run:** `scripts\train.py --tag ppo_rays5_tg --tracks training_ground --steps 2000000` → `R001`, 2,002,944 steps in 20.1 min (1,665 steps/s), seed 0. 20 model folders `models/R001_ppo_rays5_tg@{100..2000}k/` (33 MB, untracked so far); TensorBoard in `runs/R001_ppo_rays5_tg_1`.
* **Checkpoint sweep** (deterministic, 6 evenly spaced standing starts on Training Ground):

  | Checkpoint | Completion | Best lap | Off-road |
  |---|---|---|---|
  | @100k–300k | 0/6 (off-road) | – | ~7 % |
  | @400k | 6/6 | 43.80 s | 0.7 % |
  | @500k | 6/6 | 45.30 s | 0.0 % |
  | @900k | 6/6 | 43.60 s | 0.1 % |
  | **@1200k (M1 model)** | **6/6** | **42.70 s** | **0.2 %** |
  | @1300k | 6/6 | 42.75 s | 0.1 % |
  | @1600k | 0/6 | – | 6.7 % |
  | @2000k (final) | 2/6 | 43.70 s | 5.1 % |

  Baseline (centreline + curvature speed plan): 46.6 s. R001@1200k is 8 % faster, with 1.0 g of lateral grip used through corners and an apex-seeking line (screenshot checked in `drive.py`).
* **Generalisation (R001@500k, expected to be poor — single track):** Technical 0/3 (off-road in the hairpins), Fast Flow 3/3 but slow (≈ 91 s, 13–14 % off-road). This is the E5 overfitting story; Week 2's multi-track curriculum addresses it.
* **Files (edited):** `racer/agents/evaluation.py` (bug: `reset()` picked a random track from the list, so multi-track evals could repeat one track and skip another; now the track is passed explicitly), `tests/test_env.py` (+ test), `README.md` (status, command table), `PROJECT_CONTEXT.md` (§4 layout: new scripts/modules, `outputs/scratch/`; §8 measured throughput and logging; §16 M1 row), `log.md`
* **Files (deleted):** `outputs/scratch/S001_smoke@55k/` and its TensorBoard folder (smoke-test artefacts)
* **Notes / Errors:**
  - **Reliability oscillates between checkpoints** (100 % → 0 % → 100 %): the agent learns to drive at the limit with wheels on the grass, up to the 1 s all-off-road limit. Week 2 must select the **best checkpoint by evaluation** (§8 already plans this), not use the last one. Candidate fixes to try as experiments: a larger off-road penalty, a lower final learning rate / entropy, or more evaluation starts.
  - `config.yaml` records `git_commit: c8a7914…-dirty`: the Week 1 code isn't committed yet. Commit before the next serious run so snapshots point at real code.
  - The owner edited `CLAUDE.md` during this session (log-reading rule: read up to the latest Change Log record); left as is.
  - Every process this run started has stopped (training, monitors); no TensorBoard or other servers were started.

### 2026-10-10 — Claude Code (Opus 5.5) — T27: Minimal train.py + 50k smoke test

* **Status:** DONE. The 50k smoke run (`--scratch`, S001) trains, saves a §9 folder, reloads it, runs the deterministic eval (crashes at 55k steps, as expected), and plays in `drive.py --model`.
* **Files (created):** `scripts/train.py`, `racer/agents/evaluation.py` (deterministic eval episodes; Week 2's `evaluate.py` builds on it)
* **Files (edited):** `.gitignore` (`outputs/scratch/`)
* **Details:**
  - **Model folders:** `models/R{run:03d}_{tag}@{steps}k/` holds `model.zip` (1.7 MB), `vecnormalize.pkl` and `config.yaml`. The config is a full snapshot: env config (physics, obs_spec, raycast, episode, reward, track settings, track list), train config, seed, git commit, versions, command. Folders are never overwritten (`FileExistsError`). A checkpoint is saved every 100k steps; the last checkpoint doubles as the final model if it is less than one rollout old.
  - **Options:** `--set key=value` gives strict overrides of the train preset (e.g. `--set ppo.ent_coef=0.01`). TensorBoard logs go to `runs/<run>_<tag>_1`, with extra scalars `race/lap_time`, `race/laps`, `race/progress_m` and `end/{barrier,offroad,stuck,wrong_way,truncated}`.
  - **Throughput:** 1,699 steps/s with 6 envs. Per 6,144-step iteration, the rollout takes 1.6 s (SubprocVecEnv round trip ≈ 1 ms per step for 6 envs, policy 0.6 ms) and the PPO update takes 2.4 s (240 minibatch updates on 256×256, compute-bound). 4 torch threads made it slightly slower (1,570 steps/s). So ≈ 10 min per 1M steps, inside §8's 15–30 min estimate; the hyperparameters stay as specified.
* **Notes / Errors:**
  - First smoke attempt: `yaml.safe_dump` can't serialise `torch.__version__` (a `TorchVersion` object), so it's now wrapped in `str()`. The broken folder was deleted.
  - `outputs/scratch/` holds S001 (git-ignored). The S002 thread test was deleted.
  - **Week 2 concern:** 20 checkpoints × 1.7 MB ≈ 34 MB of tracked files per 2M-step run. Decide on Git LFS, or keep only selected checkpoints in git, before the overnight queue starts.

### 2026-10-10 — Claude Code (Opus 5.5) — T26: Debug renderer + drive.py

* **Status:** DONE (built and checked headless with screenshots); the owner's keyboard drive-feel check is pending. `pytest`: 42 passed (incl. headless `drive.py` run and a "simulation never imports pygame" check).
* **Files (created):** `racer/render/debug_view.py`, `scripts/drive.py`, `racer/agents/loader.py` (minimal model-folder loader for `--model`), `configs/demo/render.yaml` (`debug:` section), `tests/test_render.py`
* **Files (edited):** `racer/env/racing_env.py` (`record_substeps` flag → `substep_poses`, so the window animates every 60 Hz physics substep; off in training), `racer/track/field.py` (`rasterise` kernel)
* **Details:**
  - The world is rasterised from the track field at 4 px/m: grass, road, red/white kerb stripes (3 m), walls where territory jumps (D3) and at the barrier, dashed centreline, chequered start line. Fast Flow: 3648×1936 px, built in 1.1 s; drawing the view costs ~3.5 ms per frame (overview is cached: 0.5 ms).
  - The car is a Kenney sprite scaled to 4.5 m (rectangle fallback), with brake lights. Rays are re-cast from the displayed pose; the trail is coloured by speed.
  - HUD: speed, lateral/longitudinal g, lap, s, lateral offset, lap times (current / last / best), wheel surfaces, grip scale, μ, sliding, step and episode reward, fps; bars for steer / gas / brake.
  - **Modes:** `1` human, `2` baseline, `3` model. **Keys:** `R` rays, `T` trail, `F` overview, `+/−`/wheel zoom, `[ ]` sim speed ×0.25–×4, `Space` pause, `Enter` reset at the start line, `Backspace` random start, `N` next showcase track, `H` help, `Esc` quit.
  - On termination a banner shows the reason for 1.5 s. Human mode then respawns on the centreline where the car left the track; the baseline and model modes restart randomly.
  - Default zoom 8 px/m (the doc says 6–8; the car looked small at 6).
* **Notes / Errors:** `--screenshot out.png --frames N` runs with `SDL_VIDEODRIVER=dummy` for headless checks.

### 2026-10-10 — Claude Code (Opus 5.5) — T24, T25: Ray sensors, Gymnasium env, baseline

* **Status:** DONE. `pytest`: 40 passed (incl. Gymnasium + SB3 `check_env`, obs layout, wrong-way / stuck / crash terminations, baseline laps on all 3 showcase tracks). `scripts/bench_env.py`: **13,148 env steps/s on one core** with random actions (target 3k); the baseline completes 2 laps on all 13 tracks.
* **Files (created):** `racer/env/sensors.py`, `racer/env/obs_spec.py`, `racer/env/reward.py`, `racer/env/racing_env.py`, `racer/agents/baseline.py`, `configs/train/baseline.yaml`, `scripts/bench_env.py`, `tests/test_sensors.py`, `tests/test_env.py`
* **Files (edited):** `configs/env/rays5.yaml` (`hit_eps_m` comment)
* **Details:**
  - **Rays:** 5 rays cost 6.6 µs. vs exact ray–circle intersections: p50 0.001 m, p99 0.007 m, p99.9 0.018 m. A ray stops only at a real crossing; the first version stopped when |sdf| < eps, which made grazing rays up to 0.3 m short and could skip a crossing. The remaining outliers are rays that dip < 3 mm past an edge (touches).
  - **Env config:** one plain dict (`make_env_config`: presets + track list) that gets snapshotted into model folders. `RacingEnv.reset(options=...)` accepts `track`, `start_s`, `lateral`, `heading_offset`, `speed` for deterministic evaluation starts.
  - **Laps:** `laps` = full track lengths of net progress (bonus + truncation). `lap_time` is reported only when the start/finish line is crossed after a full lap of net progress since the previous crossing.
  - **Baseline lap times (seed 0, from standstill at s = 0):** Training Ground 46.6 s, Technical 81.6 s, Fast Flow 70.4 s. These are the E8 reference times.
* **Notes / Errors:** none.

### 2026-10-10 — Claude Code (Opus 5.5) — T22, T23: Showcase tracks, generator, held-out set

* **Status:** DONE. All 13 tracks validate; previews checked visually. `pytest`: 28 passed.
* **Files (created):** `tracks/showcase/{training_ground,technical,fast_flow}.json`, `tracks/heldout/heldout_{00..09}.json`, `racer/track/generator.py`, `scripts/preview_tracks.py`, `scripts/generate_tracks.py`, `tests/test_generator.py`; previews in `outputs/plots/tracks/*.png`
* **Files (edited):** `racer/track/track.py` (`build_field=False` option, so the generator validates without paying for the field)
* **Details:**
  - **Training Ground:** 1427 m, 16 m wide, min radius 50 m, kerbs on 1 % of the lap.
  - **Technical:** 1927 m, 12 m wide, two hairpins (≈ 15 m and 17 m radius) and a chicane. The right-hand hairpin's legs are ≈ 40 m apart, so a territory wall (D3) stands between them.
  - **Fast Flow:** 2200 m, 12 m wide, min radius 34 m; a ≈ 600 m straight into a ≈ 100 m-radius sweeper, so braking matters.
  - **Generator:** 42 % acceptance, 2.4 attempts per valid track, 1195 ± 228 m, 15 ms per valid track without the field (+ ~50 ms for the field). Held-out tracks use seed stream 1; training uses stream 0 (`SeedSequence(seed, spawn_key=(stream,))`), so they can't overlap.
  - Fields are 3–14 MB per track (Fast Flow is the largest).
* **Notes / Errors:** the showcase JSONs were written by a scratch script in the session scratchpad (not in the repo); the JSONs are the source of truth. To redesign a track, edit its JSON and re-run `scripts/preview_tracks.py`.

### 2026-10-10 — Claude Code (Opus 5.5) — T19–T21: Physics sanity, track geometry + field, progress

* **Status:** DONE. `physics_sanity.py`: ALL PASSED. `pytest`: 26 passed.
* **Files (created):** `scripts/physics_sanity.py`, `racer/track/spline.py`, `racer/track/io.py`, `racer/track/field.py`, `racer/track/track.py`, `racer/track/progress.py`, `tests/test_track.py`, `tests/test_progress.py`; output `outputs/plots/sanity/physics_sanity.png`
* **Details (sanity numbers, default preset):**
  - skidpad R = 50 m: peak a_lat 10.72 m/s² vs μg 10.79; loses the line at 23.1 m/s
  - braking from 30 m/s: threshold (0.88 brake) 41.0 m vs 41.7 m theory; full brake (locked) 48.8 m
  - coast-down: monotonic, no reversing, stops after 128 s / 1501 m
  - grass: top speed 57.6 vs 68.8 m/s (−16 %); cornering 4.11 vs 10.72 m/s² (−62 %)
  - stability: no NaN, no drift at rest, max yaw step 0.023 rad at full lock
  - `Car.step` costs ≈ 4.4 µs (plain Python floats)
* **Notes / Errors:**
  - The field's distance is exact to ~0.002 m except within one cell of the centreline, where bilinear interpolation rounds off the kink of the unsigned distance (≤ 0.25 m). Edges (6 m), kerbs and rays never sample there; progress uses the index, not the distance.
  - Grass top speed is only 16 % lower than road (§5.5 says "much lower"); its grip and acceleration are much lower. Left as specified; off-road time is capped at 1 s anyway.
  - Scripts that print μ, ², → call `sys.stdout.reconfigure(encoding="utf-8")`: the Windows console is cp1252.

### 2026-10-10 — Claude Code (Opus 5.5) — T30, T17, T18: Decisions recorded, configs, physics

* **Task:** Owner approved D1–D6 and asked to implement Week 1. Record the decisions, then build configs and the car physics.
* **Status:** DONE. `pytest`: 15 passed (config strictness, surface inheritance/fallback, straight-line acceleration, no reverse, steering sign, grass slower, no pygame import in `racer.physics`).
* **Files (edited):** `PROJECT_CONTEXT.md` (§4 layout: `configs/track/`, `racer/config.py`, `field.py`, `tests/`; §5.1 kerb placement; §5.5 braking test; §6 field, barrier territory, progress, clearance; §7.1 frozen v1 details + axes; §16 row; open question closed), `README.md` (layout rows), `requirements.txt` (`pytest==9.1.1`), `pyproject.toml` (pytest `testpaths`), `log.md`
* **Files (created):** `configs/physics/default.yaml`, `configs/env/rays5.yaml`, `configs/reward/progress.yaml`, `configs/train/ppo_default.yaml`, `configs/track/default.yaml`, `racer/config.py`, `racer/physics/surfaces.py`, `racer/physics/car.py`, `tests/test_config.py`, `tests/test_physics.py`
* **Installed (into `.venv` only):** `pytest 9.1.1` (+ `pluggy 1.6.0`, `iniconfig 2.3.1`). `pip check` clean.
* **Details:**
  - The loader supports `extends: <preset>` and strict deep-merged overrides; config dataclasses reject unknown **and** missing keys, so no tunable default hides in code.
  - Car: follows §5.4. One change: drag + rolling resistance are applied as a speed loss capped at the current speed, so the car can't jitter around v = 0 (instead of a force that could flip the velocity's sign).
* **Notes / Errors:** a stray `python3 -` call hung on the Windows Store stub and was stopped (no effect on files). Use `.venv/Scripts/python` only.

### 2026-10-10 13:20 — Claude Code (Opus 5.5) — T16: Week 1 (T06) implementation plan

* **Task:** The owner asked to start Week 1 by planning T06 (M1: the car completes laps on a showcase track).
* **Status:** DONE (plan only, no code). T06 is split into T17–T29 in the Task List. Decisions D1–D6 below are **proposed** and waiting for the owner's OK; they then go into PROJECT_CONTEXT (T29, or earlier if they change a requirement).
* **Files (edited):** `log.md` (T16–T29, this entry, Current Status)
* **Details (proposed decisions):**
  - **D1 Kerbs:** kerbs sit *inside* the road edge (band `[w/2 − kerb_m, w/2]` where |curvature| is above a threshold). So the road edge stays at ±w/2 for rays and track limits, and kerbs count as on-track (answers the open question).
  - **D2 obs_spec v1 details** (frozen once R001 is saved): rays start at the car centre; distance = first road-edge crossing along the ray (leaving the road when on-road, entering it when off-road), divided by range and clipped to [0, 1]. Fixed scales: `v_long / 70`, `v_lat / 10`, `yaw / 3 rad/s`, all clipped to [−1, 1]. These live in `configs/env/rays5.yaml`, not in the physics config, so later physics changes can't change what v1 means.
  - **D3 Barrier = field territory:** the track field stores the nearest centreline sample per 0.5 m cell. The car crashes if it is beyond `w/2 + runoff`, **or** if the field says it is now nearest to a different part of the track (a jump of more than W m from its tracked index). Where two sections are closer than 2 × (w/2 + runoff), e.g. the legs of a hairpin, this puts a wall halfway between them, as on real circuits. It also removes the tunnelling risk (no thin walls) and stops the agent from earning progress by cutting across the grass. The generator still rejects roads that come closer than `width + clearance`.
  - **D4 Braking test:** brake force (12000 N) is above static grip (μ·m·g ≈ 10 800 N), so full brake locks the wheels (kinetic μ, ≈ 49 m from 30 m/s). The §5.5 "≈ 42 m" holds for threshold braking (≈ 90 % brake). The test checks both, and the physics stays as specified: learning threshold braking is part of the racing-line skill.
  - **D5 New `configs/track/default.yaml`:** kerb curvature threshold, field cell size, generator ranges and clearances. This is a small §4 layout addition.
  - **D6 `pytest`:** add it to `requirements.txt` (pinned) and a `tests/` folder for the small unit tests (spline, field, progress, physics). Another §4 layout addition.
  - **Pulled forward from Week 2:** a minimal config loader (needed now because of the no-magic-numbers rule) and a minimal `train.py`. Model folders follow §9 from the first run, so Week 2 can register R001 retroactively.
* **Notes / Errors:**
  - **Performance plan:** physics stays plain-Python floats (one scalar `Car`); field building, wheel-surface lookup and ray casting are numba kernels (`cache=True`). Budget is ~300 µs per env step for ≥ 3k steps/s per core. If profiling says otherwise, the car step becomes a numba kernel too.
  - **Week 2 concern:** a field is ~4–8 MB per track, so multi-track training needs small per-worker track pools (or regenerating a track every K episodes), not large pools.
  - The owner's uncommitted `.vscode/settings.json` change (window title) was there before this run and was left alone.

### 2026-10-10 — Claude Code (Opus 5.5) — T15: Setup commit, history rewrite, force push

* **Task:** Commit the setup work. Remove `assets/_source/` from the repo and from its history, and force-push. The processed game assets stay tracked. The owner approved the force push explicitly.
* **Status:** DONE. Verified:
  - `git log --branches -- assets/_source` is empty
  - the object database has no `_source` paths
  - the old commits `f0f7ec0` and `2b806c3` are gone locally
  - the new initial commit differs from the old one only by the 3 `_source` files; all 73 game assets are kept
  - the final tree is identical to the pre-rewrite commit
  - remote `main` = local `main`
* **Files:** no working-tree changes besides `log.md`. The originals in `assets/_source/` stay on disk and are git-ignored.
* **Details:**
  - **Commit:** `2b806c3` "Initialise project: venv setup, repo skeleton, docs sync" on top of `f0f7ec0`.
  - **Rewrite:** `git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch assets/_source' -- --branches` rewrote `main` and `rootline`. The new history is `c18aa89` (initial commit) → `bbeecf9` (setup).
  - **Push:** `git push --force-with-lease=main:f0f7ec0 origin main` (`f0f7ec0...bbeecf9`, forced).
  - **Local purge:** deleted `refs/original/*`, then `git reflog expire --expire=now --all` and `git gc --prune=now`. `.git` went from 42.3 MB to 5.4 MB.
* **Notes / Errors:**
  - **First commit attempt failed:** PowerShell doesn't pipe a here-string into `git commit -F -`, so I used a message file in the scratchpad instead.
  - **`rootline`:** a local-only duplicate of the initial commit (the branch the first commit was made on), rewritten to `c18aa89`. It's not on GitHub.
  - **GitHub cache:** GitHub may keep the old `f0f7ec0` reachable by direct SHA until its own GC runs. The files aren't sensitive, so no support request is needed.
  - **Other clones:** any other clone must be re-cloned.

### 2026-10-10 — Claude Code (Opus 5.5) — T14: Restore T10's PROJECT_CONTEXT edits (3-way merge)

* **Task:** The owner supplied the T10 version of `PROJECT_CONTEXT.md` that never landed. Keep all of it, then re-apply the later T11/T13 edits.
* **Status:** DONE. Verified with diffs:
  - **base (`HEAD`) → pasted:** exactly the 11 hunks T10 had logged, so the transcription was faithful
  - **pasted → final:** only the T11/T13 edits (§4: `log.md`, `requirements.txt` note, `pyproject.toml`; §14.1: pinned setup; §15: rule; §16: 2 rows)
  - no conflict markers left; CRLF line endings kept
* **Files (edited):** `PROJECT_CONTEXT.md` (merged), `README.md` (`configs/demo/` in the layout table), `log.md`
* **Files (created):** `configs/demo/.gitkeep` (the restored §4 layout adds `configs/demo/`)
* **Details:**
  - **Method:** `git merge-file` (ours = current file, base = `HEAD`, theirs = pasted). It produced 3 conflicts:
    - **Budget line:** took T10's wording (same fact).
    - **§14.2:** took T10's audited rewrite. It supersedes the T13 patch to the old sounds row.
    - **§16:** kept T10's 2 rows, then T13's "Initialisation" and "Rule" rows. Dropped T13's "CC BY exception" row, because T10's asset-audit row already records it.
  - **Restored from T10:** Budget, §3 Audio row, §4 layout (`configs/demo/`, `audio.py`, `gearbox.py`, the `assets/` tree), §10.2 Car + Scenery, §10.3 `S` key, §10.4 Audio, §12 MoSCoW, §14.2, §16
* **Notes / Errors:** none. The merge inputs were in the session scratchpad, outside the project.

### 2026-10-10 — Claude Code (Opus 5.5) — T05, T11–T13: Doc rule, .gitignore fixes, environment, skeleton

* **Task:** Add the "goal/requirement changes must be reflected in documents" rule to `CLAUDE.md`, then initialise the project. The owner said: initialise only, don't start the milestone phases.
* **Status:** DONE. Verified:
  - every package imports
  - `pygame.IS_CE` is true and classic pygame is not installed
  - a numba JIT call works
  - a 128-step SB3 PPO smoke run on Pendulum works
  - `pip check` is clean
  - `import racer` works from outside the project folder
  - `git check-ignore` matches `runs/`, `outputs/frames/` and `assets/_source/`, and no longer matches `racer/env/`
  - 0 project processes and 0 sockets are left
* **Files (created):**
  - `.venv/` (ignored)
  - `requirements.txt`, `pyproject.toml`, `README.md`, `.vscode/settings.json`
  - `racer/__init__.py`, plus `racer/{physics,track,env,agents,registry,render}/__init__.py` (docstring-only stubs)
  - `.gitkeep` in `configs/{physics,env,reward,train}`, `scripts`, `tracks/{showcase,heldout,user}`, `models`, `registry`, `outputs/{videos,plots}`
  - empty `runs/` and `outputs/frames/` (ignored)
  - `racer.egg-info/` and `__pycache__/` (ignored build byproducts)
* **Files (edited):**
  - `CLAUDE.md`: new working rule
  - `PROJECT_CONTEXT.md`: Budget line, §4 layout, §14.1 setup and pinned requirements, §14.2 sounds licence row, §15 rule, §16 entries
  - `.gitignore`
  - `log.md`
* **Files (untracked):** `git rm -r --cached assets/_source` (3 files, about 55 MB). They are still on disk, and staged as deletions.
* **Details:**
  - **Interpreter:** `py -3.11` (3.11.9, 64-bit) was already installed, so no global install was needed. System `python` on PATH is 3.14; don't use it.
  - **Packages:** torch 2.14.1+cpu from the PyTorch CPU index. `requirements.txt` is pinned to the tested versions, so saved models stay loadable.
  - **`.gitignore` bugs fixed:**
    - inline `# comments` after `runs/` and `outputs/frames/` made those patterns match nothing
    - `assets/_source/` was missing, although T10 said it had been added
    - the unanchored `env/` would have ignored `racer/env/` and `configs/env/`; it's now `/env/` and `/venv/`
* **Notes / Errors:**
  - **Git state:** the repo already existed (`main` + `rootline`, pushed to `origin` = github.com/harishr-mit/ML-Sem5-Project). The previous note "No git repo yet" is outdated. Nothing has been committed or pushed in this run. The `_source` files are still in the pushed initial commit; removing them from history would need a force push, so that's the owner's call.
  - **T10 doc edits missing:** several PROJECT_CONTEXT edits that T10 logged are not in the file (no §10.4 Audio, no `S` key, no §3 Audio row). See T14.

### 2026-10-10 — Claude (Opus 5.5) — T10: Asset audit, renaming, audio cutting

* **Task:** Check that all the needed assets were downloaded, rename them for the project, remove unused ones, trim and loop the engine and tyre sounds, and update the docs
* **Status:** DONE. Verified visually (tiled textures, tinted cars) and with spectrograms (loop seams, pitch curves). The owner still needs to listen to the sounds.
* **Files (created):**
  - `assets/sprites/cars/car_{1..4}_{black,blue,green,orange,yellow}.png`, `car_{1..4}_tint.png`, `car_{1..4}_tint_mask.png`
  - `assets/sprites/props/*.png` (21)
  - `assets/textures/ground/asphalt.jpg` (stains removed)
  - `assets/fonts/*.ttf` (7, flattened)
  - `assets/licenses/*` (Kenney CC0 + 3 OFL)
  - `assets/sounds/engine_{rev_up,rev_down,idle_loop,high_loop}.ogg`, `engine_map.json`, `tyre_screech_{loop,short_1,short_2}.ogg`
  - `assets/LICENSES.md`
* **Files (moved/renamed):**
  - `Grass005_1K-JPG_Color.jpg` → `grass.jpg`, `Gravel035_…` → `gravel.jpg`, `Sand093C_…` → `sand.jpg`
  - original asphalt and both original WAVs → `assets/_source/` (git-ignored)
* **Files (deleted):** `assets/sprites/kenney_racing-pack/` (unused: motorcycles, characters, tiles, spritesheets, vectors, the pickup truck, small cars, rocks, oil, barrels, skidmarks, arrows, light beams), and the font subfolders (unused weights and the variable font)
* **Files (edited):** `PROJECT_CONTEXT.md` (Budget line, §3 Audio row, §4 layout, §10.2 Car + Scenery rows, §10.3 `S` key, new §10.4 Audio, §12 MoSCoW, §14.2 rewritten, §16), `.gitignore` (`assets/_source/`), `log.md`
* **Details:**
  - The engine recording (freesound #623436, CC BY 4.0) contains free revs and an accelerating drive through the gears. Pitch was tracked by log-spectral shift (80 ms steps) and anchored with the harmonic line spacing (idle ≈ 17.4 Hz).
  - **rev_up:** 1.19→2.88× idle, from the 1st-gear pull.
  - **rev_down:** 2.34→1.0×, a free-rev decay.
  - **idle loop:** cut from the end of rev_down.
  - **high loop:** the top of rev_up, pitch- and loudness-flattened.
  - All mono, 44.1 kHz, OGG q6. The engine clips share one gain, so their natural relative loudness is kept.
* **Notes / Errors:** The Kenney "red" cars are orange, so they're named `orange`. Kenney's car 5 is now `car_4`. The work was done on the owner's machine via device tools plus the cloud workspace (ffmpeg, scipy); nothing was installed into the project, and there are no running processes.

### 2026-10-10 — Claude (Opus 5.5) — T01–T04: Project initialisation

* **Task:** Set up the project context, agent rules, asset folders and the shared log
* **Status:** DONE
* **Files:** `PROJECT_CONTEXT.md` (created), `CLAUDE.md` (created, then extended with tracking, dependency and end-of-run rules), `log.md` (created; reinitialised from a template used in another project, with that project's entries removed), `.gitignore` (created), `assets/` folder tree (created, empty)
* **Details:** Planning was done in the Claude desktop app; implementation happens in VS Code with Claude Code. A copy of `PROJECT_CONTEXT.md` is also kept in the claude.ai project "ML Sem5 Project".
* **Notes / Errors:** None. No git repo yet (T05).
