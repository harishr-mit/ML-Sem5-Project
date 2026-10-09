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

* **Last updated:** 2026-10-10 (Claude Opus 5.5): project initialised; assets audited and ready (T10). No code yet. Next: T05 (environment setup), then Week 1 work toward M1.
* **Project:** RL Racing Line. A 2D top-down racing game written from scratch, in which a car learns a racing line with PPO (Stable-Baselines3). Grip-limited physics, ray sensors, a model registry (CSV → xlsx), and a multi-car demo (10–20 ghost cars, evolution race, draw-a-track, human vs AI).
* **Plan:** `PROJECT_CONTEXT.md` is the single source of truth: design, physics, env, training, registry, demo, timeline (§12) and setup (§14). `CLAUDE.md` holds the agent rules.
* **Architecture:** a headless simulation (`racer/physics`, `racer/track`, `racer/env`, with no pygame) used by the training scripts (`scripts/train.py`, `evaluate.py`) and by the pygame-ce demo (`scripts/demo.py`, `racer/render`).
* **Environment:** Windows, Python 3.11, project-local `.venv`, CPU training (i5-10200H, 16 GB; the GTX 1650 isn't used).
* **Commands:** none yet. Add them here as they are created: train, evaluate, demo, physics sanity, registry export.
* **Blockers / Notes:** assets are in place. `assets/LICENSES.md` is the manifest (files, licences, cut points, usage). The engine sound is **CC BY 4.0: credit required** in the README and the game. The code must still fall back to placeholders if an asset is missing.

## Task List

* [x] T01: Project context file (`PROJECT_CONTEXT.md`)
* [x] T02: Agent rules (`CLAUDE.md`): context tracking, local-only dependencies, run hygiene
* [x] T03: Asset folder skeleton (`assets/sprites/cars`, `sprites/props`, `textures/ground`, `fonts`, `sounds`)
* [x] T04: Initialise `log.md` and `.gitignore`
* [ ] T05: Environment setup: `.venv`, `requirements.txt` (§14.1), CPU torch, verify imports, `git init`
* [ ] T06: **M1 (Week 1):** physics, surfaces, tracks, sensors, Gymnasium env, sanity tests, debug renderer; first PPO run completes laps
* [ ] T07: **M2 (Week 2):** configs, train/evaluate, checkpoints, registry and xlsx export, loader, curriculum, basic 20-car Model Arena
* [ ] T08: **M3 (Week 3):** textures, effects, HUD, brain panel, minimap, camera, Evolution Race, Draw-a-Track, Human vs AI, video recorder
* [ ] T09: **M4 (Week 4):** ablations, plots, analysis view, highlight video, slides, README, rehearsal
* [x] T10: Asset audit: rename, prune unused, cut and loop the engine and tyre sounds, write `assets/LICENSES.md`, update PROJECT_CONTEXT §10/§14

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
