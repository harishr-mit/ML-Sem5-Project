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

* **Last updated:** 2026-10-10 (Claude Opus 5.5): initialisation complete (T05, T11–T13): `.venv` with pinned deps, folder skeleton, README. No simulation code yet. PROJECT_CONTEXT restored to include T10's audio/asset edits (T14). Setup committed; history rewritten without `assets/_source/` and force-pushed (T15). Next: T06 (Week 1 → M1), when the owner says to start.
* **Project:** RL Racing Line. A 2D top-down racing game written from scratch, in which a car learns a racing line with PPO (Stable-Baselines3). Grip-limited physics, ray sensors, a model registry (CSV → xlsx), and a multi-car demo (10–20 ghost cars, evolution race, draw-a-track, human vs AI).
* **Plan:** `PROJECT_CONTEXT.md` is the single source of truth: design, physics, env, training, registry, demo, timeline (§12) and setup (§14). `CLAUDE.md` holds the agent rules.
* **Architecture:** a headless simulation (`racer/physics`, `racer/track`, `racer/env`, with no pygame) used by the training scripts (`scripts/train.py`, `evaluate.py`) and by the pygame-ce demo (`scripts/demo.py`, `racer/render`).
* **Environment:** Windows, Python 3.11, project-local `.venv`, CPU training (i5-10200H, 16 GB; the GTX 1650 isn't used).
* **Commands:** setup is in `README.md` (`py -3.11 -m venv .venv` → CPU torch → `pip install -r requirements.txt` → `pip install -e .`). Run everything with `.venv\Scripts\python`. Add more here as they are created: train, evaluate, demo, physics sanity, registry export.
* **Blockers / Notes:** assets are in place. `assets/LICENSES.md` is the manifest (files, licences, cut points, usage). The engine sound is **CC BY 4.0: credit required** in the README and the game. The code must still fall back to placeholders if an asset is missing.

## Task List

* [x] T01: Project context file (`PROJECT_CONTEXT.md`)
* [x] T02: Agent rules (`CLAUDE.md`): context tracking, local-only dependencies, run hygiene
* [x] T03: Asset folder skeleton (`assets/sprites/cars`, `sprites/props`, `textures/ground`, `fonts`, `sounds`)
* [x] T04: Initialise `log.md` and `.gitignore`
* [x] T05: Environment setup: `.venv`, `requirements.txt` (§14.1), CPU torch, verify imports, `git init`
* [ ] T06: **M1 (Week 1):** physics, surfaces, tracks, sensors, Gymnasium env, sanity tests, debug renderer; first PPO run completes laps
* [ ] T07: **M2 (Week 2):** configs, train/evaluate, checkpoints, registry and xlsx export, loader, curriculum, basic 20-car Model Arena
* [ ] T08: **M3 (Week 3):** textures, effects, HUD, brain panel, minimap, camera, Evolution Race, Draw-a-Track, Human vs AI, video recorder
* [ ] T09: **M4 (Week 4):** ablations, plots, analysis view, highlight video, slides, README, rehearsal
* [x] T10: Asset audit: rename, prune unused, cut and loop the engine and tyre sounds, write `assets/LICENSES.md`, update PROJECT_CONTEXT §10/§14
* [x] T11: Add the rule "goal/requirement changes must be reflected in the relevant documents" to `CLAUDE.md`
* [x] T12: Fix `.gitignore` (inline `#` comments broke `runs/` and `outputs/frames/`; `assets/_source/` missing) and untrack `assets/_source/`
* [x] T13: Repo skeleton only (no milestone code): §4 folder tree, empty `racer` package stubs, `pyproject.toml` for an editable install, `README.md`
* [x] T14: Restore T10's PROJECT_CONTEXT edits that never landed in the file: §3 Audio row, §10.2 Car + Scenery rows, §10.3 `S` key, §10.4 Audio, §12 MoSCoW (the licence facts in Budget/§14.2/§16 were fixed in T13). Needs the owner's audio design, or the T10 session notes.
* [x] T15: Commit the setup; rewrite history so `assets/_source/` was never in the repo; force-push `main` (owner-approved). The processed game assets stay tracked.

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
