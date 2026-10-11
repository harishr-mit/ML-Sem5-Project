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

* **Last updated:** 2026-10-11 (Claude Opus 5.5): T52–T58 done on branch `feat/maps-modular-sandbox` (worktree `.claude/worktrees/maps-modular/`): 9-map library (7 train, 2 test), evaluation protocol v2 (default for new runs), upgradable engine (obs_spec v2 features, action layouts, physics versions, PPO/SAC registry, model manifest, compatibility check), AI Sandbox plan (`docs/SANDBOX_PLAN.md`); 113 tests pass. The `night1` queue (T46) ran on `main` meanwhile. **Merge steps (T60):** commit the queue's registry results on `main`, merge the branch, run the golden test, evaluate the baseline with v2, `registry.py describe`. Finding: the best model so far (R003@900k) completes 0 % of the new test maps (tight corners), which supports the owner's ray-angle concern; `rays7_narrow` is ready to train on `maps7`. **Next: T50 faster models (now on maps7 / v2), T51 arena fps, then the AI Sandbox.**
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
* [x] T31: Fix `drive.py` crashing on every episode end; add free drive for human play (solid walls, no resets), AI modes unchanged
* [x] T32: README Quickstart: how to play, train and replay a model; status updated to M1 done
* [x] T33: drive.py: wrap the clipped help text; check whether sim speed changes model behaviour (it doesn't); camera follows per simulated time
* [x] T34: Owner: keep models local only. Back up `models/` to `C:\Users\Jack\`, untrack `models/` in a new commit (gitignore, docs), push
* [x] T35: Owner finds braking weak (keyboard driving). Diagnose; human-only aids need no retrain; physics changes only if retraining is < 15 min and the owner locks the values by driving first
* [x] T36: ABS status badge on the debug HUD (ON / OFF / ACTIVE)
* [x] T36: Plan T07 (Week 2 → M2): split it into subtasks with "done when" checks; list the decisions that need the owner's OK
* *T07 is split into T37–T48 below (build order). Decisions D7–D18 (T36 Change Log entry) are **proposed** and wait for the owner's OK; record them in PROJECT_CONTEXT before any code.*
* [x] T37: Pre-flight: commit the T35 work (owner OK); `train.py` refuses `R` runs on a dirty tree unless `--allow-dirty` (D12); golden test: `R001@1200k` deterministic eval still gives 6/6 and 42.70 s on Training Ground (skips if `models/` is missing). **Done when:** tree clean, golden test passes.
* [x] T38: Track sets (D7): `configs/trackset/{tg,gen_wide,gen_mixed,final_mix}.yaml` (fixed tracks + weights; generated pool from stream 0 with a seed range and generator overrides); per-worker rotating pool (`pool_size`, `refresh_episodes`); bounded track cache keyed by name + config hash (today it uses `id(cfg)`); generator accepts `width_m: [lo, hi]`. Old snapshots (`tracks: [names]`) still work. **Done when:** tests pass (old config builds; same seed → same track sequence; held-out tracks regenerate bit-identically; cache stays bounded), and multi-track `bench_env` ≥ 3k steps/s including track generation.
* [x] T39: Ablation presets, config-only where possible (D13): `physics/no_grip_limit.yaml`, `physics/downforce.yaml`, `env/rays9.yaml`, `env/rays5_short.yaml` (all obs_spec v1, different ray lists), `reward/speed_naive.yaml` (new `speed` reward type), `train/ppo_finetune.yaml` (lower LR for curriculum stages). **Done when:** every preset loads, `check_env` passes for each env/reward combination, `physics_sanity.py` still passes for `default`.
* [x] T40: Evaluation protocol v1 (D9, D10): `configs/eval/v1.yaml`; `racer/agents/evaluation.py` full §8 metrics (completion, crash, held-out completion, best/mean lap per showcase track, mean speed, off-road %, mean |lat g|, lap ratio vs baseline); speed-coloured trajectory thumbnails; `eval.json`; `scripts/evaluate.py --model ID | --run R00N | --baseline` with a process pool. **Done when:** `R001@1200k` reproduces its Week 1 numbers on Training Ground; one model ≤ ~30 s; 20 checkpoints ≤ ~3 min on 6 processes.
* [x] T41: Registry (D11): `racer/registry/registry.py` (§9 schemas, upsert by id, atomic rewrite, scratch runs go to `outputs/scratch/registry/`), `racer/registry/export_xlsx.py` (Runs, Models, one leaderboard per showcase track, Ablations pivot; frozen header, filters, green best lap, hyperlinks to folders and thumbnails), `scripts/registry.py` (`backfill`, `export`, `list`). Backfill R001 (20 checkpoints) and the baseline (`BASE_pid`). **Done when:** R001 is registered and evaluated with `is_best` set, and `registry.xlsx` opens in Excel with every sheet.
* [x] T42: `train.py` v2: `--trackset`, `--parent <model_id>` (fine-tune: policy + VecNormalize stats, obs_spec must match, parent recorded), registers the run and every checkpoint, evaluates all checkpoints after training and marks `is_best` (`--no-eval` to skip). **Done when:** 50k scratch smoke runs (multi-track, and with `--parent`) go end to end into the scratch registry.
* [x] T43: Loader v2: lookup by model id through the registry (fallback `models/<id>`), algorithm dispatch from the snapshot (PPO now, SAC later), direct torch inference with batching. **Done when:** batched actions equal SB3 `predict` exactly (test), and 20 models load in a few seconds.
* [x] T44: Curriculum runs (D8): `R002` stage 2 (parent `R001@1200k`, `gen_wide`, ~3M); `R003` stage 3 (parent = best of R002, `final_mix`, ~3–5M); `R004` from scratch on `final_mix` (~5M) for comparison. **Done when:** results logged; target: best model ≥ 8/9 showcase starts and ≥ 80 % held-out completion.
* [x] T45: `scripts/queue_overnight.py` (D18): queue YAML in `configs/queue/`, runs `train.py` one after another, one log per run in `outputs/queue_logs/` (git-ignored), keeps going after a failure, resumes (skips finished entries), keeps Windows awake only while it runs. **Done when:** a 2-entry 20k scratch queue with one deliberately failing entry finishes and reports both.
* [~] T46: First overnight queue (owner starts it; `configs/queue/night1.yaml`, started manually after 2026-10-10): E2 no grip limit, E3 speed reward, E4 rays9 / rays5_short, E5 single- vs multi-track; from scratch on the T44 track set, seeds sized to fit ~8 h (from T44's measured steps). **Done when:** all runs are registered and evaluated; failures noted here.
* [~] T47: Basic Model Arena (D14–D16): pygame-free `racer/env/race.py` (N `RacingEnv`s on one shared track, each with its model's own physics and obs spec; grid, countdown, laps, gaps, respawns); `racer/render/world.py` + `camera.py` split out of `debug_view.py` (`drive.py` keeps working); `scripts/demo.py arena` with a basic model picker, 20 distinct cars (4 shapes × 5 colours), labels, leaderboard, `Tab`/`F`/`R`/`T`/`Space`/`+−`, `--models`, `--fair`. **Done when (M2):** 20 registry models race on each showcase track at a steady 60 FPS, and a headless test runs a 20-car race to the finish.
* [x] T50: **NEXT (owner priority): faster models.** *(Done 2026-10-11: target met by `R015_t50_narrow7_s1@4600k` (best) and `@4800k`; see Change Log. Was: queue `configs/queue/t50.yaml`, `rays7_narrow` and `rays9` on `maps7`, 5M steps each; target now v2 `lap_ratio` < 1.0 and `unseen_completion` ≥ 80 %.)* R003@900k is reliable (100 % showcase, 80 % held-out) but ≈ 24 % slower than the baseline. Try from-scratch runs on `final_mix` aimed at lap time (longer training, lower final LR / entropy, off-road / speed reward tweak); compare with the E1 reference runs from the overnight queue. **Done when:** a model beats the baseline lap time on the showcase tracks with held-out completion ≥ 80 %.
* *Owner decisions 2026-10-11 (before going away; work continues up to T62 without the owner, stopping on any critical decision):* T50 trains `rays7_narrow` and `rays9` from scratch on `maps7`, 5M steps each, plus follow-ups if a clear fix shows (≤ ~4 h of training in total). **T50 target is now:** protocol v2 `lap_ratio` < 1.0 **and** `unseen_completion` ≥ 80 %. T62 may install the CUDA torch build into a project-local `.venv-gpu` (git-ignored). Commit locally after each verified step; no push.
* [x] T51: **NEXT (owner priority): arena frame rate.** Re-measure 20 cars at ×1 on an idle CPU (the 66 fps was measured while training used all cores); fix if below a steady 60 fps (batched inference every 3rd frame, text caches, off-screen culling). Last check for M2.
* [ ] T48: Docs: PROJECT_CONTEXT (§4 layout, §8 curriculum and track sets, §9 eval artefacts, §11 run IDs, §16), README (commands, registry, arena), `log.md` Current Status.
* [x] T49: Owner approved D7–D18 with changes (any tracks may be mixed or added; tracks can be added at any time, final showcase = a faculty-drawn track; evals also copied to `outputs/eval/`; respawn rule as proposed; overnight queue must run without a Claude session). Record in PROJECT_CONTEXT (+ §16), CLAUDE.md (model-folder rule), before code.
* [x] T50: CLAUDE.md rules for periodic check-ins while waiting on a process: keep gaps under 60 min so the prompt cache stays warm; read only the status file and log tail (log.md and CLAUDE.md excepted).
* *T52–T59 (2026-10-11, owner request): more maps with a train/test split, an upgradable model design, and a plan for the AI Sandbox app. Work happens on branch `feat/maps-modular-sandbox` in the worktree `.claude/worktrees/maps-modular/` while the `night1` queue finishes on `main`. Merge after the queue: commit the queue's registry results on `main` first, then merge the branch. (Note: T50 was used twice above; IDs continue at T52.)*
* [x] T52: Record decisions D19–D24 in PROJECT_CONTEXT (§4, §6, §7, §8, §9, §10, §16) before code.
* [x] T53: Maps (D19): 6 new hand-made tracks, 4 for training (`tracks/train/`) and 2 for testing (`tracks/test/`, never trained on, protected like `heldout/`); validated, PNG previews. **Done when:** all 9 named maps pass validation and a track set that names a test track raises.
* [x] T54: Track set `maps7` (7 training maps + generated pool) and evaluation protocol v2 (D20): train maps, test maps and the 10 generated held-out tracks; per-group metrics; v1 results unchanged. **Done when:** v1 metrics for a model are bit-identical to before, and v2 runs for the baseline and one model.
* [x] T55: Upgradable observations (D21): `obs_spec` v2 = an ordered list of named features (rays, heading, speeds, …) from a feature registry, each with dims, bounds, labels and a description. v1 untouched. **Done when:** a v2 spec copying `rays5` gives exactly the v1 observation (test), and adding or removing a feature changes only the obs size.
* [x] T56: Versioned action layouts (D22): `action_spec` (`steer_gas` = v1 default for old snapshots, plus `steer_throttle_brake`); env, race, reward and `prev_action` read the layout. **Done when:** old models race unchanged and a 3-output scratch model trains and races.
* [x] T57: Engine compatibility (D23): physics `version` (absent = 1), algorithm registry (PPO, SAC), model manifest (obs dim, action dim, parameter count, versions) in the snapshot, and an engine check that reports what a model needs instead of crashing; picker greys out unsupported models. **Done when:** tests cover a model needing an unknown version, and every registered model passes the check.
* [x] T58: AI Sandbox implementation plan (D24): `docs/SANDBOX_PLAN.md` (Streamlit app, standalone; model explorer with parameter explanations, compare, configure + queue a new run, launch demo instances). Plan only; built after T50, T51 and Week 2.
* [x] T59: Docs and wrap-up: README, PROJECT_CONTEXT, Current Status; full test suite; merge instructions.
* [x] T60: Merge after the `night1` queue finishes: commit the queue's registry results and eval copies on `main`, merge `feat/maps-modular-sandbox`, full tests incl. the golden test (needs `models/`), `evaluate.py --baseline` (v2), `registry.py describe`, remove the worktree. **Done when:** `main` has both, tests pass, the registry lists v2 baseline laps.
* [~] T62: **GPU benchmark (owner, 2026-10-11).** Measure before switching: the GTX 1650 (4 GB, driver 610.60, already installed) only speeds up the PPO update (≈ 60 % of training time, §8); env stepping stays on the CPU. Install the CUDA build of the **same** torch version (2.14.1) into a separate `.venv-gpu` (project-local, git-ignored; the main `.venv` and `requirements.txt` stay untouched), then time 200k steps of `ppo_default` on CPU vs GPU, also with larger minibatches (512, 1024) where a GPU helps most. Check that a GPU-trained model loads and drives identically on the CPU (demo and evaluation stay on the CPU). **Also measure parallel runs (owner):** total steps/s of 1 run × 6 envs vs 2 runs × 3 envs vs 3 runs × 2 envs, on CPU and with updates on the GPU, plus RAM per run (≈ 1.5 GB for 6 env workers today). Today a single run's env workers sit idle during its PPO update (single-threaded), so overlapping runs may raise total throughput even without the GPU. **Done when:** steps/s for each setting are logged here with a recommendation (device, runs in parallel, envs per run).
* [ ] T64: **Parallel queue (only if T62 shows a higher total throughput).** `queue_overnight.py --parallel N` runs up to N queue entries at once (each with its own log; `state.json` stays the single source of progress; evaluation after training limited so N runs don't each start 6 eval workers); `n_envs` per run from the queue entry or T62's recommendation. Run ids stay unique when two runs start together (`next_run_id` must reserve its id, e.g. by creating the folder/lock atomically). **Done when:** a 2-slot scratch queue with 3 short entries finishes with distinct run ids and correct state, and total throughput matches T62.
* [ ] T63: **Train on the GPU (only if T62 shows ≥ 1.3× faster training).** `device: cpu|cuda` in the train presets (default `cpu`, so old configs mean the same), `train.py --device`, recorded in the snapshot; models are always saved for and loaded on the CPU; `requirements-gpu.txt` with the CUDA torch pin; README setup section; queue entries may use `--device cuda`. Requirements stay as they are: CPU-only machines still train and run everything, saved models load anywhere. **Done when:** a GPU run and a CPU run of the same config give comparable results, and every existing model still loads in the CPU `.venv`.
* [x] T61: Re-evaluate the best checkpoint of every run with protocol v2 (`evaluate.py --model <best ids>`), so all models are comparable on the map library (≈ 1 min per model). Optional: every checkpoint overnight (≈ 430 checkpoints ≈ 1 h on 6 workers).

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

### 2026-10-11 — Claude Code (Opus 5.5) — T50 done: follow-up R017 result

* **R017_t50_narrow7_ft** (`--parent R015_t50_narrow7_s1@4600k`, `ppo_finetune`, 3M steps, maps7, seed 2; 30 checkpoints, v2 evaluation 206 s): **no improvement.** Best @1300k: seen 57 %, test 67 %, held-out 100 %, unseen 87.5 %, overall 73 % (parent: 77 %), lap ratio 0.970; later checkpoints drift down (63–67 % overall). Fine-tuning at the lower LR doesn't fix Fast Flow / Technical.
* **T50 closed:** target (v2 `lap_ratio` < 1.0 and `unseen_completion` ≥ 80 %) met by **R015_t50_narrow7_s1@4600k** (is_best) and **@4800k** (93.8 % unseen). Training used ≈ 3.5 h of the 4 h budget; no more runs.
* **Next ideas for reliability (not run; owner to decide):** a second seed of R015's setup (one seed per layout so far); more training from scratch (8M+); a curriculum stage with more weight on Fast Flow / Technical; a larger off-road penalty.
* **Files:** `registry/*.csv`, `outputs/eval/R017_*`, `log.md`

### 2026-10-11 — Claude Code (Opus 5.5) — T50: R016 (9 rays on maps7) result, follow-up started

* **R016_t50_rays9_s1** (5M steps, v2): clearly weaker than R015. Best @2600k: seen 50 %, test 50 %, held-out 90 %, unseen 75 %, overall 63 %, lap ratio 0.987; no checkpoint reaches 80 % unseen. **The 7 narrow rays (R015) win** on the same tracks and budget (one seed each).
* **Follow-up (owner-approved budget, "a clear fix"):** R015's best checkpoints were its last ones, so continue `R015_t50_narrow7_s1@4600k` for 3M steps on `maps7` with `ppo_finetune` (LR 1e-4, linear decay) → R017.
* **Files:** `registry/*.csv`, `outputs/eval/R015_*`, `outputs/eval/R016_*` (v2 eval copies), `log.md`

### 2026-10-11 — Claude Code (Opus 5.5) — T50: R015 (7 narrow rays on maps7) result

* **R015_t50_narrow7_s1** (5M steps, 50 checkpoints, v2 evaluation 300 s). **The T50 target is met** (v2 `lap_ratio` < 1.0 and `unseen_completion` ≥ 80 %) by @4600k (best by D9: overall 77 %; seen 64 %, test 67 %, held-out 100 %, unseen 87.5 %, lap ratio 0.969), @4800k (unseen **93.8 %**, test 83 %, lap ratio 0.973), @4300k and @4400k. Every earlier model had 0 % on the test maps (best was R014, 17 %).
* **Faster than the PID baseline on the unseen test maps:** Alpine 75.35 s (@4800k) vs 78.95 s, Coastline 61.45 s vs 61.8 s; on seen maps e.g. Esses 67.05 vs 70.35, City Blocks 66.6 vs 69.05, Oval Infield 71.25 vs 74.2, Hairpin Valley 60.7 vs 63.15 s.
* **Not yet reliable:** @4600k fails Fast Flow (2/2 off-road at speed), Technical (2/2) and 1 of 2 Hairpin Valley starts; Alpine 1 of 3. `lap_ratio` only covers maps it laps. The best checkpoints are the latest ones (4.3–4.9M), so training was still improving.

### 2026-10-11 — Claude Code (Opus 5.5) — T50 started, T62 prepared

* **T50:** queue `configs/queue/t50.yaml` running in the background (R015 `t50_narrow7_s1`: 15 inputs, 70,404 policy parameters; then `t50_rays9_s1`). The first start failed before training: unquoted `: ` in the run notes broke the YAML; fixed with block scalars (no run was created).
* **T62 prep (benchmark waits for an idle CPU after T50):** `.venv-gpu` created from the same Python 3.11.9 (owner-approved, project-local, git-ignored): `torch==2.14.1+cu126` (index `whl/cu126`; `cu130` also exists), then `requirements.txt` and `pip install -e .`. `torch.cuda.is_available()` → True, GTX 1650. `pip freeze` identical to `.venv` except torch; `pip check` clean. No global installs; driver 610.60 was already present.
* **Files (created):** `scripts/bench_train.py` (PPO throughput: steps/s split into collection vs update; `--device`, `--batch-size`, `--n-envs`, `--runs K` in parallel; nothing saved). **Edited:** `racer/agents/algorithms.py` (`build(..., device="cpu")`, no TensorBoard dir when none is given; defaults unchanged), `.gitignore` (`.venv-gpu/`), `log.md`
* **Smoke test** (tiny, under training load, numbers not meaningful): CPU, CUDA and 2 parallel runs all work.

### 2026-10-11 — Claude Code (Opus 5.5) — T61: Best checkpoints on protocol v2

* **Status:** DONE (best checkpoint of each run; 14 models in 92 s on 6 workers). Registry rows of these checkpoints now hold v2 metrics (`eval_protocol` = v2); the other checkpoints keep v1. Scoring every checkpoint stays optional (overnight).
* **Results (v2; seen = 7 training maps, unseen = 2 test maps + 10 generated held-out):**

  | best checkpoint | seen | test | held-out | unseen | lap ratio |
  |---|---|---|---|---|---|
  | BASE_pid | 100 % | 100 % | 100 % | 100 % | 1.00 |
  | R003 mix_scratch@900k | 57 % | 0 % | 80 % | 50 % | 1.19 |
  | R004 / R010 e1_main (rays5) | 43 % / 43 % | 0 / 0 | 80 / 60 % | 50 / 38 % | 1.16 / 1.18 |
  | R007 / R013 e4_short | 29 / 43 % | 0 / 0 | 100 / 90 % | 63 / 56 % | 1.26 / 1.18 |
  | **R008 / R014 e4_rays9** | 43 / **64 %** | 0 / **17 %** | 100 / 100 % | 63 / **69 %** | 0.97 / 1.03 |
  | R006 / R012 e5_single (TG only) | 14 / 14 % | 0 / 0 | 0 / 0 | 0 / 0 | 0.92 / 0.91 |
  | R005 / R011 e2_nogrip | 100 % | 100 % | 100 % | 100 % | 0.40 (physics without a grip limit: not comparable) |

* **Reading:** with real grip, no model handles the tight corners of the new maps (their training never had them); 9 rays (with ±10° / ±30° rays) generalise best. This supports T50's plan: train on `maps7` with shallow rays.
* **Files (edited):** `registry/models.csv`; **created:** `outputs/eval/<14 best ids>/eval_v2.json` + `thumb_v2_*.png` (also added to each `models/<id>/` folder, D10)

### 2026-10-11 — Claude Code (Opus 5.5) — T51: Arena frame rate (spread decisions)

* **Status:** DONE. Measured headless, uncapped, idle CPU, **20 distinct models** (worst case: no batching), 3-lap race, frames while racing only. **Before:** mean 92–99 fps but 1 % low 45–48 fps (worst frames 26–29 ms): when most cars decide in the same tick, the sim tick takes 13 ms (p99; 20 forward passes + env steps) on top of ~8 ms of drawing. **After:** 1 % low **59–63 fps** on Training Ground, Technical, Fast Flow and Alpine, mean 86–91 fps; sim tick p99 6.1 ms (only the start tick is heavy). A rare single 70–86 ms frame remains in some runs (not reproducible; seen inside both `race.tick` and `world.draw` in the same run, none of the per-car steps slow) → OS-level stalls, not the game; `gc.freeze()` was tried and didn't change it (reverted).
* **Fix:** `Race(spread_decisions=True)` steps a car up to one agent step early (poses are buffered and played back), with a per-tick budget of ⌈Σ 1/substeps⌉ cars, so decisions spread evenly over the 3 substeps. Results are unchanged (test: identical lap times, finish times and crashes with spreading on/off; peak decisions per tick 3 instead of 9 for 9 cars). The gap table now keeps the minimum time per metre (a car may be stepped before an earlier one).
* **Files (edited):** `racer/env/race.py` (`spread_decisions`, per-tick budget, `np.fmin` gap table), `scripts/demo.py` (headless mode prints mean / 1 % low fps and worst frame over the racing frames), `tests/test_race.py` (+ same-result / less-peak-work test)
* **Tests:** 117 passed.

### 2026-10-11 — Claude Code (Opus 5.5) — T60: Merge, baseline v2, registry columns

* **Status:** DONE. The `night1` queue finished (all 11 runs done, R004–R014; no queue processes left). On `main`: committed the queue's registry rows and 330 eval-copy folders (`5cabf9d`), merged `feat/maps-modular-sandbox` (`d51474b`, no conflicts), full suite **116 passed** (golden + loader tests run with `models/`). `evaluate.py --baseline` (v2): the PID baseline completes 30/30 episodes (laps as in T54). `registry.py describe`: every run R001–R014 can run on this engine (13 inputs / 69,892 policy parameters; R008 and R014 with rays9: 17 inputs / 70,916). Worktree and branch removed (`.claude/worktrees/` stays in `.git/info/exclude`, local only).
* **Files (edited):** `registry/runs.csv` (obs_dim, action_layout, n_params filled), `registry/models.csv` (BASE_pid on v2), `log.md`
* **Files (created):** `outputs/eval/BASE_pid/eval_v2.json`, `thumb_v2_<9 maps>.png`

### 2026-10-11 — Claude Code (Opus 5.5) — T58, T59: AI Sandbox plan, docs

* **Status:** T58 DONE; T59 docs done, commit pending.
* **Files (created):** `docs/SANDBOX_PLAN.md` (standalone Streamlit app: Overview, Model Explorer, Compare, Build & Queue, Launch, Glossary; code layout `racer/sandbox/`; changes elsewhere: Streamlit pin, `.gitignore`, queue re-reads its file, `demo.py --title`; build steps S0–S6 with "done when"; risks; 3 owner decisions before S0)
* **Files (edited):** `README.md` (status, training options, eval v2, any-model arena, registry columns, commands, layout), `PROJECT_CONTEXT.md` (§9 registry columns, open question on `tracks/user/` in `final_mix`), `log.md` (Current Status, T60, T61)
* **Files (deleted):** `outputs/eval/BASE_pid/eval_v2.json` + `thumb_v2_*.png` in the worktree (written by the smoke runs' baseline reference; regenerated on `main` after the merge, T60)

### 2026-10-11 — Claude Code (Opus 5.5) — T55–T57: Upgradable engine (obs v2 features, action layouts, physics versions, algorithms, manifest, compatibility)

* **Status:** DONE. Full suite 113 passed, 3 skipped (need `models/`, absent in the worktree); `physics_sanity.py` ALL PASSED. v1 unchanged: `R001@1200k` and `R003@900k` (main `models/`) reproduce their saved `eval.json` metrics exactly; a v2 spec copying `rays5` gives bit-identical observations **and rewards** to v1 (test).
* **Smoke tests (scratch, git-ignored, worktree `outputs/scratch/`):** `S001_smoke_v2_stb` (maps7, `rays7_narrow`, `--action-layout steer_throttle_brake`, 20k steps): 16 inputs, 3 outputs, 70,918 policy parameters; trained, evaluated with v2, registered in the scratch registry. Headless race on Alpine with R003@900k (13 in / 2 out), S001 (16 in / 3 out) and the PID baseline together: works (88 fps). `S002_smoke_sac` (SAC, 4k steps): trains, evaluates and loads.
* **Design:** obs_spec v2 = ordered named features (`rays`, `heading_sincos`, `v_long`, `v_lat`, `yaw_rate`, `steer_angle`, `onroad_frac`, `prev_action`, `curvature_ahead`), each with params, bounds, labels and a description; v2 presets are standalone files (strict `extends` can't turn v1 into v2). Action layouts `steer_gas` (default; missing in old snapshots) and `steer_throttle_brake` (pedals [−1,1]→[0,1], gas = throttle − brake). Physics `version` (absent = 1) → (params, car class). Algorithms `ppo`, `sac` (train + load). Manifest in every new snapshot (inputs, outputs, labels, net_arch, policy/total parameter counts; analytic counts match SB3 exactly, test). `check()` reports missing obs version / feature / layout / physics version / reward type / algorithm; `LoadedModel` raises `IncompatibleModel`; the picker dims such models with the reason; `demo.py` skips them instead of crashing.
* **Files (created):** `racer/env/features.py`, `racer/env/actions.py`, `racer/physics/versions.py`, `racer/agents/algorithms.py`, `racer/agents/compat.py`, `configs/env/rays7_narrow.yaml` (7 rays: ±60, ±20, ±7, 0; 15 inputs), `configs/train/sac_default.yaml`, `tests/test_obs_v2.py`, `tests/test_compat.py`
* **Files (edited):** `racer/env/obs_spec.py` (ObserverV2, `describe()`, `tracker` arg, v1 requires steer_gas), `racer/env/racing_env.py` (action layout, physics versions, `controls`, `prev_steer`), `racer/env/race.py` (neutral action per layout), `racer/agents/loader.py` (compat check, algorithm registry, `manifest`), `scripts/train.py` (algorithm registry, `--action-layout`, manifest, generic update size), `racer/registry/registry.py` (runs.csv + `obs_dim`, `action_layout`, `n_params`), `scripts/registry.py` (`describe`: fills those for old runs, lists models the engine can't run), `racer/render/ui.py` (`blocked`), `scripts/demo.py` (`blocked_models`, skip incompatible), `tests/test_presets.py` (any spec version)

### 2026-10-11 — Claude Code (Opus 5.5) — T54: Track set maps7, evaluation protocol v2

* **Status:** DONE. v1 is unchanged: `R003_mix_scratch@900k` (main `models/`) reproduces its saved `eval.json` metrics exactly with the branch code. v2 runs end to end (checked without writing eval files). `tests/test_eval_v2.py` + `tests/test_registry.py`: 8 passed.
* **v2 numbers (not recorded yet; run `evaluate.py --baseline` then the models after the merge):** baseline completes all 30 episodes; lap times TG 46.6, Technical 81.55, Fast Flow 70.4, Hairpin Valley 63.15, Esses 70.35, Oval Infield 74.2, City Blocks 69.05, Alpine 78.95, Coastline 61.8 s. **R003@900k (best model so far): 57 % on the training maps, 0 % on the 2 test maps, 80 % generated held-out, unseen 50 %**; it fails Esses, Oval Infield and City Blocks (all tight corners, which it never trained on). This backs the owner's late-corner-detection concern (rays).
* **Files (created):** `configs/eval/v2.yaml` (groups train ×2 / test ×3 / heldout ×1; `metrics`, `lap_groups`, `thumbnail_groups`), `tests/test_eval_v2.py`
* **Files (edited):** `racer/agents/evaluation.py` (`DEFAULT_PROTOCOL = "v2"`, `protocol_version`, `lap_groups`/`lap_tracks`, v2 metrics; v1 path untouched), `racer/registry/registry.py` (`LIBRARY_MAPS`; appended columns `test_completion`, `unseen_completion`, `best_lap_<6 new maps>`; `metrics_row` sets every metric column so rows never mix protocols; `newer_or_same_protocol`; `set_best` ranks within the newest protocol of the run), `racer/registry/record.py` (newest protocol wins; default v2), `racer/registry/export_xlsx.py` (leaderboards for new maps once lapped, thumbnail names per protocol, unseen/test columns), `scripts/evaluate.py` (default v2), `scripts/train.py` (`--protocol`, default v2)

### 2026-10-11 — Claude Code (Opus 5.5) — T53: Map library (6 new maps, train/test split)

* **Status:** DONE. All 6 new maps pass validation (tightest radius ≥ 16 m, closest sections 28 m); previews checked by eye; `tests/test_trackset.py` 9 passed.
* **Maps:** training (`tracks/train/`): Hairpin Valley (1458 m, chicane + long straight into a 180° hairpin), Esses (1519 m, 10 m wide snake), Oval Infield (1915 m, 14 m wide oval with a tight infield), City Blocks (1411 m, right-angle corners). Test (`tracks/test/`): Alpine (1737 m, hairpins between fast sections), Coastline (1753 m, 780 m straight into a hard braking zone, sweepers).
* **Files (created):** `scripts/make_maps.py` (rounded-polygon outlines → track JSON; `--check`), `tracks/train/{hairpin_valley,esses,oval_infield,city_blocks}.json`, `tracks/test/{alpine,coastline}.json`, `configs/trackset/maps7.yaml` (T54)
* **Files (edited):** `racer/env/trackset.py` (`PROTECTED_DIRS` = heldout + test; `is_heldout` covers both), `tests/test_trackset.py` (+ test maps refused, + map library split/validity)
* **Notes:** previews in `outputs/plots/tracks/` of the worktree (git-ignored).

### 2026-10-11 — Claude Code (Opus 5.5) — T52: Decisions D19–D24 recorded

* **Task:** Owner asked for 9+ maps (2 for testing), an upgradable model design (inputs, network size, features, action layout, physics, algorithm) and a plan for an AI Sandbox app. Owner chose: 7 train + 2 test maps keeping the 10 generated held-out tracks; algorithm, action layout and car/physics variants as upgrade axes; Streamlit for the sandbox; work on a branch in a worktree while the `night1` queue finishes on `main` (code edits on `main` would fail the queue's dirty-tree check).
* **Status:** DONE (docs only).
* **Setup:** worktree `.claude/worktrees/maps-modular/` on branch `feat/maps-modular-sandbox` (from `f3dc82a`); `.claude/worktrees/` added to `.git/info/exclude` (local only, no tracked change). Tests in the worktree run with `PYTHONPATH=<worktree>` so the shared `.venv` imports the worktree's `racer`.
* **Files (edited):** `PROJECT_CONTEXT.md` (§4 layout, §5.2 car variants / physics version, §6 map library, §7.1 obs_spec v2, §7.2 action layouts, §8 eval v2 + `maps7`, §9 upgradable engine + comparing models, §10.5 AI Sandbox, §12 timeline + MoSCoW, §16 D19–D24), `log.md` (T52–T59)

### 2026-10-10 — Claude Code (Sonnet 5.5) — Next steps recorded, overnight queue handed over

* **Task:** Owner's usage limit is nearly used up: faster models (T50) and arena frame rate (T51) are the immediate next tasks; the owner runs the overnight queue by hand and we meet afterwards.
* **Status:** DONE (docs only). The owner starts: `.venv\Scripts\python scripts\queue_overnight.py configs\queue
ight1.yaml` (11 runs × 3M steps, ≈ 6 h; `--dry-run` previews; re-running resumes). Needs a clean tree for code/config paths, laptop plugged in, lid open.
* **Files (edited):** `log.md` (T50, T51, T46 now `[~]`, Current Status)

### 2026-10-10 — Claude Code (Sonnet 5.5) — T44: R003 (from scratch on final_mix) result

* **Status:** DONE. 5M steps, 50 checkpoints, all evaluated (435 s). Best by D9: **R003_mix_scratch@900k: 100 % showcase, 80 % held-out, 89 % overall.**
* **Details:** several checkpoints reach 100 % showcase / 80 % held-out (@900k, @1000k, @1100k, @1700k, @1900k), so mixed-track training from scratch generalises within ~1M steps, unlike R002's fine-tune (10 % held-out). Best laps @900k: TG 57.95 s, Technical 89.95 s, Fast Flow 97.40 s, i.e. ≈ 24 % **slower** than the baseline (46.6 / 81.55 / 70.4 s; `lap_ratio` 1.24). Reliable but not yet fast: lap time is the next thing to improve (longer training, lower final LR / entropy, or an off-road / speed reward tweak). The D8 stage-3 chain run from R002 was dropped (R002 is a weak parent); `gen_wide` fine-tuning stays documented as a negative result for the report.
* **Consequence for the overnight queue:** 3M-step runs from scratch on `final_mix` are enough to reach reliability, so `night1.yaml` (11 × 3M ≈ 5.5 h + evaluation ≈ 30 min) stands as drafted. The owner starts it; no Claude session is needed.
* **Housekeeping:** no background tasks or servers left running.

### 2026-10-10 — Claude Code — T44: R002 result, R003 (from scratch on final_mix) started

* **Status:** WORKING. T44 stays `[~]`.
* **R002** (`--parent R001@1200k`, `gen_wide`, `ppo_finetune`, 3M steps, 30 checkpoints, all evaluated): **did not generalise.** Best by the D9 rule is @200k (56 % showcase, 10 % held-out), essentially the parent. Late checkpoints: @3000k 44 % showcase / 10 % held-out, @2900k 56 % / 0 %. Training reward rose (≈ 220 → 1375) but `end/offroad` stayed 50–75 % of training episodes; in evaluation 9–10 of 10 held-out tracks end by the all-wheels-off-road rule (1 s). The agent drives faster but still leaves the road.
* **Reading:** a low-LR fine-tune from a single-track model is a weak curriculum step. Candidate fixes (not tried yet, owner/next run to decide): a larger off-road penalty, higher fine-tune LR, more steps, or simply training from scratch on mixed tracks.
* **Started:** `train.py --tag mix_scratch --trackset final_mix --steps 5000000` (it gets id **R003**, since the planned R003 chain stage was not run; D8's "R004 from scratch" is therefore R003). ≈ 50 min plus evaluation. R002's chain stage 3 is on hold until R003's result shows whether the curriculum is worth continuing.

### 2026-10-10 — Claude Code (Opus 5.5) — T50: Check-in rules in CLAUDE.md

* **Task:** Owner asked for rules for sessions that wait on a process (e.g. supervising the overnight queue) and check in periodically.
* **Status:** DONE (docs only).
* **Files (edited):** `CLAUDE.md` (new section "Waiting on a process": check-in gaps under 60 min so the cache stays warm; read only the status file and log tail, except `log.md` and `CLAUDE.md`), `log.md`
* **Notes:** Written while another session was building Week 2 (T45/T47); that session's entries were left untouched. D18 (the queue runs without a Claude session) is unchanged: these rules apply only if the owner chooses to have a session supervise.

### 2026-10-10 — Claude Code (Opus 5.5) — T45, T47: Overnight queue, Model Arena (in progress)

* **Status:** T45 DONE; T47 built and tested, the 20-car FPS check on an idle CPU is still to do (R002 is training meanwhile).
* **T45:** `scripts/queue_overnight.py` tested with a 2-entry queue: a deliberately broken entry (unknown physics preset) FAILED, the next entry still ran and finished; a second start skipped both (resume); exit code 1 because one failed. Test artefacts deleted (`outputs/queue_logs/qtest`, scratch run S001, scratch registry).
* **Files (created):** `scripts/queue_overnight.py`, `racer/env/race.py` (pygame-free race: grid, countdown, laps with interpolated crossing times, gaps from the earliest time each metre was reached, crash → hold `respawn_s` → restart on the centreline, finish timeout; policies batched per model; `act_batch(obs, envs)` so the PID baseline can race too), `racer/render/camera.py`, `racer/render/world.py` (split out of `debug_view.py`), `racer/render/cars.py` (tinted Kenney bodies, 20 palette colours × 4 shapes, rectangle fallback, ghost when respawning), `racer/render/hud.py` (leaderboard, followed-car readout, countdown, results, short labels `R002@1500k`), `racer/render/ui.py` (model picker), `scripts/demo.py` (picker → race → back; `--models`, `--run [--every N]`, `--best`, `--baseline`, `--laps`, `--fair`, `--screenshot`), `configs/demo/arena.yaml`, `tests/test_race.py`
* **Files (edited):** `racer/render/debug_view.py` (uses `Camera` + `TrackWorld`; same public API, `drive.py` unchanged and checked with a screenshot), `tests/test_render.py` (+ headless arena race, + picker logic)
* **Details:** headless, uncapped (while R002 used all cores): 20 cars at ×1 → 66 fps; 11 cars at ×4 → 270 fps. A 2-lap race of 10 R001 checkpoints + the baseline on Training Ground: R001@1300k won (best lap 43.09 s), the baseline finished 9th (46.63 s), @100k and @300k crashed 7 and 6 times.
* **Notes / Errors:** first frames had overlapping labels (full ids) and missing ▲▼▶ glyphs in Share Tech Mono; fixed with short labels and ASCII marks. `race.first_finish` initially took the first car *processed* in a tick, not the earliest interpolated finish (test caught a −0.0002 s gap).

### 2026-10-10 — Claude Code (Opus 5.5) — T42: train.py v2

* **Status:** DONE. `pytest`: 86 passed. Scratch smoke runs: S001 (`final_mix`, 50k, 1,604 steps/s, end to end into the scratch registry, both checkpoints evaluated) and S002 (`--parent R001@1200k`, `gen_wide`, `ppo_finetune`; its 25k checkpoint keeps the parent's skill: TG 42.75 s, 56 % showcase). An obs-spec mismatch with the parent is refused. Smoke artefacts deleted afterwards.
* **Files (created):** `racer/registry/record.py` (`evaluate_and_record`, shared by train.py and evaluate.py)
* **Files (edited):** `scripts/train.py` (`--trackset`, `--parent`, `--curriculum`, `--no-eval`, `--eval-workers`; run row at start, `total_steps`/`wall_clock_min` at the end, each checkpoint registered when saved, all checkpoints evaluated after training, best marked, xlsx regenerated; snapshot gains `parent_model` and `curriculum`), `scripts/evaluate.py` (uses `record.py`), `tests/test_env.py` (episodes report track keys, not display names)
* **Details:** fine-tuning builds a fresh PPO with the new preset's hyperparameters, then loads the parent's policy weights (optimiser state is not carried) and continues the parent's VecNormalize statistics.

### 2026-10-10 — Claude Code (Opus 5.5) — T39, T40, T41, T43: Presets, evaluation, registry, loader

* **Status:** DONE. `pytest`: all pass (new: `test_presets.py`, `test_loader.py`, `test_registry.py`). `physics_sanity.py`: ALL PASSED. R001 (20 checkpoints) + baseline registered and evaluated; `registry/registry.xlsx` has Runs, Models, LB_training_ground, LB_technical, LB_fast_flow, Ablations.
* **Files (created):** `configs/physics/{no_grip_limit,downforce}.yaml`, `configs/env/{rays9,rays5_short}.yaml`, `configs/reward/speed_naive.yaml`, `configs/train/ppo_finetune.yaml`, `configs/eval/v1.yaml`, `racer/registry/registry.py`, `racer/registry/export_xlsx.py`, `scripts/evaluate.py`, `scripts/registry.py`, `registry/runs.csv`, `registry/models.csv`, `outputs/eval/<model_id>/` (21 folders: eval.json + 3 thumbnails), `tests/test_presets.py`, `tests/test_loader.py`, `tests/test_registry.py`; `eval.json` + `thumb_*.png` added to the 20 R001 model folders (D10)
* **Files (edited):** `racer/env/reward.py` (`SpeedReward`, type `speed`), `racer/agents/evaluation.py` (protocol runner, metrics, thumbnails, add-once files + copies, baseline reference, process pool; `evaluate()`/`format_summary()` kept), `racer/agents/loader.py` (`resolve_model` by id: registry → `models/` → `outputs/scratch/`; algorithm dispatch PPO/SAC; `act_batch` runs the torch policy directly), `tests/test_config.py` (every preset in `configs/` must load), `.gitignore` (`registry/registry.xlsx`, `registry/*.csv.tmp`, `outputs/queue_logs/`)
* **Details:**
  - `no_grip_limit`: road/kerb μ = 100 (μ = 10 still bound at full lock at 30 m/s, a 13 g manoeuvre; test caught it).
  - Loader: one row at a time is bit-identical to SB3 `predict`; batches differ by ≤ 2e-7 (BLAS summation order). act ≈ 0.35 ms, env step ≈ 0.11 ms.
  - Protocol v1: 19 episodes per model ≈ 13 s; `evaluate.py --run R001` (20 checkpoints, 6 workers) 81 s.
  - **Baseline (BASE_pid):** 100 % everywhere; Training Ground 46.60 s, Technical 81.55 s, Fast Flow 70.40 s.
  - **R001 under protocol v1** (overfitting, as expected for one track): best = @1800k (67 % showcase, 30 % held-out). @1200k: 44 % showcase (all Training Ground), 0 % held-out, TG 42.80 s (3 starts; 6/6 and 42.70 s with the Week 1 six starts, golden test).
  - `lap_ratio` averages only the showcase tracks a model lapped, so it is only a tie-break among equal completion.
* **Notes / Errors:** first evaluation outputs had float noise in lap times (46.5999…); deleted (my own files from a minute earlier) and regenerated with rounding.

### 2026-10-10 — Claude Code (Opus 5.5) — T37, T38: Pre-flight, track sets

* **Status:** DONE. `pytest`: 56 passed. `bench_env.py --trackset final_mix`: 6,092 steps/s on one core with random actions (short episodes, so many track switches and generations; worst case), 9,155 on Training Ground alone.
* **T37:** commit `aad9d44` (ABS assist + badge, Week 2 decisions; not pushed). `train.py --allow-dirty`: real runs now refuse a dirty tree. `tests/test_golden.py`: R001@1200k still gives 6/6 and 42.70 s on Training Ground (22 s; skips without the model folder).
* **Files (created):** `racer/env/trackset.py` (`resolve_trackset`, `TrackSampler`, LRU `get_track` keyed by name + config hash, max 24), `configs/trackset/{tg,gen_wide,final_mix}.yaml`, `tests/test_golden.py`, `tests/test_trackset.py`
* **Files (edited):** `racer/env/racing_env.py` (`make_env_config(trackset=...)` stores the expanded set in `tracks`; old list snapshots still work and sample with the same RNG calls; `set_track` takes a name or a `Track`), `racer/track/io.py` (`track_folders()`: names resolve in every `tracks/` folder, showcase/heldout/user first), `racer/track/track.py` (`Track.key` = file stem; `name` is the display name), `racer/track/generator.py` (`width_m` may be a range, drawn last so held-out tracks regenerate bit-identically, tested), `scripts/bench_env.py` (`--trackset`), `scripts/train.py` (dirty guard)
* **Details:** generator acceptance with the overrides: `gen_wide` 2.0 attempts, 30 ms per track (width 14–18 m, median min radius 24 m); `final_mix` 4.4 attempts, 71 ms (width 10–16 m, median min radius 19 m). The planned `gen_mixed` preset was dropped: `final_mix` covers it.
* **Notes / Errors:** a heredoc edit of `bench_env.py` hit the backslash problem again; redone with the Edit tool.

### 2026-10-10 — Claude Code (Opus 5.5) — T49: Week 2 decisions recorded

* **Task:** Owner approved the Week 2 plan: (1) commit the ABS work (ABS is human-only and doesn't affect training); (2) any tracks may be mixed into training, more may be created, tracks must be addable at any time, and the final showcase is a faculty-drawn track converted into a track; (3) evaluations stay with the model **and** are copied to `outputs/eval/`; (4) respawn rule as proposed; (5) overnight queues only if they use no tokens, i.e. no Claude session.
* **Status:** DONE (docs only).
* **Files (edited):** `PROJECT_CONTEXT.md` (§4 layout; §6 open track library + final showcase; §8 curriculum chain, track sets, checkpoint `is_best`, dirty-tree rule, overnight queue, eval protocol file; §9 immutability rule + `outputs/eval/`, ablation IDs, registry upsert; §10.1 race rules and code split; §15 config-schema rule; §16 two rows; new open question on paper vs editor drawing), `CLAUDE.md` (model-folder rule exception for eval files), `log.md`
* **Notes / Errors:** a parallel session logged the ABS HUD badge as **T36**, the same ID as the Week 2 plan. Both entries are kept as they are (IDs are never renumbered); "T36 (plan)" and "T36 (ABS badge)" are both done.

### 2026-10-10 — Claude Code (Opus 5.5) — T36: Week 2 (T07) implementation plan

* **Task:** The owner asked for a Week 2 plan (M2: load any 20 models from the registry and race them), with no code until approved.
* **Status:** DONE (plan only, no code). T07 is split into T37–T48 in the Task List. Decisions D7–D18 below are **proposed** and wait for the owner's OK; they then go into PROJECT_CONTEXT (§4, §8, §9, §16) before any code.
* **Files (edited):** `log.md` (T36–T48, this entry, Current Status)
* **Build order:** T37 pre-flight → T38 track sets → T39 presets → T40 evaluation → T41 registry (+ R001 backfill) → T42 train v2 → T43 loader → T44 curriculum runs (CPU busy while T45/T47 are built) → T45 queue → T46 overnight ablations → T47 arena → T48 docs.
* **Proposed decisions:**
  - **D7 Track sets:** new config kind `configs/trackset/`: fixed tracks with weights plus a generated pool (training stream 0, a seed range, generator overrides such as width range and corner tightness). Fields cost 3–14 MB and ~65 ms each, so every worker keeps a small rotating pool (e.g. 4 tracks, one swapped every N episodes) instead of a new track per episode. Held-out tracks are never sampled.
  - **D8 Curriculum:** a chain of separate runs linked by `--parent` (fine-tune policy and VecNormalize stats, lower LR), so every stage is evaluated and registered on its own: R001 (done) → R002 generated wide → R003 final mix. Plus R004 from scratch on the final mix as the "mixed sampling" comparison. **The final mix includes the 3 showcase tracks (~30 %)**, so showcase lap times are on seen tracks; generalisation is reported on held-out (and user-drawn) tracks only.
  - **D9 Evaluation protocol v1** (`configs/eval/v1.yaml`): §8 protocol (3 showcase × 3 starts + 10 held-out × 1 start), standing starts on the centreline at evenly spaced points, deterministic policy. **Best checkpoint** = highest completion over all 19 episodes, ties broken by mean showcase lap time relative to the baseline. The PID baseline is registered as pseudo-model `BASE_pid` for the E8 comparison.
  - **D10 Eval artefacts in model folders:** `model.zip`, `vecnormalize.pkl` and `config.yaml` are never touched. `evaluate.py` may **add** `eval.json` and `thumb_<track>.png` once (as §9 lists them); it never overwrites them, and a new protocol writes `eval_v2.json`. This clarifies the CLAUDE.md rule "never edit a saved model folder".
  - **D11 Registry:** the CSVs are upserted by id with an atomic rewrite (evaluation updates rows, so pure append isn't enough). Scratch runs use their own registry in `outputs/scratch/registry/`. `registry.xlsx` is regenerated, so it is git-ignored (proposed).
  - **D12 Reproducibility:** commit the pending T35 work first; `train.py` refuses real (`R`) runs on a dirty tree unless `--allow-dirty`, so `git_commit` in snapshots always points at real code.
  - **D13 Presets without schema changes:** ablations are config-only where possible (`no_grip_limit` = road/kerb μ 10, grass unchanged, so leaving the road still costs), so no new required keys appear in existing schemas and old snapshots stay valid. If a new key is ever unavoidable, add an explicit snapshot migration. `speed_naive` is a new reward type. Deferred: SAC (E7), `rays5_curvature` / obs_spec v2 (E6).
  - **D14 Arena race rules:** ghost cars; a staggered 2-wide grid behind the start line; 3-2-1 countdown; N-lap race (default 3), then a results table. A crash respawns the car on the centreline where it crashed, from standstill after 1 s, and counts on the leaderboard (no DNF). The gap is in seconds (the time since the leader passed the same distance). Each car uses its own trained physics; `--fair` gives everyone the default physics.
  - **D15 Arena code split:** the multi-car race is simulation, so it lives in pygame-free `racer/env/race.py`. Rendering splits `world.py` and `camera.py` out of `debug_view.py`, ready for Week 3's textures.
  - **D16 Model picker:** a basic in-window registry list (sort by column, type to filter, multi-select) in Week 2, plus `--models` on the command line. Week 3 then only restyles it.
  - **D17 Run IDs:** curriculum runs R002–R004; ablation runs are tagged by experiment (e.g. `e2_nogrip_s1`), so the xlsx Ablations sheet groups them by tag.
  - **D18 Overnight queue:** keeps Windows awake through `SetThreadExecutionState` only while it runs (no power-plan or global changes). The laptop must stay plugged in with the lid open (or lid-close set to "do nothing" by the owner).
* **Estimated compute:** R002 ≈ 30 min, R003 ≈ 40 min, R004 ≈ 50 min (≈ 10 min per 1M steps), plus ~2–3 min of evaluation per run; one night ≈ 45M steps (e.g. 5 ablation configs × 3 seeds × 3M).
* **Risks:** multi-track training from scratch may need more than 5M steps (R004 measures it before the ablation queue is sized); `speed_naive` may not show visible hacking while the wrong-way / stuck terminations are on (if so, try a looser episode preset as a separate, logged variant); checkpoint reliability oscillates (seen in R001), which the eval-based `is_best` handles.

### 2026-10-10 — Claude Code (Opus 5.5) — T36: ABS badge on the HUD

* **Task:** Owner: show whether ABS is on or off in the UI.
* **Status:** DONE. Checked with a headless screenshot; `pytest`: 48 passed.
* **Files (edited):** `scripts/drive.py` (badge next to the brake bar in human mode: green "ABS ON (K)", red "ABS OFF (K)", amber "ABS ACTIVE" while ABS is limiting the brake), `racer/render/debug_view.py` (`badge()`), `configs/demo/render.yaml` (colour `abs_active`), `log.md`

### 2026-10-10 — Claude Code (Opus 5.5) — T35: Braking feel → ABS assist for human driving

* **Task:** Owner: braking feels weak. Change physics only if retraining costs < 15 min and it improves UX; otherwise leave it.
* **Status:** DONE; the owner's drive check of the ABS values is pending. `pytest`: 48 passed (+2 ABS tests). `physics_sanity.py`: ALL PASSED (physics untouched).
* **Diagnosis (100 km/h to 0, default physics):** keyboard full brake (wheels lock) 42.0 m; threshold braking 35.3 m; brake force raised to 20 000 N: still 42.0 m. Brake force is not the limit, grip is, and a key can only ask for 100 % brake, which locks the wheels (kinetic μ 0.9, steering scaled down). The model outputs an analog brake, so it doesn't have this problem.
* **Decision:** no physics change and no retraining. Instead, an ABS assist for human input only.
* **Files (edited):** `racer/physics/car.py` (`Car.abs_brake_limit(wheels, margin)`: a pure query of the largest brake command inside the friction circle, accounting for the current lateral demand; it doesn't change the step), `scripts/drive.py` (human brake key capped by ABS; `K` toggles; HUD shows the state), `configs/demo/render.yaml` (`debug.abs`: enabled, margin 0.97, min_brake 0.25), `tests/test_physics.py`, `README.md`, `PROJECT_CONTEXT.md` (§16), `log.md`
* **Notes / Errors:** if the owner still wants race-car braking (≈ 1.5 g+), that is a *grip* change (road μ or the downforce preset), which changes cornering too and needs a retrain (≈ 12–15 min to R001's quality); not done. Heredoc Python edits keep mangling Windows backslashes: write edit scripts to a file instead.

### 2026-10-10 — Claude Code (Opus 5.5) — T34: Models local only, backup, untracked

* **Task:** Owner: keep trained models local. Back them up under `C:\Users\Jack\`, remove them from GitHub with a new commit.
* **Status:** DONE. Backup verified identical with `diff -rq` (20 folders, 33 MB). `git check-ignore` matches `models/R001_*/model.zip`; the 20 model folders are still on disk.
* **Backup:** `C:\Users\Jack\ML_Sem5_models_backup\` (outside the project, so it is not touched by git).
* **Files (edited):** `.gitignore` (`/models/*` ignored, `!/models/.gitkeep`), `PROJECT_CONTEXT.md` (§4 layout note, §16 row), `README.md` (layout row), `log.md`
* **Git:** `git rm -r --cached models` (60 files of R001 untracked; `models/.gitkeep` re-added), committed and pushed to `origin/main`.
* **Notes / Errors:**
  - The models are gone from the latest commit, but still exist in the history of `42189b3` and `99e8947` on GitHub. Removing them from history would need a rewrite and force push (as in T15); not done, not requested.
  - A heredoc Python edit failed on `\U` in a Windows path; the `git rm --cached` that followed still ran (intended anyway). Edits redone with the Edit tool.

### 2026-10-10 — Claude Code (Opus 5.5) — T33: Help text wrap, sim-speed check, camera

* **Task:** Owner: (1) the help line is clipped at the window edge; (2) changing the sim speed seems to make the model behave differently. Explain, and fix if unintended.
* **Status:** DONE. `pytest`: 46 passed.
* **Findings (sim speed):** the simulation does **not** depend on sim speed. Sim speed only changes how many recorded physics substeps are *displayed* per frame; `env.step` and `model.act` get exactly the same inputs. Verified with new `--trace` CSVs (one row per env step): R001@1200k at ×0.25 / ×1 / ×4 for 800 steps, and R001@100k across 2 episodes with a termination and reset at ×0.25 vs ×4, are **bit-identical**. What *did* change with speed was the camera: it closed 12 % of the gap to the car per rendered **frame**, so at ×4 (4× more motion per frame) it lagged far behind and the car drifted off-centre, which can look like different driving.
* **Files (edited):** `racer/render/debug_view.py` (`update_camera(..., sim_speed)` uses `1 − (1 − lerp)^sim_speed`, so the camera follows per simulated time; new `wrap()`), `scripts/drive.py` (help is a list of items wrapped to the window width, shown by default, `H` hides it; new `--sim-speed` and `--trace` options), `README.md` (command row), `log.md`
* **Notes / Errors:** a heredoc turned `\n` escapes into real newlines inside two string literals (SyntaxError); fixed with exact edits. Trace CSVs were written to the session scratchpad, not the project.

### 2026-10-10 — Claude Code (Opus 5.5) — T32: README Quickstart

* **Task:** Owner asked for a Quickstart in `README.md`: how to train, how to play, how to replay a model.
* **Status:** DONE (docs only).
* **Files (edited):** `README.md` (new "Quickstart" section with Play / Train / Replay subsections, between Setup and Commands; status line now says M1 is done, with R001's lap time), `log.md`

### 2026-10-10 — Claude Code (Opus 5.5) — T31: drive.py crash fix + free drive for humans

* **Task:** Owner: `drive.py` crashes after a second, for both human and model driving. Asked for humans to be able to hit walls without the session ending, while the model keeps the training behaviour.
* **Status:** DONE. `pytest`: 46 passed (+3: walls are solid in free drive and nothing terminates; free drive is off by default; human-mode window runs 600 frames headless). A weak checkpoint (R001@100k) in model mode now runs through several terminations and resets without crashing.
* **Root cause:** `drive.py` sliced a `deque` (`poses[1:]`) in the termination-banner check → `TypeError` on **every** episode end (wall, off-road, wrong way, or "stuck" after 3 s of standing still). The headless test had only run the baseline, which never terminates, so it missed this.
* **Files (edited):** `racer/env/racing_env.py`, `scripts/drive.py`, `tests/test_env.py`, `tests/test_render.py`, `log.md`
* **Details:**
  - New `RacingEnv.free_drive` flag (default `False`; training never sets it). When on: a substep that would hit the barrier is undone and the car stops against the wall; no termination and no truncation; the would-be reason (`wall`, `offroad`, `stuck`, `wrong_way`) goes into `info["warning"]`.
  - `drive.py` sets `free_drive` from the mode every frame: human (`1`) = free drive; baseline (`2`) and model (`3`) = training rules, i.e. banner then random reset. The HUD shows the active rule and any warning.
* **Notes / Errors:** first edit put a statement inside the `elif` chain of key handling (SyntaxError); moved to once per frame.

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
