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

* **Last updated:** 2026-10-24 (Claude Opus 5.5). **Where we are:** start of Week 3. M1 and M2 reached; AI Sandbox built (T65–T75, T101, T106); map library + eval protocol v2; upgradable engine (D21–D23). **Neuroevolution (D26) built and merged to `main`** (PR #1, 2026-10-24; 256 tests pass at T107). Best PPO: `R015_t50_narrow7_s1@4600k` (seen 64 %, unseen 88 %, lap ratio 0.97) but it doesn't reproduce (R018, R019). GA σ test: σ 0.02 (S004: seen 100 %, unseen 87.5 %, lap ratio 0.997; one seed). Nothing is running. **Next:** T97 (owner starts `configs/queue/e10_ga.yaml`, then evaluate, back up, record E10); T83 (owner decision: run-to-run variance); **T97 queue ready on `main`** (4 runs incl. GA seed 2 and PPO under GA rules; GA tries start on the start line, T115). T08 / M3 Week 3 visuals (effects, brain panel, minimap, Evolution Race, Draw-a-Track, Human vs AI, video recorder); then T09 / M4. History of earlier status lines: see the Change Log.
* **E10 done and merged (T121, 2026-10-24):** the queue finished (R020–R023), results are in PROJECT_CONTEXT §11/§16 (PPO under GA rules beat both GA seeds; single seeds), `feat/draw-a-track` is merged into `main` (281 tests pass). Nothing is running. **Next:** T110–T114 (Draw-a-Track in the game window, T112–T114 still open); T83 (variance decision, plus a second PPO seed of R023).
* **(Superseded by the merge above) Two working copies, until the worktree is removed (owner, 2026-10-24, T117):**
  - **This folder (`ML Sem5 Project`, branch `main`) runs the E10 queue** (`configs/queue/e10_ga.yaml`). Don't edit `racer/`, `scripts/`, `configs/`, `tracks/` here and don't switch branches: every queue entry refuses to start on uncommitted code and runs whatever is checked out. Check-ins read only `outputs/queue_logs/e10_ga/state.json` and the log tail. The queue writes `registry/*.csv`, `registry/run_notes.yaml`, `outputs/eval/` and `models/` here.
  - **Development happens in the worktree `C:\Users\Jack\Desktop\ML Sem5 Project - dev`** (branch `feat/draw-a-track`). It has its **own `.venv`** (owner, 2026-10-24: Python 3.11.9, same package versions as this folder's, `racer` installed editable from the worktree): run `.venv\Scripts\python ...` there, no `PYTHONPATH` needed. Remove it with the worktree.
  - **Keep the two mergeable:** the dev branch never edits `registry/`, `outputs/eval/`, `models/` or `configs/queue/e10_ga.yaml`, and never changes what an existing preset/setting means (extend, don't change). In `log.md` it only adds tasks at the bottom and Change Log entries at the top; "Current Status" is rewritten at merge time.
  - **When the queue finishes:** evaluate/back up and record E10 results here on `main` and commit; then merge `feat/draw-a-track` into `main` (resolve `log.md` by keeping both sets of entries), run the full suite, and remove the worktree (`git worktree remove`).
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
* [x] T07: **M2 (Week 2):** configs, train/evaluate, checkpoints, registry and xlsx export, loader, curriculum, basic 20-car Model Arena
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
* [x] T46: First overnight queue (owner starts it; `configs/queue/night1.yaml`, started manually after 2026-10-10): E2 no grip limit, E3 speed reward, E4 rays9 / rays5_short, E5 single- vs multi-track; from scratch on the T44 track set, seeds sized to fit ~8 h (from T44's measured steps). **Done when:** all runs are registered and evaluated; failures noted here.
* [x] T47: Basic Model Arena (D14–D16): pygame-free `racer/env/race.py` (N `RacingEnv`s on one shared track, each with its model's own physics and obs spec; grid, countdown, laps, gaps, respawns); `racer/render/world.py` + `camera.py` split out of `debug_view.py` (`drive.py` keeps working); `scripts/demo.py arena` with a basic model picker, 20 distinct cars (4 shapes × 5 colours), labels, leaderboard, `Tab`/`F`/`R`/`T`/`Space`/`+−`, `--models`, `--fair`. **Done when (M2):** 20 registry models race on each showcase track at a steady 60 FPS, and a headless test runs a 20-car race to the finish.
* [x] T50: **NEXT (owner priority): faster models.** *(Done 2026-10-11: target met by `R015_t50_narrow7_s1@4600k` (best) and `@4800k`; see Change Log. Was: queue `configs/queue/t50.yaml`, `rays7_narrow` and `rays9` on `maps7`, 5M steps each; target now v2 `lap_ratio` < 1.0 and `unseen_completion` ≥ 80 %.)* R003@900k is reliable (100 % showcase, 80 % held-out) but ≈ 24 % slower than the baseline. Try from-scratch runs on `final_mix` aimed at lap time (longer training, lower final LR / entropy, off-road / speed reward tweak); compare with the E1 reference runs from the overnight queue. **Done when:** a model beats the baseline lap time on the showcase tracks with held-out completion ≥ 80 %.
* *Owner decisions 2026-10-11 (before going away; work continues up to T62 without the owner, stopping on any critical decision):* T50 trains `rays7_narrow` and `rays9` from scratch on `maps7`, 5M steps each, plus follow-ups if a clear fix shows (≤ ~4 h of training in total). **T50 target is now:** protocol v2 `lap_ratio` < 1.0 **and** `unseen_completion` ≥ 80 %. T62 may install the CUDA torch build into a project-local `.venv-gpu` (git-ignored). Commit locally after each verified step; no push.
* [x] T51: **NEXT (owner priority): arena frame rate.** Re-measure 20 cars at ×1 on an idle CPU (the 66 fps was measured while training used all cores); fix if below a steady 60 fps (batched inference every 3rd frame, text caches, off-screen culling). Last check for M2.
* [x] T48: Docs: PROJECT_CONTEXT (§4 layout, §8 curriculum and track sets, §9 eval artefacts, §11 run IDs, §16), README (commands, registry, arena), `log.md` Current Status.
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
* [x] T62: **GPU benchmark (owner, 2026-10-11).** Measure before switching: the GTX 1650 (4 GB, driver 610.60, already installed) only speeds up the PPO update (≈ 60 % of training time, §8); env stepping stays on the CPU. Install the CUDA build of the **same** torch version (2.14.1) into a separate `.venv-gpu` (project-local, git-ignored; the main `.venv` and `requirements.txt` stay untouched), then time 200k steps of `ppo_default` on CPU vs GPU, also with larger minibatches (512, 1024) where a GPU helps most. Check that a GPU-trained model loads and drives identically on the CPU (demo and evaluation stay on the CPU). **Also measure parallel runs (owner):** total steps/s of 1 run × 6 envs vs 2 runs × 3 envs vs 3 runs × 2 envs, on CPU and with updates on the GPU, plus RAM per run (≈ 1.5 GB for 6 env workers today). Today a single run's env workers sit idle during its PPO update (single-threaded), so overlapping runs may raise total throughput even without the GPU. **Done when:** steps/s for each setting are logged here with a recommendation (device, runs in parallel, envs per run).
* [x] T64: **NEXT. Condition met by T62: 2 runs × 6 envs on the CPU give 1.62× the total throughput (2754 vs 1704 steps/s).** **Parallel queue (only if T62 shows a higher total throughput).** `queue_overnight.py --parallel N` runs up to N queue entries at once (each with its own log; `state.json` stays the single source of progress; evaluation after training limited so N runs don't each start 6 eval workers); `n_envs` per run from the queue entry or T62's recommendation. Run ids stay unique when two runs start together (`next_run_id` must reserve its id, e.g. by creating the folder/lock atomically). **Done when:** a 2-slot scratch queue with 3 short entries finishes with distinct run ids and correct state, and total throughput matches T62.
* [x] T63: *Decided by T62: not done. The GPU trains at 0.64–0.90× the CPU's speed, so the ≥ 1.3× condition is not met; training stays on the CPU.* **Train on the GPU (only if T62 shows ≥ 1.3× faster training).** `device: cpu|cuda` in the train presets (default `cpu`, so old configs mean the same), `train.py --device`, recorded in the snapshot; models are always saved for and loaded on the CPU; `requirements-gpu.txt` with the CUDA torch pin; README setup section; queue entries may use `--device cuda`. Requirements stay as they are: CPU-only machines still train and run everything, saved models load anywhere. **Done when:** a GPU run and a CPU run of the same config give comparable results, and every existing model still loads in the CPU `.venv`.
* [x] T61: Re-evaluate the best checkpoint of every run with protocol v2 (`evaluate.py --model <best ids>`), so all models are comparable on the map library (≈ 1 min per model). Optional: every checkpoint overnight (≈ 430 checkpoints ≈ 1 h on 6 workers).
* *T65–T72 (2026-10-11): the AI Sandbox (D24), built in the steps S0–S6 of `docs/SANDBOX_PLAN.md` §5. Owner decisions (T65): Streamlit 1.65.0; the sandbox may start the queue in its own console; localhost only; retroactive run notes in `registry/run_notes.yaml`.*
* [x] T65: Sandbox implementation plan v2: owner's 4 decisions recorded (PROJECT_CONTEXT §10.5, §16), plan refined against the code (data sources checked, build steps with "done when"), tasks T66–T72.
* [x] T66: **S0** Install Streamlit 1.65.0 into `.venv` (dry run: +26 packages, no pin changes); pin it and psutil 7.2.2 (already in `.venv`, never recorded); `.gitignore`; `scripts/sandbox.py` launcher (localhost, headless, no usage stats, no files written to the user profile) + empty pages. **Done when:** the app opens on `localhost:8501` with every page listed; `pip check` clean; no `~/.streamlit/` created; the server stops cleanly.
* [x] T67: **S1** `racer/sandbox/data.py`, `explain.py`, `glossary.yaml`, `registry/run_notes.yaml` (why + outcome for R001–R017, from PROJECT_CONTEXT §11). **Done when:** tests pass, including: every key of every preset in `configs/{physics,reward,train,env,trackset}` and of every snapshot in `models/` has a glossary entry; the diff of R004 vs R008 is only the ray list (+ tag/seed/ids).
* [x] T68: **S2** Model Explorer + `diagrams.py` (ray diagram on a straight and on a real map with the real sensor code; network diagram) + `curves.py` (TensorBoard). **Done when:** R001 (v1), R015 (v2 obs) and a 3-output scratch model display correctly; the ray-diagram hit distances match the sensor within 0.1 m (test).
* [x] T69: **S3** Compare page (2–4 models: config diff with glossary notes, shapes, metrics within one protocol, overlaid curves, thumbnails, *Launch these*). **Done when:** R004 vs R008 vs R015 shows the expected differences; mixed protocols give a warning.
* [x] T70: **S4** Launch page + `racer/sandbox/procs.py` (psutil; pid + creation time) + `launcher.py` + `demo.py --title`. **Done when:** two game windows run at once from the sandbox; *Stop* closes one and leaves the other; incompatible models can't be picked; instances survive a sandbox restart in the list.
* *Owner check-in during T71 (2026-10-11): the end-to-end queue test also queues the R015 reliability runs through the sandbox form: R015's setup with seed 2 (5M) and seed 1 for 8M steps, plus a short trial run, 2 at a time (≈ 2–2.5 h). The sandbox server is stopped at the end (not left running).*
* [x] T71: **S5** Build & Queue + `builder.py` (v1 → v2 obs conversion, new `configs/env/sb_<tag>.yaml`, `--set` for hyperparameters, live obs/param count, time estimate) + `queue.py` (git-ignored `configs/queue/sandbox.yaml`, atomic writes, *Start queue* in its own console) + `pid` in the queue's `state.json`. **Done when:** a 20k-step trial entry built in the form trains through the queue and shows in the Explorer; *Start queue* is refused while a queue or training runs.
* [x] T73: Record the R015 reliability runs R018 (seed 2, 5M) and R019 (seed 1, 8M), queued through the sandbox (owner): commit registry rows + eval copies, PROJECT_CONTEXT §11/§16, `registry/run_notes.yaml`, README status.
* [x] T72: **S6** Overview page (counts, leaderboard, queue status, running games), polish, README + PROJECT_CONTEXT, full tests. **Done when:** the owner can do all 4 goals without the terminal.
* *Owner decision (2026-10-11, D25): the sandbox opens in its own native window (pywebview on Edge WebView2, a "mini browser") instead of a browser tab. Streamlit stays the app; `--browser` keeps the old behaviour as a fallback.*
* [x] T74: Record D25 in the docs: PROJECT_CONTEXT §10.5, §14.1, §16; `docs/SANDBOX_PLAN.md`; README.
* [x] T75: Sandbox window: dry-run + install pywebview into `.venv`, pin it, `.gitignore`; `scripts/sandbox.py` opens a pywebview window (storage in git-ignored `outputs/sandbox/webview/`, private mode), closing the window stops the server; `--browser` fallback; tests. **Done when:** the window opens the sandbox, closing it leaves no process or socket on 8501, nothing new in the user profile, `pip check` clean, tests pass.
* [x] T76: **Flaky registry lock (found in T75).** `tests/test_queue.py::test_registry_writes_from_parallel_processes_are_not_lost` fails about 1 in 5 runs (also without T75's changes): on Windows, creating `registry.lock` while another process is still deleting it raises `PermissionError`, but `file_lock` (`racer/registry/registry.py:85`) only retries on `FileExistsError`. Can also hit real parallel queue runs. Fix: treat that `PermissionError` as "held" and retry. **Done when:** the test passes 50 runs in a row.
* [x] T77: README: a task-based command list under Quickstart ("what can I do / show, and which command"), every showcase command checked headless.
* [x] T78: Project integrity check (owner): tests, deps vs pins, configs, tracks, registry vs model folders vs eval copies, every model loads, docs vs files.
* [x] T79: Integrity test (only if T78 finds gaps the suite doesn't cover).
* [x] T80: Week 3 readiness scan: open tasks, running processes, git state, blockers; report to the owner.
* [x] T81: **Model backups (owner).** The backup in `C:\Users\Jack\ML_Sem5_models_backup\` stopped at R001. Add `scripts/backup_models.py` (copies model folders that are missing from the backup; never deletes or overwrites; checks file sizes), the backup path in a config, and instructions (CLAUDE.md + README): back up after every completed run. Then bring the backup up to date (R002–R019).
* [x] T82: **Key list in the Model Arena (owner)**, also for the Evolution Race (`--run`): an on-screen keys panel, toggled with `H`, plus a hint so it can be found.
* [ ] T83: **Open decision for a later session (owner): R015's run-to-run variance.** Choose how to make results reliable: more seeds per setting, keeping exploration up (entropy, e.g. `--set ppo.ent_coef=…`), a curriculum, or a mix. Inputs: R015/R018/R019 (PROJECT_CONTEXT §11, §16). Decides what the overnight queue runs during Week 3.
* *T84–T86 (2026-10-13, owner): try neuroevolution (a genetic algorithm over network weights, as in a YouTube video) next to PPO. Owner choices: one track first (Training Ground), then multi-track; network 256×256 like PPO; plan + quick benchmark before building.*
* [x] T84: Record D26 (neuroevolution, E10) in PROJECT_CONTEXT (§3, §8, §11, §16) before any code.
* [x] T85: Plan `docs/NEUROEVOLUTION_PLAN.md`: algorithm, fitness, saving as an SB3 model, integration, build steps with "done when" checks, open decisions.
* [x] T86: Benchmark generation speed with a throwaway script (scratchpad, not the repo): 20 random 256×256 cars on Training Ground, 1 process vs a 6-process pool. **Done when:** time per generation and the estimate for 4000 generations are logged.
* [x] T87: Owner decisions on D26 (2026-10-13): mixed maps from the start (K maps per generation, + a single-track control run), population sized to a run budget, stop on a validation plateau with a cap, elites = top 25 %, a try ends at one lap or a failure, pause/resume at any moment. Benchmark 20/50/100 cars × 3 maps; record in PROJECT_CONTEXT, then the plan.
* [x] T88: Owner decisions on D26 (2026-10-13): **one random map per generation**, from the 7 training maps (slower plateau accepted; pause/resume covers it); 5 workers (≈ 60 % CPU); **10 inputs** (6 rays ±5.5°/±17°/±55°, heading sin/cos, forward + sideways speed; yaw rate, on-road fraction, previous action dropped); **track-limits failure** (car centre over the road edge). Recorded in PROJECT_CONTEXT, plan, log. Open: a 0° ray, 50 cars, validation settings, 200 s cap.
* [x] T89: Owner question: required stopping distance = top speed → speed of the tightest corner over all maps; does ±5° instead of ±5.5° fix the forward sight? Analysis (throwaway `stopping.py` in the session scratchpad) recorded in the plan's open proposal 1.
* [x] T90: Owner decisions (2026-10-13): add a 0° ray (200 m); drop sideways speed (counter-argument given; a with-v_lat ablation proposed). Owner question: can every generation be recorded (SQLite)? Answer: yes, as a lineage database (parent + noise seed per car), proposed in the plan. Record in PROJECT_CONTEXT, plan, log.
* [x] T91: Owner confirmed the remaining proposals (50 cars, validation settings, 200 s cap, lineage + checkpoint every 100 generations; no full-weight record, disk space). Plan finalised as the implementation contract; build tasks added; pre-implementation commit.
* *T92–T98: build steps S1–S7 of `docs/NEUROEVOLUTION_PLAN.md` ("done when" checks are there). Uncertain decisions (dropped inputs, one map per generation, σ) are revised if results show them to be poor.*
* [x] T92: S1 env: optional track-limits termination (car centre over the road edge, off by default); obs preset `rays7_lean` (7 rays incl. 0°, heading sin/cos, forward speed).
* [x] T93: S2 `racer/agents/neuroevolution.py`: weights, numpy forward, fitness, selection + mutation from seeds, lineage rebuild.
* [x] T94: S3 export to a model folder (SB3 PPO zip + identity `vecnormalize.pkl`), `ga` algorithm entry, compat.
* [x] T95: S4 evolution loop: workers, validation + stopping, checkpoints + registry, TensorBoard, lineage SQLite, pause/resume, `ga_default.yaml`, `train.py` dispatch, glossary, `.gitignore`, backup of the lineage.
* [x] T96: S5 σ test: σ 0.02 vs 0.05, scratch runs on `train7` with the same budget of 10M env steps each (owner, 2026-10-15: run now; equal compute instead of 200 generations).
* [ ] T97: S6 control run (Training Ground) and main run (7 maps); evaluate; back up. Owner decision (2026-10-15): queued in `configs/queue/e10_ga.yaml` for the owner to start; then evaluate (automatic), back up, record results in PROJECT_CONTEXT §11 and `registry/run_notes.yaml`.
* [x] T98: S7 docs (README, PROJECT_CONTEXT, log). The E10 results are added with T97.
* [x] T99: Owner request (2026-10-15): check CLAUDE.md, PROJECT_CONTEXT.md, README.md and log.md against the D26 decisions and fix what is out of line (docs only).
* [x] T100: Owner decisions (2026-10-15): selection stated exactly (all 50 ranked, top 12 kept unchanged, the same 12 are the only parents of the other 38); model ids of GA runs get `GA_` before the tag. Docs updated; pre-implementation commit; then T92 onwards.
* *T101–T107 (2026-10-15, owner): docs for presets and agents; GA budget in generations; steadier validation, checkpoints and evaluation; export any car.*
* [x] T101: README: list of every model preset and how to use it; owner (mid-task): also a way to see them visually, for all models → sandbox **Presets** page.
* [x] T102: CLAUDE.md: agent workflow for any change, the elements to update or keep in mind (queue, backup, train, tests, sandbox, physics, …), the architecture to follow; owner's three working lines (ask for critical decisions midrun, fix bugs found midrun, add tests).
* [x] T103: GA budget in generations (`ga.max_generations`, `train.py --generations`); PPO/SAC keep `--steps`; sandbox Build & Queue and queue display.
* [x] T104: Validation every 20 generations of all elites (max of their mean scores), a checkpoint at every validation = the best-validated elite.
* [x] T105: After a GA run, evaluate the 10 best-validated checkpoints + the last 10 (settings).
* [x] T106: Export any car of any generation from the lineage (`scripts/export_ga.py`, and in the sandbox).
* [x] T107: Docs (PROJECT_CONTEXT, README, plan, glossary) for T103–T106; full suite; review; commit.
* [x] T108: Owner (2026-10-24): `feat/neuroevolution` merged to `main` (PR #1); "Current Status" rewritten to match.
* [x] T109: Owner (2026-10-24): `REVIEW_LATER.md` listing every pending owner decision (owner leaning: drop PPO if GA wins; decided after T97).
* [x] T110 (editor part dropped by the owner, D28; sandbox page done in T119): T08 Draw-a-Track mode (owner: first Week 3 feature, PROJECT_CONTEXT §6, §10.1). Branch `feat/draw-a-track`. Drawing → track (T111) and the sandbox page (T119) are done; **open (owner): is the game-window editor (T112–T113) still wanted, now that drawing lives in the sandbox?**
* [x] T111: `racer/track/sketch.py`: drawn points (metres) → smoothed control points → `TrackSpec`, same validation as the generator; auto-scale a too-small drawing up to the minimum length / corner radius; direction as drawn. Settings in `configs/demo/editor.yaml`. Done when: tests for circle, cw/ccw, small doodle, sharp square, figure-eight, too few points pass.
* [!] T112 (dropped by the owner, D28): `racer/render/editor.py`: draw freehand (drag) or click points, undo, clear, width ±, direction flip, live validation message with the clash marked, preview of the built road. Done when: a headless test drives it with synthetic events into a valid track.
* [!] T113 (dropped by the owner, D28): `demo.py --draw`: editor → model picker (drawn track preselected) → race; Esc returns to the editor with the drawing kept; `S` saves to `tracks/user/`. Done when: headless screenshot run on a scripted drawing races a model.
* [!] T114 (dropped by the owner, D28): Docs (PROJECT_CONTEXT §6/§10.1, README, glossary if needed), full suite, review, commit.
* [x] T115: Owner (2026-10-24, mid-T110): GA training tries always start on the start/finish line (centred, aligned, standing) on every map, not at a random point. New setting `ga.start: line | random` (`ga_default`: line; missing = random, so older runs resume unchanged). Done when: test shows line starts and an unchanged map sequence; docs (PROJECT_CONTEXT §8, §16, glossary, README). T110 paused meanwhile.
* [x] T116: Owner decisions for the E10 queue (2026-10-24): control run (TG, 300 gen), GA main seeds 1 and 2 (no generation cap, plateau stop), PPO under GA rules (rays7_lean + track limits, train7, 5M steps) last; σ stays 0.02; commit and merge locally; the owner starts the queue.
* [x] T117: Owner (2026-10-24): the E10 queue runs in this folder on `main`; development continues in a git worktree (`ML Sem5 Project - dev`, branch `feat/draw-a-track`); both merged when the queue finishes. Rules noted under Current Status.
* [x] T118: Owner (2026-10-24): D27, every feature is reflected in the AI Sandbox; rule in `CLAUDE.md`, PROJECT_CONTEXT §10.5/§16, memory.
* [x] T119: Draw-a-Track in the sandbox (D27; owner: draw inside the sandbox, save to `tracks/user/`): *Draw a Track* page with a canvas (`streamlit-drawable-canvas` 0.13.0), validated preview, save, *Race on it* → Launch. Done when: headless page test passes (done) and a real browser run works (done).
* [x] T120: Owner (2026-10-24): own `.venv` in the dev worktree, rebuilt on Python 3.11.9 (the first one was 3.14 and empty); package versions identical to the main venv.
* [x] T121: E10 queue finished (2026-10-24); backup checked; E10 results recorded (PROJECT_CONTEXT §11/§16, `registry/run_notes.yaml`); `feat/draw-a-track` merged into `main`, full suite 281 passed. Worktree removed in T122.
* [x] T122: Remove the dev worktree and copy the drawn track `wiggly_wiggly` to `main` (owner, 2026-10-24).
* [x] T123: Owner (2026-10-24): finish all demo and analysis work, then experiment freely. Plan `docs/DEMO_PLAN.md`, decision D28. Worktree `ML Sem5 Project - dev`, branch `feat/demo-complete`. Done when: plan and decisions committed.
* [x] T124: `demo.py` shell: mode menu (arena, Evolution Race, Human vs AI, swarm), shared window and state. Done when: headless, the menu starts each mode and returns.
* [x] T125: Audio (§10.4): `gearbox.py`, `audio.py`, toggle `S` + button in every race window; on when watching a race, off for training/sandbox/analysis; `--sound/--no-sound`; Launch checkbox. Done when: tests for gearbox ratios and RPM→offset (dummy audio driver) and the toggle pass.
* [x] T126: Evolution Race: pick a run and checkpoints (every N, evenly spaced, hand-picked, best-validated), labels by step; sandbox Launch option. Done when: a scripted race of 4 checkpoints runs headless.
* [x] T127: Human vs AI: keyboard and gamepad, ABS assist, physics choice; sandbox *Drive it yourself* (a game window). Done when: key events move a car headless and Launch builds the command.
* [x] T128: Brain panel: activation capture, network graph, input bars, outputs, key `B`. Done when: labels and finite activations are right for every compatible model.
* [x] T129: Effects: trails, skid marks, smoke and dust, brake lights, start lights. Done when: screenshot check and cost measured.
* [x] T130: Visuals: textures, kerbs, start line, barrier, scenery, shadows, tinted cars. Done when: screenshots on 3 tracks look right.
* [x] T131: Menus and HUD: leaderboard, results screen, minimap, key list `H`, camera (speed zoom, overview, cycle cars). Done when: the on-screen key list test passes.
* [ ] T132: Swarm view. Done when: many cars from one policy run at 60 FPS.
* [ ] T133: Performance: 20 cars with all effects and audio at 60 FPS. Done when: a bench script and a test threshold pass.
* [ ] T134: Rays table: Delete column + *Remove ticked rays* button in Build & Queue. Done when: a page test removes chosen rows.
* [ ] T135: `racer/agents/telemetry.py` + `racer/sandbox/plots.py`: racing line, speed trace, g-g diagram, braking zones, registry plots (curves, ablation bars, E10). Done when: unit tests on small data pass.
* [ ] T136: Sandbox *Workbench* page (live plots from the same runtime) + default plots on Overview; Compare links to it. Done when: headless page tests pass.
* [ ] T137: `scripts/make_plots.py`: the PNG report set from the same functions. Done when: it writes the set from the registry.
* [ ] T138: Docs (PROJECT_CONTEXT §10/§12/§16, README, glossary, `assets/LICENSES.md`), full suite, merge `feat/demo-complete` into `main`, remove the worktree. Done when: the suite passes on `main`.

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

### 2026-10-24 — Claude Code (Sonnet 5.5) — T131: Menus, HUD, minimap and camera

* **Status:** DONE. Fonts: Rajdhani for the HUD, Orbitron for big numbers and titles (`render.yaml: fonts`). Dashboard for the followed car (`dashboard.py`: rpm arc with redline, speed, gear from the cosmetic gearbox, lap/last/best, lateral g, gas/brake/steer bars). Minimap (`minimap.py`, key N). Start lights (three reds, then green) instead of a number. Results screen on top of everything with medals, the fastest lap in purple, and Enter to race again (`run_race` restarts). The camera zooms out with speed (`camera_speed_zoom`, up to 30 % at 70 m/s) and eases back. Mode menu with an animated track-and-cars background, the wrapped credit line, accent bars. Tests: dashboard, minimap, camera and menu in `tests/test_render.py`; 71 passed in the demo test files.
* **Files:** Created `racer/render/dashboard.py`, `racer/render/minimap.py`; edited `racer/render/hud.py`, `camera.py`, `arena.py`, `app.py`, `menu.py`, `configs/demo/render.yaml`, `tests/test_render.py`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T129: Race effects

* **Status:** DONE. `racer/render/effects.py`: skid marks (rear wheels sliding on tarmac, drawn as fading lines), tyre smoke (sliding), grass dust (wheels on grass), crash sparks plus a camera kick on the followed car; a pooled numpy particle field (700) and a capped segment list (6000); key `E` toggles them. All numbers in `arena.yaml: effects`. Tests: `tests/test_effects.py` (7: pool cap and expiry, skid chaining, shake decay, tarmac vs grass emission, crash, headless drawing). Brake lights existed already.
* **Files:** Created `racer/render/effects.py`, `tests/test_effects.py`; edited `racer/render/arena.py`, `configs/demo/arena.yaml`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T130: Textured world, barrier and scenery

* **Status:** DONE. The pre-rendered world now has tiled textures (grass, a gravel run-off band next to the road, asphalt with edge lines and slow brightness drift to hide the tiling), kerbs as before, a thicker red/white barrier with a shadow, an `outside` colour that matches the grass beyond the barrier, and trackside scenery placed from the track geometry (`racer/render/scenery.py`: grandstands by the start line, tyre walls on the outside of corners, tents, tree clumps; deterministic per track, never inside the barrier). `world_px_per_m` is 6 (builds take 3-4 s per track). Cars have drop shadows. A missing texture or sprite falls back (flat colours / noise), `textures.enabled: false` gives the old flat world. All numbers are in `configs/demo/render.yaml` (`textures`, `scenery`). Tests: `tests/test_world.py` (9).
* **Files:** Created `racer/render/textures.py`, `racer/render/scenery.py`, `tests/test_world.py`; edited `racer/render/world.py`, `racer/render/cars.py`, `configs/demo/render.yaml`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T128: Brain panel

* **Status:** DONE. Key `B` (on by default, `arena.yaml: brain`) shows the followed car's network live: input bars with their labels, one square per neuron for each hidden layer (blue negative, orange positive, brighter = stronger; 16×16 for 256 neurons), steering wheel and gas/brake outputs, and lines between the columns that brighten with activity. `racer/agents/activations.py` runs the real `policy._predict` with hooks on the hidden activations (PPO, GA and SAC shapes; an unknown shape still shows inputs and outputs; hand-coded and human cars say they have no network); colours are smoothed over frames. Labels come from the model's manifest. The sandbox part (a live brain view in the Workbench) is T136. Tests: `tests/test_brain.py` (7: probe vs the real action for PPO and SAC, tanh/ReLU ranges, normaliser, unknown shape, grid and colours, drawing); 24 passed (brain, render, human).
* **Files:** Created `racer/agents/activations.py`, `racer/render/brain_panel.py`, `tests/test_brain.py`; edited `racer/render/arena.py`, `racer/render/app.py`, `configs/demo/arena.yaml`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T127: Human vs AI and Drive it yourself

* **Status:** DONE. Mode `human` (menu entry 3): the arena picker plus your car's physics (Shift+C cycles the presets) and 0–19 opponents; your car is the first ghost car (`racer/render/human.py`). Keyboard (arrows/WASD) or gamepad (left stick, triggers or A/B, dead zone in `arena.yaml: human`), ABS assist for the all-or-nothing keyboard brake (K toggles, as in `drive.py`; the HUD shows ABS on / ACTIVE). CLI: `demo.py --human [PHYSICS]` adds your car to any race. **Sound key is M (was S in the doc) because S is the brake in WASD; M for the minimap moves to N (docs updated in T138).** Sandbox: Launch has a third tab *Drive it yourself* (physics, opponents, map, laps, baseline, sound) that opens a game window (Streamlit can't read 60 Hz keys). Tests: `tests/test_human.py` (input, gamepad, ABS, picker, headless race), `demo_argv` human; 97 passed, 7 skipped (human, render, sandbox, audio, checkpoints).
* **Files:** Created `racer/render/human.py`, `tests/test_human.py`; edited `racer/render/app.py`, `arena.py`, `ui.py`, `audio.py`, `scripts/demo.py`, `configs/demo/arena.yaml`, `configs/demo/audio.yaml`, `racer/sandbox/launcher.py`, `racer/sandbox/pages/launch.py`, `tests/test_render.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T125: Audio

* **Status:** DONE. Engine and tyre sound for the followed car, in every race window. Pure logic: `racer/render/gearbox.py` (cosmetic 5-speed box: shifts at 2.85, lands at 2.0, down at 1.4, clutch slip 1.2), `engine_sound.py` (rev_up/high_loop/rev_down/idle state machine with seek-to-rpm, 6 % drift re-seek, hysteresis and a minimum re-seek time so continuous AI gas doesn't chatter; tyre loop volume + one-shots on slide onset or hard braking). `audio.py` runs it on the mixer (clips decoded once, seek by slicing; no device or missing assets → silent, never an error). `configs/demo/audio.yaml` holds every number. Defaults (owner, D28): on when watching a race, off for headless captures unless `--sound`; `--no-sound` starts muted; the `S` key toggles everywhere and the window shows `sound: on · gear N`. Sandbox: a Sound checkbox on both Launch tabs (passes `--no-sound` when off). The title screen carries the CC BY 4.0 credit line. Tests: `tests/test_audio.py` (15, dummy audio device) + render/sandbox additions: 91 passed, 7 skipped (audio, render, sandbox, checkpoints).
* **Files:** Created `configs/demo/audio.yaml`, `racer/render/gearbox.py`, `engine_sound.py`, `audio.py`, `tests/test_audio.py`; edited `racer/render/app.py`, `arena.py`, `menu.py`, `scripts/demo.py`, `configs/demo/arena.yaml`, `racer/sandbox/pages/launch.py`, `tests/test_render.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T126: Evolution Race

* **Status:** DONE. Mode `evolution` in the menu: a picker (`racer/render/evorace.py`) lists runs with 2+ raceable checkpoints, preselects them by strategy (every N-th / N evenly spaced / N best by score; `racer/agents/checkpoints.py`), lets you add or remove checkpoints by click, and races them with cars coloured blue (early) to red (late). CLI: `demo.py --run R0xx --spaced 6 | --top 6 | --every 5` (`--gradient` for any race). Sandbox: the Launch page's evolution tab now offers the same strategies and passes `--gradient` (`demo_argv` gained `extra`). Tests: `tests/test_checkpoints.py` (5), picker/gradient/headless race in `tests/test_render.py`, `demo_argv` extra; 74 passed, 7 skipped (render, checkpoints, sandbox).
* **Files:** Created `racer/agents/checkpoints.py`, `racer/render/evorace.py`, `tests/test_checkpoints.py`; edited `racer/render/app.py`, `arena.py`, `ui.py`, `scripts/demo.py`, `racer/sandbox/launcher.py`, `racer/sandbox/pages/launch.py`, `tests/test_render.py`, `tests/test_sandbox.py`, `log.md`. Four R023 checkpoints were copied into the worktree's git-ignored `models/` for headless checks (delete at merge).

### 2026-10-24 — Claude Code (Sonnet 5.5) — T124: Demo shell and mode menu

* **Status:** DONE. The arena code moved from `scripts/demo.py` into `racer/render/arena.py`; `racer/render/app.py` holds the shared `Context`, the `MODES` registry and the run loop; `racer/render/menu.py` is the mode menu (Up/Down, number keys, click, Esc). `demo.py` is now thin (`--mode` opens a mode directly; the old options work as before). Later modes register in `MODES`. Tests: menu navigation, command-line selection (`tests/test_render.py`, 8 passed, including the headless baseline race).
* **Files:** Created `racer/render/app.py`, `arena.py`, `menu.py`; edited `scripts/demo.py`, `tests/test_render.py`, `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T123: Plan for finishing the demo and the analysis workbench (D28)

* **Owner decisions (answers to my questions):** finish the demo and analysis first, then experiment freely. In: mode menu, Evolution Race (top priority), Human vs AI, swarm view, brain panel as a live network graph, audio everywhere a race is watched (on for races, off for training/sandbox/analysis, toggle in every window), real polish (car and track visuals, effects, menus and HUD, camera and feel), a sandbox Workbench for live plots, *Drive it yourself* from the sandbox, a Delete option for the rays table. Out: the game-window drawing editor, the video recorder, slides and the highlight video.
* **Worktree:** `ML Sem5 Project - dev`, branch `feat/demo-complete` (from `main` @ `7202ac8`), because the sandbox queue (`R024_GA_rays6_narrow_s1`) runs in `main` and real runs refuse uncommitted code. Python: the main `.venv` with `PYTHONPATH` set to the worktree (checked).
* **Files:** Created `docs/DEMO_PLAN.md`. Edited `log.md` (T110, T112–T114 closed as dropped, T123–T138 added, this entry), `PROJECT_CONTEXT.md` (§16, D28).
* **Notes:** merge rules are in `docs/DEMO_PLAN.md` §5.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T122: Dev worktree removed

* **Owner:** remove the worktree and copy `wiggly_wiggly` here. Copied `tracks/user/wiggly_wiggly.json` and `outputs/plots/tracks/wiggly_wiggly.png` into `main` (byte-identical, `cmp`), then `git worktree remove --force` (its only untracked files were those two; its `.venv` went with it) and deleted the merged branch `feat/draw-a-track`. The two new files and the owner's `swiggly` track are untracked in `main`, not committed.
* **Files:** Created `tracks/user/wiggly_wiggly.json`, `outputs/plots/tracks/wiggly_wiggly.png`; edited `log.md`.

### 2026-10-24 — Claude Code (Sonnet 5.5) — T121: E10 queue finished, backup checked

* **Task:** owner said the queue run completed; do the after-queue backup and log it.
* **Status:** DONE. `e10_ga.yaml`: all 4 entries `done`, exit code 0, no Python processes left. Backup: `backup_models.py` copied 0 files (8008 already backed up, 3 finished GA lineages already there); R022 (45 folders) and R023 (50 folders) are in both `models/` and the backup. No CONFLICT lines.
* **Runs:** R020 `GA_ctrl_tg_s1` (14.4 min); R021 `GA_main_s1` (47.4 min, stopped at generation 919, best validation 1.368, best evaluated `@46282k`); R022 `GA_main_s2` (38.1 min, generation 899, best validation 1.367, best evaluated `@40055k`); R023 `e10_ppo_lean_s1` (5.0M steps in 46.6 min, evaluation 345 s, best `@1700k`: seen/held-out/unseen/overall 100 %).
* **Then (owner: "continue the rest"):** committed the queue outputs and the owner's `CLAUDE.md` edit (advisor rules) separately; merged `feat/draw-a-track` (conflicts only in `log.md` and `CLAUDE.md`, resolved by keeping both; the dev branch had already taken T118–T120, so this task was renumbered T121). Installed `streamlit-drawable-canvas==0.13.0` (pinned in `requirements.txt` by the dev branch) into the main `.venv` only; `pip check` clean. `tests/test_integrity.py::test_run_notes_match_the_runs` failed (R020–R023 had no notes: the queue doesn't write `run_notes.yaml`); added the four entries. Full suite: 281 passed.
* **E10 result:** PPO under GA rules (R023, 5M steps) beat both GA seeds on unseen completion (100 % vs 75 % / 94 %) and lap ratio (0.975 vs 1.05 / 1.07); caveats (one seed, best-of-many checkpoints, GA validation did not predict evaluation) are in §11.
* **Files:** Edited `log.md`, `PROJECT_CONTEXT.md` (§11 table and E10 results, MoSCoW line, §16), `README.md`, `registry/run_notes.yaml`; merge of the dev branch's files.
* **Notes:** the dev worktree is NOT removed: it holds an untracked `tracks/user/wiggly_wiggly.json` (+ preview PNG) the owner drew; `git worktree remove` would delete it, and copying it to `main` makes it `final_mix` training data (open question, §16). Asked the owner. The advisor was consulted on the merge/doc order (it caught the T118 clash and the worktree data risk).

### 2026-10-24 — Claude Code (Opus 5.5) — T120: Dev worktree venv

* **Owner:** created a `.venv` in the dev worktree. It was Python 3.14.6 and empty; the project is pinned to 3.11.9 (torch 2.14.1+cpu, numba, SB3; §14.1), so it was rebuilt with `py -3.11 -m venv .venv`, then torch (CPU index), `requirements.txt`, `pip install -e .`. Unpinned sub-dependencies came out newer (e.g. python-dateutil 3.0.0) and were set to the main venv's versions: `pip freeze` now identical except the editable `racer` path. `pip check` clean. Local to the worktree, git-ignored (`.venv/`); no global change.
* **Checks:** `racer` and torch import from the worktree; test_golden/test_sketch/test_sandbox_draw/test_physics: 30 passed, 1 skipped.
* **Files:** Edited `log.md` (Current Status rule for the worktree, T120, this entry).

### 2026-10-24 — Claude Code (Opus 5.5) — T119: Draw a Track page in the sandbox

* **Owner decisions:** draw inside the sandbox (not a button opening a game editor); save to `tracks/user/`.
* **Install (project venv only, no global change):** `streamlit-drawable-canvas==0.13.0` (MIT, requires only `streamlit>=1.53`; `pip check` clean). Installed from a wheel downloaded to `%TEMP%`, deleted afterwards. Pinned in `requirements.txt` and PROJECT_CONTEXT §14.1. Plotly (lasso selection) was the alternative but isn't installed either; Altair can't return a freehand path.
* **Built:** `racer/sandbox/draw.py` (canvas JSON → pixels → metres, file-safe names, save with a preview PNG, never overwrites a name used in any tracks folder), `diagrams.track_figure` (road, start arrow, too-close sections / too-tight corner in red; `_road` shared with the map ray diagram), page `pages/draw.py`, Launch takes a hand-over `launch_track`.
* **Bugs found on the way (fixed, tested):** Fabric.js 6 names objects `Path`/`Polygon` (capitalised), so nothing was read at first; a straight stroke gave a confusing "corner radius 0.0 m": now refused as "a line, not a loop" (`sketch.min_thickness`); the save message showed off-screen at the top: moved next to the buttons; canvas wider than its column: now 640×420, 1000 m across.
* **Checked in a real browser (worktree server on port 8599, stopped):** clicked rectangle → valid 3834 m track → saved → *Race on it* opened Launch with it chosen. Headless race on that track (`demo.py --models <R015@4600k folder of the main checkout> --baseline --screenshot`): both cars on lap 1 at 37 s, 0 crashes, 139 fps mean. The throwaway track and preview were deleted.
* **Tests:** `tests/test_sandbox_draw.py` (7), page list updated in `tests/test_sandbox.py`, 2 line cases added to `tests/test_sketch.py`.
* **Files:** Created `racer/sandbox/draw.py`, `racer/sandbox/pages/draw.py`, `tests/test_sandbox_draw.py`. Edited `racer/sandbox/diagrams.py`, `racer/sandbox/pages/__init__.py`, `racer/sandbox/pages/launch.py`, `racer/track/sketch.py`, `configs/demo/editor.yaml`, `requirements.txt`, `tests/test_sandbox.py`, `tests/test_sketch.py`, `PROJECT_CONTEXT.md` (§6, §10.1, §10.5, §14.1, §16), `README.md`, `log.md`.

### 2026-10-24 — Claude Code (Opus 5.5) — T118: Every feature in the sandbox (D27)

* **Owner rule:** any feature added must be reflected in the AI Sandbox; noted somewhere seen often.
* **Status:** DONE. `CLAUDE.md`: new section above *Working with the owner*, workflow step 7, parts table row. PROJECT_CONTEXT §10.5 and §16 (D27). Memory `every-feature-in-sandbox`. T119 added for Draw-a-Track's sandbox part.
* **Files:** Edited `CLAUDE.md`, `PROJECT_CONTEXT.md`, `log.md`.

### 2026-10-24 — Claude Code (Opus 5.5) — T111: Drawing → track (`racer/track/sketch.py`)

* **Status:** DONE (not committed yet; commit with T114). Worktree `ML Sem5 Project - dev`, branch `feat/draw-a-track`.
* **How it works:** `sketch_track(points_m, name, cfg["sketch"])` → `(Track without field | None, problems)`. The drawing is a closed loop starting at the first point, direction as drawn; resampled to 1000 points, smoothed with a window that is a fraction of the loop (so drawing size and hand jitter scale together), scaled into [800, 4000] m and thinned to a control point every 40 m. Up to 10 fix steps: scale up for length / corner radius / section gaps (shape is fixed, so scaling widens corners), double the smoothing (capped at 20 % of the loop) when corners are too sharp for any allowed size. Unfixable problems (figure-eight) are returned.
* **Found on the way:** a fixed smoothing window in metres failed: jitter grows with the scale; unbounded smoothing collapsed spiky shapes to < 3 control points (now capped, regression test).
* **Measured:** 191/200 random jagged polygons valid (the rest: corners still < 15 m, reported); 0.12 s mean, 0.58 s worst per conversion.
* **Tests:** `tests/test_sketch.py` (12). Also ran test_integrity, test_sandbox, test_config, test_presets: 125 passed, 10 skipped.
* **Files:** Created `racer/track/sketch.py`, `configs/demo/editor.yaml`, `tests/test_sketch.py`. Edited `log.md`.

### 2026-10-24 — Claude Code (Opus 5.5) — T117: Queue folder and dev worktree

* **Owner decision:** run the E10 queue in the background in the project folder (`main`) and keep working in a separate git worktree; merge both when the queue finishes.
* **Why a worktree:** `train.py` refuses a real run on uncommitted code (`CODE_PATHS`), checked when *each* queue entry starts, and it runs whatever branch is checked out; so editing here would fail entries 2–4. Docs, the registry and eval copies don't count as dirty.
* **Status:** DONE. Rules for both copies under "Current Status". Worktree `C:\Users\Jack\Desktop\ML Sem5 Project - dev` on `feat/draw-a-track`.
* **Files:** Edited `log.md`.

### 2026-10-24 — Claude Code (Opus 5.5) — T115: GA tries start on the start line

* **Owner request (mid-T110):** GA runs never start at random positions; always the beginning of the track, on every track.
* **Status:** DONE. Full suite 258 passed. New setting `ga.start: line | random` (`GAParams.start`, default `random` so snapshots without it, e.g. paused scratch runs S004–S007, resume identically); `ga_default.yaml` sets `line`: start_s 0, lateral 0, heading 0, speed 0 (validation already used this). The map is still drawn first from the same RNG, so the map sequence is unchanged. Invalid values are refused.
* **Tests:** new `test_line_start_puts_every_generation_on_the_start_line_with_the_same_maps`, `test_runs_without_a_start_setting_keep_random_starts`; the resume test's "new start each generation" assertion now expects the start line.
* **Files:** Edited `racer/agents/evolve.py`, `configs/train/ga_default.yaml`, `racer/sandbox/glossary.yaml`, `tests/test_evolve.py`, `PROJECT_CONTEXT.md` (§8 Generation bullet, §16), `README.md`, `log.md`. Branch `feat/draw-a-track` (T110 paused; nothing of T110 written yet).
* **T116 (same session):** `configs/queue/e10_ga.yaml` gained `e10_ga_main_s2` and `e10_ppo_lean_s1` (GA runs pause and resume; the PPO run restarts if interrupted, so it is last). Dry run OK; a 12k-step PPO scratch check on `rays7_lean`/`train7` trained fine (`outputs/scratch/S008_ppo_lean_check@12k`, git-ignored, kept like earlier scratch runs). Committed and merged into `main` locally for the owner to start the queue (D12: clean tree).

### 2026-10-24 — Claude Code (Opus 5.5) — T108: Status after the neuroevolution merge

* **Status:** DONE. Owner merged `feat/neuroevolution` into `main` (PR #1, 3df2d5d). "Current Status" rewritten (it still said the branch was unmerged and listed history up to 2026-10-15). No other doc stated the branch as unmerged.
* **T109:** created `REVIEW_LATER.md` (8 pending decisions: drop PPO / E10 fairness, T83, GA inputs, faculty-track folder, editor vs photo, demo physics, barrier rays, `final_mix` and `tracks/user/`).
* **Files:** Edited `log.md` (Current Status, T108–T110, this entry). Created `REVIEW_LATER.md`. Branch `docs/status-2026-10-24`.
* **Left running:** nothing.

### 2026-10-15 — Claude Code (Opus 5.5) — T101–T107: Presets, agent workflow, GA generations, elite validation, export

* **Owner requests:** README preset list (clarified mid-task: the model presets, viewable in the sandbox for all models); a workflow, parts checklist and architecture for agents in CLAUDE.md plus three working lines; a generation budget for GA (PPO keeps steps), CLI and sandbox; fix the noisy validation; options for showing the best cars and the evolution; question on starting weights.
* **Owner decisions:** checkpoints every 20 generations = the best-validated elite; after a run evaluate the 10 best-validated + the last 10 checkpoints ("last N only" rejected after the evidence that later isn't reliably better: S004@10017k 36 % seen vs @8738k 100 %); an export tool for any generation's car, also in the sandbox.
* **Status:** DONE. Full suite 251 passed. A tiny real run (S006_GA_gen_check: `--trackset train7 --generations 6`, validation every 3) stopped at the budget, saved the best-validated elite at each validation and at the end, evaluated both checkpoints; `export_ga.py --list` and `--generation 4` worked.
* **T103 (budget):** `ga.max_generations` (`train.py --generations N`); `--steps` is refused for GA and `--generations` for PPO/SAC. `total_steps` removed from `ga_default.yaml`. Resume may change `ga.workers`, `ga.max_generations`, `ga.evaluate` (`RESUME_FREE`). Sandbox: Build & Queue budgets GA in generations (No limit / 200 / 500 / 1000 / 2000 / custom), time estimate from `configs/sandbox/default.yaml` (`ga_minutes_per_generation`, `ga_eval_minutes`); the queue list shows generations. `configs/queue/e10_ga.yaml` control run: `--generations 300`.
* **T104 (validation, checkpoints):** every 20 generations (`ga.validation.every`) all elites drive the validation tracks; score = the best elite's mean; per-elite scores in a new lineage table `validations`; a checkpoint at every validation and at the end (the budget's last generation is always validated) = the best-validated elite (snapshot: `generation`, `car`, `fitness`, `validation`). `ga.checkpoint_every` removed. The plateau is only checked at validations, so a finished run always ends with a validated checkpoint.
* **T105 (evaluation):** `ga.evaluate: {top_validated: 10, last: 10}`; `train.py:checkpoints_to_evaluate`.
* **T106 (export):** `racer/agents/evolve.py`: `write_model_folder` (shared by checkpoints and exports: `.{id}.partial` + rename), `export_car`, `pick_car` (best-validated elite if validated, else best on its map), `generation_table`, `run_dir_of`. Refuses on a different numpy version. `scripts/export_ga.py --run R020 --list | --generation G [--car C] [--evaluate]`. Sandbox: Model Explorer **Evolution** tab (fitness/validation curves, *Save as a model*).
* **T101 (presets):** README "Model presets" (every preset of the five model kinds with what it is, how to combine them, `--set`, Build & Queue, other config kinds). Sandbox **Presets** page (`racer/sandbox/pages/presets.py`, logic in `explain.py`: `PRESET_KINDS`, `preset_header`, `runs_using`): values with notes, rays to scale, track previews, algorithm and network, runs trained with it.
* **T102 (CLAUDE.md):** "Working with the owner" (the three lines), "Workflow for any change", "Architecture (follow it)" (layers, dependency direction, extend-don't-change, self-describing models, pausable jobs, one source of truth), "Parts to keep in mind" table.
* **Bugs fixed on the way:** (1) Explorer: opening a trial (S) model by link showed it only once; the next interaction hid it again ("No trained models"), because the "Trial runs" toggle stayed off. (2) README: an earlier scripted edit had written `scripts<TAB>rain.py` in the GA command.
* **Answer recorded (owner question):** every first run starts from random weights, never the baseline: PPO from SB3's random initialisation (exploring with noise); GA generation 0 = 50 random networks from their own seeds. Only `--parent` (PPO) starts from a trained model.
* **Files:** Created `scripts/export_ga.py`, `racer/sandbox/pages/presets.py`. Edited `racer/agents/evolve.py`, `scripts/train.py`, `configs/train/ga_default.yaml`, `configs/sandbox/default.yaml`, `configs/queue/e10_ga.yaml`, `racer/sandbox/builder.py`, `racer/sandbox/queue.py`, `racer/sandbox/explain.py`, `racer/sandbox/pages/__init__.py`, `racer/sandbox/pages/build.py`, `racer/sandbox/pages/explorer.py`, `racer/sandbox/glossary.yaml`, `tests/test_evolve.py`, `tests/test_sandbox.py`, `CLAUDE.md`, `README.md`, `PROJECT_CONTEXT.md`, `docs/NEUROEVOLUTION_PLAN.md`, `log.md`. Scratch (git-ignored): `outputs/scratch/S006_*`.
* **Review** (atomic-reviewer, Opus 5.5): CHANGES_REQUESTED (1 critical: train.py's docstring still described `ga.checkpoint_every`). All fixed: `ga.evaluate.last: 0` selected every checkpoint (`snaps[-0:]`); `--generations` accepted 0 and negatives; the Overview page showed GA queue entries as "0M steps" (shared `builder.budget_label`, GA time estimate); `pick_car` broke validation ties by index, not by rank like the checkpoint (now joined with fitness; test); a checkpoint whose id clashes with another car (same thousand of env steps) is saved as `_g<gen>c<car>` instead of being dropped; the temporary export folder carries the PID; exports set `saved_at`; README `sac_default` (ReLU, 1M steps); glossary `run.generation`/`run.car`; Presets page counts GA parameters with 10 inputs; CLAUDE.md workflow points to the existing sections instead of repeating them, agents/registry drawn as one layer; small cleanups. New tests: flag refusals, generation-0 refusal, queue generations display, tie-break, `last: 0`. Full suite 256 passed; tiny real run S007_GA_gen_check2 saved and renamed its checkpoints.
* **Left running:** nothing.

### 2026-10-15 — Claude Code (Opus 5.5) — T96, T98: σ test, docs; T97 queued

* **T96 (owner: run the σ test now):** two scratch runs on `train7`, 10M env steps each (equal compute instead of 200 generations), seed 0: S004_GA_sigma_002 (224 generations, ≈ 30 min) and S005_GA_sigma_005 (295 generations). Evaluated v2 automatically. **σ 0.02 wins:** best checkpoint S004@8738k seen 100 %, unseen 87.5 % (test maps 67 %, held-out 100 %), lap ratio 0.997; S005@6126k 86 % / 62.5 % / 1.083. Finishers per generation (last 50): 27.8 vs 16.4 of 50. Best validation 1.35 vs 1.06. `ga_default.yaml` keeps σ 0.02.
* **Observations for the owner:** (1) the validation score swings a lot between checks (0.47–1.35), because the car it tests is the best on that generation's single map; it only drives the stopping rule. (2) Checkpoint quality varies the same way (S004@10017k: 36 % seen). Possible changes: checkpoint the car with the best validation score, or checkpoint at every validation (25 generations). Not changed: the checkpoint interval is an owner decision.
* **E10 judging (owner):** GA checkpoints are evaluated and raced under their own rules (track limits); recorded in PROJECT_CONTEXT §16.
* **T97:** queue file `configs/queue/e10_ga.yaml` (control: Training Ground, 10M steps; main: `train7`, until the plateau; one run at a time). Dry run OK. Not started (owner starts it).
* **T98:** README (new "Train with neuroevolution" section, status, layout), PROJECT_CONTEXT (§1, §4 layout, §8 neuroevolution bullets updated to what was built, §11 first results, §12, §16), CLAUDE.md (backup line), plan status "built".
* **Left running:** nothing; 0 background tasks.
* **Files:** Created `configs/queue/e10_ga.yaml`. Edited `PROJECT_CONTEXT.md`, `README.md`, `CLAUDE.md`, `docs/NEUROEVOLUTION_PLAN.md`, `log.md`. Scratch outputs (git-ignored): `outputs/scratch/S004_*`, `S005_*`, `outputs/scratch/evolve/`.

### 2026-10-15 — Claude Code (Opus 5.5) — T95: Evolution loop, train.py, queue, backup, sandbox (S4)

* **Status:** DONE. Full suite 237 passed. All S4 "done when" checks met:
  * **End to end:** `train.py --train ga_default --tracks training_ground --env rays7_lean --scratch` (S001_GA_e2e_a) ran 20 generations of 50 cars, finished at the validation plateau (test settings: patience 15), saved 4 checkpoints and evaluated them with protocol v2 (24 s).
  * **Pause/resume:** S002_GA_e2e_b, the same settings, paused with `stop.request` before generation 14 (exit 3) and resumed with 3 workers instead of 5: lineage identical to S001 (1000 car rows, 20 generation rows), same checkpoint ids. Also a test (`test_paused_and_resumed_with_other_workers_equals_an_uninterrupted_run`) and a crash-recovery test (resume from an older periodic save).
  * **Arena:** S001@509k raced R015 headless in `racer/env/race.py` (the demo's race code): 3 laps of Training Ground in 53.5 s each (R015 46.3 s, baseline 46.6 s), 2 track-limits respawns.
* **Loop (`racer/agents/evolve.py`):** spawn pool at below-normal priority, workers ignore Ctrl+C; generation g draws map + start from `default_rng([seed, g, 0])` and breeds from `default_rng([seed, g, 1])`; each car drives from a fully specified start until one lap, a failure, or the env's step limit; validation of the best car on generated tracks (stream 2) every 25 generations; plateau stop; optional env-step budget; lineage SQLite (`INSERT OR REPLACE`, so re-driven generations after a crash rewrite identical rows); atomic `state.npz` (also saved at generation 0); `finished.json` and state deletion at the end.
* **train.py:** GA dispatch (`train_ga`): the same command resumes the run with that tag, from the run's own saved settings (only `ga.workers` and `total_steps` come from the command; differences are listed); a finished run is not restarted; model ids `R###_GA_<tag>@<env steps>k`; checkpoints written as `<id>.partial` and renamed, with the commit current at save time; evaluation after finishing; exit 3 when paused. `base_snapshot_of` shared with PPO.
* **Elsewhere:** the queue maps exit 3 to `interrupted` (resumed on the next start); `backup_models.py` copies finished evolve runs to `<backup>/evolve/`; `.gitignore` `outputs/evolve/`; sandbox Build & Queue supports GA (no-limit budget, nested `ga.validation` inputs, a track set without a generated pool, no time estimate); Explorer curves `ga/validation`, `ga/best_fitness`; glossary `run.generation`, `run.fitness`; the integrity test skips runs whose GA run is unfinished (they are evaluated when they finish).
* **Reviewer** (atomic-reviewer, Opus 5.5): CHANGES_REQUESTED (0 critical, 9 risks). Fixed: a killed checkpoint export could leave a folder that is skipped forever (now `.partial` + rename); resume broke when a map is added to a track-set folder (now uses the run's own settings); a second Ctrl+C couldn't abort on Windows while `pool.map` waits (now polled with `map_async`); no state before generation 10, so a crash orphaned the run (now saved at 0); checkpoint `git_commit`; each worker generated all 5 validation tracks (now per index); GA time estimate in the sandbox; duplicate error. Accepted: the sandbox's *Stop queue* kills a GA run (resumes from its last periodic save); the queue runner now imports `racer.agents.evolve` for one constant.
* **Re-reviews (2 more passes):** checkpoint export now under a dot name `.{id}.partial` (skipped by every scan; `registry.backfill` now skips dot folders too, with a test), fails loudly if a leftover can't be removed, and retries the rename on `PermissionError`; GA estimate test; `_diff_keys` test. Last pass: one finding (a false "registry skips it" comment), fixed by making it true. Full suite 239 passed. A tiny real run (S003_GA_ckpt_check) wrote and renamed 2 checkpoints and refused to restart once finished.
* **Bug fixed on the way (pre-existing):** `racer/sandbox/queue.py` read `state.json`/`runner.json` without retrying; on Windows a read that meets the runner's atomic replace raises `PermissionError` (seen once in `test_stop_request_interrupts_a_running_queue`). Reads now retry briefly.
* **Left running:** nothing. Scratch runs S001/S002 stay in `outputs/scratch/` (git-ignored).
* **Files:** Edited `racer/agents/evolve.py`, `scripts/train.py`, `scripts/queue_overnight.py`, `scripts/backup_models.py`, `racer/registry/backup.py`, `racer/sandbox/builder.py`, `racer/sandbox/pages/build.py`, `racer/sandbox/curves.py`, `racer/sandbox/glossary.yaml`, `racer/sandbox/queue.py`, `.gitignore`, `README.md` (command row), `docs/NEUROEVOLUTION_PLAN.md` (pause line), `tests/test_evolve.py`, `tests/test_sandbox.py`, `tests/test_backup.py`, `tests/test_integrity.py`, `log.md`.

### 2026-10-15 — Claude Code (Opus 5.5) — T94: GA model folders, `ga` algorithm (S3)

* **Status:** DONE. Full suite 223 passed. Reviewer (atomic-reviewer, Opus 5.5): CHANGES_REQUESTED, 1 critical fixed: registering `ga` put it on the sandbox Build page, where picking it crashed (`KeyError: checkpoint_every`). The page now lists only `builder.BUILDABLE` (ppo, sac; GA is added in T95), with a test that every listed algorithm validates. Other fixes: `update_steps` is `None` for GA (no gradient updates); `export_model` derives the layer sizes itself; `hparams_json` in runs.csv keeps the `ga:` section; the Explorer caption says GA weights are evolved, not trained; the GA layer table joined the existing count test.
* **Export:** `racer/agents/evolve.py:export_model` copies the weights into an SB3 PPO actor (`vf: []`), saves `model.zip` (≈ 0.3 MB) and an identity `vecnormalize.pkl`. `LoadedModel.act_batch` matches the numpy forward pass (atol 1e-5, inputs inside and outside the input range).
* **Config:** `configs/train/ga_default.yaml` (all GA settings, glossary entries for each). The GA's maps are the run's own tracks; new track set `train7` = the 7 training maps without a generated pool. Plan change: the optional cap is an env-step budget (`total_steps` / `--steps`, the same unit as PPO and the sandbox queue) instead of `max_generations`.
* **Files:** Created `racer/agents/evolve.py`, `configs/train/ga_default.yaml`, `configs/trackset/train7.yaml`, `tests/test_evolve.py`. Edited `racer/agents/algorithms.py`, `racer/agents/compat.py`, `racer/sandbox/diagrams.py`, `racer/sandbox/builder.py`, `racer/sandbox/pages/build.py`, `racer/sandbox/pages/explorer.py`, `racer/sandbox/glossary.yaml`, `racer/registry/registry.py`, `tests/test_sandbox.py`, `docs/NEUROEVOLUTION_PLAN.md`, `log.md`.

### 2026-10-15 — Claude Code (Opus 5.5) — T93: GA core (S2)

* **Status:** DONE. 14 tests in `tests/test_neuroevolution.py` pass; the full suite passed (215) before the review fixes. Reviewer (atomic-reviewer, Opus 5.5): CHANGES_REQUESTED (0 critical), fixed: wrong-length scores now raise (a lost worker result would otherwise silently drop cars from selection); the tie test now uses a population-sized case that fails without the stable sort; separate `NO_NOISE` marker for elites; keyword-only `fitness`; a test that pins the torch (out, in) layout.
* **Design:** flat float32 weights in SB3 actor order; tanh hidden, linear clipped output; init W ~ N(0, 1/fan_in), b = 0 (generation 0 drives in varied ways instead of stalling); child = parent + σ × N(0, 1) from its own seed; ranking is stable (elites, which sit first, win ties). Per-generation randomness will come from `default_rng([seed, generation])`, so the pause state needs no RNG state (plan updated).
* **Files:** Created `racer/agents/neuroevolution.py`, `tests/test_neuroevolution.py`. Edited `docs/NEUROEVOLUTION_PLAN.md` (state line), `log.md`.

### 2026-10-15 — Claude Code (Opus 5.5) — T92: Track-limits rule, rays7_lean (S1)

* **Status:** DONE. Full suite 202 passed (golden R001 test ran and passed), `physics_sanity.py` ALL PASSED, `check_env` passes for `rays7_lean`. Reviewer (atomic-reviewer, Opus 5.5): CHANGES_REQUESTED (0 critical), all findings fixed (weak "stops at the edge" test tightened to one substep of travel; comment trimmed; glossary end-reason lists; imports).
* **Track limits:** `env.episode.track_limits` (default false). Checked after every physics substep: the car's centre beyond `half_width` (field distance to the centreline; kerbs lie inside it) ends the episode with `termination: track_limits` and the crash penalty; the barrier takes precedence. In free drive it is a warning. TensorBoard `end/track_limits`.
* **Config:** `from_dict` now lets a dataclass field with a default be missing (only `track_limits` has one), so old snapshots load unchanged; fields without a default stay required.
* **`rays7_lean`:** 7 rays (±55° 60 m, ±17° 120 m, ±5.5° 200 m, 0° 200 m), heading sin/cos, v_long = 10 inputs; `track_limits: true`. Ray labels now show decimals (`ray+5.5`); integer angles keep their old labels.
* **Files:** Created `configs/env/rays7_lean.yaml`. Edited `racer/config.py`, `racer/env/racing_env.py`, `racer/env/features.py`, `racer/sandbox/glossary.yaml`, `scripts/train.py`, `tests/test_env.py`, `tests/test_config.py`, `log.md`.

### 2026-10-15 — Claude Code (Opus 5.5) — T100: Selection wording, GA in the model id

* **Task (owner):** for the next generation, the top 25 % (12) are kept unchanged, and the same overall top 25 % are mutated to make the other 38; add GA to the model id. Update the docs, make a pre-implementation commit, then start implementing.
* **Status:** DONE. The selection matches the T91 design; it is now stated without ambiguity (ranking over all 50, previous elites included; children only from these 12). Model id: `R{run}_GA_{tag}@{steps}k` (uppercase `GA_` added by `train.py`; tags stay lowercase). Checked that the id patterns in the code accept it (`hud.py` label regex, `train.py` run-id scan).
* **Files:** Edited `PROJECT_CONTEXT.md` (§8 selection, §9 model id, §16 row), `docs/NEUROEVOLUTION_PLAN.md` (diagram, selection row, model id), `log.md`.

### 2026-10-15 — Claude Code (Sonnet 5.5) — T99: Docs aligned with the D26 decisions

* **Task (owner):** read CLAUDE.md, PROJECT_CONTEXT.md, README.md and log.md and make sure they reflect the decisions taken so far.
* **Status:** DONE (docs only; no code, nothing running). `docs/NEUROEVOLUTION_PLAN.md` and the §16 rows already matched; the gaps were elsewhere.
* **Fixed in `PROJECT_CONTEXT.md`:** §8 said the GA used "the same inputs" as PPO's `rays7_narrow`, which contradicts the final decision (its own 10-input preset `rays7_lean`; only outputs and hidden layers match); §1 summary, §4 layout (`ga_default.yaml`, `neuroevolution.py`, `ga` in `algorithms.py`, `outputs/evolve/`), §7.4 (optional `track_limits` ending, off by default), §9 (algorithm list), §12 (E10 under *Could*).
* **Fixed in `README.md`:** status note on the planned GA and the plan in the `docs/` row. No commands added, because none exist yet.
* **Fixed in `CLAUDE.md`:** backup bullet: the lineage database is backed up, `state.npz` is kept while a run is unfinished, and a run is paused with Ctrl+C or `stop.request`.
* **Not changed on purpose:** `.gitignore` (`outputs/evolve/` is added in T95), the T84–T90 entries and §16 rows below, which are history and were superseded by the later rows. Current Status: added the branch note.
* **Files:** Edited `PROJECT_CONTEXT.md`, `README.md`, `CLAUDE.md`, `log.md`.

### 2026-10-13 — Claude Code (Opus 5.5) — T91: Neuroevolution ready to implement, pre-implementation commit

* **Task (owner):** confirm the open proposals, prepare the implementation, commit; no implementation or training run this session.
* **Status:** DONE. Planning only: no code, nothing running.
* **Owner decisions:** 50 cars (12 elites); validation 5 generated tracks every 25 generations, stop after 500 generations without a 1 % gain; 200 s simulated cap per try; lineage SQLite + a checkpoint every 100 generations; no full-weight record of the run (≈ 50 GB free on C:). Sideways speed stays dropped (owner: breeding removes sliders either way; counter-argument: without the input, a reaction to a slide can't evolve; first ablation if driving is poor).
* **Disk estimate:** lineage ≈ 30 MB per 10,000 generations; checkpoint folder ≈ 1.8 MB (R015's: model.zip 1683 KB + thumbnails); pause state ≈ 14 MB, overwritten.
* **Plan:** `docs/NEUROEVOLUTION_PLAN.md` rewritten as the implementation contract: final decisions, pause/resume, recording, integration points (things to check: `from_dict` dataclass defaults for the new episode key; `vf: []` for a small zip; compat for `algorithm: ga`), build steps S1–S7 = T92–T98 with "done when" checks. The track-limits work became its own step (S1), so there are 7 steps instead of 6.
* **Files:** Edited `PROJECT_CONTEXT.md` (§4, §8 bullets, §16 row), `docs/NEUROEVOLUTION_PLAN.md` (rewritten), `log.md`.

### 2026-10-13 — Claude Code (Opus 5.5) — T90: Center ray, sideways speed dropped, lineage recording

* **Task (owner):** add a 0° ray; drop sideways speed (the owner's reasoning: selection for faster laps reduces slip without the car sensing it); can every generation's checkpoint be recorded, e.g. in SQLite?
* **Status:** DONE (planning only). Inputs final: 7 rays (0° 200 m, ±5.5° 200 m, ±17° 120 m, ±55° 60 m), heading sin/cos, forward speed = 10 inputs, 69,122 weights.
* **Counter-argument recorded (sideways speed):** selection does favour less slip, but with yaw rate and sideways speed both gone the network can't perceive a slide at all (it is memoryless and sees only body angle, not travel direction), so it can't correct one; expected result: a safety margin below the grip limit (slower) and uncorrected slides on lower-grip kerbs/grass and unseen tracks. Owner's "try, change later" applies; first ablation: sideways speed back in.
* **Recording:** a network is 69,122 weights (512 is the neuron count), 276 KB. Full weights for every car ≈ 138 GB per 10,000 generations of 50, so the plan proposes a lineage SQLite (parent + noise seed per car, ≈ 30 MB per 10,000 generations) from which any car can be rebuilt and exported, plus a registered checkpoint every 100 generations. Rebuilding needs the pinned numpy (RNG streams are only guaranteed within a version); model folders stay the permanent record.
* **Files:** Edited `PROJECT_CONTEXT.md` (§8 inputs bullet, §16 row), `docs/NEUROEVOLUTION_PLAN.md` (inputs row, recording section, open proposals), `log.md`.

### 2026-10-13 — Claude Code (Opus 5.5) — T89: Forward-sight analysis (ray angles)

* **Task (owner):** compute the braking distance from top speed to the tightest corner's speed over all maps, and check whether ±5° rays (instead of ±5.5°) see far enough.
* **Status:** DONE (analysis only; the ray choice is the owner's).
* **Method:** `configs/physics/default.yaml` (μ_static road 1.10, CdA 0.70, 150 kW); top speed solved from power = drag + rolling (68.8 m/s); corner speed √(μ·g·R) at each map's minimum centreline radius; braking integrated with drag; a forward/backward centreline speed profile per map for realistic braking zones; ray sight on a straight = lateral gap ÷ sin(angle).
* **Results:** top speed → tightest corner (Technical, R 15.1 m, 12.8 m/s) = 191 m (212 m without drag). Realistic longest braking zone 116 m (Coastline, from 56 m/s; others 68–102 m). Sight with the car centred: ±5.5° 52–83 m, ±5° 57–92 m; on the outside line: 94–157 m and 103–172 m. ±5° adds 10 % but covers neither 191 m nor (centred) 68–116 m; seeing 191 m from the centre of a 12 m road needs ≤ 1.8° (≤ 1.5° on 10 m).
* **Files:** Edited `docs/NEUROEVOLUTION_PLAN.md` (open proposal 1), `log.md`. Nothing left running.

### 2026-10-13 — Claude Code (Opus 5.5) — T88: One map per generation, CPU load, inputs discussion

* **Task (owner):** one random map per generation from the 7 training maps (instead of K maps); workers sized to ≈ 60 % CPU load instead of a fixed 8; the owner proposes dropping some inputs.
* **Status:** DONE (planning only). Recorded in PROJECT_CONTEXT §8/§16 and the plan.
* **Inputs (owner):** 10: 6 rays at ±5.5° (200 m), ±17° (120 m), ±55° (60 m) (ranges by analogy with `rays7_narrow`), heading sin/cos, forward and sideways speed. Dropped: yaw rate, on-road fraction, previous steer and gas ("try; change later if worse"). Flagged: with no 0° ray the car sees ≈ 52–83 m ahead on a straight (the ±5.5° rays hit the edges), against ≈ 180 m needed to brake from top speed; proposed adding a 0° ray, owner to decide.
* **Track limits (owner):** a try fails when the car's centre crosses the road edge (half the car over the line). This removes the off-road ray ambiguity and grass shortcuts. To be an optional episode key, off for existing presets.
* **Time cap:** the earlier 120 s was a guess, not checked against the baseline; BASE_pid's best laps are 46.6–81.6 s (Technical slowest). Proposed instead: the env's existing 4000 steps = 200 s of simulated time (≈ 0.6 s of one worker).
* **CPU benchmark** (`bench_ga2.py`, session scratchpad, with `psutil` CPU load): 4 processes ≈ 52 % CPU, ≈ 29.5k steps/s; 5 ≈ 64–73 %, ≈ 33k; 8 ≈ 96 %, ≈ 37k. The i5-10200H has 4 cores / 8 threads, so 8 workers is the whole CPU. Default: 5 workers, below-normal priority, changeable on resume. Nothing left running.
* **Files:** Edited `PROJECT_CONTEXT.md`, `docs/NEUROEVOLUTION_PLAN.md` (rewritten), `log.md`.

### 2026-10-13 — Claude Code (Opus 5.5) — T87: Neuroevolution decisions refined, 3-map benchmark

* **Task (owner):** discuss the T85 plan. Owner decisions: mixed maps from the start (no master-one-map-first) plus a single-track control run; population sized to a run budget; stop on a validation plateau with a cap; top 25 % seed the next generation; a try ends at **one lap or a failure**; training **pausable at any moment** (the generation in progress finishes and is saved) and **resumable** by rerunning the command.
* **Status:** DONE (planning only; no code yet). PROJECT_CONTEXT §8 (D26 bullet expanded) and §16 (new row) updated first, then the plan.
* **Benchmark** (throwaway `bench_ga2.py` in the session scratchpad, not in the repo): R015's actions recorded per map and replayed (all three showcase maps were driven the full 60 s), a random 256×256 numpy net on every step, weights sent to the workers. Per generation: 20 × 3 maps 1.97 s / 1.84 s (6 / 8 processes); 50 × 3 4.98 / 4.58 s; 100 × 3 9.52 / 9.03 s. 8 processes ≈ 39k env steps/s (T86's 24.8k was measured with R015's torch inference in the same processes). Nothing left running.
* **Plan decisions:** score per map = lap fraction if failed, 1 + mean speed ÷ 70 m/s if finished; fitness = sum over K maps. Network stays 256×256 (R015 generalises with it). σ by two short runs. Run state for pause/resume in `outputs/evolve/<run_id>/` (to be git-ignored in S3), never in model folders.
* **No generation cap by default (owner):** pausing makes it unnecessary; the plateau rule needs a minimum improvement margin so it always fires; an optional `max_generations` stays for unattended queue entries.
* **Open proposals for the owner:** K = 3, 50 cars, validation (5 tracks every 25 generations, patience 500, minimum improvement 1 %), time cap ≈ 120 s per try.
* **Files:** Edited `PROJECT_CONTEXT.md`, `docs/NEUROEVOLUTION_PLAN.md` (rewritten), `log.md`.

### 2026-10-13 — Claude Code (Opus 5.5) — T84–T86: Neuroevolution decision, plan, speed benchmark

* **Task (owner):** try the genetic-algorithm method from a YouTube video (20 cars per generation, the best become parents with mutated weights) next to PPO. Owner choices: Training Ground first, then multi-track; 256×256 network; plan + benchmark before building.
* **Status:** DONE (planning only; no code in the repo yet). Build steps S1–S6 wait for the owner's OK on the plan's open decisions.
* **T84:** D26 recorded in PROJECT_CONTEXT §3 (algorithm row), §4 (docs/), §8 (neuroevolution bullet), §11 (E10), §16.
* **T85:** `docs/NEUROEVOLUTION_PLAN.md`. Fitness = progress in 60 s (one score for "get round" and "be fast"), same start for the whole generation, top 5 kept unchanged, the rest mutated copies. Saved as an SB3 PPO `model.zip` + identity `vecnormalize.pkl`, so loader, demo, eval, registry and backup work unchanged (`tests/test_integrity.py` requires all three model files). `train.py` dispatches on `algorithm: ga`, so the queue and sandbox can run it.
* **T86 (benchmark):** throwaway script in the session scratchpad (not in the repo). 20 cars × 60 s on Training Ground, driven by R015 so all laps were full length; R015 averaged 1759 m in 60 s. Env step 104 µs, numpy forward of a 256×256 net 37 µs; 7.1k steps/s on 1 process, 24.8k on 6 → ≈ 1.0–1.2 s per generation worst case, **≤ ≈ 80 min for 4000 generations** (96M env steps). Nothing was left running.
* **Files:** Created `docs/NEUROEVOLUTION_PLAN.md`. Edited `PROJECT_CONTEXT.md`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T76, T81, T82, T83: Lock fix, model backups, arena key list

* **Task (owner):** fix T76; make regular model backups (the backup had stopped at R001); add a key list to the 20-model race and the evolution race; record the variance decision as a task for a later session.
* **Status:** DONE. Full suite: **191 passed**. The owner committed T74–T80 in the meantime (`ac2f2a5`, "week 2 done - checked"); this work isn't committed yet.
* **T76:**
  * `file_lock` (`racer/registry/registry.py`) now also treats `PermissionError` as "held". Windows raises it while the previous holder's lock file is still being deleted.
  * The test that failed about 1 in 5 runs passed 50 out of 50.
* **T81 (backups):**
  * `racer/registry/backup.py`, `scripts/backup_models.py` and `configs/backup/default.yaml` (`dest`, and the scripts that block a backup while running).
  * The backup is additive: it copies missing folders and files (including eval files added later), through a `.partial` name with a size check. It never overwrites or deletes; a file whose size differs is reported as `CONFLICT`. It refuses while `train.py`, `queue_overnight.py` or `evaluate.py` runs (`--force` overrides). `--dry-run` and `--run` are available.
  * Ran it: 6470 files (1222.8 MB) copied, 670 new folders, no conflicts. `diff -rq models <backup>` shows them identical except `.run_ids` (skipped on purpose).
  * Instructions: a new `CLAUDE.md` section "After a training run or queue completes", a README section "Back up the models (after every run)" plus a line in the queue section and a row in the Commands table, and PROJECT_CONTEXT §4, §9 and §16.
  * Tests: `tests/test_backup.py` (4).
* **T82 (key list):**
  * `RaceHud.keys()` (`racer/render/hud.py`) draws a key list in the bottom-right corner, or an "H keys" hint when hidden. `demo.py` `arena_keys()` builds the list, and `H` toggles it.
  * `Esc` reads "quit" when the race was started from the command line (`--models`/`--run`/`--best`/`--baseline`), otherwise "back to the picker".
  * `configs/demo/arena.yaml` `show_keys_at_start: true`.
  * Checked: a 20-car screenshot shows the list fits below the full leaderboard, with no overlap; 108 fps mean uncapped.
  * Test: every key `run_race` handles must be in the list (`tests/test_render.py`).
  * Docs: README arena keys; PROJECT_CONTEXT §10.3 (`H` key list, and every game mode shows its keys on screen).
* **T83:** recorded as an open task (the variance decision, for a later session).
* **Files:**
  * Created: `configs/backup/default.yaml`, `racer/registry/backup.py`, `scripts/backup_models.py`, `tests/test_backup.py`.
  * Edited: `racer/registry/registry.py`, `racer/render/hud.py`, `scripts/demo.py`, `configs/demo/arena.yaml`, `tests/test_render.py`, `CLAUDE.md`, `PROJECT_CONTEXT.md`, `README.md`, `log.md`.
  * Outside the project: 670 model folders copied to `C:\Users\Jack\ML_Sem5_models_backup\` (the existing, owner-approved backup location).

### 2026-10-11 — Claude Code (Opus 5.5) — T78–T80: Integrity check, integrity test, Week 3 readiness

* **Task:** Owner asked for a quick integrity check of the whole project, an integrity test if needed, and whether it's clear to start Week 3.
* **Status:** DONE.
  * Full suite: 175 passed (the flaky T76 test passed this time). With the new file: 185.
  * `physics_sanity.py`: all 5 pass (skidpad 10.72 vs μg 10.79 m/s², threshold braking 41.0 m, full lock 48.8 m).
  * `pip check` clean.
* **T78 (one-off check script, in the scratchpad, read-only):**
  * Every installed package matches its pin; torch is 2.14.1.
  * Nothing tracked that should be ignored (only `models/.gitkeep`, on purpose). All 19 tracks load, and their names are unique.
  * Registry: 20 runs and 691 models. The ids are unique, every model belongs to a run, and each run has exactly one best checkpoint.
  * All 690 model folders match their registry rows. Each has its 3 files, the eval file of its protocol, and its copy in `outputs/eval/`. All of them pass `compat.check`.
  * `run_notes.yaml` covers every run.
  * Found: 2 made-up example ids in PROJECT_CONTEXT §9 (`R012_ppo_rays5_curric@1500k`, `R010_e2_nogrip_s1@3000k`) didn't match the real runs. Replaced with real ids (`R015_t50_narrow7_s1@4600k`, `R005_e2_nogrip_s1@3000k`).
  * `configs/demo/audio.yaml` is named in the §4 layout but is planned (audio is Week 3), so it isn't a fault.
* **T79:** `tests/test_integrity.py` (10 tests, < 1 s):
  * registry ids unique and linked; one best per run; an eval copy for every evaluated model (tracked, so this runs everywhere); run notes match the runs.
  * Model folders ↔ registry, and folder completeness. These skip without a local `models/`.
  * Model ids named in README/PROJECT_CONTEXT exist; presets used in README commands exist; track names are unique; installed packages match the pins.
* **T80 (readiness):**
  * Open tasks: T08 (M3, Week 3), T09 (M4), T76 (flaky lock test).
  * Git: `main` is 10 commits ahead of `origin` (not pushed). This session's changes (T74–T80) are not committed yet. `rootline` is the old local initial-commit branch (already in main). No stashes or extra worktrees.
  * Running: only the owner's own sandbox (8501). No training or queue.
  * Disk: 73 GB free; `models/` 1.2 GB.
  * **Risk:** the models backup (`C:\Users\Jack\ML_Sem5_models_backup\`) holds only R001. R002–R019 exist only in `models/`.
  * The R015 variance question (PROJECT_CONTEXT §16) is still open. It doesn't block Week 3 visuals, but it decides what the nights are used for.
* **Files:** created `tests/test_integrity.py`; edited `PROJECT_CONTEXT.md` (§4 tests line, §9 example ids), `README.md` (layout row), `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T77: README command list

* **Task:** Owner asked for one list of commands covering everything the project can do and show, under the README Quickstart.
* **Status:** DONE.
  * New README section *Everything at a glance* (first in Quickstart). It has a 12-step showcase table in presentation order, with what each step shows: sandbox, driving, evolution race (R001), generalisation on a test map (R015/R016 vs baseline), arena `--best`, E2 grip with and without `--fair` (R004 vs R005), E5 overfitting (R006 vs R015), E3 speed reward (R009), one model up close, TensorBoard, the xlsx registry and the track gallery. Below it are short tables for understand/compare, train/queue, evaluate, and tracks/checks. It links to the existing full-flag *Commands* table and says which features aren't built yet.
  * Checks: all 9 showcase game commands ran headless (`--screenshot`, 240 frames), and one screenshot was inspected (the R001 evolution race). `registry.py list --best` / `list --run R015` / `describe`, `make_maps.py --check`, the queue `--dry-run` and the `train.py` flags all ran OK. None of them changed a file. `evaluate.py`, a real `train.py` run, TensorBoard and `preview_tracks.py` were not run (they write files or start servers); their flags were checked against the argparse definitions.
  * The showcase models are each run's best checkpoint by D9 (`registry/models.csv`) and exist in `models/`.
* **Files (edited):** `README.md`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T74, T75: Sandbox in its own window (D25)

* **Task:** Owner asked whether the sandbox was meant to be a pygame desktop app. It never was (D24/T65: Streamlit). Owner then chose to keep Streamlit but show it in a native window, a "mini browser" (D25, pywebview).
* **Status:** DONE.
  * New tests (3) pass. Full suite: 174 passed, 1 failed: the pre-existing flaky registry-lock test (T76). It fails about 1 in 5 runs on clean code as well.
  * Real check on port 8599: the window opened (title "AI Sandbox — RL Racing Line"), WebView2 held live connections to the app, and closing the window stopped the server's whole tree. No sockets or processes were left.
  * With a server already running (the owner's own sandbox on 8501), the launcher only opens a window. Closing it leaves that server running (checked).
  * No `%APPDATA%\pywebview`, `%LOCALAPPDATA%\pywebview` or `~/.streamlit` was created. WebView2 data went to `outputs/sandbox/webview/EBWebView` (git-ignored via `outputs/sandbox/`).
* **Install (`.venv` only):** pywebview 6.2.1 + bottle 0.13.4, cffi 2.1.1, clr_loader 0.3.1, proxy_tools 0.1.0, pycparser 3.11, pythonnet 3.2.1. `pip check` clean, no pinned package changed. WebView2 runtime 154.0.4258.62 is already part of Windows 11 (nothing installed globally).
* **Details:**
  * `scripts/sandbox.py`: default view is the window. `--browser` opens a browser tab (old behaviour), `--no-browser` only runs the server. If the window can't open (no pywebview or WebView2), it falls back to the browser.
  * Closing the window stops the server with `procs.stop` (whole tree: the venv's `python.exe` is a launcher). Ctrl+C does the same. A server that is still starting is given `server_wait_s`.
  * Window settings (title, size, minimum size, storage folder, server wait) are in `configs/sandbox/default.yaml` `window:`.
* **Files (edited):** `scripts/sandbox.py`, `configs/sandbox/default.yaml`, `tests/test_sandbox.py`, `requirements.txt`, `PROJECT_CONTEXT.md` (§4, §10.5, §14.1, §16 D25), `docs/SANDBOX_PLAN.md` (§1), `README.md`, `log.md`.
* **Notes:** The owner's own sandbox (started before this task, PIDs 7144/19816 on 8501) was left running. It still runs the old launcher code; restart it to get the window.

### 2026-10-11 — Claude Code (Opus 5.5) — T73: R015 reliability runs (R018, R019)

* **Status:** DONE.
  * The sandbox queue (`configs/queue/sandbox.yaml`, 2 at a time) finished at 12:30. All 3 entries are `done`; the runner exited and removed `runner.json`.
  * Full suite on the idle machine afterwards: **172 passed**, including the Launch tests that once failed under load.
  * No Python processes and no sockets on 8501 are left.
* **Results (protocol v2, best checkpoint by D9):**
  * **R018_rel_narrow7_s2** (seed 2, 5M, 57 min):
    * Best @3400k: seen 64 %, held-out 100 %, **test maps 0 %**, unseen 62.5 %, lap ratio 0.998.
    * Only @1000k (33 %) and @3000k (17 %) finish any test-map episodes. R015 managed 67 % there.
  * **R019_rel_narrow7_8m_s1** (seed 1, 8M, 86 min; 80 checkpoints evaluated in 189 s):
    * Failed to learn. Best @1700k: overall 6.7 %; crash rate ≈ 93–100 % at every checkpoint.
    * Training curves: episode reward plateaued at 500–600 (R015/R018 > 1650), barrier endings stayed ≈ 50 % (R015/R018 fell below 10 %), and policy std collapsed to 0.04, so it stopped exploring.
    * Same seed as R015, but `total_steps` sets the linear learning-rate decay, so its path differs from the first update.
  * The pipeline isn't to blame: R018 was launched the same way and trained normally, and nothing in training, env or physics changed.
* **Conclusion:** R015's T50 result is partly seed luck. The narrow-ray setup on `maps7` varies strongly between runs, so claims need several seeds. Recorded in PROJECT_CONTEXT §11 (table + note) and §16 (with an open question for the owner), `registry/run_notes.yaml` (R015 outcome updated, R018/R019 added) and the README status.
* **Files:**
  * Edited: `registry/runs.csv`, `registry/models.csv` (R018: 50 checkpoints, R019: 80, written by the runs), `PROJECT_CONTEXT.md`, `README.md`, `registry/run_notes.yaml`, `log.md`.
  * Created: `outputs/eval/R018_*`, `outputs/eval/R019_*` (eval copies).
  * Models (`models/R018_*`, `models/R019_*`) stay local, as always.

### 2026-10-11 — Claude Code (Opus 5.5) — T72: Sandbox S6 (Overview, docs)

* **Status:** DONE (code and docs).
  * The sandbox tests (51 with `test_queue.py`) pass twice in a row while R018/R019 train.
  * The full suite on an idle machine (170) passed before the runs started. I'll rerun it once they finish.
  * The owner's walkthrough ("all 4 goals without the terminal") is theirs to do. Every page and every button path is tested headless (AppTest), and goals 3 and 4 were also done for real (T70, T71).
* **Overview page:**
  * Cards for the 4 goals.
  * Counts: runs, checkpoints, trial runs, models the engine can't run.
  * **Leaderboard** of best checkpoints (v2 or v1): sorted by unseen completion, then lap ratio. Runs with another car or reward (R005/R011 no grip limit, R009 speed reward) are listed last with a *not comparable* note. Select rows to Explore (1), Compare (2–4) or Race them (1–20).
  * **Running now:** the queue (entries, minutes running, ≈ training time left, waiting entries) and trainings started from a terminal, plus game windows with *Stop*.
* **Flaky test fixed:**
  * Under training load, `test_stop_request_interrupts_a_running_queue` once timed out waiting for the runner (20 s). It now waits up to 60 s, and `runner.json` is written atomically.
  * The test's cleanup only killed the venv launcher, so the runner's child lived on and recreated its log folder (`outputs/queue_logs/pytest_stop_…`). Cleanup now kills the whole process tree, and I deleted that leftover folder.
  * The same full run had 2 Launch-page test failures that 5 later runs never reproduced. They were probably side effects of that orphaned runner under load. Noted in case they come back.
* **Docs:**
  * README: AI Sandbox section, commands (`sandbox.py`, `demo.py --title`, `stop.request`), layout.
  * PROJECT_CONTEXT: §4 layout (`racer/sandbox`, `configs/sandbox/`, `run_notes.yaml`), §8 queue runner control, §10.5 later decisions and what was built, §12 Week 3, §16.
  * `docs/SANDBOX_PLAN.md`: status set to built, with the differences from the plan.
* **Files:**
  * Edited: `racer/sandbox/pages/overview.py`, `scripts/queue_overnight.py` (atomic `runner.json`), `tests/test_sandbox.py`, `README.md`, `PROJECT_CONTEXT.md`, `docs/SANDBOX_PLAN.md`, `log.md`.
  * Deleted: `outputs/queue_logs/pytest_stop_88dbe5e5/` (git-ignored test leftover).

### 2026-10-11 — Claude Code (Opus 5.5) — T71: Sandbox S5 (Build & Queue)

* **Status:** DONE. Full suite: **170 passed**.
* **End-to-end check:**
  * I drove the real Build & Queue page headless (AppTest) to add 3 runs to `configs/queue/sandbox.yaml`, then pressed *Start queue*. The runner opened in its own console window, 2 at a time.
    * `sbtest_trial_s1`: trial, 100k steps.
    * `rel_narrow7_s2`: R015's setup with seed 2, 5M steps.
    * `rel_narrow7_8m_s1`: R015's setup with seed 1, 8M steps.
  * The generated commands match R015's original setup (`--trackset maps7 --env rays7_narrow`, no `--set`), plus the form's `--notes` and `--curriculum "AI Sandbox (from R015_t50_narrow7_s1@4600k)"`.
  * The trial (S001) trained, was evaluated (v2) and was marked best within 2 min. The Explorer showed it with its notes, without errors. I then deleted `outputs/scratch/` (it held only this trial).
  * The real runs started as **R018_rel_narrow7_s2** (11:00) and **R019_rel_narrow7_8m_s1** (11:01). The 8M run started only after the tree was confirmed clean (D12).
  * The `sbtest_trial_s1` entry stays in the git-ignored queue file as `done`.
* **What was built:**
  * `racer/sandbox/builder.py`:
    * `Plan`, created from a model's snapshot or from presets.
    * `v1_to_v2` (test: identical observations to `rays9` over a 150-step rollout).
    * Reuses an existing env preset when inputs, outputs, ray marching and episode rules all match; otherwise writes a standalone v2 `configs/env/sb_<tag>.yaml` (never overwrites one with different settings).
    * Hyperparameter and network changes become `--set` against `<algo>_default`. Floats are written so YAML reads them back as numbers (`3.0e-05`, not `3e-05`); a test checks this through train.py's own `apply_sets`.
    * One entry per seed (`<tag>_s<seed>`).
    * Validation with train.py's code (tag, why, unique tag, steps ≥ one checkpoint, presets exist, `build_manifest`), plus warnings and time estimates.
  * `racer/sandbox/queue.py`:
    * Reads, adds to and removes from `configs/queue/sandbox.yaml`, atomically, with args as lists. A `--notes` with `: ` and quotes round-trips through `queue_overnight.load_queue`.
    * Per-entry status, with a `running` left by a dead runner shown as `interrupted`.
    * Checks whether the runner is alive (`runner.json` + its command line).
    * `busy()` (process scan), `start()` (own console window, refused while anything trains), `request_stop()`.
  * `scripts/queue_overnight.py`:
    * Writes `runner.json` while it runs and each entry's `pid` to `state.json`.
    * `stop.request` acts like Ctrl+C.
    * Stopping now kills each run's whole process tree (psutil): a venv's `python.exe` is a launcher, so `terminate()` alone left `train.py` and its env workers running when no console Ctrl+C reached them.
    * Real test: 2 running stand-in entries, a stop request, exit 130, both `interrupted`, the processes gone, `runner.json` removed.
  * **Build & Queue page:**
    * Start from a model's setup or from presets.
    * Tabs: Inputs (ray table editor with a live diagram, the other features with their parameters and descriptions), Outputs & network (layout, layers, activation, live parameter count), Algorithm (each hyperparameter with its glossary note, typical range and default), Car / reward / tracks (preset differences, map previews and shares), Budget & why (1/2/3/5M or custom, seeds, *Trial* checkbox: off = real run, tag, required why).
    * Summary with sizes and time estimates, errors and warnings, and the exact train.py commands.
    * Queue: status per entry, *Remove* (waiting), *Explore* (done), log tails, runs at a time, *Start queue* (with the reason when it's refused), *Stop queue* behind a confirmation checkbox.
  * **Tests:** `tests/test_sandbox.py` restores `sys.modules["__main__"]` after each test. AppTest replaces it, and Windows' spawn-based process pools in `test_queue.py` then re-ran the AppTest script (`NameError: __args`).
* **Notes:**
  * The first process scan in a process takes ~8 s on this machine (psutil); later scans take ~0.25 s, and the page caches the result for 10 s.
  * The "20k-step trial" in the done-when check becomes 100k: checkpoints are saved every 100k, so a 20k run has nothing to show in the Explorer.
* **Files:**
  * Created: `racer/sandbox/{builder,queue}.py`.
  * Edited: `racer/sandbox/pages/build.py`, `scripts/queue_overnight.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T70: Sandbox S4 (Launch)

* **Status:** DONE. 38 sandbox tests pass, and no processes are left over.
* **Real check:** through the sandbox's launcher code, I opened two real game windows at once: 3 cars on Technical, and R004 on Alpine. Each had its own title (`AI Sandbox #1 · technical · R015@4600k, R008@1400k + baseline`). *Stop* on one closed only that window; the other kept running until it was stopped too.
* **How stopping works:**
  * During a race, *Stop* closes the window gracefully (`WM_CLOSE` → pygame QUIT).
  * While a window is still loading models it doesn't handle events, so it is force-stopped after `stop_grace_s` (3 s).
  * I removed the probe windows' state (`outputs/sandbox/`) afterwards.
* **Tests:**
  * The launcher with a stand-in game script: start 2, the list survives a sandbox restart, stopping one leaves the other running, game output goes to the log.
  * A PID with a different creation time is never treated as ours.
  * `demo_argv`.
  * The Launch page, headless: *Launch* then *Stop*.
  * Explorer *Launch in a race* → the Launch page with the model pre-selected.
* **What was built:**
  * `racer/sandbox/procs.py` (psutil):
    * Processes are identified by PID + creation time.
    * Finds running `train.py` / `queue_overnight.py` (a venv launcher and its child count once).
    * Stops gracefully: `WM_CLOSE` to the process tree's windows, then kills the tree.
  * `racer/sandbox/launcher.py`:
    * `demo.py` runs with no console (`CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, so Ctrl+C on the sandbox doesn't reach it). Output goes to `outputs/sandbox/logs/game_NNN.log`.
    * Windows open in a cascade (`SDL_VIDEO_WINDOW_POS`).
    * Instances are kept in `outputs/sandbox/instances.json`.
  * **Launch page:**
    * Model multi-select (best checkpoints, or any checkpoint or trial runs via toggles). Models the engine can't run are hidden and listed with the reason. Up to 20 cars.
    * Map with preview, laps, baseline car, same physics for all.
    * Evolution race: every N-th checkpoint of a run, always ending with the last one.
    * Open windows with *Stop* / *Stop all*; closed windows with their log tail.
  * `scripts/demo.py --title`.
  * `configs/sandbox/default.yaml`: window cascade, stop grace time, queue settings and the measured minutes per million steps. Tunable numbers stay in configs.
  * Page switching (`pages.goto`) now records the target and switches on the next run (`st.switch_page` isn't safe inside button callbacks).
* **Files:**
  * Created: `racer/sandbox/{procs,launcher}.py`, `configs/sandbox/default.yaml`.
  * Edited: `racer/sandbox/pages/{launch,__init__}.py`, `scripts/demo.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T69: Sandbox S3 (Compare)

* **Status:** DONE. 32 sandbox tests pass.
  * R004 vs R008 vs R015 (real data) shows exactly the config differences that changed: total steps, env preset, track set and its fixed maps, obs spec version/name/rays/features. Shared settings (e.g. `physics.mass_kg`) are hidden by default.
  * Two models without a common protocol give a warning instead of numbers.
* **Compare page:**
  * Pick 2–4 models: best checkpoints by default, or any checkpoint or trial runs via toggles. The Explorer's *Compare with others* button pre-fills the selection.
  * **Launch these in a race** button.
  * **What is different:** only config rows that differ, each with its glossary note. Toggles show unchanged settings (with differences in bold) and identity / derived rows.
  * **Shape:** algorithm, inputs, outputs, hidden layers, parameters, presets, steps and engine support, plus rays overlaid on a straight road and on any map.
  * **Results:** only protocols every model has. Metric table, grouped bar chart, best lap per map (★ = fastest, with the baseline) and driven lines side by side.
  * **Learning curves:** overlaid per run, with smoothing.
* **Also:** `explain.snapshot_rows` fills in `env.action_spec.layout` for snapshots from before D22 (no `action_spec` means `steer_gas`), so old and new models don't show a false difference. The cached figure helpers moved to `pages/_ui.py` so both pages share them.
* **Files:** edited `racer/sandbox/pages/{compare,explorer,_ui}.py`, `racer/sandbox/explain.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T68: Sandbox S2 (Model Explorer)

* **Status:** DONE. 30 sandbox tests pass.
  * The straight-road ray diagram matches the real sensor (`cast_rays` on a synthetic 12 m stadium straight) within 0.1 m at offsets 0, +2 and −3.5 m.
  * Per-layer parameter tables add up to `compat.count_params` for PPO and SAC.
  * The Explorer renders headless, without exceptions or warnings, for R001 (v1 inputs, an old snapshot with no stored manifest and a plain track list), R015 (v2 features) and a 3-output `steer_throttle_brake` trial model.
  * I checked the diagram PNGs by eye: straight road, overlay of R015 + R004, rays on Technical, and the network.
  * Timing: first render ≈ 6 s (TensorBoard parsing + building the track field), reruns ≈ 0.9 s.
* **Explorer page:**
  * Run picker (trial runs behind a toggle) and checkpoint slider (★ = best).
  * Headline metrics: algorithm, steps, inputs → outputs, parameters, unseen completion, lap ratio. A warning appears when the engine can't run the model.
  * 10 tabs:
    * **Summary & why:** tag, experiment group, purpose, notes, run notes with an edit form, parent chain, other seeds, results at a glance, identity, training command, *Compare* / *Launch* buttons.
    * **What it senses:** feature table plus rays on a straight road (road width and offset sliders) and rays on any map (position, offset and heading sliders), cast by the model's own sensor.
    * **Its brain:** network diagram and actor/critic layer table, plus "What it controls".
    * **The car**, **Reward**, **How it trained**, **Episode rules:** every value with its note; values that differ from the default preset are marked.
    * **Where it trained:** map previews with each map's share of episodes, plus the generated pool.
    * **Results:** any protocol; per-group finished/ended counts, lap table against the baseline, thumbnails, completion and lap ratio over every evaluated checkpoint.
    * **Learning curves:** reward, lap time, progress, laps, episode length, how episodes ended, and optimiser statistics on demand, with smoothing.
* **Files:**
  * Created: `racer/sandbox/{diagrams,curves}.py`.
  * Edited: `racer/sandbox/pages/explorer.py`, `racer/sandbox/{data,explain}.py` (`section`, `reference_rows`, folder-existence cache), `tests/test_sandbox.py`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T67: Sandbox S1 (data, glossary, run notes)

* **Status:** DONE. `tests/test_sandbox.py` has 21 tests and all pass.
* **Glossary coverage:** every key of every preset in `configs/{physics,reward,train,env,trackset,track}`, of one snapshot per run in `models/` (R001–R017), and every metric column of `models.csv` has an entry with "what it is" and "what it affects".
* **Diff check:** comparing R004 with R008 (real data) shows only `presets.env`, `env.obs_spec.name` and `env.obs_spec.rays` as config differences.
* **What was built:**
  * `racer/sandbox/data.py`: `Store`, read-only, with caches keyed on file mtime and size. It reads real and trial (scratch) runs and checkpoints with a model folder, best checkpoints, snapshots, manifests (stored or derived) and engine problems (`check`). It also reads every protocol's eval files, from the model folder or `outputs/eval/`, plus thumbnails per protocol, per-checkpoint metrics from the eval files (so v1 numbers stay visible after a v2 re-evaluation), parent chains, track previews, TensorBoard folders and run notes. Reading all 560 snapshots takes ~6 s (CSafeLoader); the pages load one run at a time.
  * `racer/sandbox/explain.py`: one key namespace (`physics.*`, `reward.*`, `env.*`, `tracks.*`, `track.*`, `presets.*`, `train.*`, `manifest.*`, `run.*`, `metric.*`), glossary lookup with wildcards, snapshot flattening, a diff that sorts each row into identity / derived / config, and compact value formatting (ray lists as `+7°/200 m`).
  * `racer/sandbox/glossary.yaml`: 183 entries in 11 groups. Feature, layout and algorithm texts stay in their code registries.
  * `registry/run_notes.yaml`: why + outcome for R001–R014, and the outcome for R015–R017 (their "why" is already in the snapshot notes), from PROJECT_CONTEXT §11.
  * **Glossary page:** searchable, with the input features, output layouts and algorithms taken from the code registries.
* **Files:**
  * Created: `racer/sandbox/{data,explain}.py`, `racer/sandbox/glossary.yaml`, `racer/sandbox/pages/_ui.py`, `registry/run_notes.yaml`.
  * Edited: `racer/sandbox/pages/glossary.py`, `tests/test_sandbox.py`, `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T66: Sandbox S0 (install, skeleton)

* **Owner decisions before leaving (T66–T72 run without the owner):**
  * The Build form defaults to a **real run**, with a *Trial* checkbox for scratch runs.
  * The sandbox may **stop a running queue**, with a confirmation step. It works like Ctrl+C: running entries become `interrupted` and resume on the next start.
  * **Run outcome notes can be edited** in the Explorer (they save to `registry/run_notes.yaml`).
  * On anything critical: **skip that piece, log it and continue.**
* **Status:** DONE.
  * The app opens on `localhost:8501` (listening on 127.0.0.1 only) with all 6 pages.
  * `pip check` is clean.
  * The before/after `pip freeze` shows the existing packages unchanged; only the 26 new ones were added.
  * `~/.streamlit/` was not created, and after stopping no process or socket remains.
  * `tests/test_sandbox.py` has 9 tests (launcher settings, routing, each page headless with `AppTest`, the app with navigation); all pass.
* **Install:** `streamlit==1.65.0` into `.venv` only. It added altair 6.3.0, anyio, attrs, certifi, charset_normalizer, click, h11, httptools, idna, itsdangerous, jsonschema (+ specifications), narwhals, pyarrow 25.0.1, pydeck, python-multipart, referencing, requests, rpds-py, starlette 1.7.0, toml, urllib3, uvicorn 0.54.0, watchdog, websockets. `psutil==7.2.2` was already installed (unrecorded) and is now pinned.
* **Files:**
  * Created: `scripts/sandbox.py` (re-runs itself under `streamlit run` with all settings on the command line: localhost, headless, no usage stats, no file watcher; waits for `/_stcore/health`, opens the browser, `--port`, `--no-browser`; reuses a running server), `racer/sandbox/__init__.py`, `racer/sandbox/pages/{__init__,overview,explorer,compare,build,launch,glossary}.py` (placeholders), `tests/test_sandbox.py`.
  * Edited: `requirements.txt` (streamlit, psutil), `.gitignore` (`outputs/sandbox/`, `.streamlit/`, `configs/queue/sandbox.yaml`), `PROJECT_CONTEXT.md` (§14.1 pins), `log.md`.

### 2026-10-11 — Claude Code (Opus 5.5) — T65: AI Sandbox plan v2

* **Task:** owner asked to plan the sandbox's implementation and to ask any critical decisions.
* **Status:** DONE (plan only; no code, no installs).
* **Owner decisions:**
  * Streamlit 1.65.0.
  * The sandbox may start the queue in its own console.
  * Localhost only.
  * Retroactive "why" and outcome notes go in `registry/run_notes.yaml`.
* **Checked against the code before planning:**
  * `pip install --dry-run streamlit` in `.venv` gives 1.65.0 plus 26 new packages and changes no existing pin (numpy, pandas and protobuf stay). Nothing was installed.
  * The glossary has to cover about 135 config keys: physics 32, reward 8, train 41 (including the baseline keys), env 27, trackset 8, plus snapshot-only keys.
  * TensorBoard logs have 21 scalar tags per run (`rollout/ep_rew_mean`, `race/lap_time`, `race/progress_m`, `end/*`, `train/*`), with about 800 points for R015.
  * `describe_features()` already provides the feature descriptions.
  * Only R015–R017 have a real "why" note; R004–R014 have none. This is the reason for `run_notes.yaml`.
  * There is no v1 → v2 obs conversion yet; the builder needs one.
  * `queue_overnight.py` `state.json` has no PIDs, so S5 adds them.
  * The queue already re-reads its file (T64), so that item was removed from the plan.
* **Found:** `psutil 7.2.2` is installed in `.venv` but missing from `requirements.txt`, and no log entry records installing it. T66 pins it; the sandbox uses it for process checks.
* **Plan changes v1 → v2:**
  * `scripts/sandbox.py` launches itself headless with every setting on the command line, so nothing is written to `~/.streamlit/`.
  * Ray diagram on a real map, using the model's own sensor code.
  * Actor and critic both shown in the network diagram.
  * The sandbox queue file is git-ignored, so it never makes the tree dirty (D12).
  * New env presets stay untracked (D12 ignores untracked files).
  * Queue args are written as YAML lists.
  * Game windows are matched by PID + creation time and stopped gracefully.
  * Demo output goes to log files instead of console windows.
  * A trial-run (`--scratch`) toggle.
  * Page tests with `streamlit.testing.v1.AppTest`.
  * Build tasks T66–T72 with "done when" checks.
* **Files (edited):** `docs/SANDBOX_PLAN.md` (rewritten as v2), `PROJECT_CONTEXT.md` (§4 layout, §10.5 decisions, §16), `log.md` (T65–T72, Current Status, this entry)

### 2026-10-11 — Claude Code (Opus 5.5) — T64: Parallel queue

* **Status:** DONE. Real check: a 2-slot scratch queue with 3 runs (200k steps each on `maps7`, with v2 evaluation) finished in 6 min (≈ 9 min one after another): distinct ids S001–S003, state `done` ×3, 6 checkpoints evaluated in the scratch registry, no lock left. Throughput while two trained together: 1287 + 1273 = **2560 steps/s** (T62 benchmark: 2754); the run that trained mostly alone: 1608. Scratch outputs and the test queue's logs deleted afterwards.
* **What changed:**
  * `queue_overnight.py`: `--parallel N` or `parallel:` in the queue file (default 1, so existing queues behave as before); a polling scheduler fills free slots (starts staggered by a few seconds); re-reads the queue file whenever a slot is free (new entries are picked up, a half-saved file is ignored); with N > 1 each run gets `--eval-workers 6 // N` unless the entry sets it; Ctrl+C stops every running entry and marks it `interrupted`; `state.json` is written atomically; queue files are checked for unknown keys.
  * `train.py`: `reserve_run_id` reserves the id with an atomic `mkdir` marker in `<out_dir>/.run_ids/` (a run's first model folder only appears at its first checkpoint, so two runs started together would have got the same id). Ids are never reused, even if a run fails early.
  * `racer/registry/registry.py`: `file_lock` (exclusive lock file, stale after 10 min) around every read-modify-write (`upsert_run`, `upsert_model`, `set_best`); `export_xlsx.export` holds it too.
* **Tests:** `tests/test_queue.py` (2 slots never exceeded with 3 entries, one failing → state done/failed/done, shared eval workers; 4 processes × 25 upserts → 100 rows; 8 concurrent reservations → S001–S008). Full suite **122 passed** (7 clean runs in a row). The very first full run, started seconds after the scratch queue ended, had 1 failure that never reproduced; the only timing-based test is `test_sensors.py`'s ray-cost check (< 15 µs for 5 rays), which can fail while other processes still load the CPU. Run the suite on an idle machine.
* **Files (created):** `tests/test_queue.py`. **Edited:** `scripts/queue_overnight.py`, `scripts/train.py`, `racer/registry/registry.py`, `racer/registry/export_xlsx.py`, `.gitignore` (`registry/registry.lock`), `README.md`, `PROJECT_CONTEXT.md` (§8 overnight queue), `log.md`

### 2026-10-11 — Claude Code (Opus 5.5) — T07, T46, T47, T48 closed (M2)

* **Task:** owner noticed T07/T46/T47/T48 still open; check each against its "done when" and close it.
* **T46:** DONE. All 11 `night1` runs (R004–R014) finished, registered and evaluated (v1; best checkpoints also v2, T61). No failures.
* **T47:** DONE. The FPS part was met in T51 (20 distinct models, 1 % low 59–63 fps on every showcase track). The missing headless check was added: `test_twenty_car_race_finishes` (20 cars, 10-row grid, everyone finishes, standings ordered by finish time, gaps ≥ 0; 4 s).
* **T48:** DONE. PROJECT_CONTEXT: §8 records what happened to the D8 curriculum (R002 negative → from scratch on mixed tracks since; R017 fine-tune no gain); §11 "Runs so far" table (every run id → experiment, setting, v2 seen/unseen/lap ratio); §4 + `bench_train.py`; §16 M2 entry. README status. (§9 eval artefacts and the README commands, registry and arena sections were already updated in T52–T62.)
* **T07 (M2):** DONE, all subtasks T37–T48 complete.
* **Files (edited):** `tests/test_race.py`, `PROJECT_CONTEXT.md`, `README.md`, `log.md`

### 2026-10-11 — Claude Code (Opus 5.5) — T62: GPU and parallel-run benchmark

* **Status:** DONE. `scripts/bench_train.py`, 200k steps per setting, `ppo_default`, `maps7`, `rays5`, idle machine (nothing else running). GPU: GTX 1650 with `torch 2.14.1+cu126` in `.venv-gpu`.

  | setting | total steps/s | vs today | collect / update (s) |
  |---|---|---|---|
  | CPU, 6 envs, batch 256 (today) | 1704 | 1.00× | 62 / 57 |
  | GPU, 6 envs, batch 256 | 1093 | 0.64× | 118 / 68 |
  | CPU, 6 envs, batch 1024 | 2087 | 1.22× | 63 / 34 |
  | GPU, 6 envs, batch 1024 | 1451 | 0.85× | 118 / 22 |
  | GPU, 6 envs, batch 2048 | 1529 | 0.90× | 118 / 14 |
  | CPU, 2 runs × 3 envs | 2206 | 1.29× | each 1102 |
  | GPU, 2 runs × 3 envs | 1193 | 0.70× | each 596 |
  | CPU, 3 runs × 2 envs | 2410 | 1.41× | each ~803 |
  | GPU, 3 runs × 2 envs | 1140 | 0.67× | each ~380 |
  | **CPU, 2 runs × 6 envs** | **2754** | **1.62×** | each ~1377 |

* **Why the GPU loses:** it only speeds up the update with large minibatches (57 s → 14 s at batch 2048), but every env step also runs the policy to pick an action, and on the GPU each of those tiny batches (6 observations) pays transfer and launch overhead: collection takes 118 s instead of 62 s. Batch 1024 on the CPU is 1.22× faster but changes learning (4× fewer gradient steps per rollout), so it's an experiment, not a free speed-up.
* **Recommendation:** keep training on the CPU (T63 not needed). Run **2 trainings at once with 6 envs each** (1.62× more experiments per night; each run ~19 % slower on the clock) → T64. RAM: ≈ 0.2–0.25 GB per env worker + ≈ 0.4 GB per main process (task manager), so 2 runs ≈ 3.5 GB of 16 GB.
* **GPU-trained models:** a model trained on CUDA loads in the CPU `.venv` and gives **identical** actions to a CPU load in `.venv-gpu`; vs CUDA inference the difference is 1.5e-7 (float rounding). So GPU training would never have broken model loading.
* **Cleanup:** `.venv-gpu` deleted (≈ 3 GB; GPU not useful, as agreed); `.gitignore` keeps the `.venv-gpu/` line in case it is recreated. Worktree `outputs/scratch` (fps screenshots) deleted. No processes or servers left running.
* **Files (edited):** `log.md`, `PROJECT_CONTEXT.md` (§8 device, §16), `README.md` (benchmark command)

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
