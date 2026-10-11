# AI Sandbox: implementation plan

> **Status:** **built** (T66–T72, 2026-10-11); run it with `.venv\Scripts\python scripts\sandbox.py`. This is plan v2 (T65), with the owner's decisions in §7; v1 was T58. The build follows this plan. The differences are listed in `log.md` T66–T72:
> - The end-to-end trial run used 100k steps, not 20k: a checkpoint is saved every 100k, so a 20k run has nothing to show in the Explorer.
> - Overview lists runs with another car or reward (e.g. no grip limit) last, as not comparable.
> - Tunable numbers live in `configs/sandbox/default.yaml`.
> **Owner's goals, in order:** (1) understand a model: see what it was trained with, and why; (2) compare models; (3) configure a new model and queue it, choosing the number of steps; (4) pick trained models and launch a game window with them, several at once.

---

## 1. What it is

A **standalone local web app** (Streamlit 1.65.0) that runs next to the game, never inside it.

```
.venv\Scripts\python scripts\sandbox.py          → opens http://localhost:8501
```

`scripts/sandbox.py` is both the launcher and the app. Run with plain `python`, it re-runs itself through `streamlit run`, passing the settings on the command line: `--server.address localhost`, `--server.port 8501`, `--server.headless true` and `--browser.gatherUsageStats false`. It then opens the browser itself. Headless mode skips Streamlit's first-run e-mail prompt, so nothing is written to `~/.streamlit/`, and the project needs no global config.

| Principle | Consequence |
|---|---|
| Standalone | Its own process. Game windows (`demo.py`) and the queue (`queue_overnight.py`) are started as separate processes. Closing the sandbox never closes a game or stops a queue. |
| Read-only on models | It never writes to `models/` or the registry CSVs. It writes only: `configs/queue/sandbox.yaml` (git-ignored), new input presets `configs/env/sb_<tag>.yaml`, `registry/run_notes.yaml` (outcome notes), and its own state in `outputs/sandbox/` (git-ignored). |
| One source of truth for explanations | Feature, action-layout and algorithm descriptions come from the code registries (`features.py` `describe_features()`, `actions.py`, `algorithms.py`). Config keys are explained in one glossary file. A test fails if any key has no explanation. |
| Works for any model | Every model is read through `manifest_of()` and `check()` (D23). Models with other inputs, outputs, algorithms or physics versions display correctly, and models the engine can't run are listed with the reason. |
| Local only (owner) | Bound to `localhost`; no telemetry. The pages can start processes, so they are not exposed on the network. |

---

## 2. Pages

### 2.1 Overview (home)
- Counts: runs, checkpoints, how many were evaluated with v2, models the engine can't run.
- **Leaderboard of best checkpoints** by `unseen_completion` (protocol v2), then lap ratio. A toggle switches to v1 numbers. Rows link to the Explorer.
- **What is running:** queue entries from `state.json` (running / done / failed, ETA), trainings started from a terminal (process scan), and game windows with *Stop*.

### 2.2 Model Explorer (main purpose: understand a model)
Pick a run, then a checkpoint (a select slider over its steps, with the best pre-selected). A toggle includes trial (scratch) runs.

| Section | Shows | Source |
|---|---|---|
| **Summary** | id, algorithm, steps, parent, created, git commit; the headline metrics (seen / test / held-out / unseen completion, lap ratio) | snapshot, `eval_v2.json` (else `eval.json`) |
| **Why** | The run's purpose and what it showed: tag, `curriculum`, `notes`, the outcome from `registry/run_notes.yaml`, the parent chain and the experiment group (E1–E9 / T50) | snapshot, `runs.csv`, `run_notes.yaml` |
| **What it senses** | (a) **Ray diagram on a straight:** the car to scale on a 12 m road, each ray at its angle, drawn to where it hits the edge (or to its range), with the distance labelled. (b) **On a real map:** pick a map and a position slider; the model's own sensor code casts its rays on that track, drawn over the track. (c) The feature list in input order with each feature's description, parameters and dimensions; the input count. | `manifest.inputs`, `describe_features()`, `racer/env/sensors.py` |
| **Its brain** | **Network diagram:** inputs → hidden layers → outputs for the actor (drives the car) and the critic (training only), with parameters per layer and in total, and the activation | `manifest` (`net_arch`, `n_params`, `n_params_total`) |
| **What it controls** | The action layout and what each output does | `actions.py` |
| **The car** | Physics preset and version; each parameter with its value and a plain-language note (grip per surface, power, braking, steering lock, drag, downforce); differences from `default` highlighted | snapshot `env_config.physics`, glossary |
| **What it is rewarded for** | Each reward term with its weight and what it encourages or punishes | snapshot `env_config.reward`, glossary |
| **Where it trained** | The track set: map previews with weights, and the generated pool settings | snapshot `env_config.tracks`, `outputs/plots/tracks/` |
| **How it trained** | Hyperparameters, each with "what it affects" and the typical range; differences from the algorithm's default preset highlighted | snapshot `train`, glossary |
| **Episode rules** | Start randomisation and the termination limits | snapshot `env_config.env.episode`, glossary |
| **Results** | Metrics per evaluation group, lap table against the baseline, trajectory thumbnails | `eval*.json`, `outputs/eval/<id>/` |
| **Learning curves** | Episode reward, lap time, progress and termination reasons over training, with an optional train-loss panel | TensorBoard `runs/<run_id>_*` |
| **Over training** | Completion and lap ratio of every evaluated checkpoint against steps; the chosen checkpoint marked | `models.csv` |

### 2.3 Compare (2–4 models)
- **Config diff:** each snapshot is flattened to `section.key` rows, and by default only the rows that differ are shown, each with its glossary note. Identity rows (ids, dates, git commit, seed) are hidden behind a toggle. This answers "what is different between these?" directly.
- **Shape:** inputs, outputs and parameter counts side by side; the ray diagrams overlaid in one plot.
- **Metrics** as grouped bars, within one protocol only. If the models were evaluated with different protocols, a warning appears and only their common protocol is used.
- Learning curves overlaid; thumbnails side by side for each map.
- *Launch these in a race* opens the Launch page with the selection filled in.

### 2.4 Build & Queue (configure a new model)
A form that starts from an existing model ("start from R015@4600k") or from presets:

| Group | Controls | Live feedback |
|---|---|---|
| Inputs | Add, remove and reorder features; a ray editor (an angle and range per ray, add or remove rays); the parameters of each feature | Ray diagram, input count |
| Outputs | Action layout | Output count |
| Network | Hidden layers and widths, activation | Parameter count (`compat.count_params`), compared with the current models |
| Algorithm | PPO / SAC and their hyperparameters (glossary note and typical range next to each) | Changes from the default preset, shown as `--set` |
| Car and reward | Physics preset, reward preset (existing presets; a new car is a new preset file, as before) | What the preset changes compared with `default` |
| Tracks | Track set preset (`maps7`, `final_mix`, …); test and held-out maps can't be chosen | Map previews |
| Budget | **Steps** (1M / 2M / 3M / 5M or a custom value), seeds (one queue entry per seed), real run or trial (`--scratch`) | Estimated time: ≈ 10 min per million steps alone, ≈ 13 min per million with 2 in parallel (T62/T64); the queue's total ETA |
| Why | Tag and a required "why" note (passed as `--notes`, so it is stored in the snapshot and in `runs.csv`) | The model id it will get (`R0NN_<tag>`, estimated) |

**On *Add to queue*:**
1. **Validate** with `build_manifest()` + `check()` (the code `train.py` uses), plus `train.py`'s own rules: the tag is unique, and the track set has no test or held-out maps.
2. **Inputs:** if the feature list equals an existing env preset, use that preset's name. Otherwise write a new v2 preset `configs/env/sb_<tag>.yaml`, a standalone file because `extends` can't change v1 into v2. A model whose obs spec is v1 is converted to the equivalent v2 feature list (`builder.v1_to_v2`; a test checks that it gives the identical observation). The new file stays untracked. D12 ignores untracked files, so real runs still start, and the snapshot stores the full spec anyway. The page suggests committing the file for the record.
3. **Hyperparameter and network changes** become `--set` arguments against the algorithm's default preset (`--set policy.net_arch=[128,128]`). No file is written.
4. **Append** an entry to `configs/queue/sandbox.yaml` (git-ignored; `parallel: 2`). Args are written as a YAML list, so `: ` and quotes can't break the file. Writes are atomic (temp file + replace); the runner ignores a half-written file anyway. The exact `train.py` command is shown as well.
5. **Pending entries** (not started yet) are listed with *Remove*. The runner re-reads the file whenever a slot frees (T64), so edits apply while it runs.
6. ***Start queue*** opens `queue_overnight.py configs/queue/sandbox.yaml` in **its own console window** (owner), so it keeps running after the sandbox closes and can be stopped with Ctrl+C there. It is disabled while any queue or `train.py` is running (process scan), so the CPU is never oversubscribed.

### 2.5 Launch (pick trained models, press *Launch*)
- Multi-select from the registry: searchable, sorted by the leaderboard, models the engine can't run disabled with the reason, at most 20 cars. Options: map (any track in `tracks/`, with preview), laps, *add baseline*, *fair physics*.
- *Launch* starts `demo.py --track <map> --models <ids…> [--baseline] [--laps N] [--fair] --title "<label>"` as a **separate process** with no console; its output goes to `outputs/sandbox/logs/<instance>.log`. Pressing it again opens another window.
- *Evolution race:* pick a run and every Nth checkpoint (`demo.py --run R0NN --every N`).
- **Running instances:** a list of PID, label, models, map and start time, each with *Stop*. The list is kept in `outputs/sandbox/instances.json`. A process is matched by PID **and creation time**, so a recycled PID is never stopped by mistake. *Stop* closes the window gracefully (`WM_CLOSE`), then force-kills after 3 s. Dead entries are pruned.

### 2.6 Glossary
Every parameter the sandbox shows: what it is, what it affects, the typical range, trade-offs, and the related experiment (E1–E9). It is searchable and grouped (car, reward, inputs, episode, training, tracks). This is the page to open when presenting.

---

## 3. Code layout

```
scripts/sandbox.py                 # launcher (re-runs itself under streamlit) + st.navigation routing
racer/sandbox/                     # no Streamlit imports outside pages/ → unit-testable
    data.py        registry rows (real + scratch), snapshots, manifests, check() results, eval docs,
                   thumbnails, run notes; cached by file mtime
    curves.py      learning curves from TensorBoard events (EventAccumulator, already a
                   dependency); downsampled; cached by event-file mtime
    explain.py     glossary lookup (wildcards such as physics.surfaces.*.mu_static), snapshot
                   flattening, config diff
    diagrams.py    ray diagram (straight / real map via the model's own sensor), network diagram
                   (matplotlib)
    builder.py     form state → env preset dict + train.py argv; v1 → v2 obs conversion;
                   validation via compat; time estimate
    queue.py       read / append / remove entries in the queue YAML (atomic), read state.json, start
                   the runner
    procs.py       process helpers (psutil): alive(pid, created), scan for train.py /
                   queue_overnight.py, graceful stop
    launcher.py    start / list / stop demo.py instances (outputs/sandbox/instances.json)
    glossary.yaml  explanations for every physics, reward, train, env and track-set key
    pages/         overview.py, explorer.py, compare.py, build.py, launch.py, glossary.py
registry/run_notes.yaml            # why + outcome per run (tracked; old runs written from §11)
tests/test_sandbox.py              # data, glossary coverage, diff, diagrams, builder, queue,
                                   # launcher (no browser needed)
```

The Streamlit app itself is checked with `streamlit.testing.v1.AppTest` (headless, no browser): each page renders without an exception for a v1 model, a v2 model and an empty registry.

---

## 4. Changes needed elsewhere

| Where | Change | Why |
|---|---|---|
| `requirements.txt` | `streamlit==1.65.0` (+26 packages, no existing pin changes; dry run 2026-10-11) and `psutil==7.2.2` (already in `.venv` but never recorded) | New dependencies, `.venv` only |
| `.gitignore` | `outputs/sandbox/`, `.streamlit/`, `configs/queue/sandbox.yaml` | Run state. The sandbox queue file changes all the time and must not make the tree dirty for D12. |
| `scripts/demo.py` | `--title` for the window caption | Telling several game windows apart |
| `scripts/queue_overnight.py` | Write each running entry's `pid` into `state.json` | So the sandbox can tell a live queue from a stale `running` state after a crash |
| `scripts/train.py` | Nothing: `--env`, `--action-layout`, `--set`, `--steps`, `--seed`, `--notes`, `--scratch` cover every form option | Built in T55–T57 |

---

## 5. Build steps (each one testable; tasks in `log.md`)

| Step | Task | Work | Done when |
|---|---|---|---|
| S0 | T66 | Install and pin; `.gitignore`; launcher + empty pages | The app opens on `localhost:8501` with every page listed; `pip check` clean; no `~/.streamlit/`; the server stops cleanly |
| S1 | T67 | `data.py`, `explain.py`, `glossary.yaml`, `run_notes.yaml` | Tests pass, including glossary coverage of every preset and snapshot key, and R004 vs R008 differing only in the ray list (plus identity rows) |
| S2 | T68 | Model Explorer, `diagrams.py`, `curves.py` | R001 (v1), R015 (v2) and a 3-output scratch model display correctly; diagram hit distances match the sensor within 0.1 m |
| S3 | T69 | Compare | R004 vs R008 vs R015 shows the expected differences; mixed protocols give a warning |
| S4 | T70 | Launch, `procs.py`, `launcher.py`, `demo.py --title` | Two game windows at once; *Stop* closes one only; incompatible models can't be picked |
| S5 | T71 | Build & Queue, `builder.py`, `queue.py`, queue `pid` | A 20k-step trial built in the form trains through the queue and appears in the Explorer; *Start queue* is refused while training runs |
| S6 | T72 | Overview, polish, docs, full tests | The owner does all 4 goals without the terminal |

**Order:** S0 → S1 → S2 is the critical path to goal 1. S3 and S4 are independent of each other. S5 comes last because it is the only step that writes files other code reads.
**Estimate:** S0 ≈ 1 h, S1 ≈ half a day (most of it writing the glossary), S2 ≈ 1 day, S3 + S4 ≈ half a day, S5 ≈ 1 day, S6 ≈ half a day. **≈ 3.5 days.** Each step is committed locally once verified.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Reading 400+ snapshots and TensorBoard files is slow | `st.cache_data` keyed by file mtimes; snapshots per run on demand; curves downsampled to ≤ 500 points |
| A queue started from the sandbox competes with a running training for the CPU | Process scan; *Start queue* disabled while any queue or `train.py` runs |
| A recycled PID is stopped by mistake | Match PID and creation time (psutil) |
| Explanations drift from the code | Descriptions live in the code registries; the glossary test fails on any unexplained key, including keys in old snapshots |
| A new input preset or queue file makes the tree dirty | The queue file is git-ignored; new presets are untracked files, which D12 ignores |
| Streamlit writes to the user profile | Launched headless with every setting on the command line; S0 checks that `~/.streamlit/` doesn't appear |
| Game windows or the server left running after a session | Covered by the end-of-run checklist: stop instances and the server, then check sockets |

---

## 7. Owner decisions (2026-10-11, T65)

1. **Streamlit 1.65.0.** The dry run showed no conflict with the pinned numpy 2.4.6 or pandas 3.0.6, so NiceGUI isn't needed.
2. **The sandbox may start the queue.** *Start queue* runs the same `queue_overnight.py` in its own console window.
3. **Local only.** Bound to `localhost`, not shared on the LAN.
4. **Run notes:** a tracked `registry/run_notes.yaml` holds the why and outcome of each run. Old runs are filled from PROJECT_CONTEXT §11. New runs from the form get their "why" through `--notes`, and their outcome is added later.
5. **New runs default to real runs.** The Build form has a *Trial* checkbox for scratch runs.
6. **The sandbox may stop a running queue**, after a confirmation step. It writes a stop request (`stop.request` next to `state.json`), and the runner treats it like Ctrl+C: running entries become `interrupted`, finished checkpoints stay, and the entries resume when the queue starts again.
7. **Outcome notes can be edited** in the Explorer. They save to `registry/run_notes.yaml`; snapshots are never touched.
8. **Owner away during T66–T72:** anything critical is skipped or given the safe option, and logged.
