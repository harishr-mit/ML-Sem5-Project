# AI Sandbox: implementation plan

> **Status:** plan only (T58, 2026-10-11). Build it right after M2, T50 and T51 (PROJECT_CONTEXT §10.5, §12, D24).
> **Owner's goals, in order:** (1) understand a model: see what it was trained with, and why; (2) compare models; (3) configure a new model and queue it, choosing the number of steps; (4) pick trained models and launch a game window with them, several at once.

---

## 1. What it is

A **standalone local web app** (Streamlit) that runs next to the game, never inside it.

```
.venv\Scripts\python -m streamlit run scripts\sandbox.py      → http://localhost:8501
```

| Principle | Consequence |
|---|---|
| Standalone | Its own process. The game (`demo.py`) is launched from it as separate processes, so several game windows can run at once, and closing the sandbox never closes a game. |
| Read-only on models | It never writes to `models/` or the registry CSVs. It writes only: queue files (`configs/queue/sandbox.yaml`), new input presets (`configs/env/sb_*.yaml`), and its own state (`outputs/sandbox/`, git-ignored). |
| One source of truth for explanations | Feature, action-layout and algorithm descriptions come from the code registries (`racer/env/features.py`, `actions.py`, `racer/agents/algorithms.py`). Everything else comes from one glossary file. No explanation is written twice. |
| Works for any model | It reads every model through `manifest_of()` and `check()` (D23), so models with other inputs, outputs, algorithms or physics versions show up correctly, and models the engine can't run yet are listed with the reason. |
| Local only | Streamlit bound to `localhost`; no telemetry (`browser.gatherUsageStats = false`). |

---

## 2. Pages

### 2.1 Overview (home)
- Counts: runs, checkpoints, models the engine can't run.
- **Leaderboard of best checkpoints** by `unseen_completion` (protocol v2), then lap ratio. Toggle: v1 / v2 numbers.
- Queue status: the running queue's `state.json` (current run, done, failed, ETA from the measured ~10 min per million steps).
- Running game windows (from the launcher, §2.5) with *Stop* buttons.

### 2.2 Model Explorer (main purpose: understand a model)
Pick a run, then a checkpoint (slider over its steps; the best is pre-selected). Sections:

| Section | Shows | Source |
|---|---|---|
| **Summary** | id, algorithm, steps, parent, created, git commit; v2 metrics (seen / test / held-out / unseen completion, lap ratio) | snapshot, `eval_v2.json` (or `eval.json`) |
| **What it senses** | A **ray diagram**: the car drawn to scale on a 12 m road, every ray at its angle and range, with the distance at which it hits the road edge on a straight. The feature list in input order with each feature's description and parameters; the input count. | `manifest.inputs`, `features.py` descriptions |
| **Its brain** | A **network diagram**: input count → hidden layers → outputs, with the parameter count per layer and in total, and the activation | `manifest` (`net_arch`, `n_params`) |
| **What it controls** | The action layout and what each output does | `actions.py` |
| **The car** | Physics preset, with the parameters that matter shown with plain-language notes (grip, power, braking, steering lock) and the physics version | snapshot `env_config.physics`, glossary |
| **What it is rewarded for** | Each reward term with its weight and what it encourages or punishes | snapshot `env_config.reward`, glossary |
| **Where it trained** | The track set: each map with a thumbnail and weight, plus the generated pool settings | snapshot `env_config.tracks`, `outputs/plots/tracks/` |
| **How it trained** | Hyperparameters, each with "what it affects" and the typical range | snapshot `train`, glossary |
| **Why** | The run's tag, `curriculum` text and `notes`, the parent chain, and the ablation group it belongs to | snapshot, `runs.csv` |
| **Results** | Metrics per evaluation group, lap table vs the baseline, trajectory thumbnails, and **learning curves** (episode reward, lap time, termination reasons) | `eval*.json`, `outputs/eval/`, TensorBoard logs in `runs/` |
| **Over training** | Completion and lap ratio of every checkpoint of the run against steps (where the policy got good, where it got worse) | `models.csv` |

### 2.3 Compare (2–4 models)
- **Config diff:** the snapshots are flattened to `section.key` rows; only rows that differ are shown by default, each with its glossary note. This answers "what is different between these two?" directly.
- Shape: inputs, outputs, parameter counts side by side, with both ray diagrams overlaid.
- Metrics as grouped bars (same protocol only; a warning if the models were evaluated with different protocols).
- Learning curves overlaid; thumbnails side by side per map.
- *Launch these in a race* button (goes to §2.5 with the selection filled in).

### 2.4 Build & Queue (configure a new model)
A form that starts from an existing model ("start from R008") or from presets:

| Group | Controls | Live feedback |
|---|---|---|
| Inputs | Add, remove and reorder features; a ray editor (angle, range per ray) | Ray diagram, input count, look-ahead distance on a 12 m road |
| Outputs | Action layout | Output count |
| Network | Hidden layers and widths, activation | Parameter count; a warning if it is far larger than the current models |
| Algorithm | PPO / SAC and their hyperparameters (glossary note and safe range next to each) | |
| Car and reward | Physics preset, reward preset | Short note on what changes |
| Tracks | Track set (`maps7`, `final_mix`, …); test and held-out maps can't be chosen | Map thumbnails |
| Budget | **Steps** (presets 1M / 2M / 3M / 5M, or a custom value), seed(s) | Estimated time (~10 min per million steps with 6 envs, from §8; updated from measured runs) |
| Why | Tag, and a required "why" note (stored as the run's notes) | The model id it will get |

**On *Add to queue*:**
1. Validate with `build_manifest()` + `check()`, the same code that `train.py` uses.
2. If the inputs or outputs differ from every existing preset, write a new v2 preset `configs/env/sb_<tag>.yaml`. Hyperparameter and network changes need no file: they become `--set` arguments (`--set policy.net_arch=[128,128]`).
3. Append an entry to `configs/queue/sandbox.yaml` (same format as `night1.yaml`). Show the exact `train.py` command too.
4. *Start queue* launches `queue_overnight.py configs/queue/sandbox.yaml` in its own console window, unless it is already running.

Real runs need a clean git tree (D12), and a new preset is a new file. The page shows "commit these files before the queue starts" with the file list; it doesn't run git itself.

### 2.5 Launch (pick trained models, press *Launch*)
- Multi-select from the registry (searchable; models the engine can't run are disabled, with the reason), a map, laps, *add baseline*, *fair physics*.
- *Launch* runs `demo.py --track <map> --models <ids…> [--baseline] [--laps N] [--fair]` as a **separate process** (`subprocess.Popen`, new console on Windows). It can be pressed again for another window with other models.
- *Evolution race:* pick a run and every Nth checkpoint (`demo.py --run R00N --every N`).
- Running instances are listed (PID, models, map, started) with *Stop*. The list is kept in `outputs/sandbox/instances.json` and pruned of dead PIDs.

### 2.6 Glossary
Every parameter the sandbox shows: what it is, what it affects, typical range, trade-offs, and the related experiment (E1–E9). Searchable. This is the page to open when presenting.

---

## 3. Code layout

```
scripts/sandbox.py                 # entry point: page routing only (Streamlit multipage)
racer/sandbox/                     # no Streamlit imports outside pages/ → unit-testable
    data.py        registry rows, snapshots, manifests, eval docs, thumbnails; cached by file mtime
    curves.py      learning curves from TensorBoard event files (tensorboard's EventAccumulator,
                   already a dependency); downsampled
    explain.py     glossary lookup, snapshot flattening, config diff
    diagrams.py    ray diagram and network diagram (matplotlib, already a dependency)
    builder.py     form state → env preset dict + train.py argv; validation via compat
    queue.py       read/append queue YAML, read state.json, start the runner
    launcher.py    start / list / stop demo.py processes
    glossary.yaml  explanations for physics, reward, train, episode and track keys
    pages/         overview.py, explorer.py, compare.py, build.py, launch.py, glossary.py
tests/test_sandbox.py              # data, diff, builder, queue, launcher (no browser needed)
```

---

## 4. Changes needed elsewhere

| Where | Change | Why |
|---|---|---|
| `requirements.txt` | Add `streamlit==<pinned>`, installed in `.venv` only | New dependency. Check first that it accepts the pinned `numpy 2.4.6` / `pandas 3.0.6` without changing them; if it doesn't, fall back to NiceGUI (same plan, different UI calls). |
| `.gitignore` | Add `outputs/sandbox/` (instance list, caches) and `.streamlit/` (local settings); neither is ignored today | Run hygiene |
| `scripts/queue_overnight.py` | Re-read the queue file after each run, so entries added from the sandbox while it runs are picked up | Today it reads the file once at start |
| `scripts/demo.py` | Optional `--title` for the window caption, so several windows can be told apart | Several game instances |
| `scripts/train.py` | Nothing: `--env`, `--action-layout`, `--set`, `--steps`, `--notes` already cover every option | (built in T55–T57) |

---

## 5. Build steps (each one testable)

| Step | Work | Done when |
|---|---|---|
| S0 | Install Streamlit into `.venv` (pin it, log it, check `.gitignore`); skeleton `scripts/sandbox.py` with empty pages | The app opens on localhost and lists the pages; `pip check` is clean |
| S1 | `data.py`, `explain.py`, `glossary.yaml`; a test that **every key of every physics, reward, train and episode preset has a glossary entry** | Tests pass; no unexplained parameter can appear |
| S2 | Model Explorer + `diagrams.py` + `curves.py` | Explorer shows R001, R003, R008 and a v2 scratch model correctly; the ray diagram matches the ray angles |
| S3 | Compare | Diff of R004 vs R008 shows only the ray list; metrics side by side |
| S4 | Launch + `launcher.py` + `demo.py --title` | Two game windows run at once from the sandbox; *Stop* closes one |
| S5 | Build & Queue + `builder.py` + `queue.py` + queue re-read | A 20k-step scratch entry built in the form trains through the queue and shows up in the Explorer |
| S6 | Overview page, polish, README and PROJECT_CONTEXT update | The owner walks through all 4 goals without the terminal |

**Estimate:** S0–S2 ≈ 1 day, S3–S4 ≈ half a day, S5 ≈ 1 day, S6 ≈ half a day.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Streamlit's dependencies conflict with the pinned numpy / pandas | Check before installing (S0); fall back to NiceGUI |
| Reading 400+ snapshots and TensorBoard files is slow | Cache by file mtime; load per run on demand; downsample curves |
| A queue started from the sandbox fights a running training for the CPU | Show what is running; *Start queue* is disabled while another queue or training runs |
| A new input preset makes the tree dirty, so real runs refuse to start (D12) | List the files to commit; never run git from the app |
| Explanations drift from the code | Descriptions live in the code registries; the glossary test fails on any unexplained key |

---

## 7. Decisions for the owner before S0

1. **Streamlit version:** latest that works with the pinned numpy and pandas (proposed), or a specific one.
2. **Start queue from the sandbox:** allowed (proposed, it starts the same `queue_overnight.py` in its own console), or show the command only.
3. **Who may use it:** local only (proposed). Sharing the sandbox on the LAN during the presentation would need `server.address=0.0.0.0`.
