# Demo completion and analysis workbench: plan (D28)

> Owner decisions of 2026-10-24 (task T123). Finish all demo and analysis work first, then experiment freely.
> Built in the worktree `ML Sem5 Project - dev` (branch `feat/demo-complete`) while the sandbox queue runs in `main`.

## 1. Scope

| In | Out (owner) |
|---|---|
| Mode menu; Evolution Race; Human vs AI; swarm view | Game-window drawing editor (T112–T114; the sandbox page *Draw a Track* stays) |
| Brain panel (live network graph) | Video recorder |
| Audio (engine, gearbox, tyres) | Slides and highlight video |
| Real polish: car and track visuals, effects, menus and HUD, camera and feel | |
| Sandbox: **Workbench** (live plots), drive-it-yourself launch, delete rows in the rays table | |
| Default plots and the racing line / telemetry view | |

## 2. Decisions (owner answers)

- **Audio** works in every window that shows a race (arena, Evolution Race, Human vs AI), toggled with `S` and an on-screen button. **Default: on when watching a race; off for training, the sandbox and analysis.** `demo.py --sound/--no-sound`; the sandbox's Launch has a Sound checkbox. Design: PROJECT_CONTEXT §10.4.
- **Brain panel** = a live network graph: input nodes, the hidden layers as groups of neurons, output nodes, lit by the current activations, with the input bars and steering wheel / pedal outputs. Works for every model through its manifest labels.
- **Workbench** = a sandbox page that plots live from the same Python runtime (no export step): registry metrics, learning curves, and simulations of any model on any track.
- **Drive in the sandbox**: Streamlit cannot take 60 Hz key presses, so *Drive it yourself* on the Launch page opens a game window (as for races) with the chosen physics preset, track and opponents.
- **Rays table**: a Delete column with a *Remove ticked rays* button (Streamlit's own row delete exists but is hidden).

## 3. Design

- Logic in `racer/` (pure, testable); pygame only in `racer/render/`; Streamlit only in `racer/sandbox/pages/`.
- New modules: `racer/render/{app,menu,effects,minimap,brain_panel,audio,gearbox,evorace,human}.py`, `racer/agents/telemetry.py` (run a model on a track, return per-step arrays: position, speed, steer, gas, lateral g, surface, progress), `racer/sandbox/plots.py` (data frames and Altair charts, no Streamlit), `racer/sandbox/pages/workbench.py`, `scripts/make_plots.py` (thin CLI, PNGs from the same functions).
- Charts: Altair (ships with Streamlit, interactive, no new dependency) for the Workbench; matplotlib (pinned) for PNG files and the speed-coloured racing line.
- Every new preset, setting and key gets a glossary entry (a test fails otherwise). Every feature gets its sandbox part (D27).

## 4. Tasks (IDs in `log.md`)

| ID | Step | Done when |
|---|---|---|
| T124 | `demo.py` shell: mode menu (arena, Evolution Race, Human vs AI, swarm), shared window and state | headless: menu → each mode starts and returns |
| T125 | Audio: `gearbox.py`, `audio.py`, `audio.yaml` use, toggle, defaults, `--sound`, Launch checkbox | tests for gearbox ratios and the RPM→offset map (dummy audio driver); toggle test |
| T126 | Evolution Race: pick a run, pick checkpoints (every N, evenly spaced, hand-picked, best-validated), labels by step; Launch option | scripted race of 4 checkpoints |
| T127 | Human vs AI: keyboard and gamepad, ABS assist, physics choice; sandbox *Drive it yourself* | key events move a car (headless); Launch builds the command |
| T128 | Brain panel: activation capture (SB3 MLP), graph, bars, outputs; key `B` | right labels and finite activations for every compatible model |
| T129 | Effects: trails, skid marks, smoke and dust, brake lights, start lights | screenshot check; cost measured |
| T130 | Visuals: textures, kerbs, start line, barrier, scenery props, shadows, tinted cars | screenshot check on 3 tracks |
| T131 | Menus and HUD: leaderboard, results screen, minimap, key list (`H`), camera (speed zoom, overview, cycle cars) | on-screen key list test |
| T132 | Swarm view | many cars from one policy at 60 FPS |
| T133 | Performance: 20 cars with all effects and audio at 60 FPS | `bench` script and a test threshold |
| T134 | Rays table: Delete column and button | page test removes chosen rows |
| T135 | `telemetry.py` + `plots.py`: racing line, speed trace, g-g diagram, braking zones; registry plots (curves, ablation bars, E10) | unit tests on small data |
| T136 | Workbench page + default plots on Overview; Compare links to it | headless page tests |
| T137 | `scripts/make_plots.py` (PNG report set) | script writes the set from the registry |
| T138 | Docs (PROJECT_CONTEXT §10, §12, README, glossary, `assets/LICENSES.md`), full suite, merge to `main`, remove the worktree | suite passes |

Order: T124 → T126 → T125 → T127 → T128 → T129–T131 → T132 → T133 → T134 → T135–T137 → T138.

## 5. Keeping the merge easy

- The queue runs in `main`. This branch never edits: `registry/`, `outputs/eval/`, `models/`, `configs/queue/`, `configs/env/sb_*`, `configs/trackset/sb_*`, or what an existing preset means (extend, don't change).
- `log.md`: add tasks only at the bottom of the task list and Change Log entries only at the top; "Current Status" is rewritten at merge time. `PROJECT_CONTEXT.md`: append rows (§16), edit only sections of this plan.
- Merge `main` into this branch at least once per step (`git merge main`); resolve conflicts at once, while they are small.
- Run Python here with the main `.venv` and `PYTHONPATH` set to the worktree (checked: `racer` imports from the worktree). Heavy test runs wait for a quiet moment, so the queue keeps its CPU.
- New dependencies: none planned. If one is needed, install it into the project `.venv` only, pin it in `requirements.txt` and say so in the log.
