# RL Racing Line

A 2D top-down racing game written from scratch, where a car learns to drive a racing line with reinforcement learning (PPO, Stable-Baselines3).

- **What the car senses:** 5 distance rays to the road edge, its heading relative to the track, its speed and a few physics readings. It never sees the map.
- **What it outputs:** steering and gas/brake.
- **Why a racing line appears:** grip-limited physics with momentum and per-wheel surfaces. Corner too fast and the car slides, so it has to learn to brake, hit the apex and use the full track width.

The demo (pygame-ce) loads any trained model from a registry and races up to 20 of them as ghost cars (Model Arena). Week 3 adds the evolution race, draw-a-track mode and human vs AI.

> **Status:** Week 2 (milestone M2). Track sets with generated tracks, a curriculum (`--parent` fine-tuning), the evaluation protocol, the model registry (CSV → xlsx), an overnight experiment queue and the Model Arena are in place. Week 1's model `R001` laps Training Ground faster than the hand-coded baseline (42.7 s vs 46.6 s).
> **New (2026-10-11):** a map library of 9 hand-designed maps (7 for training, 2 test maps never trained on), evaluation protocol v2, and an upgradable model design: inputs are built from named features, the outputs follow a versioned action layout, and PPO and SAC are both supported. Any model the engine can run races in the arena. The AI Sandbox app is planned next: [docs/SANDBOX_PLAN.md](docs/SANDBOX_PLAN.md).
> Full design, plan and timeline: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md). Progress and change log: [log.md](log.md).

## Setup (Windows)

Requires Python 3.11 (64-bit). All dependencies stay in the project-local `.venv`.

```bat
cd "%USERPROFILE%\Desktop\ML Sem5 Project"
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

- Training runs on the CPU. The CPU build of torch is intentional.
- `pip install -e .` makes the `racer` package importable from `scripts/`.
- Versions in `requirements.txt` are pinned so saved models stay loadable.

## Quickstart

Run these from the project folder.

### Play the game

```bat
.venv\Scripts\python scripts\drive.py
.venv\Scripts\python scripts\drive.py --track technical
```

- **Driving:** arrow keys or WASD. Up accelerates, down brakes (there is no reverse).
- **Walls:** in human mode they are solid, and nothing resets your car.
- **Starting over:** `Enter` puts you back on the start line, `N` switches to the next track.
- **Watching the baseline:** press `2` to hand the car to the hand-coded baseline driver, then `1` to take it back.

### Train a model

```bat
.venv\Scripts\python scripts\train.py --tag ppo_rays5_tg --tracks training_ground --steps 2000000
```

- **Time:** about 10 minutes per million steps on the CPU, using 6 parallel environments.
- **Output:** a checkpoint every 100k steps, saved as `models\R###_<tag>@<steps>k\`. The run number goes up by itself (`R001`, `R002`, …).
- **Tracks:** `--tracks` names tracks directly; `--trackset` uses a preset from `configs\trackset\` (fixed tracks, whole folders, and freshly generated tracks). `--trackset maps7` trains on the 7 training maps plus generated tracks. Test maps (`tracks\test\`) and held-out tracks are never trained on.
- **Inputs, outputs, network, algorithm:** `--env` picks the inputs (e.g. `rays7_narrow`: 7 rays with shallow angles that see bends early), `--action-layout steer_throttle_brake` gives the car separate pedals, `--set policy.net_arch=[128,128]` changes the network size, and `--train sac_default` uses SAC instead of PPO. The run prints its input count, output count and parameter count.
- **Fine-tuning:** `--parent <model_id>` starts from a saved model (curriculum, e.g. `--trackset final_mix --parent R002_curric_wide@2700k --train ppo_finetune`). Its algorithm, inputs, outputs and network must match.
- **Evaluation:** afterwards every checkpoint is evaluated with protocol v2: the 7 training maps × 2 starts, the 2 test maps × 3 starts and 10 generated held-out tracks. Results are written to the registry, and the best checkpoint is marked. The final checkpoint isn't always the best. `unseen_completion` (test maps + held-out tracks) is the generalisation number that matters most. Results also go to `outputs\eval\<model_id>\`.
- **Commit first:** real runs refuse to start with uncommitted code or config changes, so each model records the exact code it came from (`--allow-dirty` overrides).
- **Live curves:** in a second terminal, run `.venv\Scripts\python -m tensorboard.main --logdir runs` and open http://localhost:6006.
- **Smoke test:** add `--scratch` to write to `outputs\scratch\` (git-ignored, with its own registry) instead of `models\`.

### Race models (Model Arena)

```bat
.venv\Scripts\python scripts\demo.py
.venv\Scripts\python scripts\demo.py --track technical --best --baseline
.venv\Scripts\python scripts\demo.py --track fast_flow --run R002 --every 3
```

- **Picker:** pick a track on the left and up to 20 models on the right (click rows; click a column header to sort; type to filter; `Tab` best checkpoints only; `Shift+B` adds the PID baseline; `Shift+P` fair physics; `+/−` laps), then `Enter`.
- **Race:** ghost cars start from a grid after a countdown. A crashed car restarts where it crashed after 1 s, and its crashes are counted.
- **Keys:** `Tab` (or click a leaderboard row) follows the next car · `F` overview · `R` rays · `T` all trails · `L` labels · `Space` pause · `+/−` sim speed · mouse wheel zoom · `Esc` back to the picker.
- **Physics:** each car uses the physics it was trained with; `--fair` (or `Shift+P`) puts every car on the default physics.
- **Any model:** each car uses its own inputs and outputs, so models with different sensors, action layouts or algorithms race together. A model the engine can't run yet is shown dimmed in the picker with the reason.

### Run experiments overnight

```bat
.venv\Scripts\python scripts\queue_overnight.py configs\queue\night1.yaml --dry-run
.venv\Scripts\python scripts\queue_overnight.py configs\queue\night1.yaml
```

- Start it yourself from a normal terminal; it needs no Claude session. It runs the queue's training runs one after another, logs each to `outputs\queue_logs\`, keeps going if one fails, and resumes where it left off if you start it again.
- It keeps Windows awake while it runs. Keep the laptop plugged in, with the lid open.

### Replay a trained model

```bat
.venv\Scripts\python scripts\drive.py --model "models\R001_ppo_rays5_tg@1200k"
.venv\Scripts\python scripts\drive.py --model "models\R001_ppo_rays5_tg@1200k" --track fast_flow
```

- **The car's config:** the model drives with the physics, sensors and rules saved in its own folder.
- **Episode ends:** an episode ends exactly as in training (wall, off-road, stuck, wrong way), then restarts from a random point.
- **Switching drivers:** press `3` for the model, `2` for the baseline, and `1` to drive yourself.
- **Visual aids:** `R` shows the model's rays, `T` its speed-coloured trail.
- **Model id or folder:** `--model` takes a model id (looked up in the registry) or a folder path. Keep quotes around paths, because the `@` can confuse some shells.

### Registry

- `registry\runs.csv` and `registry\models.csv` are the source of truth: one row per run and per checkpoint, with evaluation metrics.
- `registry\registry.xlsx` is regenerated from them (Runs, Models, a leaderboard per map, Ablations). Never edit it by hand.
- Evaluation files are also added to each model folder and copied to `outputs\eval\`: protocol v1 writes `eval.json` and `thumb_<track>.png`, and v2 writes `eval_v2.json` and `thumb_v2_<track>.png`. A registry row shows the newest protocol a model was evaluated with (`eval_protocol`), so compare models evaluated with the same protocol. Re-evaluate older models with `scripts\evaluate.py --run R003`.
- `runs.csv` also records each run's input count, action layout and parameter count. `scripts\registry.py describe` fills these in for runs from before they existed.

## Commands

Run everything with the project interpreter, `.venv\Scripts\python`.

| What | Command |
|---|---|
| Drive / watch (debug window) | `scripts\drive.py [--track technical] [--mode human\|baseline\|model] [--model models\<model_id>] [--sim-speed 4] [--trace out.csv]` |
| Train | `scripts\train.py --tag … (--tracks … or --trackset maps7) [--parent <model_id>] [--train ppo_default\|ppo_finetune\|sac_default] [--physics …] [--env rays5\|rays7_narrow\|…] [--action-layout steer_gas\|steer_throttle_brake] [--reward …] [--steps N] [--seed N] [--set ppo.ent_coef=0.01] [--scratch] [--no-eval] [--protocol v2]` |
| Evaluate (protocol v2) | `scripts\evaluate.py --model <id> …` · `--run R002` (all checkpoints, marks the best) · `--baseline` · `[--protocol v1]` |
| Registry | `scripts\registry.py list [--run R002] [--best]` · `backfill` · `export` (regenerate the xlsx) · `describe` (inputs, outputs, parameters per run; models the engine can't run) |
| Model Arena | `scripts\demo.py [--track …] [--models … or --run R002 [--every N] or --best] [--baseline] [--laps 3] [--fair]` |
| Overnight queue | `scripts\queue_overnight.py configs\queue\<queue>.yaml [--dry-run] [--retry-failed]` |
| TensorBoard | `-m tensorboard.main --logdir runs` |
| Unit tests | `-m pytest -q` |
| Physics sanity tests | `scripts\physics_sanity.py` (run after every physics change) |
| Track previews + validation | `scripts\preview_tracks.py [showcase\|train\|test\|heldout\|<name>]` |
| Rebuild the hand-designed maps | `scripts\make_maps.py [--check] [names…]` (outlines are defined in the script) |
| Regenerate held-out tracks | `scripts\generate_tracks.py --heldout` (`--stats 200` for the acceptance rate) |
| Env speed + baseline laps | `scripts\bench_env.py [--trackset final_mix]` |
| Training speed (CPU vs GPU, parallel runs) | `scripts\bench_train.py --steps 200000 [--n-envs 6] [--batch-size 256] [--runs 2] [--device cuda]` (GPU needs a CUDA build of torch; measured slower than the CPU) |

**Debug window keys:** arrows/WASD drive · `1` human · `2` baseline · `3` model · `R` rays · `T` trail · `F` overview · `+/−` or mouse wheel zoom · `[ ]` sim speed · `Space` pause · `Enter` reset at the start line · `Backspace` random start · `N` next track · `K` ABS on/off · `H` help · `Esc` quit.

In human mode (`1`) the walls are solid and nothing resets the car; baseline and model modes end episodes exactly as in training. Keyboard braking uses an ABS assist by default (a key is all-or-nothing, so full brake would lock the wheels); it only affects human driving.

## Layout

| Folder | Contents |
|---|---|
| `racer/` | `physics`, `track`, `env` (headless, no pygame); `agents`, `registry`, `render` (pygame-ce, demo only) |
| `configs/` | YAML presets: `physics/`, `env/`, `reward/`, `train/`, `track/` (geometry and generator), `trackset/` (what training samples), `eval/` (evaluation protocol), `queue/` (overnight experiment lists), `demo/` (render, arena and audio settings) |
| `scripts/` | train, evaluate, registry, demo (Model Arena), queue_overnight, drive (debug window), physics sanity, track previews / generation, env benchmark; later record video, plots |
| `tests/` | pytest unit tests |
| `tracks/` | `showcase/` and `train/` (the 7 training maps), `test/` (2 test maps) and `heldout/` (generated), both never used in training, `user/` (drawn in the editor). Any other folder works too: a track JSON added at any time is found by name |
| `docs/` | `SANDBOX_PLAN.md`: implementation plan of the AI Sandbox app |
| `models/` | self-contained, immutable model folders; **local only** (git-ignored), back them up by copying the folder |
| `registry/` | `runs.csv`, `models.csv` → `registry.xlsx` |
| `assets/` | sprites, textures, fonts, sounds; see [assets/LICENSES.md](assets/LICENSES.md) |
| `runs/`, `outputs/` | TensorBoard logs; `eval/` (copies of every model's evaluation), videos, plots, frames, `scratch/` (smoke runs) and `queue_logs/` (both git-ignored) |

## Credits

Engine sound: "Delorean DMC-12 V6 PRV engine" by skyernaklea ([freesound.org/s/623436](https://freesound.org/s/623436/)), CC BY 4.0, edited. Tyre sound by johnnydekk (CC0). Sprites by Kenney (CC0). Textures by ambientCG (CC0). Fonts: Rajdhani, Share Tech Mono, Orbitron (SIL OFL 1.1).
