# RL Racing Line

A 2D top-down racing game written from scratch, where a car learns to drive a racing line with reinforcement learning (PPO, Stable-Baselines3).

- **What the car senses:** 5 distance rays to the road edge, its heading relative to the track, its speed and a few physics readings. It never sees the map.
- **What it outputs:** steering and gas/brake.
- **Why a racing line appears:** grip-limited physics with momentum and per-wheel surfaces. Corner too fast and the car slides, so it has to learn to brake, hit the apex and use the full track width.

The demo (pygame-ce) will load any trained model from a registry and race 10–20 of them as ghost cars. It will also have an evolution race, a draw-a-track mode and human vs AI.

> **Status:** Week 1 (milestone M1) in progress: physics, tracks, sensors, the Gymnasium env, a debug driving window and PPO training are in place.
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

## Commands

Run everything with the project interpreter, `.venv\Scripts\python`.

| What | Command |
|---|---|
| Drive / watch (debug window) | `scripts\drive.py [--track technical] [--mode human\|baseline\|model] [--model models\<model_id>]` |
| Train PPO | `scripts\train.py --tag ppo_rays5_tg --tracks training_ground [--steps 2000000] [--set ppo.ent_coef=0.01] [--scratch]` |
| TensorBoard | `-m tensorboard.main --logdir runs` |
| Unit tests | `-m pytest -q` |
| Physics sanity tests | `scripts\physics_sanity.py` (run after every physics change) |
| Track previews + validation | `scripts\preview_tracks.py [showcase\|heldout\|<name>]` |
| Regenerate held-out tracks | `scripts\generate_tracks.py --heldout` (`--stats 200` for the acceptance rate) |
| Env speed + baseline laps | `scripts\bench_env.py` |

**Debug window keys:** arrows/WASD drive · `1` human · `2` baseline · `3` model · `R` rays · `T` trail · `F` overview · `+/−` or mouse wheel zoom · `[ ]` sim speed · `Space` pause · `Enter` reset at the start line · `Backspace` random start · `N` next track · `H` help · `Esc` quit.

## Layout

| Folder | Contents |
|---|---|
| `racer/` | `physics`, `track`, `env` (headless, no pygame); `agents`, `registry`, `render` (pygame-ce, demo only) |
| `configs/` | YAML presets: `physics/`, `env/`, `reward/`, `train/`, `track/` (geometry and generator), `demo/` (render and audio settings) |
| `scripts/` | train, drive (debug window), physics sanity, track previews / generation, env benchmark; later evaluate, demo, record video, plots |
| `tests/` | pytest unit tests |
| `tracks/` | `showcase/`, `heldout/` (never used in training), `user/` (drawn in the editor) |
| `models/` | self-contained, immutable model folders |
| `registry/` | `runs.csv`, `models.csv` → `registry.xlsx` |
| `assets/` | sprites, textures, fonts, sounds; see [assets/LICENSES.md](assets/LICENSES.md) |
| `runs/`, `outputs/` | TensorBoard logs; videos, plots, frames, `scratch/` (smoke runs, git-ignored) |

## Credits

Engine sound: "Delorean DMC-12 V6 PRV engine" by skyernaklea ([freesound.org/s/623436](https://freesound.org/s/623436/)), CC BY 4.0, edited. Tyre sound by johnnydekk (CC0). Sprites by Kenney (CC0). Textures by ambientCG (CC0). Fonts: Rajdhani, Share Tech Mono, Orbitron (SIL OFL 1.1).
