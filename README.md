# RL Racing Line

A 2D top-down racing game written from scratch, where a car learns to drive a racing line with reinforcement learning (PPO, Stable-Baselines3).

- **What the car senses:** 5 distance rays to the road edge, its heading relative to the track, its speed and a few physics readings. It never sees the map.
- **What it outputs:** steering and gas/brake.
- **Why a racing line appears:** grip-limited physics with momentum and per-wheel surfaces. Corner too fast and the car slides, so it has to learn to brake, hit the apex and use the full track width.

The demo (pygame-ce) will load any trained model from a registry and race 10–20 of them as ghost cars. It will also have an evolution race, a draw-a-track mode and human vs AI.

> **Status:** project initialised (environment, folder skeleton, assets). No simulation code yet; Week 1 (milestone M1) is next.
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

## Layout

| Folder | Contents |
|---|---|
| `racer/` | `physics`, `track`, `env` (headless, no pygame); `agents`, `registry`, `render` (pygame-ce, demo only) |
| `configs/` | YAML presets: `physics/`, `env/`, `reward/`, `train/`, `demo/` (render and audio settings) |
| `scripts/` | train, evaluate, demo, record video, plots, physics sanity tests |
| `tracks/` | `showcase/`, `heldout/` (never used in training), `user/` (drawn in the editor) |
| `models/` | self-contained, immutable model folders |
| `registry/` | `runs.csv`, `models.csv` → `registry.xlsx` |
| `assets/` | sprites, textures, fonts, sounds; see [assets/LICENSES.md](assets/LICENSES.md) |
| `runs/`, `outputs/` | TensorBoard logs, videos, plots, frames |

## Credits

Engine sound: "Delorean DMC-12 V6 PRV engine" by skyernaklea ([freesound.org/s/623436](https://freesound.org/s/623436/)), CC BY 4.0, edited. Tyre sound by johnnydekk (CC0). Sprites by Kenney (CC0). Textures by ambientCG (CC0). Fonts: Rajdhani, Share Tech Mono, Orbitron (SIL OFL 1.1).
