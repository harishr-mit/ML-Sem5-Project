# PROJECT CONTEXT: RL Racing Line

> **What this file is:** the single source of truth for this project: goals, design decisions, architecture, physics, RL setup, demo features, timeline and setup.
> Humans read it to stay oriented, and Claude Code reads it (via `CLAUDE.md`) before writing any code.
> When a decision changes, update this file first, then the code. Log the change in §16.

- **Course:** Machine Learning, Semester 5 (solo project)
- **Started:** 2026-10-10
- **Target:** 3–4 weeks, so a feature-complete demo by about 2026-11-07
- **Budget:** zero. Only free and open-source tools. Assets are CC0 or OFL, plus one CC BY 4.0 engine sound that **must be credited** (see `assets/LICENSES.md`).
- **Tooling:** VS Code with the Claude Code extension for implementation and debugging; the Claude desktop app for brainstorming and writing prompts.

---

## 1. Summary

A 2D top-down racing game written from scratch, with a car that learns to drive with **reinforcement learning (PPO)**.
- **What the car senses:** 5 distance rays to the edge of the road, its heading relative to the track, its speed, and a few physics readings. It does not see the map.
- **What it outputs:** steering in [−1, 1] and gas in [−1, 1], where a negative value means brake.
- **What makes a racing line appear:** the physics has mass, momentum, gravity and a grip limit that depends on the surface. Taking a corner too fast makes the car slide, so the agent has to learn to brake before corners, take the apex, use the full track width and accelerate out.

**Main goal: the demo has to look impressive.** The plan is:
- a live game in which **any trained version** can be loaded from a spreadsheet-style registry
- **10–20 chosen models racing together** at the same time, as ghost cars that pass through each other
- an "evolution race" between checkpoints of one run
- draw-a-track mode, so the AI drives a track it has never seen
- human vs AI
- solid ML analysis (learning curves and ablations) for the report and slides

---

## 2. Goals and constraints

| # | Goal | Requirement |
|---|------|-------------|
| G1 | Show RL learning | Learning curves; the same car shown early, mid and late in training |
| G2 | Show a real racing line | Speed-coloured trajectory compared with the centreline; braking zones visible |
| G3 | Live demo | A pygame app with several modes, steady at 60 FPS with 20 cars |
| G4 | Record of every version | A registry (CSV as the source of truth, exported to `.xlsx`) listing every run and checkpoint with its configs and evaluation metrics |
| G5 | Load any version | The demo picks models from the registry. Each model carries its own sensor, physics and normaliser configs. |
| G6 | Run 10–20 models at once | No car-to-car collisions, rays ignore other cars, live leaderboard |
| G7 | ML depth | Ablations (grip, reward, rays, number of tracks, algorithm), generalisation to held-out tracks, a baseline controller |

**Constraints:** solo, 3–4 weeks, zero budget. Any library is allowed.

**Hardware:**
- i5-10200H (4 cores, 8 threads)
- 16 GB DDR4-2666
- GTX 1650 4 GB
- Windows

The GPU is not needed for training. PPO with a small MLP is faster on the CPU (see §8).

---

## 3. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Game | **Write our own**, don't adapt an existing game | Existing games have no `reset()` or `step()`, tie the game loop to rendering and run at real-time speed. Our own engine runs headless and fast. |
| RL interface | **Gymnasium** `Env` | Standard; works directly with Stable-Baselines3 |
| Algorithm | **PPO** (Stable-Baselines3); SAC as a comparison | PPO is robust, simple and works well with parallel environments |
| Observations | Rays and the car's own readings only, **no map knowledge** | Lets one model generalise to unseen tracks, which makes the draw-a-track demo possible |
| Rays measure | Distance to the **edge of the road** (not to the outer barrier) | The road is what matters for track limits and the racing line |
| Physics | Arcade-sim: momentum, mass, gravity, friction circle, per-wheel surface grip | Creates the racing line; cheap to compute |
| Surfaces | `Surface` base class (= grass physics) with `Road` and `Kerb` subclasses that override values | Matches the user's idea. Anything that isn't road behaves as grass, whatever it looks like. |
| Rendering | **pygame-ce**, kept completely separate from the simulation | Training never renders; the demo renders everything |
| Configs | YAML; a full snapshot saved with every model | Any old model can always be loaded and run correctly |
| Registry | `runs.csv` and `models.csv`, exported to `registry.xlsx` | Easy to append; Excel looks good on a slide |
| Visual style | Code-generated effects, CC0 textures and sprites, OFL fonts | Free, legal, and looks good |
| Audio | Engine clips seeked by RPM + a **cosmetic gearbox**; tyre screech driven by slip (§10.4) | Sounds like a real car accelerating through the gears; the physics and RL are untouched |

---

## 4. Architecture

Two halves that never mix:

```
┌────────────────────────── SIMULATION (headless, fast) ──────────────────────────┐
│ physics/  (car, surfaces)   track/  (spline, generator, progress)                │
│ env/      (Gymnasium env: sensors, reward, termination)                          │
└──────────────────────────────────────────────────────────────────────────────────┘
        ▲ used by                                   ▲ used by (same code)
┌───────┴────────── TRAINING ─────────┐   ┌─────────┴────────── DEMO APP ──────────┐
│ scripts/train.py  (SB3 PPO, VecEnv) │   │ scripts/demo.py (pygame-ce)            │
│ scripts/evaluate.py                 │   │ render/ (world, effects, HUD, camera)  │
│ registry/ (CSV → XLSX)              │   │ modes: arena, evolution, editor,       │
└─────────────────────────────────────┘   │        human-vs-AI, analysis           │
                                          └────────────────────────────────────────┘
```

**Rule:** nothing in `physics/`, `track/` or `env/` may import pygame. The demo **wraps** simulation objects and draws them.

### Repository layout

```
ML Sem5 Project/
├── CLAUDE.md                 # short; imports this file + coding conventions
├── PROJECT_CONTEXT.md        # this file
├── README.md
├── log.md                    # task list + change log shared by all agents
├── requirements.txt          # pinned versions (torch comes from the CPU index)
├── pyproject.toml            # makes racer/ importable: pip install -e .
├── configs/
│   ├── physics/  default.yaml, no_grip_limit.yaml, downforce.yaml
│   ├── env/      rays5.yaml, rays9.yaml, rays5_curvature.yaml
│   ├── reward/   progress.yaml, speed_naive.yaml (for the reward-hacking demo)
│   ├── train/    ppo_default.yaml, sac_default.yaml, baseline.yaml (hand-coded driver)
│   ├── track/    default.yaml (kerb threshold, field cell size, generator ranges, clearances)
│   ├── trackset/ tg.yaml, gen_wide.yaml, final_mix.yaml … (which tracks training samples, §8)
│   ├── eval/     v1.yaml (evaluation protocol, §8)
│   ├── queue/    *.yaml (experiment lists for queue_overnight.py)
│   └── demo/     render.yaml (texture scales, colours; `debug:` window), audio.yaml (volumes, gearbox)
├── racer/config.py           # YAML preset loader (strict keys, deep-merged overrides)
├── racer/
│   ├── physics/  car.py, surfaces.py
│   ├── track/    track.py, spline.py, field.py, generator.py, progress.py, io.py
│   ├── env/      racing_env.py, sensors.py, reward.py, obs_spec.py, trackset.py,
│   │             race.py (multi-car race for the demo, pygame-free)
│   ├── agents/   loader.py (registry → policy + normaliser + configs), baseline.py,
│   │             evaluation.py (eval protocol, metrics, thumbnails, eval files)
│   ├── registry/ registry.py, export_xlsx.py, record.py (evaluate + record)
│   └── render/   debug_view.py (Week 1 debug window), world.py, camera.py, cars.py (tinted sprites),
│                 effects.py, hud.py, brain_panel.py,
│                 minimap.py, ui.py (menus/model picker), editor.py,
│                 audio.py (engine/tyre sound controller), gearbox.py (cosmetic RPM)
├── scripts/
│   ├── train.py  evaluate.py  demo.py  record_video.py  make_plots.py  registry.py
│   ├── queue_overnight.py    # runs a list of experiments one after another
│   ├── physics_sanity.py     # skidpad / braking / coast-down tests
│   ├── drive.py              # debug window: keyboard / baseline / model driving
│   ├── preview_tracks.py  generate_tracks.py  bench_env.py
├── tests/                    # pytest unit tests (spline, field, progress, physics, env)
├── tracks/
│   ├── showcase/   *.json    # 3 hand-designed demo tracks
│   ├── heldout/    *.json    # fixed generated tracks, NEVER used in training
│   └── user/       *.json    # tracks saved from draw-a-track mode
├── assets/
│   ├── LICENSES.md           # manifest, credits, usage notes (READ before using assets)
│   ├── sprites/cars/  sprites/props/  textures/ground/  fonts/  sounds/  licenses/
│   └── _source/              # original downloads (git-ignored), for re-cutting
├── models/<model_id>/        # self-contained model folders (see §9); local only, git-ignored
├── registry/  runs.csv  models.csv  registry.xlsx
├── runs/                     # TensorBoard logs
└── outputs/   videos/  plots/  frames/  eval/ (copies of every model's eval files)
               scratch/ (smoke runs, git-ignored)  queue_logs/ (git-ignored)
```

---

## 5. Physics model

Units are SI: metres, seconds, kilograms, newtons. The renderer converts with `PIXELS_PER_METRE` (about 6–8 at the default zoom).

Physics runs at a fixed **60 Hz** (`dt = 1/60`). The agent decides at **20 Hz**: each action is held for 3 physics substeps.

### 5.1 Surfaces (inheritance, as the user suggested)

```python
class Surface:            # base = GRASS. Anything that isn't road falls back to this.
    name = "grass"
    mu_static   = 0.45    # tyre-surface friction coefficient
    mu_kinetic  = 0.35    # when sliding (lower → losing grip is punished)
    rolling_res = 0.12    # Crr, high → grass slows you down a lot
    visual      = "grass" # visual variants (sand, gravel, dirt) = same physics

class Road(Surface):      # overrides
    name = "road"; mu_static = 1.10; mu_kinetic = 0.90; rolling_res = 0.015; visual = "asphalt"

class Kerb(Road):         # thin band at road edges on corners
    name = "kerb"; mu_static = 0.95; mu_kinetic = 0.80; rolling_res = 0.025; visual = "kerb"
```

- **No pixel lookup.** The surface is worked out from geometry: the distance from the centreline compared with the road half-width and the kerb band.
- **Kerbs sit inside the road edge** (band `[w/2 − kerb_m, w/2]`, only where |curvature| is above a threshold). The road edge stays at ±w/2 for rays and track limits, and kerbs count as on-track.
- **Sampled per wheel.** Each of the 4 wheel positions is checked separately, so a car with two wheels on the grass loses about half its extra grip. This looks realistic and costs very little.
- The barrier (an outer wall at the end of the run-off) is a **collision object, not a surface**.

### 5.2 Car state and parameters (`configs/physics/default.yaml`)

**State:** `pos (x, y)`, `heading θ`, `vel (vx, vy)` in world coordinates (this is the momentum), `steer_angle`, `sliding` flag.

| Parameter | Default | Notes |
|---|---|---|
| mass `m` | 1000 kg | |
| gravity `g` | 9.81 m/s² | constant |
| wheelbase `L` | 2.6 m | car is 4.5 m × 1.9 m |
| max steer | 30° at low speed, reducing to about 8° at top speed | speed-dependent limit keeps it stable |
| steer rate | 120°/s | rate limit, so steering is smooth (better behaviour, better looks) |
| engine force max | 6000 N | |
| engine power max | 150 kW | power limit: `F_drive ≤ P/v` gives a realistic top speed |
| brake force max | 12000 N | the grip limit applies on top of this |
| drag `CdA` | 0.70 m² (`0.5·ρ·CdA` with ρ = 1.225) | top speed ≈ 65–70 m/s (about 240 km/h) |
| lateral damping `c_lat` | 0.9 | fraction of sideslip removed per step when grip allows |
| downforce `k_down` | 0 (on in the `downforce` preset: about 1.5 N·s²/m²) | adds normal force ∝ v² |

### 5.3 Forces: how mass, gravity and grip interact

- **Normal force:** `N = m·g + k_down·v²`
- **Grip limit (friction circle):** `|F_tyre| ≤ μ_eff · N`, where `μ_eff` is the average over the 4 wheels' surfaces. Use `mu_kinetic` instead of `mu_static` if the car was sliding on the previous step.
- **Honest note for the report:** without downforce, both the grip and the inertia scale with *m*, so the **maximum cornering acceleration is μ·g whatever the mass**. Mass still matters for acceleration (fixed engine force), braking, momentum when sliding, and rolling resistance. Downforce makes high-speed corners grippier than low-speed ones, which is an interesting ablation.

### 5.4 Step pseudo-code (a reference sketch; adjust as needed)

```python
fwd  = (cos θ, sin θ);  left = (−sin θ, cos θ)
v_long = vel·fwd;  v_lat = vel·left;  speed = |vel|

# controls
steer_target = action_steer * max_steer(speed)
steer_angle  = move_towards(steer_angle, steer_target, steer_rate*dt)
throttle, brake = max(a_gas,0), max(−a_gas,0)        # negative gas = brake, never reverse

# grip available
mu   = mean(surface_at(w).mu_kinetic if sliding else .mu_static for w in wheels)
F_max = mu * (m*g + k_down*v_long**2)

# tyre force DEMAND (car frame)
F_drive = throttle * min(F_engine_max, P_max / max(v_long, 1.0))
F_brake = brake * F_brake_max if v_long > 0.1 else 0
F_long  = F_drive − F_brake
F_lat   = −m * v_lat / dt * c_lat                     # force needed to remove sideslip

# friction circle
demand = hypot(F_long, F_lat)
scale  = min(1, F_max / demand) if demand > 0 else 1
sliding = scale < 0.999                                # → skid marks, smoke, kinetic μ next step
F_long *= scale;  F_lat *= scale

# resistances (not limited by grip)
F_drag = −0.5*ρ*CdA * speed * vel
F_roll = −rolling_res_eff * m * g * unit(vel)          # surface-dependent

acc  = (F_long*fwd + F_lat*left + F_drag + F_roll) / m
vel += acc*dt
if vel·fwd < 0 and throttle == 0: vel -= (vel·fwd)*fwd # braking can't send the car backwards
yaw_rate = v_long * tan(steer_angle) / L * scale        # saturated grip → understeer
θ   += yaw_rate*dt
pos += vel*dt
```

This produces momentum (velocity persists), understeer when entering a corner too fast, drifting when sliding, slow driving on grass, and braking limited by grip.

### 5.5 Physics sanity tests (`scripts/physics_sanity.py`)
1. **Skidpad:** a constant-radius circle at increasing speed. The car loses the line at `a_lat ≈ μ·g` (about 10.8 m/s² on road).
2. **Braking:** with **threshold braking** (≈ 90 % brake, just under the grip limit), the stopping distance from 30 m/s ≈ v²/(2μg) ≈ 42 m. Full brake (12000 N) exceeds static grip (μ·m·g ≈ 10 800 N), so the wheels lock, kinetic μ applies and the car stops in ≈ 49 m. Learning threshold braking is part of the racing-line skill.
3. **Coast-down:** speed decays smoothly to 0, with no oscillation and no reversing.
4. **Grass:** top speed and cornering grip are much lower on grass.
5. **Stability:** no NaNs or jitter at near-zero speed with full steering lock.

---

## 6. Tracks

- **Representation:** a closed **Catmull-Rom spline** through control points, resampled every **1 m of arc length**. For each sample, precompute: position, tangent, normal, curvature and cumulative distance `s`.
- **Track JSON:** `{name, control_points[[x,y]...], width_m (≈12), runoff_m (≈15), kerb_m (≈1.2), direction (cw/ccw)}`
- **Geometry derived from the spline:** road edges at ±width/2, kerbs (inside the road edge, only where curvature is above a threshold, drawn red and white), grass run-off, and an outer barrier at ±(width/2 + runoff).
- **Track field (`field.py`):** a grid of 0.5 m cells holding each cell's distance to the centreline and its nearest centreline sample, built once per track with numba. Surfaces, rays, progress and barrier checks all read it, and the debug renderer draws the world from it, so what you see is what the physics uses.
- **Barrier = field territory:** the car crashes if it is beyond `width/2 + runoff`, **or** if the field says it is now nearest to a different part of the track (a jump of more than a set arc distance from its tracked index). Where two sections are closer than 2 × (width/2 + runoff), e.g. the legs of a hairpin, this puts a wall halfway between them, as on real circuits. There are no thin walls, so nothing can tunnel through, and the agent can't earn progress by cutting across the grass.
- **Progress:** take the car's nearest centreline sample from the field and refine it on the segment. Progress `Δs` handles the lap wrap-around correctly, and driving backwards gives negative `Δs`.
- **Generator:**
  1. Take 8–16 points around a circle with random radius and angle jitter.
  2. Build the spline.
  3. **Reject** tracks that intersect themselves (or whose roads come closer than `width + clearance`), have a corner radius below the minimum (about 15 m), or are shorter or longer than allowed.
  4. Randomly choose driving direction and mirroring.
- **Showcase tracks (3, hand-designed):**
  - "Training Ground": wide and simple
  - "Technical": hairpins and a chicane
  - "Fast Flow": high-speed sweepers where braking matters
- **Held-out set:** about 10 generated tracks with fixed seeds, saved in `tracks/heldout/`. **Never** used in training, only to test generalisation.
- **Open track library (owner):** tracks can be added at any time in the project. Any valid track JSON dropped into a `tracks/<folder>/` is picked up by name everywhere (training track sets, evaluation, demo). Training may mix any non-held-out tracks, and new hand-made or generated tracks may be added for variety. Adding a track never changes an existing model (each model's snapshot lists what it trained on).
- **Final showcase (owner):** faculty members draw a track, it is converted into a track JSON (Draw-a-Track, §10.1), and the trained models drive it live. This is the strongest generalisation demo, so held-out performance is the metric that matters most.
- **Editor (draw-a-track):**
  1. Click points, or drag to sketch freehand. Freehand input is smoothed and simplified down to control points.
  2. Run the same validation as the generator.
  3. Press Enter and the AI drives it.
  4. Optionally save it to `tracks/user/`.

---

## 7. Environment (Gymnasium)

### 7.1 Observation (default spec `rays5`, all values scaled to about [−1, 1] or [0, 1])

| Feature | Dims | Notes |
|---|---|---|
| Ray distances to the road edge | 5 | angles −90°, −45°, 0°, +45°, +90° relative to the car. Range: **200 m** for the forward ray (braking from 65 m/s to 17 m/s needs about 180 m), 60 m for the side rays. Value = distance / range. |
| Heading error vs the track tangent | 2 | sin and cos of the angle (avoids the wrap-around jump at ±180°) |
| Forward speed | 1 | `v_long / v_max` |
| Lateral velocity | 1 | sideslip, so the agent can feel a slide |
| Yaw rate | 1 | |
| Fraction of wheels on road | 1 | |
| Previous action | 2 | helps smooth control |

That's **13 dimensions** in total.

**Frozen details of `obs_spec` v1** (they live in `configs/env/rays5.yaml`, not in the physics config, so later physics changes can't change what v1 means):
- Rays start at the car's centre. A ray's distance is the first road-edge crossing along it: leaving the road when the car is on-road, entering it when off-road. Value = distance / range, clipped to [0, 1].
- Scales: `v_long / 70`, `v_lat / 10`, `yaw_rate / 3 rad/s`, each clipped to [−1, 1]. Kerbs count as road for the on-road fraction.
- World axes: x right, y up; angles are counter-clockwise, so a ray at −90° points to the car's right.
- Order: rays (in the listed order, −90° first), sin/cos heading error, `v_long`, `v_lat`, yaw rate, on-road fraction, previous steer, previous gas.
- **Ablation specs:** `rays9`, `rays5_short` (60 m forward), and `rays5_curvature` (adds the track curvature 25, 50 and 100 m ahead, i.e. map knowledge, to compare).
- **Implementation:** ray casting uses a precomputed **signed-distance grid** of the road (0.5 m cells) with sphere tracing, or vectorised ray–segment intersection with numba. Target **at least 3k env steps per second per core**.

### 7.2 Action
`Box([-1,-1], [1,1])` → `[steer, gas]`. A negative gas value means brake. There is no reverse gear.

### 7.3 Reward (`configs/reward/progress.yaml`)
```
r = 1.0  * Δs_metres                      # progress along the centreline (main signal)
  − 0.05 * dt_agent                       # time pressure (each 1/20 s step)
  − 0.1  * |Δsteer|                       # smooth steering (looks better, drives better)
  − 0.5  * offroad_fraction * dt_agent    # discourage cutting across the grass
  + 5.0  on lap completion
  − 20   on crash / termination
```
The `speed_naive.yaml` variant (reward = speed) exists **only** to show reward hacking (circling or weaving) in the presentation.

### 7.4 Episode rules
- **Start:** a random point along the track, with small random offsets in lateral position and heading, and a random starting speed of 0–15 m/s. This stops the agent learning only the first corner.
- **Episode ends (with a penalty) if:**
  - the car touches the barrier
  - all 4 wheels are off the road for more than 1.0 s
  - the car makes less than 2 m of progress in 3 s
  - the car points the wrong way (heading error over 110°) for more than 1 s
- **Truncate** after 2 laps or 4000 agent steps.
- Random track per episode during multi-track training (curriculum, see §8).

---

## 8. Training

- **Library:** Stable-Baselines3 PPO, `MlpPolicy`, `net_arch=[256,256]`, tanh activations.
- **Parallelism:** `SubprocVecEnv` with **6 environments** (4 cores / 8 threads; leaves headroom). `VecNormalize` for observations and rewards, with its statistics saved alongside each model.
- **Starting hyperparameters:** `n_steps=1024` per env, `batch_size=256`, `n_epochs=10`, `gamma=0.995`, `gae_lambda=0.95`, `lr=3e-4` (linear decay), `clip=0.2`, `ent_coef=0.002`.
- **Device:** **CPU**, with `torch.set_num_threads(1)` in each worker. The GTX 1650 is not useful for an MLP this small.
- **Windows:** SubprocVecEnv needs the `if __name__ == "__main__":` guard in every script.
- **Overnight queue (D18, owner):** `queue_overnight.py` is started by the owner from a normal terminal and runs **without a Claude session** (no token use). It runs the queue file's runs one after another, logs each to `outputs/queue_logs/`, continues after a failure, resumes finished queues, and keeps Windows awake only while it runs (no power-plan changes).
- **Expected budget:** 2–5M steps per serious run. **Measured (2026-10-10):** ≈ 1.7k steps/s with 6 envs, so ≈ 10 min per million steps. The PPO update is ~60 % of the time (compute-bound at batch 256; more torch threads don't help); one env alone runs at 13k steps/s. Run long experiments **overnight** using `queue_overnight.py`.
- **Curriculum:**
  1. A single wide showcase track.
  2. Random generated tracks (wide).
  3. Random tracks with mixed widths and tighter corners.

  Each stage either fine-tunes from the previous model (recorded as the `parent` in the registry) or uses mixed sampling.
  **Chosen (D8):** a chain of separate runs linked by `train.py --parent <model_id>` (policy and VecNormalize statistics carried over, lower learning rate via `ppo_finetune`), so each stage is evaluated and registered on its own: R001 (Training Ground) → R002 (generated wide) → R003 (final mix). R004 trains on the final mix from scratch for comparison.
- **Track sets (D7, `configs/trackset/*.yaml`):** which tracks a run samples: named tracks with weights, folders of tracks, and a generated pool (training seed stream 0, a seed range, generator overrides such as a width range). Generated fields cost 3–14 MB and ~65 ms each, so each worker keeps a small rotating pool of generated tracks instead of a new one per episode. Held-out tracks are never sampled. The final mix may include showcase tracks and any other added tracks (owner, §6), so showcase lap times are on seen tracks; generalisation is measured on held-out and user-drawn tracks only.
- **Checkpoints:** every 100k steps. Every checkpoint is registered (see §9) and evaluated after training, and the best is marked `is_best` (D9: highest completion over all evaluation episodes, ties broken by mean showcase lap time relative to the baseline). Evolution races and comparisons can be built from any step.
- **Reproducibility (D12):** real (`R`) runs refuse to start on a dirty git tree unless `--allow-dirty`, so the `git_commit` in every snapshot points at real code.
- **Evaluation protocol (`evaluate.py`, runs automatically after training and on demand):**
  - deterministic policy and fixed seeds
  - 3 showcase tracks × 3 starts, plus 10 held-out tracks × 1 start
  - **Metrics:**
    - lap completion rate
    - best and mean lap time for each showcase track
    - crash rate
    - mean speed
    - off-road time %
    - mean |lateral g|
    - held-out completion rate (generalisation)
  - Saves a trajectory PNG thumbnail for each showcase track.
  - The protocol is versioned in `configs/eval/v1.yaml` (standing starts on the centreline, evenly spaced). The hand-coded baseline is registered as pseudo-model `BASE_pid` (E8 reference).
- **Logging:** TensorBoard (`runs/`) for episode reward, length, lap time and losses, plus `race/*` (lap time, laps, progress) and `end/*` (termination reasons).

---

## 9. Model registry and versioning

**Rule:** every model folder is **self-contained and immutable**. The demo must be able to load and run correctly any model ever trained, even after the code has moved on.

```
models/R012_ppo_rays5_curric@1500k/
├── model.zip            # SB3 policy
├── vecnormalize.pkl     # observation normaliser statistics
├── config.yaml          # FULL snapshot: obs_spec, physics, reward, train hparams, seed, git hash
├── eval.json            # metrics from evaluate.py
└── thumb_<track>.png    # trajectory coloured by speed
```

- **What "immutable" means (D10, owner):** `model.zip`, `vecnormalize.pkl` and `config.yaml` are never changed after saving. `evaluate.py` **adds** `eval.json` and `thumb_<track>.png` once per protocol version (a later protocol writes `eval_v2.json`; nothing is overwritten), and copies them to `outputs/eval/<model_id>/`. Models stay local (git-ignored); the copies in `outputs/eval/` and the registry CSVs are tracked.
- **Model ID:** `R{run:03d}_{short_tag}@{steps}k` (e.g. `R012_ppo_rays5_curric@1500k`). Ablation runs carry the experiment in the tag (D17, e.g. `R010_e2_nogrip_s1@3000k`).
- **Observation specs are versioned** (`obs_spec: {version: 1, rays: [...], extras: [...]}`). Never change what an existing version means; add a new version instead. The loader builds each car's sensor from that car's own spec.

**`registry/runs.csv`:** one row per training run
`run_id, created_at, tag, algorithm, parent_model, obs_spec, physics_preset, reward_preset, track_set, curriculum, total_steps, wall_clock_min, seed, hparams_json, notes`

**`registry/models.csv`:** one row per saved checkpoint
`model_id, run_id, steps, path, is_best, completion_rate, crash_rate, heldout_completion, best_lap_<showcase1..3>, mean_speed, offroad_pct, mean_lat_g, overall_completion, lap_ratio, score, eval_protocol, eval_date, tags, notes`
(`completion_rate` = showcase episodes, `heldout_completion` = held-out episodes, `overall_completion` = all; `lap_ratio` = best showcase lap ÷ the baseline's, averaged over the tracks the model lapped; `mean_speed` in m/s.)

**`registry/registry.xlsx`:** generated by `export_xlsx.py` (openpyxl)
- **Sheets:** `Runs`, `Models`, a leaderboard sheet for each showcase track, and `Ablations` (pivot-style summary).
- **Formatting:** frozen header, filters, conditional formatting (green for the best lap time), hyperlinks to model folders and thumbnails.
- The CSVs stay the source of truth; the xlsx is regenerated, never edited by hand (and git-ignored, D11). Rows are upserted by id (evaluation updates them). Scratch runs use a separate registry in `outputs/scratch/registry/`.

---

## 10. Demo application (`scripts/demo.py`, pygame-ce)

### 10.1 Modes (menu launcher)
1. **Model Arena (the centrepiece)**
   - Pick a track, then pick **1–20 models** from a registry browser (searchable and sortable, showing each model's metrics and thumbnail).
   - All the chosen cars race at once as ghost cars with no collisions. Each car gets a distinct colour and label.
   - **Live leaderboard:** position, gap, current, last and best lap.
   - **Physics:** each car uses the physics it was trained with by default. A toggle switches to a common "fair race" preset.
   - **Race rules (D14):** staggered 2-wide grid behind the start line, 3-2-1 countdown, N-lap race (default 3), then a results table. A crash respawns the car on the centreline where it crashed, from standstill after 1 s, and the crash is counted on the leaderboard (no DNF). Gaps are in seconds (time since the leader passed the same distance).
   - **Code (D15, D16):** the race itself is simulation (`racer/env/race.py`, pygame-free); rendering uses `render/world.py` and `camera.py`. Models are picked in a basic in-window registry list (sort, filter, multi-select) or with `--models`.
2. **Evolution Race:** pick one run, and its checkpoints (e.g. 100k, 300k, 1M, 3M) race together. This shows learning in about 20 seconds.
3. **Draw-a-Track:** sketch a track, validate it, and pick models to drive it live. This demonstrates generalisation.
4. **Human vs AI:** keyboard (WASD or arrow keys; optional gamepad through pygame joystick) against chosen models. Ghost cars, same physics.
5. **Analysis view:**
   - racing line compared with the centreline
   - **telemetry trace** (speed against lap distance, F1-style) comparing 2–4 models
   - braking zones highlighted
   - built with matplotlib, rendered into pygame or exported as PNGs
6. **Swarm view (optional):** many environments from one policy drawn at once. Used for early-training chaos shots and recorded videos.

### 10.2 Visual features (all effects code-generated)

| Feature | Details |
|---|---|
| World | Ground texture tiled across the map. Road texture masked by the road polygon. Red and white kerbs, a checkered start/finish line and the barrier. Drawn **once per track** to a large surface, then blitted every frame. |
| Car | Kenney sprites (`car_{1..4}_{colour}.png`). For more than 5 distinct colours, use `car_N_tint.png` + `car_N_tint_mask.png` (tint the body only; see `assets/LICENSES.md`). Brake lights glow when braking. Small drop shadow. |
| Rays | Lines from the car, coloured green → yellow → red by distance, with a dot where each ray hits. Shown for the selected car only (20 cars' rays would be clutter); toggle with `R`. |
| Trail | Fading trail behind each car, coloured by speed (blue = slow, red = fast). Toggle `T`. |
| Racing line overlay | Best lap path coloured by speed, with the centreline dashed for comparison. Toggle `L`. |
| Skid marks | Drawn to a persistent layer when `sliding`. |
| Smoke and dust | Particles: grey on tarmac when sliding, green/brown dust on grass |
| HUD | Speed (km/h), gear-style speed bar, steering and throttle/brake bars, lap number, current/last/best lap, lateral g, surface indicator |
| Brain panel | Live bars for the 13 inputs, plus the 2 outputs shown as a steering wheel and pedal bars. Shows what the network senses and decides. |
| Camera | Follow the selected car (smooth lerp, zoom with speed), overview mode, cycle cars with `Tab` |
| Minimap | Corner minimap with coloured dots for all cars |
| Scenery | Kenney props outside the barrier: grandstands, tents, trees, tyre stacks and barriers along the run-off, and a `start_lights` gantry at the start line (countdown before a race). Placed procedurally from the track geometry and never inside the barrier. |
| Recording | `record_video.py` renders offscreen (no window) to MP4 with imageio-ffmpeg, for highlight videos |

**Performance target:** 60 FPS with 20 cars. Inference is batched per model and runs every third frame (20 Hz), so it costs almost nothing.

### 10.3 Controls (summary)
`Tab` cycle car · `R` rays · `T` trails · `L` racing line · `B` brain panel · `M` minimap · `F` follow/overview · `S` sound on/off · `Space` pause · `+/−` speed ×0.25–×4 · `Esc` menu

### 10.4 Audio (`racer/render/audio.py`, `configs/demo/audio.yaml`)

All clips and their RPM curves are described in `assets/LICENSES.md` and `assets/sounds/engine_map.json`.

**Whose sound plays:** only the **followed car** (or the human's car) is audible. With 20 cars, everything else stays silent. Audio is mixed with pygame-ce `mixer`, and each clip is decoded once into a numpy array.

**Cosmetic gearbox (`gearbox.py`):** the physics engine has no gears (it is power-limited, §5.2). The gearbox exists **only for sound and the HUD**, and doesn't affect physics or RL.
- `rpm_ratio` (idle = 1.0) comes from speed and the current gear: 5 gears up to about 70 m/s.
- It shifts up when `rpm_ratio` reaches **2.85** (the top of `engine_rev_up`), and the ratio steps are chosen so `rpm_ratio` lands at about 2.0 after a shift. It shifts down at about 1.4.
- At standstill or very low speed in 1st gear, `rpm_ratio = max(1.2, speed-derived)` (clutch slip).

**Engine logic (what the user described):**
1. **Throttle pressed from standstill:** play `engine_rev_up` from the start, so it rises from idle.
2. **Throttle pressed again while moving:** play `engine_rev_up` from the offset whose ratio in `engine_map.json` equals the current `rpm_ratio`, i.e. from somewhere in the middle. It rises from there.
3. **Reaching the top** of `rev_up`: crossfade (about 60 ms) into `engine_high_loop` and hold at redline.
4. **Gear shift up:** the RPM drops, so re-seek `rev_up` to the new, lower RPM. Repeated through the gears, this gives the classic accelerating-through-the-gears sound.
5. **Throttle released:** play `engine_rev_down` from the offset matching the current RPM. If the RPM is above 2.34 (the top of rev_down), start it at 0. When it reaches `idle_loop_entry_s`, switch to `engine_idle_loop` (seamless, because both come from the same take).
6. **Keeping sound and car in sync:** a clip rises or falls at its own recorded speed, which won't always match the car. If the clip's current ratio drifts more than about 100 cents (6 %) from `rpm_ratio`, re-seek with a short crossfade (30–50 ms).
7. **Volume:** use per-clip gains from config, because the clips keep their natural loudness (idle is about 20 dB below a full rev). Scale volume slightly with throttle.

**Tyres:**
- `tyre_screech_loop` volume follows the slip intensity: lateral sliding speed, or the friction-circle overload when `sliding`. Fade it in and out over about 80 ms.
- On the **onset** of a short slide or hard braking, play `tyre_screech_short_1` or `_2` at random.

**Recording:** `record_video.py` stays silent by default. Mixing the audio into the MP4 is a *Could*.

---

## 11. Experiments and analysis (for the report and slides)

| # | Experiment | Expected story |
|---|---|---|
| E1 | Main run: PPO, rays5, grip on, curriculum | Learning curves; lap time falls; a racing line appears |
| E2 | **Grip limit off** (`no_grip_limit`) | Flat-out everywhere and no braking, so no real racing line. **Shows that the physics is what creates the behaviour.** |
| E3 | **Naive speed reward** | Reward hacking: circling or weaving. A funny video with a serious lesson. |
| E4 | Rays: 5 vs 9, and short vs long forward range | Short rays mean late braking and crashes: the agent needs to see ahead |
| E5 | Single-track vs multi-track training | Overfitting: perfect on its own track, fails on held-out tracks |
| E6 | Rays only vs rays + curvature look-ahead | How much map knowledge helps |
| E7 | PPO vs SAC | Sample efficiency and stability compared |
| E8 | RL vs hand-coded baseline (a PID controller that follows the centreline at a fixed safe speed) | RL is faster because it uses the track width and brakes at the right time |
| E9 (optional) | Downforce on vs off | A different racing line on fast corners |

**Plots (`make_plots.py`):**
- reward and lap time against training steps (mean ± std over seeds where possible)
- completion rate on held-out tracks
- a bar chart of the ablations
- trajectory maps coloured by speed
- telemetry traces
- a lateral g–longitudinal g scatter plot (the "g-g diagram"), which shows the agent using the friction circle

---

## 12. Timeline (4 weeks) and priorities

| Week | Focus | Milestone |
|---|---|---|
| **1** (Oct 10–16) | Physics, surfaces, spline tracks, generator, progress, sensors, Gymnasium env, physics sanity tests, minimal debug renderer, first PPO run | **M1:** the car completes laps on a showcase track |
| **2** (Oct 17–23) | Config system, train/evaluate scripts, checkpoints, **registry and xlsx export**, loader, curriculum and multi-track training, **basic Model Arena with 20 cars** | **M2:** load any 20 models from the registry and race them |
| **3** (Oct 24–30) | Textures, effects, HUD, brain panel, minimap, camera, Evolution Race, Draw-a-Track, Human vs AI, video recorder | **M3:** full live demo |
| **4** (Oct 31–Nov 7) | Ablations (overnight queue from week 2 onwards), plots, telemetry/analysis view, highlight video, slides, README, rehearsal, buffer | **M4:** ready to present |

**Keep the CPU busy every night from week 2:** queue the experiment runs.

**Priorities (MoSCoW):**
- **Must:** grip physics env, PPO training, registry (CSV and xlsx), load any model, multi-car arena (10–20), rays, trails, HUD, learning curves, showcase tracks, held-out evaluation
- **Should:** Evolution Race, textures and effects, Draw-a-Track, Human vs AI, ablations E2–E5, telemetry trace, highlight video, engine and tyre audio (§10.4)
- **Could:** brain panel, swarm view, SAC (E7), curvature (E6), downforce (E9), trackside scenery, audio in recorded videos, gamepad
- **Won't (this time):** car-to-car collisions, multi-agent competitive training, pixel/CNN observations, 3D

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reward hacking (backwards laps, exploits at the start line) | Signed `Δs` with wrap-around handling; wrong-way termination; watch videos of every new run |
| Agent learns to stop or crawl | Time penalty, no-progress termination, random starting speed |
| Twitchy steering looks bad | Steering rate limit, `|Δsteer|` penalty, previous action in the observation |
| Physics unstable at low speed | Force-based sideslip removal (no slip-angle division by v); clamp at v ≈ 0; sanity tests |
| Training too slow | SDF ray casting / numba, profile `env.step`, 20 Hz decisions with action repeat |
| Overfits to one track | Generated tracks, random direction and mirroring, held-out evaluation |
| Old models break after code changes | Immutable model folders, versioned `obs_spec`, configs loaded per model |
| Running out of time | MoSCoW above; M2 (arena and registry) is a presentable fallback on its own |
| Tunnelling through walls at high speed | Check the swept segment from the previous to the current position against the barrier, not just the end point |

---

## 14. Setup: software and assets

### 14.1 Software (Windows)
- **Python 3.11** (64-bit, "Add to PATH" ticked). 3.12 also works.
- **VS Code** with the **Claude Code** extension and the Python extension
- **Git for Windows** (and optionally a GitHub account for the public repo and README GIFs)
- **Excel**, or the free **LibreOffice Calc**, to open `registry.xlsx`
- Optional: **OBS Studio** to screen-record the live demo; a gamepad
- A separate ffmpeg install is **not** needed: `imageio[ffmpeg]` bundles its own.

```bat
cd "%USERPROFILE%\Desktop\ML Sem5 Project"
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

- `pip install -e .` (via `pyproject.toml`) makes the `racer` package importable from `scripts/` without `sys.path` hacks.
- `.vscode/settings.json` points VS Code at `.venv`.

`requirements.txt`: the packages below, **pinned to exact versions** (the set verified on 2026-10-10 with Python 3.11.9). This keeps saved models loadable (rule in §9). Change a pin only on purpose, and log it.
```
gymnasium==1.4.0
stable-baselines3==2.9.0   # NOT [extra]: it pulls in classic pygame, which conflicts with pygame-ce
tensorboard==2.21.0
tqdm==4.70.1
rich==15.0.0
numpy==2.4.6
numba==0.68.0
scipy==1.17.1
pygame-ce==2.5.8
pyyaml==6.0.3
pandas==3.0.6
openpyxl==3.1.5
matplotlib==3.11.2
imageio[ffmpeg]==2.38.1
# torch==2.14.1+cpu is installed separately from the CPU index (above)
```

### 14.2 Assets (in place; audited 2026-10-10)

The assets are downloaded, renamed and pruned. **`assets/LICENSES.md` is the manifest:** it lists every file, its source and licence, how it was cut or modified, and how to use it. Read it before writing rendering or audio code.

| Folder | Contents |
|---|---|
| `sprites/cars/` | `car_{1..4}_{black,blue,green,orange,yellow}.png` (Kenney, facing up) + `car_N_tint.png` / `car_N_tint_mask.png` (tintable) |
| `sprites/props/` | barriers, tyre stacks, cones, trees, grandstands, tents, `start_lights` |
| `textures/ground/` | `asphalt.jpg` (stains removed, seamless), `grass.jpg`, `gravel.jpg`, `sand.jpg`: 1024², seamless, ambientCG CC0 |
| `fonts/` | `rajdhani_{regular,medium,semibold,bold}.ttf`, `share_tech_mono.ttf`, `orbitron_{bold,black}.ttf` |
| `sounds/` | `engine_{rev_up,rev_down,idle_loop,high_loop}.ogg` + `engine_map.json`; `tyre_screech_{loop,short_1,short_2}.ogg` |
| `licenses/` | Kenney CC0 licence and the OFL texts for each font |
| `_source/` | original downloads (git-ignored) |

- **Licences:** everything is CC0 or OFL **except the engine sound, which is CC BY 4.0**. The README and the demo's title or credits screen must show the credits line from `assets/LICENSES.md`.
- **Textures:** in code, darken and slightly desaturate them, and scale them to world metres (suggested tile sizes are in `LICENSES.md`), so photographic ground and Kenney's flat sprites look consistent.
- **Generated in code:** kerbs, start/finish line, skid marks, smoke, trails, rays and UI panels.
- **Fallback:** if an asset is missing, use a procedural placeholder (noise ground, polygon car, silence). The game must never crash because an asset is missing.

---

## 15. Conventions for implementation (Claude Code)

- Read this file before starting any task, and keep the simulation (`physics/`, `track/`, `env/`) **free of pygame**.
- **No magic numbers in code.** Tunable values belong in `configs/*.yaml`, with defaults matching this document.
- Every training run is reproducible: seed, full config snapshot, git commit hash in `config.yaml`.
- Any script that uses `SubprocVecEnv` needs the `if __name__ == "__main__":` guard (Windows).
- Use type hints and short docstrings. Use `pathlib` for paths, because the project path contains spaces.
- After physics changes, run `scripts/physics_sanity.py`. After env changes, run a 50k-step smoke test of training.
- When adding an observation feature, create a **new `obs_spec` version**; never change an existing one.
- Never add a required key to an existing config schema (physics, env, reward, track) unless an explicit snapshot migration fills it for old models; prefer new presets or new types.
- Never hand-edit `registry.xlsx`; regenerate it from the CSVs.
- When a goal, requirement or design decision changes, update this file first (and §16), then every other document it affects (`README.md`, `log.md`, `assets/LICENSES.md`, configs).

---

## 16. Decisions log and open questions

| Date | Decision / note |
|---|---|
| 2026-10-10 | Build a custom environment instead of adapting an open-source game. PPO (SB3) on CPU. |
| 2026-10-10 | Inputs: 5 rays + heading (relative to the track) + speed (+ lateral velocity, yaw rate, on-road fraction, previous action). Outputs: steer and gas in [−1, 1]; negative gas = brake, no reverse. |
| 2026-10-10 | Physics: mass, momentum, constant gravity, friction-circle grip `μ·m·g`, per-wheel surfaces with `Surface` (grass) → `Road` → `Kerb` inheritance. |
| 2026-10-10 | Demo: registry (CSV → xlsx), load any model, 10–20 ghost cars, evolution race, draw-a-track, human vs AI. |
| 2026-10-10 | Assets: CC0 (Kenney, ambientCG / Poly Haven) and OFL fonts (Rajdhani, Share Tech Mono, Orbitron); all effects code-generated. |
| 2026-10-10 | Asset audit: renamed and pruned; Kenney pickup truck dropped (4 car shapes); tintable car bases with masks added; asphalt stains removed. Engine sound (freesound #623436, **CC BY 4.0**, credit required) cut into rev_up / rev_down / idle / high loops with an RPM map. Tyre screech (CC0) trimmed into a 2 s loop + 2 one-shots. |
| 2026-10-10 | Audio design: a cosmetic gearbox (sound and HUD only) drives `rpm_ratio`; clips are seeked by RPM (§10.4). Scenery props added as a *Could*. |
| 2026-10-10 | Initialisation: Python 3.11.9 `.venv`, torch 2.14.1+cpu, all versions pinned in `requirements.txt`; `racer` installed editable via `pyproject.toml`. |
| 2026-10-10 | Rule: whenever a goal or requirement changes, update this file first, then every document it affects (`CLAUDE.md` working rules, §15). |
| 2026-10-10 | Week 1 decisions (owner-approved, `log.md` T16 D1–D6): **D1** kerbs inside the road edge, on-track (§5.1); **D2** `obs_spec` v1 details frozen (§7.1); **D3** track field + barrier = field territory, walls between close sections (§6); **D4** braking test: threshold ≈ 42 m, full lock ≈ 49 m, physics unchanged (§5.5); **D5** `configs/track/default.yaml`; **D6** `pytest` + `tests/` (§4). A minimal config loader and `train.py` are pulled forward from Week 2. |
| 2026-10-10 | **M1 reached:** R001 (PPO, Training Ground, 2M steps, ≈ 20 min). Best checkpoint `R001_ppo_rays5_tg@1200k`: 6/6 deterministic starts, best lap 42.70 s vs 46.6 s for the hand-coded baseline. Reliability oscillates between checkpoints (agent drives at the limit with wheels on the grass), so models are chosen by evaluation, not by recency (§8). |
| 2026-10-10 | **Models stay local** (owner): `models/` is git-ignored and removed from tracking (≈ 33 MB per 2M-step run). Back up by copying the folder (current backup: `C:\Users\Jack\ML_Sem5_models_backup\`). The registry CSVs stay in git. |
| 2026-10-10 | **Braking feel (owner):** brake force is not the limit (12 000 N already exceeds grip; 20 000 N stops in the same 42 m from 100 km/h). Keyboard braking is all-or-nothing, so it locks the wheels (kinetic μ, little steering). Fix: an **ABS assist for human input only** (`Car.abs_brake_limit`, settings in `configs/demo/render.yaml`): 35 m instead of 42 m, steering kept. Physics, training and saved models are unchanged; no retraining. |

| 2026-10-10 | Week 2 decisions (owner-approved, `log.md` T36 D7–D18): **D7** track sets in `configs/trackset/` with per-worker rotating generated pools; **D8** curriculum as a `--parent` chain R001 → R002 → R003, plus R004 from scratch; any non-held-out tracks may be mixed into training; **D9** eval protocol v1 + `is_best` rule + `BASE_pid`; **D10** evaluation adds `eval.json`/thumbnails to model folders (the three model files never change) and copies them to `outputs/eval/`; **D11** registry upsert, scratch registry, xlsx git-ignored; **D12** no real runs on a dirty tree; **D13** ablations config-only, no new required keys in existing schemas without a snapshot migration; **D14** arena: grid, countdown, N laps, respawn + crash count; **D15** `racer/env/race.py` + `render/world.py`/`camera.py`; **D16** basic in-window model picker; **D17** ablation tags; **D18** overnight queue runs without a Claude session. |
| 2026-10-10 | **Tracks (owner):** the track library stays open; tracks can be added at any time and mixed into training. The final showcase is a track drawn by faculty, converted into a track and driven live (§6). |

**Open questions:**
- Faculty track: drawn in the Draw-a-Track editor, or on paper and converted from a photo? Currently: the editor (Week 3); photo import is a possible extension.
- Should rays also detect the barrier, or only the road edge? Currently: road edge only.
- Should the demo default to each model's own physics or a common one? Currently: each model's own, with a toggle.
- ~~Should kerbs be "road" for track limits?~~ Decided (D1): yes, kerbs sit inside the road edge and count as on-track.
