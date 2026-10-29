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

A 2D top-down racing game written from scratch, with a car that learns to drive with **reinforcement learning (PPO)**. A genetic algorithm (neuroevolution, D26) trains the same kind of network without gradients, as a comparison (E10).

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
| --- | ------ | ------------- |
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
| --- | --- | --- |
| Game | **Write our own**, don't adapt an existing game | Existing games have no `reset()` or `step()`, tie the game loop to rendering and run at real-time speed. Our own engine runs headless and fast. |
| RL interface | **Gymnasium** `Env` | Standard; works directly with Stable-Baselines3 |
| Algorithm | **PPO** (Stable-Baselines3); SAC and neuroevolution (D26) as comparisons | PPO is robust, simple and works well with parallel environments |
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
│   ├── env/      rays5.yaml, rays9.yaml, rays7_narrow.yaml, rays7_lean.yaml (GA), rays7_pen.yaml (rays through the road edge, D33) …
│   ├── reward/   progress.yaml, speed_naive.yaml (for the reward-hacking demo)
│   ├── train/    ppo_default.yaml, sac_default.yaml, baseline.yaml (hand-coded driver), ga_default.yaml (neuroevolution, D26)
│   ├── track/    default.yaml (kerb threshold, field cell size, generator ranges, clearances)
│   ├── trackset/ tg.yaml, gen_wide.yaml, final_mix.yaml, maps7.yaml, train7.yaml (GA) … (which tracks training samples, §8)
│   ├── eval/     v1.yaml, v2.yaml (evaluation protocols, §8)
│   ├── queue/    *.yaml (experiment lists for queue_overnight.py; sandbox.yaml is the sandbox's, git-ignored)
│   ├── sandbox/  default.yaml (AI Sandbox: game-window cascade, stop grace, queue time estimates)
│   ├── backup/   default.yaml (model backup folder, T81)
│   └── demo/     render.yaml (texture scales, colours; `debug:` window), audio.yaml (volumes, gearbox)
├── racer/config.py           # YAML preset loader (strict keys, deep-merged overrides)
├── racer/
│   ├── physics/  car.py, surfaces.py
│   ├── track/    track.py, spline.py, field.py, generator.py, progress.py, io.py
│   ├── env/      racing_env.py, sensors.py, reward.py, obs_spec.py, trackset.py,
│   │             race.py (multi-car race for the demo, pygame-free),
│   │             features.py (obs v2 feature registry), actions.py (action layouts)
│   ├── agents/   loader.py (registry → policy + normaliser + configs), baseline.py,
│   │             evaluation.py (eval protocol, metrics, thumbnails, eval files),
│   │             algorithms.py (PPO/SAC/GA registry), compat.py (manifest + engine check),
│   │             neuroevolution.py (GA core: selection, mutation, lineage rebuild), evolve.py (GA runs, D26)
│   ├── registry/ registry.py, export_xlsx.py, record.py (evaluate + record)
│   ├── sandbox/  data, explain, diagrams, curves, builder, queue, procs, launcher, glossary.yaml,
│   │             pages/ (AI Sandbox, §10.5; Streamlit is imported only in pages/)
│   └── render/   app.py (shell, Context, MODES), menu.py, arena.py (picker + race window), evorace.py, human.py, swarm.py,
│                 brain_panel.py, dashboard.py, minimap.py, effects.py, scenery.py, textures.py, audio.py + engine_sound.py + gearbox.py,
│                 debug_view.py (Week 1 debug window), world.py, camera.py, cars.py (tinted sprites),
│                 effects.py, hud.py, brain_panel.py,
│                 minimap.py, ui.py (menus/model picker), editor.py,
│                 audio.py (engine/tyre sound controller), gearbox.py (cosmetic RPM)
├── scripts/
│   ├── train.py  evaluate.py  demo.py  make_plots.py  bench_demo.py  registry.py   (record_video.py dropped, D28)
│   ├── queue_overnight.py    # runs a list of experiments one after another
│   ├── physics_sanity.py     # skidpad / braking / coast-down tests
│   ├── drive.py              # debug window: keyboard / baseline / model driving
│   ├── preview_tracks.py  generate_tracks.py  make_maps.py  bench_env.py
│   ├── backup_models.py      # copy new model folders/files to the backup (after every run, T81)
│   ├── export_ga.py          # save any car of a neuroevolution run as a model folder (D26)
│   ├── bench_train.py        # training throughput: CPU vs GPU, minibatch, parallel runs (T62)
│   ├── sandbox.py            # AI Sandbox (Streamlit in its own window, §10.5): understand, compare, build/queue, launch
├── tests/                    # pytest unit tests (spline, field, progress, physics, env, …; test_integrity.py: registry ↔ model folders ↔ eval copies ↔ docs ↔ pins)
├── tracks/
│   ├── showcase/   *.json    # 3 hand-designed demo tracks (training maps)
│   ├── train/      *.json    # more hand-designed training maps (D19)
│   ├── test/       *.json    # hand-designed test maps, NEVER used in training (D19)
│   ├── heldout/    *.json    # fixed generated tracks, NEVER used in training
│   └── user/       *.json    # tracks saved from draw-a-track mode
├── docs/          SANDBOX_PLAN.md (AI Sandbox implementation plan, D24), NEUROEVOLUTION_PLAN.md (D26, built)
├── assets/
│   ├── LICENSES.md           # manifest, credits, usage notes (READ before using assets)
│   ├── sprites/cars/  sprites/props/  textures/ground/  fonts/  sounds/  licenses/
│   └── _source/              # original downloads (git-ignored), for re-cutting
├── models/<model_id>/        # self-contained model folders (see §9); local only, git-ignored
├── registry/  runs.csv  models.csv  registry.xlsx  run_notes.yaml (why + outcome per run)
├── runs/                     # TensorBoard logs
└── outputs/   videos/  plots/  frames/  eval/ (copies of every model's eval files)
               scratch/ (smoke runs, git-ignored)  queue_logs/ (git-ignored)
               evolve/<run_id>/ (neuroevolution state.npz + lineage.sqlite, git-ignored, D26)
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
| --- | --- | --- |
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

**Car variants and physics versions (D23).** A different car (mass, power, grip, steering) is just a new physics preset; each model keeps its own copy in its snapshot. A change to the physics *equations* (e.g. real gears, a tyre model) gets a new physics `version` with its own car class; a preset without `version` is version 1, so old models keep driving exactly as they were trained.

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
- **Map library (D19, owner):** at least 9 hand-designed maps. **7 training maps:** the 3 showcase tracks + 4 in `tracks/train/`. **2 test maps** in `tracks/test/`, never used in training (the loader refuses a track set that names one, as for `heldout/`). More maps can be added to either folder at any time; `scripts/make_maps.py` builds the hand-designed maps from rounded-polygon outlines, so they are reproducible.
- **Held-out set:** about 10 generated tracks with fixed seeds, saved in `tracks/heldout/`. **Never** used in training, only to test generalisation. Together with the 2 test maps, this is the "unseen" set.
- **Open track library (owner):** tracks can be added at any time in the project. Any valid track JSON dropped into a `tracks/<folder>/` is picked up by name everywhere (training track sets, evaluation, demo). Training may mix any non-held-out tracks, and new hand-made or generated tracks may be added for variety. Adding a track never changes an existing model (each model's snapshot lists what it trained on).
- **Final showcase (owner):** faculty members draw a track, it is converted into a track JSON (Draw-a-Track, §10.1), and the trained models drive it live. This is the strongest generalisation demo, so held-out performance is the metric that matters most.
- **Editor (draw-a-track; built in the AI Sandbox, T111/T119, owner):**
  1. Draw freehand, or click points (click the first point again to close the loop). The first point is the start/finish line; the cars drive the way it was drawn.
  2. `racer/track/sketch.py` smooths the drawing (a window that is a fraction of the loop, so the drawing size doesn't matter), scales it into 800–4000 m and thins it to control points; a corner still too tight gets more smoothing. Then the generator's validation; what can't be fixed (crossings, sections too close) is marked in red. A line instead of a loop is refused. Settings: `configs/demo/editor.yaml`.
  3. Save it to `tracks/user/` (a preview PNG goes to `outputs/plots/tracks/`), then *Race on it* opens Launch with the track chosen.

---

## 7. Environment (Gymnasium)

### 7.1 Observation (default spec `rays5`, all values scaled to about [−1, 1] or [0, 1])

| Feature | Dims | Notes |
| --- | --- | --- |
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

**`obs_spec` v2: features as building blocks (D21).** v1 stays frozen for the models that use it. v2 is an ordered list of named features, each with its own parameters:

```yaml
obs_spec:
  version: 2
  name: rays7_narrow
  features:
    - {feature: rays, rays: [{angle_deg: -60, range_m: 60}, {angle_deg: -20, range_m: 120}, …]}
    - {feature: heading_sincos}
    - {feature: v_long, scale: 70.0}
    - {feature: prev_action}
```

- Features live in a registry (`racer/env/features.py`). Each one declares its size, bounds, labels for the brain panel, and a one-line description (shown in the AI Sandbox). The observation is the concatenation in list order.
- **Adding or removing an input = editing the list.** A new kind of input = a new registered feature name. A feature's meaning never changes once a model uses it; a changed behaviour gets a new name.
- Features available at first: `rays`, `heading_sincos`, `v_long`, `v_lat`, `yaw_rate`, `onroad_frac`, `prev_action` (same maths as v1), plus `steer_angle` (current wheel angle) and `curvature_ahead` (track curvature at given distances ahead: map knowledge, E6).
- **Added by D30 (owner, 2026-10-26):** `accel_g` (forward and sideways acceleration), `grip_use` (tyre force demanded ÷ grip available), `wheel_surface` (road / kerb / grass for each wheel), `road_position` (offset from the centre line and gaps to both road edges) and `history` (memory: chosen earlier inputs as they were 1…n decisions ago, every k-th decision; must be the last feature; before the first decision of an episode the past is the present). `history` is stateful: observers have a `reset()` that the env calls at every episode start.

- **Added by D33 (owner idea, 2026-10-29): `rays2`, rays that see through the road edge.** A real driver sees the road beyond the first edge (the exit of a bend across the infield); a ray that stops at the first crossing is tunnel vision. `rays2` casts each ray on past its first road-edge crossing and reports, for each ray, the distance to every crossing up to `penetrations` more (default 1, so two values per ray: leaving the road, then re-entering it), each as a share of the ray's range, 1 = no such crossing within range. A ray starting off the road reports entering first, then leaving. It still measures geometry along the car's own rays only (no track look-up, so it works on any drawn track). `rays` is unchanged; the race-pace experiments use `rays2` instead of `curvature_ahead` (the owner found a curvature input unfair to a real driver). The ray length is the range (the owner asked for 200 m on all); there is no separate "penetration strength".

### 7.2 Action

`Box([-1,-1], [1,1])` → `[steer, gas]`. A negative gas value means brake. There is no reverse gear.

**Action layouts (D22).** The layout is part of each model's snapshot (`action_spec: {layout: …}`); an old snapshot without it means `steer_gas`.

- `steer_gas` (default, all models so far): `[steer, gas]` as above.
- `steer_throttle_brake`: 3 outputs in [−1, 1]; throttle and brake are mapped to [0, 1] and the car receives `gas = throttle − brake`.
- **Added by D30:** `steer_rate` (the first output turns the wheel from where it is by up to 0.3 of full lock per decision, instead of naming an angle) and `steer_throttle_brake_abs` (as `steer_throttle_brake`, with the brake capped just inside the grip limit by the env). A handbrake layout needs a new physics version and is in REVIEW_LATER.md.
- New layouts are added to a registry (`racer/env/actions.py`); `prev_action` and the reward's steering term read the layout, so they work for any layout.

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
  - **optional, off by default (D26):** the car's centre crosses the road edge (`track_limits`); only the neuroevolution preset turns it on, so every existing preset and model is unchanged
- **Truncate** after 2 laps or 4000 agent steps.
- Random track per episode during multi-track training (curriculum, see §8).

---

## 8. Training

- **Library:** Stable-Baselines3 PPO, `MlpPolicy`, `net_arch=[256,256]`, tanh activations.
- **Parallelism:** `SubprocVecEnv` with **6 environments** (4 cores / 8 threads; leaves headroom). `VecNormalize` for observations and rewards, with its statistics saved alongside each model.
- **Starting hyperparameters:** `n_steps=1024` per env, `batch_size=256`, `n_epochs=10`, `gamma=0.995`, `gae_lambda=0.95`, `lr=3e-4` (linear decay), `clip=0.2`, `ent_coef=0.002`.
- **Device:** **CPU**, with `torch.set_num_threads(1)` in each worker. The GTX 1650 is not useful for an MLP this small. **Measured (T62, 2026-10-11, `scripts/bench_train.py`):** the GPU trains at 0.64× the CPU's speed (0.90× at best with batch 2048): every env step runs the policy on tiny batches, and the transfer overhead doubles collection time. **Two runs in parallel on the CPU (6 envs each) give 1.62× the total throughput**, so the queue will run 2 at a time (T64).
- **Windows:** SubprocVecEnv needs the `if __name__ == "__main__":` guard in every script.
- **Overnight queue (D18, owner):** `queue_overnight.py` is started by the owner from a normal terminal and runs **without a Claude session** (no token use). It runs the queue file's runs, logs each to `outputs/queue_logs/`, continues after a failure, resumes finished queues, and keeps Windows awake only while it runs (no power-plan changes). **Parallel runs (T64):** `--parallel N` (or `parallel: N` in the queue file) runs N entries at once; 2 is the measured best (1.62× total throughput, T62). Run ids are reserved atomically (`models/.run_ids/`), registry writes hold a lock file, and parallel runs share the evaluation workers. The queue file is re-read whenever a slot frees, so entries can be added while it runs. **Sandbox control (T71):** while it runs it writes `runner.json` (its PID) and each running entry's PID to `state.json`. A `stop.request` file next to `state.json` stops it like Ctrl+C. Stopping kills each run's whole process tree: a venv's `python.exe` is a launcher, so `train.py` and its env workers are its children.
- **Expected budget:** 2–5M steps per serious run. **Measured (2026-10-10):** ≈ 1.7k steps/s with 6 envs, so ≈ 10 min per million steps. The PPO update is ~60 % of the time (compute-bound at batch 256; more torch threads don't help); one env alone runs at 13k steps/s. Run long experiments **overnight** using `queue_overnight.py`.
- **Curriculum:**
  1. A single wide showcase track.
  2. Random generated tracks (wide).
  3. Random tracks with mixed widths and tighter corners.

  Each stage either fine-tunes from the previous model (recorded as the `parent` in the registry) or uses mixed sampling.
  **Chosen (D8):** a chain of separate runs linked by `train.py --parent <model_id>` (policy and VecNormalize statistics carried over, lower learning rate via `ppo_finetune`), so each stage is evaluated and registered on its own: R001 (Training Ground) → R002 (generated wide) → R003 (final mix). R004 trains on the final mix from scratch for comparison.
  **What happened (2026-10-10/11):** the chain stopped after stage 2. R002 (fine-tuned from R001@1200k on generated wide tracks) **did not generalise** (10 % held-out; the agent still left the road), so stage 3 was dropped and R003 trained on `final_mix` **from scratch** instead (100 % showcase / 80 % held-out within ~1M steps). Training from scratch on mixed tracks has been the method since: the overnight ablations (R004–R014) and the T50 runs on `maps7` (R015, R016). Fine-tuning a good model on the same tracks (R017 from R015) didn't help either. `--parent` stays available; R002 is the documented negative result for the report.
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
  - **Protocol v2 (D20)** (`configs/eval/v2.yaml`) uses the map library: 7 training maps × 2 starts, 2 test maps × 3 starts, and the 10 generated held-out tracks × 1 start (30 episodes, about 50 s per model). New metrics: `test_completion` (test maps), `unseen_completion` (test maps + generated held-out; the generalisation number that matters for the faculty-drawn track), and best laps on every map. v1 stays frozen and keeps its numbers. New runs evaluate with v2; older models can be re-evaluated with v2 at any time (`evaluate.py --protocol v2`), which adds `eval_v2.json` next to `eval.json`.
- **Track set `maps7` (D20):** the 7 training maps (weight 1 each) + a generated pool with mixed widths (weight 7). It deliberately leaves out `tracks/user/`, so tracks drawn in the demo stay unseen.
- **Neuroevolution (D26, owner; plan in `docs/NEUROEVOLUTION_PLAN.md`):** a genetic algorithm as a second way to train the same network, with no gradients. Same outputs and 256×256 hidden layers as PPO's R015, but its own 10-input preset (below), saved as an SB3 policy so the loader, demo, evaluation and registry work unchanged. Compared with PPO as E10.
  - **Inputs (owner, T88–T90): 10.** 7 rays at 0° (200 m), ±5.5° (200 m), ±17° (120 m), ±55° (60 m); car direction vs track direction (sin, cos); forward speed. Yaw rate, on-road fraction, previous action and sideways speed are dropped (to be revisited if driving is worse). The 0° ray is needed for braking: top speed → the tightest corner takes 191 m, and the ±5.5° rays see only 52–83 m on a straight (T89). Obs preset `rays7_lean`. A new obs preset; PPO models are unaffected. E10 therefore compares inputs as well as methods unless a PPO run uses the same preset.
  - **Track limits (owner, T88):** a try **fails when the car's centre crosses the road edge** (half the car over the line; kerbs count as road, D1). No grass shortcuts, and the car never sees the off-road ray meaning (D2). An optional episode setting, off for every existing preset, so old models are unchanged.
  - **Generation:** every car drives the same map from the same start: **one of the 7 training maps, drawn at random each generation** (it may repeat). **Start (owner, 2026-10-24): always the start/finish line**, centred, aligned with the track, from standstill, on every map (`ga.start: line`, as in validation and evaluation). Runs before then used a random point, offset and speed (`ga.start: random`, the meaning of a snapshot without the key). A try ends when the car completes **one lap or fails** (any training termination, the track-limits rule, or a time cap of simulated time: the env's existing 4000 steps = 200 s, 2.5× the baseline's slowest lap of 81.6 s on Technical).
  - **Fitness:** the fraction of the lap covered if it failed; 1 + mean speed ÷ 70 m/s if it finished (a finisher always beats a non-finisher, a faster lap beats a slower one).
  - **Fitness v2 (D31, owner, 2026-10-28; `ga.fitness: 2`, preset `ga_v2`; v1 above stays the meaning of every older snapshot):** one speed-and-completion score on a continuous scale. A finisher's lap time is its real time; a car that fails is charged its time so far plus the unfinished distance at `fail_speed_frac` × `ref_speed` (default 0.3 × 25 m/s). With `ρ = (lap length ÷ that time) ÷ ref_speed`: `fitness = (e^{kρ} − 1) ÷ (e^k − 1) + finish_bonus × finished` (k = 2, bonus 0.5). So a fast crash late in the lap can outrank a slow finisher (the owner's choice: risk is rewarded), an early crash is not, and each extra bit of speed is worth more than the last. Selection ranks by order, so the convex curve changes the validation score and its plateau test, not who is picked. `ref_speed` is one fixed number (the baseline's mean speed, 24.95 m/s), not a per-track baseline lap time (shortcut).
  - **Staged fitness (D32, owner, 2026-10-28; `ga.fitness: 3`, preset `ga_staged`):** stage 1 uses fitness v1 (lap fraction / 1 + speed ÷ 70) until the cars complete laps consistently; stage 2 then uses fitness v2 with `fail_speed_frac` 0.1. Switch rule: at least `stage_min_finishers` (5) cars finish in at least `stage_share` (90 %) of the last `stage_window` (100) generations. It is one-way, decided from the lineage's per-generation finisher counts (so pause, resume, extend and restore agree), and at the switch the validation baseline (best score, smoothing window, patience counter) restarts because the score scale changes. Reason: R027 (fitness v2 from generation 0) never learned to finish a lap, because a failed car is charged at 7.5 m/s and slow early cars get no gradient.
  - **Mutation shrinks (D31):** `ga.sigma_floor` set → σ(g) = max(floor, σ ÷ (1 + ln(1 + g ÷ `sigma_g0`))), a function of the generation number only, so pause, resume, extend and lineage rebuild stay exact (the lineage meta stores the schedule). Preset `ga_v2`: 0.02 → 0.005, g0 = 100.
  - **Validation (D31):** `ga_v2` validates on 8 tracks, and the plateau test uses the mean of the last `validation.smooth` (3) validations. The extra check of the best checkpoints on a larger set is the existing post-run protocol-v2 evaluation of the 10 best-validated ones. One map per generation stays (noisy selection against overfitting, owner).
  - **Selection (owner, T100):** all 50 cars of the generation (including the previous elites, which drove again) are ranked by fitness. The top 25 % (12) go to the next generation unchanged. The same 12 are the only parents of the other 38: each child is one of them, picked at random, with σ × Gaussian noise added to every weight.
  - **Population:** sized to a run budget, not a fixed count: **50 cars, 12 elites** (≈ 1.8 s per generation on 5 processes, worst case, measured T87–T88).
  - **CPU (owner):** about 60 % CPU load, leaving room for other programs: 5 worker processes by default (measured 64–73 %; 4 = 52 %, 8 = 96 % for only ≈ 13 % more speed, because the CPU has 4 cores / 8 threads). Workers run at below-normal priority. The count is a setting and can be changed when resuming; it never changes the results.
  - **Validation and stopping (owner, T101–T107):** every 20 generations **all 12 elites** drive 5 validation tracks (generated from their own seed range, never trained on, not the test or held-out sets); the validation score is the best elite's mean. (Validating only the single-map winner swung between 0.47 and 1.35 in the σ test.) The run stops when that score hasn't improved by 1 % for 500 generations, or when the owner pauses it. No cap by default; **GA runs are budgeted in generations** (`--generations N`, `ga.max_generations`), PPO/SAC runs in env steps (`--steps`).
  - **Input normalisation (D33, owner, 2026-10-29):** PPO standardises its inputs with VecNormalize's running statistics; a GA car cannot, because the statistics would change under the population and a car could no longer be rebuilt from the lineage. With `ga.normalise: true` the run first drives the hand-coded baseline (clean and with action noise, random starts, every training map; seeded) to measure each input's mean and variance (a floor `ga.norm_min_std` on the spread), freezes them in the run's snapshot (`ga.obs_norm`), and standardises every input with VecNormalize's formula, clipped at `ga.clip_obs`. Exported cars carry the same numbers in a real `vecnormalize.pkl`. Default off: older snapshots mean no normalisation.
  - **Control run:** a short single-track run (Training Ground) to show overfitting next to E5.
  - **Recording (owner: limited disk space, no full-weight record of the run):** a lineage database `outputs/evolve/<run>/lineage.sqlite` (one row per car per generation: parent, noise seed, fitness, map, lap time, termination; ≈ 30 MB per 10,000 generations), from which any car can be rebuilt and exported; **a registered checkpoint at every validation (every 20 generations) and at the end: the elite with the best validation score** (a normal model folder; model.zip ≈ 0.3 MB). After the run, the 10 best-validated checkpoints and the last 10 are evaluated (protocol v2); the others can be evaluated later with `evaluate.py`. Later checkpoints are not reliably better (one map per generation: S004's last checkpoint 36 % seen, its best 100 %). **Any car of any generation** can be rebuilt from the lineage and saved as a model folder afterwards (`scripts/export_ga.py`, and in the sandbox), e.g. to hand-pick moments for an Evolution Race.
  - **Pause and resume (owner):** Ctrl+C or a `stop.request` file (as for the queue) lets the current generation finish, saves the run state (population, generation, best score, stopping counter, env steps; the random numbers come from the seed and the generation) to the git-ignored `outputs/evolve/<run>/`, and exits (code 3; the queue resumes it on its next start). Rerunning the same command continues from there, identical to an uninterrupted run. The state is also saved every 10 generations against crashes (the sandbox's *Stop queue* kills a run, which then resumes from there). A finished run is not restarted by the same command, but it stays **resumable (D29, owner)**: it keeps its `state.npz` (the next generation, bred before stopping), and `train.py --extend` continues it with a new `ga.validation.patience` and/or `max_generations`, identical to an uninterrupted run with those settings. A run that finished before D29 gets its state rebuilt from the lineage database. Model folders are never used for this. **Built (T92–T95):** a paused run resumed with another worker count was checked to be bit-identical to an uninterrupted one.
- **Logging:** TensorBoard (`runs/`) for episode reward, length, lap time and losses, plus `race/*` (lap time, laps, progress) and `end/*` (termination reasons).

---

## 9. Model registry and versioning

**Rule:** every model folder is **self-contained and immutable**. The demo must be able to load and run correctly any model ever trained, even after the code has moved on.

```
models/R015_t50_narrow7_s1@4600k/
├── model.zip            # SB3 policy
├── vecnormalize.pkl     # observation normaliser statistics
├── config.yaml          # FULL snapshot: obs_spec, physics, reward, train hparams, seed, git hash
├── eval.json            # metrics from evaluate.py
└── thumb_<track>.png    # trajectory coloured by speed
```

- **What "immutable" means (D10, owner):** `model.zip`, `vecnormalize.pkl` and `config.yaml` are never changed after saving. `evaluate.py` **adds** `eval.json` and `thumb_<track>.png` once per protocol version (a later protocol writes `eval_v2.json`; nothing is overwritten), and copies them to `outputs/eval/<model_id>/`. Models stay local (git-ignored); the copies in `outputs/eval/` and the registry CSVs are tracked. **Backups (T81):** after every completed run (or queue), `scripts/backup_models.py` copies whatever is missing to the backup folder in `configs/backup/default.yaml`; it never overwrites or deletes, so the backup is as immutable as the model folders (a GA run extended after its backup (D29) gets its changed files saved beside the old ones as `name.v2`, `.v3`).
- **Model ID:** `R{run:03d}_{short_tag}@{steps}k` (e.g. `R015_t50_narrow7_s1@4600k`). Ablation runs carry the experiment in the tag (D17, e.g. `R005_e2_nogrip_s1@3000k`). Neuroevolution runs get `GA_` before the tag (owner, T100: `R{run:03d}_GA_{short_tag}@{steps}k`, e.g. `R020_GA_main_s1@96000k`), so the id itself shows the genetic algorithm; steps = env steps used so far.
- **Observation specs are versioned** (`obs_spec: {version: 1, rays: [...], extras: [...]}`). Never change what an existing version means; add a new version instead. The loader builds each car's sensor from that car's own spec.
- **Upgradable engine (D23).** Everything that defines a model is versioned and looked up in a registry: obs spec version and features (§7.1), action layout (§7.2), physics version (§5.2) and algorithm (`racer/agents/algorithms.py`: PPO, SAC; `ga` for neuroevolution once built, D26). New snapshots carry a **manifest** (`obs_dim`, `action_dim`, `n_params`, `net_arch`, the versions above, input labels); for older snapshots the loader derives it. Before loading, the engine checks the manifest against what it supports: a model that needs something the engine doesn't have yet (e.g. an unknown feature) is listed with the reason and greyed out in the picker, instead of crashing the demo. To support it, register the missing piece; existing models are never touched.
- **Comparing models:** the registry row of a model holds the metrics of the newest protocol it was evaluated with (`eval_protocol` column); compare models only within one protocol. `runs.csv` also records `obs_dim`, `action_layout` and `n_params`, so input count and network size can be compared directly.

**`registry/runs.csv`:** one row per training run
`run_id, created_at, tag, algorithm, parent_model, obs_spec, physics_preset, reward_preset, track_set, curriculum, total_steps, wall_clock_min, seed, hparams_json, notes, obs_dim, action_layout, n_params`

**`registry/models.csv`:** one row per saved checkpoint
`model_id, run_id, steps, path, is_best, completion_rate, crash_rate, heldout_completion, best_lap_<showcase1..3>, mean_speed, offroad_pct, mean_lat_g, overall_completion, lap_ratio, score, eval_protocol, eval_date, tags, notes, test_completion, unseen_completion, best_lap_<6 library maps>`
(`completion_rate` = showcase episodes in v1, training-map episodes in v2; `heldout_completion` = generated held-out episodes; `test_completion` = test-map episodes (v2); `unseen_completion` = test + held-out (v2); `overall_completion` = all; `lap_ratio` = best lap ÷ the baseline's, averaged over the lap tracks the model lapped; `mean_speed` in m/s. Columns are only ever appended.)

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
2. **Evolution Race (built, T126):** pick one run and its checkpoints (every N-th, N evenly spaced, or the N best by score, then add or remove by click) and they race together, coloured blue (early) to red (late). Shows learning in about 20 seconds. `demo.py --run R0xx --spaced 6`.
3. **Draw-a-Track:** sketch a track, validate it, and pick models to drive it live. This demonstrates generalisation. **Built as the sandbox's *Draw a Track* page (owner, T119)**, see §6 Editor.
4. **Human vs AI (built, T127):** keyboard (WASD or arrows) or gamepad against chosen models. Ghost cars. Your car's physics preset is your choice (Shift+C in the picker); an ABS assist caps the all-or-nothing keyboard brake inside the grip limit (`K` toggles). `demo.py --human [PHYSICS]`. The sandbox's Launch page has *Drive it yourself*.
5. **Analysis view (built as the sandbox *Workbench*, T135-T137, D28):** racing line coloured by speed, speed/gas/steering traces against lap distance, g-g diagram, braking zones, lap tables, the registry as a plottable data set, training curves and a scrubber showing what the network does at each moment. Charts are interactive (Altair); `scripts/make_plots.py` writes the same figures as PNGs.
6. **Swarm view (built, T132):** one model (or the baseline) as N cars (default 24, up to 60) scattered over the track at random places and speeds, whole-track view. Early-training chaos shots. `demo.py --swarm 30 --models R0xx`.

### 10.2 Visual features (all effects code-generated)

| Feature | Details |
| --- | --- |
| World | Tiled grass, a gravel run-off band beside the road, asphalt with white edge lines and a slow brightness drift that hides the tiling, red and white kerbs, a chequered start/finish line and a red/white barrier with a shadow. Rasterised from the physics field at 6 px/m **once per track** (3-4 s), then blitted every frame. `configs/demo/render.yaml: textures`; missing files fall back to flat colours. |
| Car | Kenney sprites (`car_{1..4}_{colour}.png`). For more than 5 distinct colours, use `car_N_tint.png` + `car_N_tint_mask.png` (tint the body only; see `assets/LICENSES.md`). Brake lights glow when braking. Small drop shadow. |
| Rays | Lines from the car, coloured green → yellow → red by distance, with a dot where each ray hits. Shown for the selected car only (20 cars' rays would be clutter); toggle with `R`. |
| Trail | Fading trail behind each car, coloured by speed (blue = slow, red = fast). Toggle `T`. |
| Racing line overlay | Best lap path coloured by speed, with the centreline dashed for comparison. Toggle `L`. |
| Skid marks | Rear wheels sliding on tarmac leave dark marks, baked into the world surface as they are made (they last for the race). `E` toggles all effects. |
| Smoke, dust, sparks | Particles: grey smoke when sliding on tarmac, brown dust on grass, orange sparks and a camera kick when the followed car crashes (a pooled numpy field of 700; `arena.yaml: effects`). |
| HUD | Dashboard of the followed car (rpm arc with redline, speed, gear, lap/last/best, lateral g, gas/brake/steer bars), leaderboard, three-light start sequence, results screen (medals, fastest lap in purple, `Enter` to race again), key list. Fonts: Rajdhani and Orbitron. |
| Brain panel | `B`: input bars with their labels, each hidden layer as a grid of neurons (one square each, blue negative, orange positive), steering wheel and gas/brake outputs, lines that brighten with activity. Activations come from the real forward pass (`racer/agents/activations.py`). |
| Camera | Follow the selected car (smooth lerp, looks ahead, zooms out by up to 30 % with speed), overview mode, cycle cars with `Tab`, crash shake |
| Minimap | `N`: corner minimap with a dot per car |
| Scenery | Kenney props outside the barrier: grandstands by the start line, tyre walls on the outside of corners, tents and tree clumps, with drop shadows. Placed from the track geometry (deterministic per track, never inside the barrier). The start sequence is three lights in the HUD. |
| Recording | **Dropped (owner, D28).** The headless `--screenshot` mode stays for captures. |

**Performance target:** 60 FPS with 20 cars. Inference is batched per model and runs every third frame (20 Hz). **Measured (T133, headless, brain panel, effects, minimap and audio on):** 20 trained cars 112 fps, 30 cars 93 fps; 20 baseline cars 215 fps. `scripts/bench_demo.py` repeats it; a test asserts at least 45 fps.

### 10.3 Controls (summary)

`Tab` cycle car · `R` rays · `T` trails · `E` effects · `L` labels · `B` brain panel · `N` minimap · `F` follow/overview · `M` sound on/off · `Space` pause · `+/−` speed ×0.25–×4 · mouse wheel zoom · `Enter` race again after the finish · `H` key list · `Esc` back. Human vs AI adds arrows/WASD to drive and `K` for ABS. **`M` is sound (not `S`, which is the brake in WASD) and `N` is the minimap (D28, T127).**

Every game mode shows its keys on screen (bottom right, toggled with `H`; T82), so the demo never depends on remembering them. The arena list is built next to its key handling, and a test checks that they match.

### 10.4 Audio (`racer/render/audio.py`, `configs/demo/audio.yaml`)

All clips and their RPM curves are described in `assets/LICENSES.md` and `assets/sounds/engine_map.json`.

**Defaults (owner, D28):** sound is **on when watching a race** (arena, Evolution Race, Human vs AI, swarm), **off for training, the sandbox and analysis**, and toggleable in every game window with `M` (`demo.py --sound/--no-sound`; the sandbox's Launch has a Sound checkbox; headless `--screenshot` runs are silent unless `--sound`). Without an audio device or the asset files the game runs silently. The title screen carries the CC BY 4.0 credit line. Built in `racer/render/{gearbox,engine_sound,audio}.py`, all numbers in `configs/demo/audio.yaml`.

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

**Recording:** dropped with the video recorder (D28).

### 10.5 AI Sandbox (D24, owner; plan in `docs/SANDBOX_PLAN.md`)

A **standalone** local app (Streamlit, `scripts/sandbox.py`), separate from the game, shown in **its own desktop window** (D25: pywebview, a "mini browser" on Edge WebView2; `--browser` opens a browser tab instead). Its main purpose is to **understand a model**:

- **Model explorer:** pick any registered model and see what it was trained with: its inputs (ray diagram, feature list), network size, physics, reward terms, track set and hyperparameters. Each parameter has a short "what it affects" note, and each run its notes on why it was made.
- **Compare:** 2–4 models side by side: config differences highlighted, evaluation metrics, learning curves and thumbnails.
- **Configure and queue:** build a new run from presets (add or remove inputs, network size, steps, track set, …), see the resulting input count and parameter count, and add it to a queue file. The queue still runs through `queue_overnight.py`.
- **Launch:** pick models from the trained list and press *Launch* to open a game window (`demo.py`) racing them. Several game windows can run at once, each a separate process.
**Every feature shows in the sandbox (D27, owner):** any feature added to the project gets a way to see, explain or launch it here; a feature isn't done until it does.
The sandbox only reads the registry and model folders and writes queue files and drawn tracks (`tracks/user/`, T119) (and, on request, a new model folder exported from a GA lineage, registered in the registry); it never edits a model.

**Owner decisions (2026-10-11, T65):**

- Streamlit 1.65.0. It needs no change to the existing pins.
- **Localhost only**, with no telemetry and nothing written to the user profile. Run it with `.venv\Scripts\python scripts\sandbox.py`.
- The sandbox **may start the queue**: *Start queue* opens `queue_overnight.py` in its own console. It is refused while another queue or training is running.
- The "why" and outcome of each run live in a tracked `registry/run_notes.yaml`. Snapshots can't change, so old runs are filled in from §11. New runs get their "why" through `train.py --notes`.
- Build steps S0–S6 are tasks T66–T72.
- **Later the same day (T66):**
  - New runs default to **real runs**; a *Trial* checkbox makes a scratch run.
  - The sandbox may **stop a running queue**, after a confirmation step: it writes a `stop.request` next to the queue's `state.json`, which acts like Ctrl+C.
  - **Outcome notes can be edited** in the Explorer.

**Built (T66–T72, 2026-10-11).** Pages:

- **Overview:** leaderboard, plus what is running.
- **Model Explorer:** rays on a straight road and on any map by the model's own sensor, network diagram and layer table, every setting with its glossary note, results for each protocol, learning curves.
- **Compare**, **Build & Queue** (D30: six guided steps with a pinned review, quick-pick presets from real files in `configs/{rays,readings,memory,network,learning,budget}/`, suggested tag and reason, decimal inputs, plain-language '?' help), **Launch**, **Glossary**. Build & Queue's rays table has a *Remove* column and button (T134).
- **Workbench (T135-T136, owner, D28):** plots from the same Python runtime, no export step: the registry as a data set (any two metrics, colour by algorithm, CSV), result bars for chosen models, training curves, and fresh laps driven in the page (up to 4 drivers) with a lap table, racing line coloured by speed, traces with shared zoom, g-g diagram, braking zones and CSV; a *Brain* tab scrubs a lap and shows the network's inputs and neurons at each moment. The Overview has default plots. Logic: `racer/agents/telemetry.py`, `racer/sandbox/plots.py`.
- **Launch** has three tabs: race chosen models, an Evolution race (strategies as in the game) and **Drive it yourself** (a game window with your car on a physics preset against chosen models; Streamlit cannot read 60 Hz keys), each with a Sound checkbox.
- **Draw a Track (T119, owner, D27):** a drawing canvas (`streamlit-drawable-canvas`), the drawing shown as a validated track, *Save* to `tracks/user/`, *Race on it* (Launch with the track chosen).
- **Presets (T101, owner):** every model preset (inputs, car, reward, algorithm and network, track sets) with its values and notes, the rays to scale or the track previews, and the runs trained with it.
- **Explorer → Evolution (T106, owner):** for neuroevolution models, fitness and validation per generation from the lineage, *Save as a model* for any generation's car, *Save a range of generations* (one car each, up to 100 at a time, for an Evolution race; D29), and *Continue this run* (a queue entry for `train.py --extend` with a new patience, budget and worker count; the restore itself stays in `train.py`, so the sandbox writes no run state). This is the one place where the sandbox writes a model folder: a new one, rebuilt from the lineage; it never edits an existing one.

How it is built:

- All logic lives in `racer/sandbox/` without Streamlit, so it is unit-tested; the pages are tested headless with `streamlit.testing`.
- Explanations come from one glossary (`racer/sandbox/glossary.yaml`) plus the code registries. A test fails if any key of any preset or saved snapshot has no explanation.
- **Queue runner:** writes `runner.json` while it runs, records each entry's PID, and stops a run's whole process tree.
- **Game windows** are separate `demo.py --title …` processes, matched by PID + creation time.
- **Tunable numbers** live in `configs/sandbox/default.yaml`.
- **Window (D25, T75):** `scripts/sandbox.py` starts the Streamlit server, then opens a pywebview window on it. Closing the window stops the server (game windows and queues it started keep running). WebView2 runs in private mode with its data folder in the git-ignored `outputs/sandbox/webview/`, so nothing goes to the user profile. If the server is already running, it only opens another window.

---

## 11. Experiments and analysis (for the report and slides)

| # | Experiment | Expected story |
| --- | --- | --- |
| E1 | Main run: PPO, rays5, grip on, curriculum | Learning curves; lap time falls; a racing line appears |
| E2 | **Grip limit off** (`no_grip_limit`) | Flat-out everywhere and no braking, so no real racing line. **Shows that the physics is what creates the behaviour.** |
| E3 | **Naive speed reward** | Reward hacking: circling or weaving. A funny video with a serious lesson. |
| E4 | Rays: 5 vs 9, and short vs long forward range | Short rays mean late braking and crashes: the agent needs to see ahead |
| E5 | Single-track vs multi-track training | Overfitting: perfect on its own track, fails on held-out tracks |
| E6 | Rays only vs rays + curvature look-ahead | How much map knowledge helps |
| E7 | PPO vs SAC | Sample efficiency and stability compared |
| E8 | RL vs hand-coded baseline (a PID controller that follows the centreline at a fixed safe speed) | RL is faster because it uses the track width and brakes at the right time |
| E9 (optional) | Downforce on vs off | A different racing line on fast corners |
| E10 | PPO vs neuroevolution (D26), same 256×256 network | Gradients vs selection + mutation: steps needed, wall-clock time, lap time, and whether evolution generalises beyond its training track |

**Runs so far (registry ids; best checkpoint by D9, evaluated with protocol v2: seen = 7 training maps, unseen = 2 test maps + 10 generated held-out tracks; lap ratio vs the PID baseline over the maps the model laps):**

| Run(s) | Experiment | Setting | Seen | Unseen | Lap ratio | Note |
| --- | --- | --- | --- | --- | --- | --- |
| BASE_pid | E8 | hand-coded centreline + speed plan (sees the map) | 100 % | 100 % | 1.00 | reference |
| R001 | M1 | rays5, Training Ground only | 29 % | 19 % | 0.94 | fast on its own track only |
| R002 | D8 stage 2 | fine-tune of R001 on generated wide tracks | 21 % | 6 % | 1.11 | negative result |
| R003 | D8 | rays5, `final_mix`, from scratch | 57 % | 50 % | 1.19 | reliable on v1, slow |
| R004 / R010 | E1 reference (seeds 1/2) | rays5, `final_mix`, 3M | 43 / 43 % | 50 / 38 % | 1.16 / 1.18 | |
| R005 / R011 | E2 | no grip limit | 100 % | 100 % | 0.40 | other physics: not comparable |
| R006 / R012 | E5 | Training Ground only | 14 / 14 % | 0 / 0 % | 0.92 / 0.91 | overfits its one track |
| R007 / R013 | E4 short | forward ray 60 m | 29 / 43 % | 63 / 56 % | 1.26 / 1.18 | slowest: brakes late, drives slowly |
| R008 / R014 | E4 rays9 | 9 rays (±10°, ±30°, ±60°, ±90°) | 43 / 64 % | 63 / 69 % | 0.97 / 1.03 | best of the overnight runs |
| R009 | E3 | naive speed reward | 36 % | 63 % | 1.14 | reward-hacking demo |
| **R015** | T50 | **rays7_narrow, `maps7`, 5M** | **64 %** | **88 %** (@4800k 94 %) | **0.97** | **meets T50; faster than the baseline on the test maps** |
| R016 | T50 | rays9, `maps7`, 5M | 50 % | 75 % | 0.99 | |
| R017 | T50 follow-up | fine-tune of R015@4600k on `maps7` | 57 % | 88 % | 0.97 | no gain over its parent |
| R018 | R015 reliability (seed 2) | rays7_narrow, `maps7`, 5M | 64 % | 63 % (test maps 0 %) | 1.00 | held-out 100 %, but never finishes the test maps |
| R019 | R015 reliability (seed 1, 8M) | rays7_narrow, `maps7`, 8M | 14 % | 0 % | 0.92 | failed to learn: stuck at ~50 % barrier crashes |
| R020 | E10 control | GA, Training Ground only, 300 generations (track limits on) | 14 % | 6 % | 1.02 | overfits like E5 |
| R021 | E10 | GA seed 1, `rays7_lean`, 7 maps, 46.3M steps (gen 919) | 93 % | 75 % | 1.05 | test maps 50 % |
| R022 | E10 | GA seed 2, 45.3M steps (gen 899) | 86 % | 94 % | 1.07 | test maps 100 % |
| **R023** | E10 | **PPO under GA rules** (`rays7_lean`, track limits, 7 maps), 5M steps | **100 %** | **100 %** | **0.975** | one seed; best of 50 checkpoints |

**Neuroevolution, first results (T96, scratch runs S004/S005, 2026-10-15; not in the registry):** σ 0.02 vs 0.05, 10M env steps each (≈ 30 min each on 5 processes, rays7_lean, 7 training maps, track limits on). σ 0.02 won: best checkpoint (S004@8738k, generation 199) **100 % seen, 87.5 % unseen, lap ratio 0.997**, against 86 % / 62.5 % / 1.083 for σ 0.05; on average 28 of 50 cars finished their lap in the last 50 generations (σ 0.05: 16). One seed each, and the best checkpoint varies a lot from one checkpoint to the next (S004@10017k: 36 % seen), so these are early, single-seed numbers. The real runs were then run through `configs/queue/e10_ga.yaml` (T97); results below.

**E10 results (2026-10-24, queue `e10_ga`, evaluated with protocol v2 under each model's own rules: track limits on for all four runs; registry R020–R023):**
- **PPO beat the GA.** R023 (PPO, same 10 inputs, same rules, same 7 maps) reached 100 % seen and unseen with lap ratio 0.975 at its best checkpoint (@1700k), against 93 % / 75 % / 1.05 (R021) and 86 % / 94 % / 1.07 (R022) for the GA seeds. PPO used 5M env steps (46.6 min); the GA runs stopped on their validation plateau after about 46M and 45M steps (45 and 35 min). So the GA needed about 9× more env steps for similar wall-clock time, because its steps are cheap (no gradient updates), and still drove slower laps.
- **Control (R020):** GA on Training Ground only overfits as PPO did in E5: 14 % seen, 0 % on the test maps, 10 % on held-out.
- **Caveats:** one seed per method. The two GA seeds disagree on the test maps (50 % vs 100 %), which repeats the lesson of R015/R018. Each model's best checkpoint was picked from many evaluated ones (GA 18, PPO 50), so the numbers are optimistic, PPO's slightly more so. The GA's best evaluated checkpoint (R021@46282k) was its last one, not the best-validated one, so validation did not predict evaluation, as in S004. GA tries start on the start line (T115) and PPO's at random points; evaluation is the same for both. R015 (no track limits, `rays7_narrow`) is not a like-for-like comparator.
- **Open question for the owner:** more seeds (a second PPO seed of R023 is the cheapest test of whether PPO's 100 % is luck).

Results are one seed per setting unless two are listed. **R015's result doesn't reproduce reliably (R018, R019, 2026-10-11).**

- **Seed 2 (R018):** the same setup matches R015 on the seen maps (64 %) and the generated held-out tracks (100 %), at the baseline's lap time. But it finishes none of the test-map episodes at its best checkpoints (2 of its 50 checkpoints manage 17–33 %), where R015 managed 67 %.
- **8M steps (R019, seed 1):** a longer schedule changes the learning-rate decay from the first update, so it doesn't repeat R015's path either. It got stuck early: episode reward ≈ 500–600 against R015's > 1700, and about half its episodes ended at the barrier. Its exploration noise then collapsed (policy std 0.04) and it never recovered.
- So narrow rays on the map library *can* reach the T50 target, but the result varies strongly between runs. Claims need several seeds.

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
| --- | --- | --- |
| **1** (Oct 10–16) | Physics, surfaces, spline tracks, generator, progress, sensors, Gymnasium env, physics sanity tests, minimal debug renderer, first PPO run | **M1:** the car completes laps on a showcase track |
| **2** (Oct 17–23) | Config system, train/evaluate scripts, checkpoints, **registry and xlsx export**, loader, curriculum and multi-track training, **basic Model Arena with 20 cars** | **M2:** load any 20 models from the registry and race them |
| **3** (Oct 24–30) | **First: AI Sandbox (§10.5, owner priority, right after M2, T50 and T51; built 2026-10-11, T66–T72).** Then textures, effects, HUD, brain panel, minimap, camera, Evolution Race, Draw-a-Track, Human vs AI, video recorder | **M3:** full live demo |
| **4** (Oct 31–Nov 7) | Ablations (overnight queue from week 2 onwards), plots, telemetry/analysis view, highlight video, slides, README, rehearsal, buffer | **M4:** ready to present |

**Keep the CPU busy every night from week 2:** queue the experiment runs.

**Priorities (MoSCoW):**

- **Must:** grip physics env, PPO training, registry (CSV and xlsx), load any model, multi-car arena (10–20), rays, trails, HUD, learning curves, showcase tracks, held-out evaluation, map library with a test split (D19), AI Sandbox (D24)
- **Should:** Evolution Race, textures and effects, Draw-a-Track, Human vs AI, ablations E2–E5, telemetry trace, engine and tyre audio (§10.4) (all built; highlight video and slides dropped, D28)
- **Could:** brain panel (built), swarm view (built), SAC (E7), neuroevolution (E10, D26; built, run: PPO won, §11), curvature (E6), downforce (E9), trackside scenery, audio in recorded videos, gamepad
- **Won't (this time):** car-to-car collisions, multi-agent competitive training, pixel/CNN observations, 3D

---

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Reward hacking (backwards laps, exploits at the start line) | Signed `Δs` with wrap-around handling; wrong-way termination; watch videos of every new run |
| Agent learns to stop or crawl | Time penalty, no-progress termination, random starting speed |
| Twitchy steering looks bad | Steering rate limit, ` | Δsteer | ` penalty, previous action in the observation |
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
pytest==9.1.1
streamlit==1.65.0    # AI Sandbox (T66, 2026-10-11): +26 packages, no change to the pins above
psutil==7.2.2        # AI Sandbox process checks
pywebview==6.2.1     # AI Sandbox window (D25, T75): Edge WebView2, ships with Windows 11; +6 packages, no change to the pins above
streamlit-drawable-canvas==0.13.0  # Draw a Track page (T119): MIT, needs only streamlit>=1.53; no change to the pins above
# torch==2.14.1+cpu is installed separately from the CPU index (above)
```

### 14.2 Assets (in place; audited 2026-10-10)

The assets are downloaded, renamed and pruned. **`assets/LICENSES.md` is the manifest:** it lists every file, its source and licence, how it was cut or modified, and how to use it. Read it before writing rendering or audio code.

| Folder | Contents |
| --- | --- |
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
| --- | --- |
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
| 2026-10-11 | Owner decisions (`log.md` T52–T59): **D19** map library of 9+ hand-designed maps: 7 for training (3 showcase + 4 in `tracks/train/`), 2 for testing in `tracks/test/` (never trained on), the 10 generated held-out tracks kept (§6); **D20** track set `maps7` and evaluation protocol v2 with `test_completion` / `unseen_completion`, v1 frozen (§8); **D21** `obs_spec` v2 built from named features, so inputs can be added or removed (§7.1); **D22** versioned action layouts, `steer_gas` default (§7.2); **D23** upgradable engine: physics version, algorithm registry, model manifest (inputs, outputs, parameter count) and a compatibility check, so every model either runs or says what it needs (§5.2, §9); **D24** AI Sandbox: a standalone Streamlit app to understand, compare, configure, queue and launch models, built first in Week 3 (§10.5). Work done on branch `feat/maps-modular-sandbox` while the `night1` queue finished on `main`. |
| 2026-10-11 | **M2 reached** (T07: T37–T48 done): any registered model loads and 20 of them race at a steady 60 FPS; the night1 ablations are registered (§11 "Runs so far"). The D8 curriculum chain was replaced by training from scratch on mixed tracks after R002's negative result (§8). |
| 2026-10-11 | **Results (owner-approved runs):** 7 narrow rays (`rays7_narrow`) on `maps7` (R015) meet the new T50 target (v2 lap ratio < 1.0, unseen completion ≥ 80 %): R015@4600k is faster than the PID baseline on the unseen test maps; earlier models had 0 % on them. 9 rays on `maps7` (R016) and fine-tuning R015 (R017) did worse. The arena spreads model decisions over substeps (steady 60 FPS with 20 different models, T51). **Training stays on the CPU** (the GPU was slower, T62); parallel runs on the CPU are next (T64). |
| 2026-10-11 | **AI Sandbox decisions (owner, T65):** Streamlit 1.65.0 (no pin changes), local only (localhost), the sandbox may start the queue in its own console, run "why" and outcome notes in `registry/run_notes.yaml`. Plan v2 in `docs/SANDBOX_PLAN.md`; build tasks T66–T72 (§10.5). |
| 2026-10-11 | **AI Sandbox built (T66–T72).** Owner decisions taken during the build: the Build form defaults to real runs; the sandbox may stop a running queue (after a confirmation step); outcome notes can be edited; critical issues are skipped and logged while the owner is away (none came up). The queue runner gained `runner.json`, per-entry PIDs, `stop.request` and process-tree stopping (§8). `demo.py --title`. **Reliability runs for R015's setup (owner)**, queued through the sandbox: R018 (seed 2, 5M) and R019 (seed 1, 8M). |
| 2026-10-11 | **R015 doesn't reproduce reliably (T73).** R018 (seed 2): seen 64 %, held-out 100 %, test maps 0 %, lap ratio 1.00. R019 (8M): stuck early, best overall 7 %. The T50 target was met by one lucky run, not by a robust setting (§11). **Open question for the owner:** what to do about the variance, e.g. more seeds per setting, keeping exploration up (entropy), or the curriculum. |
| 2026-10-11 | **Model backups after every run (owner, T81).** The backup had stopped at R001. `scripts/backup_models.py` (additive: copies what is missing, never overwrites or deletes, refuses while training/queue/evaluation runs) is run after every completed run or queue; the rule is in `CLAUDE.md` and the README. Backup brought up to date: R001–R019, 690 folders (§9). The variance question is task T83 for a later session. |
| 2026-10-11 | **D25 (owner): the AI Sandbox opens in its own desktop window** (pywebview 6.2.1 on Edge WebView2, a "mini browser") instead of a browser tab. Streamlit stays the app, so pages and tests are unchanged; `scripts/sandbox.py --browser` keeps the browser tab as a fallback. WebView2 data stays in `outputs/sandbox/webview/` (git-ignored), nothing in the user profile (§10.5, T74–T75). |
| 2026-10-13 | **D26 (owner): try neuroevolution next to PPO** (a genetic algorithm over the network weights, as in a YouTube racing video): 20 cars per generation, fitness = distance in a fixed time, the best kept and mutated. Same 256×256 network as PPO. Training Ground first, then multi-track; plan and a speed benchmark before building (§8, E10 in §11, T84–T86). |
| 2026-10-13 | **D26 refined (owner, T87):** mixed maps from the start (K maps per generation, changed each generation) instead of Training Ground first, because single-track training memorises (E5, R002); a single-track run stays only as a control. A try ends at one lap or a failure (score: lap fraction, or 1 + mean speed ÷ 70 m/s for a finished lap). Elites = top 25 %. Population sized to a run budget. Stop on a validation plateau; no generation cap by default, since training is pausable (an optional cap for queue entries). Network stays 256×256 (PPO's R015 generalises with it); a bigger one only if the training score plateaus low. Mutation size σ: chosen by two short runs. Measured: ≈ 39k env steps/s on 8 processes, ≈ 4.6 s per generation for 50 cars × 3 maps × 60 s (§8). Training can be **paused at any moment** (the generation in progress finishes and the state is saved) and **resumed** by rerunning the same command. |
| 2026-10-13 | **D26, maps per generation (owner, T88):** **one random map per generation**, from the 7 training maps, instead of K maps: 3× more generations for the same time, noisier selection (the population can chase the current map); a slower plateau is accepted because training is pausable. Generated tracks are used only for validation. **CPU:** 5 worker processes by default (≈ 60–70 % CPU load; owner target 60 %), below-normal priority, changeable on resume. **Inputs:** 10 (6 rays at ±5.5°/±17°/±55°, heading sin/cos, forward and sideways speed); yaw rate, on-road fraction and previous action dropped, to be revisited if driving is worse. **Track limits:** the try fails when the car's centre crosses the road edge (optional episode setting, off for existing presets). More maps will be added over time. |
| 2026-10-13 | **D26 inputs final (owner, T89–T90):** a 0° ray at 200 m added (braking from top speed to the tightest corner needs 191 m; ±5.5° rays see 52–83 m on a straight, ±5° only 10 % more); sideways speed dropped, so the inputs are 7 rays, heading sin/cos and forward speed (10). Every car of every generation is to be recorded in a lineage database (proposal, `docs/NEUROEVOLUTION_PLAN.md`). |
| 2026-10-13 | **D26 ready to implement (owner, T91):** 50 cars (12 elites); validation = 5 generated tracks every 25 generations, stop after 500 generations without a 1 % gain; 200 s simulated time cap per try; lineage database + a checkpoint every 100 generations, no full-weight record of the run (disk space). Decisions that were uncertain (dropped inputs, one map per generation, σ) are revised if results show them to be poor. Build steps S1–S7 are tasks T92–T98. |
| 2026-10-15 | **D26 selection and model id (owner, T100):** all 50 cars are ranked; the top 25 % (12) go to the next generation unchanged, and the same 12 are the only parents of the other 38 (each child = one of them at random + σ × noise). Neuroevolution model ids carry `GA_` before the tag (`R020_GA_main_s1@96000k`). Implementation starts (T92). |
| 2026-10-15 | **D26 owner decisions (T101–T107):** GA runs are budgeted in generations (`--generations`, `ga.max_generations`), PPO/SAC in steps; validation every 20 generations of all 12 elites (score = the best elite's mean); a checkpoint at every validation = the best-validated elite; after a run evaluate the 10 best-validated + the last 10 checkpoints (evaluating only the last ones was rejected: later isn't reliably better); a tool (CLI + sandbox) exports any car of any generation from the lineage. CLAUDE.md gets an agent workflow and architecture section; README a preset list. |
| 2026-10-15 | **σ = 0.02 (T96):** at the same 10M-step budget it beat 0.05 on every measure (100 % vs 86 % seen, 87.5 % vs 62.5 % unseen, lap ratio 0.997 vs 1.083); `ga_default.yaml` keeps 0.02. |
| 2026-10-15 | **D26 built (T92–T95); owner decisions for the runs (T96):** the σ test runs now, as two scratch runs with the same env-step budget (10M each, ≈ 200–300 generations; equal compute is the fairer comparison than equal generations); the control and main runs go into a queue file the owner starts. **E10 judging:** GA checkpoints are evaluated and raced under their own rules (track limits on), i.e. stricter than PPO, which may cut onto the grass for up to 1 s; the E10 write-up states this. |
| 2026-10-24 | **D26 start rule (owner, T115):** GA training tries always start on the start/finish line (centred, aligned, standing) on every map, instead of a random point, offset and speed. New setting `ga.start` (`line` in `ga_default`; a snapshot without it means `random`, so older runs resume unchanged); the map is still drawn at random each generation, so the map sequence is unchanged. PPO training starts are unchanged (§7.4). |
| 2026-10-24 | **D27 (owner): every feature is reflected in the AI Sandbox** (a page, launch option or glossary entry, with a headless page test); a feature isn't done until it is. Rule in `CLAUDE.md` (its own section, the workflow and the parts table) and §10.5. |
| 2026-10-24 | **Draw-a-Track in the sandbox (owner, T119):** drawn *inside* the sandbox on a canvas (`streamlit-drawable-canvas` 0.13.0, a new pinned package) rather than a button opening a game-window editor; saved to `tracks/user/` (only the old `final_mix` trains on that folder; `maps7`/`train7` don't). The drawing → track logic is `racer/track/sketch.py` (T111), pygame-free, so a game-window editor can reuse it. |
| 2026-10-24 | **E10 results (T121):** under identical inputs and rules, PPO (R023, 5M steps) beat neuroevolution (R021/R022, about 46M steps): unseen 100 % vs 75 % / 94 %, lap ratio 0.975 vs 1.05 / 1.07. Single seeds, best-of-many checkpoints, so claims need more seeds (§11). No design rule changes. |
| 2026-10-24 | **D28 (owner, T123): finish the demo and analysis work first, then experiment freely.** In: mode menu, Evolution Race, Human vs AI, swarm view, brain panel as a live network graph, audio in every race window (on for watching races, off for training/sandbox/analysis, toggle everywhere), real polish, a sandbox *Workbench* for live plots, *Drive it yourself* from the sandbox, delete in the rays table. Out: the game-window drawing editor (the sandbox page stays), the video recorder, the slides and highlight video. Plan: `docs/DEMO_PLAN.md`. |
| 2026-10-24 | **D29 (owner, T139): GA runs stay resumable after they finish.** The patience stop can come too early, and a new run ("main 2") would throw away the generations already driven. A finishing run now breeds the next generation and keeps `state.npz` (about 14 MB); the backup copies it; `train.py --extend` continues the run with a new patience or generation budget (only the stopping rules may change, so the continuation equals an uninterrupted run); runs finished earlier (R020–R022 and R024) are restored from their lineage. Sandbox: *Continue this run* (Explorer → Evolution) and a bulk *Save as a model*. Plan: `docs/GA_EXTEND_PLAN.md`. Backup (owner): a changed file of an extended run is saved next to the old copy as `name.v2`, `name.v3`, … (`backup(versioned=True)` for evolve runs), so nothing is overwritten. |
| 2026-10-29 | **D33 (owner idea): `rays2` and GA input normalisation** (§7.1, §8). Reason: every model so far laps at 0.96–0.98 of the baseline with 0.55–0.7 g of ~1.1 available (docs/PPO_RACEPACE_PLAN.md); the cars drive in a tunnel and cannot see a corner past the first road edge. The curvature look-ahead was rejected as unfair to a real driver. Race-pace PPO runs use reward `racepace`, inputs `rays7_pen` (rays through the edge) and a matching GA with frozen input normalisation. |
| 2026-10-28 | **D31 (owner): GA fitness v2, shrinking σ, smoother validation** (§8). Reason: R025/R026 (4000 generations each) plateaued at validation ≈ 1.30 from generation ~500; the v1 score has a cliff between failing and finishing and a flat speed term. Opt-in through a new preset `ga_v2` (`ga.fitness: 2`, `sigma_floor`, `validation.smooth`); older snapshots mean v1, constant σ, no smoothing. Plan: `docs/GA_FITNESS_V2_PLAN.md`. |
| 2026-10-28 | **D32 (owner): staged GA fitness** (§8). R027 (fitness v2 from the start) never finished a lap. `ga.fitness: 3` = fitness v1 until cars complete laps consistently (5 of 50 finishing in 90 % of the last 100 generations), then fitness v2 with `fail_speed_frac` 0.1 (to be reviewed, REVIEW_LATER.md). Opt-in preset `ga_staged`. |
| 2026-10-26 | **D30 (owner, T154): Build & Queue polish.** The page becomes guided steps (inputs, network, learning, world, budget, name) with a pinned summary. Every step gets one-click **quick-pick presets as real files**: `configs/rays/`, `configs/readings/` (input bundles), `configs/network/`, `configs/learning/` (per algorithm), `configs/budget/` (steps or generations, seeds, trial). They only fill the form (a run still stores the full resolved settings), are listed on the Presets page, and old presets and models are untouched. The tag and the 'why' are suggested from what was changed and stay editable. The form keeps its state when you leave and return. '?' texts on the page are rewritten in plain language. |

---

name: every-feature-in-sandbox
description: Owner rule (D27) — every feature added to the RL racing project must be reflected in the AI Sandbox before it counts as done
metadata:
  type: feedback
---

Any feature added to the project must show up in the AI Sandbox (Streamlit app, `racer/sandbox`): a way to see, explain or launch it, plus a headless page test.

**Why:** the owner uses the sandbox as the one place to understand and drive everything; a feature only reachable from the CLI or the game is invisible to them.

**How to apply:** when planning a feature in log.md, add its sandbox part as its own task; don't mark the feature done without it. Recorded as D27 in PROJECT_CONTEXT §10.5/§16 and in CLAUDE.md. Related: [[e10-queue-worktree]].

**Open questions:**

- `final_mix` trains on every track in `tracks/user/`, so a track drawn and saved in the demo becomes training data for later `final_mix` runs. `maps7` leaves `user/` out. Should `final_mix` stay as it is (its meaning is frozen in old snapshots either way), and should faculty-drawn tracks be saved to `tracks/test/` instead? Currently: unchanged; use `maps7` for new runs.
- Faculty track: drawn in the Draw-a-Track editor, or on paper and converted from a photo? Currently: the editor (Week 3); photo import is a possible extension.
- Should rays also detect the barrier, or only the road edge? Currently: road edge only.
- Should the demo default to each model's own physics or a common one? Currently: each model's own, with a toggle.
- ~~Should kerbs be "road" for track limits?~~ Decided (D1): yes, kerbs sit inside the road edge and count as on-track.
