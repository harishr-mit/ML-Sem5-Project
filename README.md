# RL Racing Line

A 2D top-down racing game written from scratch, where a car learns to drive a racing line with reinforcement learning (PPO, Stable-Baselines3).

- **What the car senses:** 5 distance rays to the road edge, its heading relative to the track, its speed and a few physics readings. It never sees the map.
- **What it outputs:** steering and gas/brake.
- **Why a racing line appears:** grip-limited physics with momentum and per-wheel surfaces. Corner too fast and the car slides, so it has to learn to brake, hit the apex and use the full track width.

The demo (pygame-ce) loads any trained model from a registry and races up to 20 of them as ghost cars (Model Arena). Week 3 adds the evolution race, draw-a-track mode and human vs AI.

> **Status:** milestone M2 (Week 2) done. Track sets with generated tracks, fine-tuning (`--parent`), the evaluation protocol, the model registry (CSV → xlsx), an overnight experiment queue and the Model Arena (20 different models at a steady 60 FPS) are in place. The best model so far, `R015_t50_narrow7_s1@4600k` (7 narrow rays, trained on the map library), is faster than the hand-coded baseline and completes 88 % of the runs on maps it has never seen. A second seed and a longer run of the same setup didn't reproduce that (R018, R019), so results vary a lot between runs. Experiment results so far: [PROJECT_CONTEXT.md §11](PROJECT_CONTEXT.md).
> **New (2026-10-11):** a map library of 9 hand-designed maps (7 for training, 2 test maps never trained on), evaluation protocol v2, and an upgradable model design: inputs are built from named features, the outputs follow a versioned action layout, and PPO and SAC are both supported. Any model the engine can run races in the arena.
> **New: the AI Sandbox** (`scripts\sandbox.py`, a local app in its own window). It explains every trained model (what it senses, its network, what it was trained with and why), compares models, builds and queues new training runs, and launches races in separate game windows. See [AI Sandbox](#understand-compare-build-and-race-models-ai-sandbox) below.
> **New (2026-10-15): neuroevolution** (D26), a genetic algorithm that trains the same kind of driving network without gradients, to compare with PPO (experiment E10). See [Train with neuroevolution](#train-with-neuroevolution-genetic-algorithm). Design: [docs/NEUROEVOLUTION_PLAN.md](docs/NEUROEVOLUTION_PLAN.md).
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

### Everything at a glance

Every command below starts with `.venv\Scripts\python`, run from the project folder. For example, `scripts\sandbox.py` means `.venv\Scripts\python scripts\sandbox.py`. Keep the quotes around model ids, because the `@` can confuse some shells. All the flags for each script are in [Commands](#commands).

**Showcase (live demo, in presentation order)**

| # | What it shows | Command |
|---|---|---|
| 1 | The AI Sandbox: every model explained, compared and launched from one window | `scripts\sandbox.py` |
| 2 | The game and its physics: drive it yourself (momentum, sliding, grass) | `scripts\drive.py --track technical` |
| 3 | **Learning:** one run's checkpoints race each other (5 cars, 100k to 1.7M steps) | `scripts\demo.py --track training_ground --run R001 --every 4` |
| 4 | **Generalisation:** the best models on a test map they never trained on, against the baseline | `scripts\demo.py --track coastline --models "R015_t50_narrow7_s1@4600k" "R016_t50_rays9_s1@2600k" --baseline` |
| 5 | **Arena:** every run's best checkpoint plus the baseline (up to 20 cars) | `scripts\demo.py --track technical --best --baseline` |
| 6 | **The physics creates the racing line (E2):** a normal model next to one trained without a grip limit | `scripts\demo.py --track fast_flow --models "R004_e1_main_s1@3000k" "R005_e2_nogrip_s1@900k"` |
| 6b | The same race on one common physics: the no-grip model can't handle real grip | the same command + `--fair` |
| 7 | **Overfitting (E5):** a model trained on one track against one trained on the map library, on an unseen map | `scripts\demo.py --track alpine --models "R006_e5_single_s1@2600k" "R015_t50_narrow7_s1@4600k"` |
| 8 | **Naive speed reward (E3):** a model rewarded for speed rather than progress | `scripts\drive.py --model "R009_e3_speed_s1@2900k" --track training_ground` |
| 9 | One model up close: its rays (`R`) and speed-coloured trail (`T`) on an unseen map | `scripts\drive.py --model "R015_t50_narrow7_s1@4600k" --track alpine` |
| 10 | Learning curves (open http://localhost:6006) | `-m tensorboard.main --logdir runs` |
| 11 | The model registry as a spreadsheet (then open `registry\registry.xlsx`) | `scripts\registry.py export` |
| 12 | Track gallery: PNG previews in `outputs\plots\tracks\` | `scripts\preview_tracks.py showcase train test heldout` |

Tracks: `training_ground`, `technical`, `fast_flow` (showcase); `city_blocks`, `esses`, `hairpin_valley`, `oval_infield` (training maps); `alpine`, `coastline` (test maps, never trained on); `heldout_00` … `heldout_09` (generated, never trained on). Draw-a-track, Human vs AI, video recording and the report plots are planned for Weeks 3–4 and aren't built yet.

**Understand and compare models**

| What | Command |
|---|---|
| Explorer, Compare, Glossary (easiest) | `scripts\sandbox.py` (`--browser` opens it in a browser tab instead) |
| Each run's best checkpoint, in the terminal | `scripts\registry.py list --best` |
| Every checkpoint of one run | `scripts\registry.py list --run R015` |
| Inputs, outputs and parameter count per run; models the engine can't run | `scripts\registry.py describe` |

**Train and experiment**

| What | Command |
|---|---|
| Build and queue runs without typing commands | `scripts\sandbox.py` → *Build & Queue* |
| Quick smoke test (scratch run, nothing registered) | `scripts\train.py --tag smoke --trackset maps7 --steps 50000 --scratch` |
| A real run (commit your changes first) | `scripts\train.py --tag my_run --trackset maps7 --env rays7_narrow --steps 5000000` |
| Check an overnight queue without running it | `scripts\queue_overnight.py configs\queue\night1.yaml --dry-run` |
| Run a queue, 2 runs at a time (stop with `Ctrl+C`) | `scripts\queue_overnight.py configs\queue\night1.yaml --parallel 2` |
| Neuroevolution on the 7 training maps (pause with `Ctrl+C`; the same command resumes it) | `scripts\train.py --tag main_s1 --train ga_default --trackset train7 --env rays7_lean --seed 1` |

**Evaluate**

| What | Command |
|---|---|
| Evaluate one model (protocol v2) | `scripts\evaluate.py --model "R015_t50_narrow7_s1@4600k"` |
| Evaluate every checkpoint of a run and mark the best | `scripts\evaluate.py --run R015` |
| Evaluate the hand-coded baseline | `scripts\evaluate.py --baseline` |

**Tracks and checks**

| What | Command |
|---|---|
| Check the hand-designed maps still build | `scripts\make_maps.py --check` |
| Regenerate the held-out tracks | `scripts\generate_tracks.py --heldout` |
| Unit tests | `-m pytest -q` |
| Physics sanity tests (after any physics change) | `scripts\physics_sanity.py` |
| Simulation speed and baseline laps | `scripts\bench_env.py` |

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

### Model presets

A model is built from five presets, each a YAML file in `configs/<kind>/`. A run's snapshot keeps a full copy of every preset it used, so changing or adding a preset never changes a saved model. To try something new, add a preset (start the file with a comment saying what it is for) rather than changing one that models were trained with. The sandbox's **Presets** page shows each preset's values with notes, a picture where it helps (rays to scale, track previews) and the runs trained with it.

| Kind (`train.py` flag) | Preset | What it is |
|---|---|---|
| **Inputs and episode rules** (`--env`) | `rays5` | Default: 5 rays (−90°, −45°, 0°, +45°, +90°), heading, speed, slide, yaw rate, wheels on road, previous action (13 inputs, obs_spec v1) |
| | `rays5_short` | E4: as `rays5` with a 60 m forward ray (too short to brake from top speed) |
| | `rays9` | E4: 9 rays (±10°, ±30°, ±60°, ±90°, 0°), 17 inputs |
| | `rays7_narrow` | 7 rays with shallow angles (±7°, ±20°, ±60°, 0°) that see bends early, 15 inputs (R015, the best PPO model) |
| | `rays7_lean` | Neuroevolution: 7 rays (0°, ±5.5°, ±17°, ±55°), heading, forward speed (10 inputs); track limits on |
| **Car** (`--physics`) | `default` | The car: 1000 kg, 150 kW, grip-limited tyres, road / kerb / grass surfaces |
| | `no_grip_limit` | E2: unlimited grip on the road, so there is no reason to brake or take a racing line |
| | `downforce` | E9: grip grows with speed², so fast corners hold more than slow ones |
| **Reward** (`--reward`, PPO/SAC only) | `progress` | Default: metres of progress, minus a time cost, steering changes and grass; lap bonus, crash penalty |
| | `speed_naive` | E3: rewards speed in any direction, only to show reward hacking (circling pays too) |
| **Algorithm and network** (`--train`) | `ppo_default` | PPO, 256 × 256 tanh network, 2M steps by default (`--steps`) |
| | `ppo_finetune` | PPO with a lower learning rate, for `--parent` fine-tuning |
| | `sac_default` | SAC (E7), 256 × 256 ReLU network, 1M steps by default |
| | `ga_default` | Neuroevolution (D26): 50 cars, top 12 kept, σ 0.02, budget in generations (`--generations`) |
| | `baseline` | Settings of the hand-coded PID driver (E8); not a training preset |
| **Track sets** (`--trackset`) | `tg` | Training Ground only |
| | `gen_wide` | Mostly wide generated tracks plus a little Training Ground |
| | `final_mix` | Showcase tracks, user-drawn tracks and generated tracks |
| | `maps7` | The 7 training maps plus a generated pool with mixed widths (R015) |
| | `train7` | The 7 training maps only, no generated tracks (neuroevolution) |

- **Combine them** in one command: `scripts\train.py --tag my_run --env rays7_narrow --physics default --reward progress --train ppo_default --trackset maps7 --steps 5000000`. Flags you leave out use the defaults (`rays5`, `default`, `progress`, `ppo_default`; `--tracks training_ground` without a track set).
- **Change one value** without a new file: `--set ppo.ent_coef=0.01`, `--set policy.net_arch=[128,128]` or `--set ga.sigma=0.05` (train preset values only; a misspelt key is an error).
- **Outputs:** `--action-layout steer_throttle_brake` gives the car separate pedals (obs_spec v2 inputs only).
- **Without typing commands:** the sandbox's *Build & Queue* starts from any preset (or any trained model), lets you change inputs, network, algorithm, car, reward and tracks, and writes a new input preset `configs\env\sb_<tag>.yaml` when the inputs are new.
- **Other presets** (not part of a model): `configs\eval\` (evaluation protocols v1, v2), `configs\track\default.yaml` (track geometry, generator), `configs\demo\` (arena, rendering, sound), `configs\sandbox\default.yaml`, `configs\backup\default.yaml`, and `configs\queue\` (overnight run lists).

### Train with neuroevolution (genetic algorithm)

```bat
.venv\Scripts\python scripts\train.py --tag main_s1 --train ga_default --trackset train7 --env rays7_lean --seed 1
```

- **How it learns:** 50 cars drive the same lap of one of the 7 training maps (a random map each generation; every try starts on the start/finish line from standstill). The best 12 go on unchanged; the other 38 are copies of those 12 with small random changes to every weight. No gradients, no reward: a car scores the fraction of the lap it covered, or 1 + its mean speed / 70 m/s if it finished. Generation 0 is 50 random networks (PPO also starts from random weights; neither starts from the baseline).
- **Inputs:** `rays7_lean` (7 rays incl. one straight ahead, heading, speed: 10 inputs). A try ends at the lap, a crash, track limits (the car's centre over the road edge), stuck, wrong way, or 200 s.
- **Budget:** in generations, not steps. `--generations 1000` stops after 1000 generations; without it the run stops when its validation score hasn't improved by 1 % for 500 generations. (PPO and SAC use `--steps`; each flag is refused for the other kind.)
- **Validation and checkpoints:** every 20 generations all 12 elites drive 5 generated tracks they never train on. The best of them is saved as a checkpoint `models\R###_GA_<tag>@<env steps>k\`, so each checkpoint is a step of the evolution. They load, race and appear in the sandbox like any model; an Evolution Race shows them: `scripts\demo.py --run R020 --every 10`.
- **Evaluation:** when the run finishes, the 10 checkpoints with the best validation scores and the last 10 are evaluated with protocol v2 (later checkpoints aren't reliably better: each generation trains on one map). Evaluate the others any time with `scripts\evaluate.py --model "<id>"`.
- **Any car, afterwards:** every car of every generation is recorded in `outputs\evolve\<run>\lineage.sqlite`. `scripts\export_ga.py --run R020 --list` shows the validated generations; `--generation 750` saves that generation's best car as a model folder (`--car 3` for a specific one, `--evaluate` to evaluate it). The sandbox's Model Explorer does the same in its **Evolution** tab, with the run's fitness and validation curves.
- **Pause and resume:** press `Ctrl+C` (or create `outputs\evolve\<run>\stop.request`). The current generation finishes, the run is saved and the program exits. Run **the same command** again to continue; `--set ga.workers=8` changes the number of processes (default 5, ≈ 60–70 % CPU), and `--generations` may change too. The result is the same however often it was paused.
- **Control run (one map):** `--tracks training_ground` instead of `--trackset train7`.
- **In the queue or the sandbox:** works as a queue entry; *Build & Queue* offers `ga` with a generation budget (no limit by default). A paused run is marked *interrupted* and resumes when the queue starts again.

### Race models (Model Arena)

```bat
.venv\Scripts\python scripts\demo.py
.venv\Scripts\python scripts\demo.py --track technical --best --baseline
.venv\Scripts\python scripts\demo.py --track fast_flow --run R002 --every 3
```

- **Picker:** pick a track on the left and up to 20 models on the right (click rows; click a column header to sort; type to filter; `Tab` best checkpoints only; `Shift+B` adds the PID baseline; `Shift+P` fair physics; `+/−` laps), then `Enter`.
- **Race:** ghost cars start from a grid after a countdown. A crashed car restarts where it crashed after 1 s, and its crashes are counted.
- **Keys:** `Tab` (or click a leaderboard row) follows the next car · `F` overview · `R` rays · `T` all trails · `L` labels · `Space` pause · `+/−` sim speed · mouse wheel zoom · `H` hides or shows the on-screen key list (bottom right) · `Esc` back to the picker (or quits, when started with `--models`, `--run` or `--best`). The same keys work in the Evolution Race (`--run`).
- **Physics:** each car uses the physics it was trained with; `--fair` (or `Shift+P`) puts every car on the default physics.
- **Any model:** each car uses its own inputs and outputs, so models with different sensors, action layouts or algorithms race together. A model the engine can't run yet is shown dimmed in the picker with the reason.

### Understand, compare, build and race models (AI Sandbox)

```bat
.venv\Scripts\python scripts\sandbox.py
```

This opens the sandbox in its own window (a small built-in browser, Edge WebView2, part of Windows 11). Close the window, or press `Ctrl+C` in its console, to stop it; game windows and a queue it started keep running. Add `--browser` to open it in your normal browser at http://localhost:8501 instead. The sandbox runs on this PC only, sends no usage statistics and writes nothing to your user profile.

- **Overview:** a leaderboard of every run's best checkpoint (sorted by completion on unseen tracks, then lap time against the baseline), plus what is running now: the queue and any open game windows.
- **Model Explorer:**
  - **What it senses:** its rays drawn to scale on a straight road, and cast on any real map by the model's own sensor code.
  - **Its brain:** a network diagram with the parameters per layer.
  - **What it trained with:** the car, reward, tracks, hyperparameters and episode rules. Every value has a note on what it affects, and values that differ from the default preset are marked.
  - **Why the run was made** and what it showed, from `registry\run_notes.yaml`, which you can edit on the page.
  - **Results** for every evaluation protocol, plus learning curves from TensorBoard.
  - **Evolution** (neuroevolution models): fitness and validation per generation, and *Save as a model* for any generation's car.
- **Compare (2–4 models):** only the settings that differ, each with its note, then their shape, results (within one protocol), learning curves and driven lines side by side.
- **Build & Queue:** start from a trained model's setup or from presets, then change what you want to test:
  - **Inputs:** a ray table with a live diagram, plus the car's own readings.
  - **Network:** outputs, hidden layers and a live parameter count.
  - **Training:** algorithm and hyperparameters, car, reward, track set.
  - **Budget:** steps (1/2/3/5M or custom) and seeds.
  - **Why:** a required note on what the run should show.

  *Add to queue* reuses an existing input preset, or writes a new `configs\env\sb_<tag>.yaml`. Hyperparameter changes become `train.py --set` arguments. The entry goes into `configs\queue\sandbox.yaml` (git-ignored). *Start queue* opens the overnight queue in its own window, 2 runs at a time; *Stop queue* asks it to stop like `Ctrl+C`. Real runs still need committed code; a new untracked preset doesn't count.
- **Draw a Track:** draw a closed loop freehand or by clicking points (first point = start line, drawn direction = driving direction). It is smoothed and resized into a valid track and shown with any problem marked in red; *Save* puts it in `tracks\user\`, and *Race on it* opens Launch with that map chosen. This is how a faculty-drawn track gets raced live.
- **Launch:** pick up to 20 models (or every N-th checkpoint of one run, as an evolution race), a map, laps, the baseline car and same physics for all, then press *Launch*. Each press opens a separate game window named `AI Sandbox #N · …`, so several can run at once; *Stop* closes one.
- **Presets:** every preset a model is built from (inputs, car, reward, algorithm and network, track sets): its values with notes, the rays drawn to scale or the track previews, and the runs trained with it.
- **Glossary:** every setting, what it affects and its typical range, searchable.

### Run experiments overnight

```bat
.venv\Scripts\python scripts\queue_overnight.py configs\queue\night1.yaml --dry-run
.venv\Scripts\python scripts\queue_overnight.py configs\queue\night1.yaml
```

- Start it yourself from a normal terminal; it needs no Claude session. It runs the queue's training runs, logs each to `outputs\queue_logs\`, keeps going if one fails, and resumes where it left off if you start it again.
- **Two at a time:** add `--parallel 2` (or `parallel: 2` at the top of the queue file). Two runs at once get about 1.6× more training done per night than one after another; each run takes about 20 % longer. Entries added to the queue file while it runs are picked up.
- It keeps Windows awake while it runs. Keep the laptop plugged in, with the lid open.
- **When it has finished,** back up the new models: `.venv\Scripts\python scripts\backup_models.py` (see [Back up the models](#back-up-the-models-after-every-run)).

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

### Back up the models (after every run)

`models\` is not in git, so the backup folder is the only second copy of the trained models. **After every completed run** (training and evaluation finished, or a whole overnight queue), run:

```bat
.venv\Scripts\python scripts\backup_models.py
```

- It copies the model folders and files that are missing from `C:\Users\Jack\ML_Sem5_models_backup\` (set in `configs\backup\default.yaml`). This includes eval files added to older models.
- It never overwrites or deletes anything in the backup. A file whose size differs is reported as `CONFLICT` and left alone.
- It refuses to run while a training, queue or evaluation is still running (they are still writing model folders).
- `--dry-run` lists what it would copy; `--run R015` backs up one run only.

## Commands

Run everything with the project interpreter, `.venv\Scripts\python`.

| What | Command |
|---|---|
| Drive / watch (debug window) | `scripts\drive.py [--track technical] [--mode human\|baseline\|model] [--model models\<model_id>] [--sim-speed 4] [--trace out.csv]` |
| Train | `scripts\train.py --tag … (--tracks … or --trackset maps7) [--parent <model_id>] [--train ppo_default\|ppo_finetune\|sac_default\|ga_default] [--physics …] [--env rays5\|rays7_narrow\|…] [--action-layout steer_gas\|steer_throttle_brake] [--reward …] [--steps N (PPO/SAC) \| --generations N (GA)] [--seed N] [--set ppo.ent_coef=0.01] [--scratch] [--no-eval] [--protocol v2]` |
| Export a neuroevolution car | `scripts\export_ga.py --run R020 --list` · `--generation 750 [--car 3] [--evaluate]` (rebuilt from the lineage) |
| Evaluate (protocol v2) | `scripts\evaluate.py --model <id> …` · `--run R002` (all checkpoints, marks the best) · `--baseline` · `[--protocol v1]` |
| Registry | `scripts\registry.py list [--run R002] [--best]` · `backfill` · `export` (regenerate the xlsx) · `describe` (inputs, outputs, parameters per run; models the engine can't run) |
| Back up models (after every run) | `scripts\backup_models.py [--run R015] [--dry-run] [--dest <folder>]` (copies what's missing; never overwrites or deletes) |
| Model Arena | `scripts\demo.py [--track …] [--models … or --run R002 [--every N] or --best] [--baseline] [--laps 3] [--fair] [--title "caption"]` |
| AI Sandbox | `scripts\sandbox.py [--port 8501] [--browser \| --no-browser]` (localhost only) |
| Overnight queue | `scripts\queue_overnight.py configs\queue\<queue>.yaml [--parallel 2] [--dry-run] [--retry-failed]` (a `stop.request` file next to its `state.json` stops it like Ctrl+C) |
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
| `racer/` | `physics`, `track`, `env` (headless, no pygame); `agents`, `registry`, `render` (pygame-ce, demo only); `sandbox` (AI Sandbox: data, explanations + `glossary.yaml`, diagrams, curves, run builder, queue, launcher; Streamlit only in `sandbox/pages/`) |
| `configs/` | YAML presets: `physics/`, `env/`, `reward/`, `train/`, `track/` (geometry and generator), `trackset/` (what training samples), `eval/` (evaluation protocol), `queue/` (overnight experiment lists; `sandbox.yaml` is git-ignored), `demo/` (render, arena and audio settings), `sandbox/` (AI Sandbox settings) |
| `scripts/` | train, evaluate, registry, demo (Model Arena), sandbox (AI Sandbox), queue_overnight, drive (debug window), physics sanity, track previews / generation, env benchmark; later record video, plots |
| `tests/` | pytest unit tests; `test_integrity.py` checks that the registry, model folders, eval copies, run notes, docs and pinned packages agree |
| `tracks/` | `showcase/` and `train/` (the 7 training maps), `test/` (2 test maps) and `heldout/` (generated), both never used in training, `user/` (drawn in the editor). Any other folder works too: a track JSON added at any time is found by name |
| `docs/` | `SANDBOX_PLAN.md`: design and build plan of the AI Sandbox app; `NEUROEVOLUTION_PLAN.md`: design of the genetic-algorithm trainer (D26) |
| `models/` | self-contained, immutable model folders; **local only** (git-ignored), back them up by copying the folder |
| `registry/` | `runs.csv`, `models.csv` → `registry.xlsx`; `run_notes.yaml` (why each run was made and what it showed) |
| `assets/` | sprites, textures, fonts, sounds; see [assets/LICENSES.md](assets/LICENSES.md) |
| `runs/`, `outputs/` | TensorBoard logs; `eval/` (copies of every model's evaluation), videos, plots, frames, `scratch/` (smoke runs), `queue_logs/` and `sandbox/` (game windows started from the sandbox, with their logs) (the last three git-ignored) |

## Credits

Engine sound: "Delorean DMC-12 V6 PRV engine" by skyernaklea ([freesound.org/s/623436](https://freesound.org/s/623436/)), CC BY 4.0, edited. Tyre sound by johnnydekk (CC0). Sprites by Kenney (CC0). Textures by ambientCG (CC0). Fonts: Rajdhani, Share Tech Mono, Orbitron (SIL OFL 1.1).
