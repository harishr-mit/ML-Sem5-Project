# Neuroevolution plan (D26, E10)

**Status: built** (2026-10-15, T92–T95; σ test T96, runs T97). Train the same kind of driving network as PPO with a genetic algorithm instead of gradients, as in the owner's reference video, and compare the two (E10). The decisions and their reasons are in PROJECT_CONTEXT §8 and §16 (T84–T91). Decisions that were uncertain (the dropped inputs, one map per generation, σ) are revised if results show them to be poor.

## How it works

```
generation g:  draw 1 of the 7 training maps at random (may repeat) and a start point
               50 networks drive one lap each from that start, in parallel ──▶ 50 fitness scores
               rank all 50; the top 12 (25 %) go to g+1 unchanged (elites)
               the other 38 = one of those same 12, picked at random, + σ × Gaussian noise on every weight
               ──▶ generation g+1
every 25 generations: the best car drives 5 validation tracks (stop after 500 generations without a 1 % gain)
every 100 generations: the best car is saved as a registered checkpoint
```

## Decisions

| Part | Decision | Why |
|---|---|---|
| Inputs | **10**: 7 rays at 0° (200 m), ±5.5° (200 m), ±17° (120 m), ±55° (60 m); car direction vs track direction (sin, cos); forward speed. New obs preset `rays7_lean` (D21, config only) | Owner: yaw rate, wheels on road, previous action and sideways speed are redundant. First ablation if driving is poor: sideways speed back in. The 0° ray sees the 191 m needed to brake from top speed (T89) |
| Outputs | steer and gas in [−1, 1] (negative gas = brake), clipped like PPO's deterministic `predict` | The saved model drives identically in the demo |
| Network | 10 → 256 → 256 → 2, tanh; 69,122 weights | The PPO actor's hidden layers; R015 generalises with them |
| Track limits | the try **fails when the car's centre crosses the road edge** (kerbs count as road, D1); new termination reason `track_limits` | No grass shortcuts; the car never sees the off-road ray meaning (D2) |
| Map | **one of the 7 training maps per generation**, at random (may repeat); the same map and start for every car; a new start each generation | Owner choice; more maps are added over time. Deterministic physics and policy make the ranking within a generation exact |
| End of a try | one lap, or a failure (barrier, track limits, stuck, wrong way), or 200 s of simulated time (the env's existing 4000 steps) | The cap only stops a crawler; 2.5× the baseline's slowest lap (81.6 s) |
| Fitness | did not finish: fraction of the lap covered (0–1); finished: 1 + mean speed ÷ 70 m/s | A finisher always beats a non-finisher, a faster lap beats a slower one |
| Selection | 50 cars ranked by fitness (previous elites included, they drove again); the top 25 % (12) go to the next generation unchanged; the same 12 are the only parents of the other 38: each child = one of them, picked at random, + σ × N(0, 1) per weight (owner, T100) | Elitism: the best car is never lost |
| σ | two 200-generation scratch runs, 0.02 and 0.05; keep the faster learner | No known value for this net |
| Stopping | every 25 generations the best car drives 5 validation tracks (generated, own seed range; never trained on; not the test maps or held-out set); stop when that score hasn't improved by 1 % for 500 generations, or on pause. No cap by default; an optional env-step budget (`total_steps`, `--steps`, as for PPO, so the sandbox queue and E10 use the same unit) for queue entries | Stops when it stops getting better at driving, not at memorising |
| CPU | 5 worker processes (≈ 60–70 % CPU), below-normal priority; changeable on resume, never changes results | Owner target 60 % |
| Control | a short single-track run on Training Ground | Shows the overfitting next to E5 |
| Normaliser | none (obs are already scaled); an identity `vecnormalize.pkl` is written | Every model folder has all three files (`tests/test_integrity.py`) |

## Pause and resume

- **Pause:** Ctrl+C, or a `stop.request` file in `outputs/evolve/<run>/`. The generation in progress finishes, the run state is saved, the process exits (code 3; the queue marks the entry interrupted and resumes it on its next start). A second Ctrl+C aborts immediately. The sandbox's *Stop queue* kills the queue's runs instead: a GA run then resumes from its last periodic save (at most `state_every` generations are driven again, with identical results).
- **State:** population weights, generation (each generation's random numbers come from `default_rng([seed, generation])`, so there is no RNG state to save), best scores, stopping counter, env steps. Saved atomically to `outputs/evolve/<run_id>/state.npz` (≈ 14 MB, overwritten), also every 10 generations against crashes.
- **Resume:** rerun the same command (or the queue entry); it continues where it stopped, identical to an uninterrupted run, with any worker count.
- Model folders are never touched by this.

## Recording (owner: limited disk space)

No full-weight record of the run (all cars of every generation would be ≈ 138 GB per 10,000 generations; 50 GB is free).

- **Lineage database** `outputs/evolve/<run_id>/lineage.sqlite` (stdlib `sqlite3`): one row per car per generation: generation, car, parent, noise seed (none for an elite copy), fitness, map, start, lap time, termination. ≈ 30 MB per 10,000 generations.
- **Rebuild:** a child is its parent + σ × noise from its seed, so any car is rebuilt exactly by replaying its chain from the initial network (≈ 1.5 s for 5000 mutations) and can be exported as a model folder. Exact rebuilding needs the pinned numpy (RNG streams are only guaranteed within a version); model folders stay the permanent record (§9).
- **Checkpoints:** the best car every 100 generations and at the end, as normal model folders (model.zip ≈ 0.3 MB, plus thumbnails after evaluation), registered and evaluated with protocol v2 after the run. A 5000-generation run: ≈ 50 checkpoint folders + the lineage.
- `scripts/backup_models.py` also copies the lineage database.

## Measured cost (T86–T88)

Full-length laps (R015's recorded actions replayed), a random 256×256 numpy net on every step, weights sent to the workers.

| Processes | CPU load | Env steps/s | 50 cars × 1 lap, worst case |
|---|---|---|---|
| 4 | ≈ 52 % | ≈ 29.5k | ≈ 2.0 s |
| **5** | **≈ 64–73 %** | **≈ 33k** | **≈ 1.8 s** |
| 8 | ≈ 96 % | ≈ 37k | ≈ 1.6 s |

Worst case = every car survives a 60 s lap; early generations crash within seconds. A worker simulates ≈ 330 s of driving per second.

## Fits into what exists

- **Model folder:** an SB3 PPO `model.zip` with the evolved weights in the actor (the value net is never trained and is reduced to one layer with `vf: []`; ≈ 0.3 MB), the identity `vecnormalize.pkl`, and a full `config.yaml` snapshot (`algorithm: ga`, the GA settings, generation, the manifest). Loader, demo, Evolution Race, `evaluate.py`, registry, backup and sandbox then work unchanged.
- **Model id:** `R0xx_GA_<tag>@<steps>k` (owner, T100: `GA_` is added by `train.py`, so the id shows the genetic algorithm; the tag stays lowercase and carries the seed, e.g. `R020_GA_main_s1@96000k`), steps = env steps used so far (E10's shared x-axis).
- **Algorithm registry:** a `ga` entry in `racer/agents/algorithms.py` loading through SB3's PPO class; `racer/agents/compat.py` must accept it.
- **Entry point:** `train.py` dispatches to the evolution loop when the train preset's `algorithm` is `ga`, so `queue_overnight.py` and the sandbox's Build & Queue page run it unchanged.
- **Config:** `configs/train/ga_default.yaml` (population, elite fraction, σ, fitness speed scale, validation tracks / stream / interval / patience / margin, optional `total_steps` budget, checkpoint interval, state-save interval, workers). The training maps are the run's own tracks (`--tracks`, or a track set without a generated pool: `train7` = the 7 training maps, `tg` = Training Ground for the control run). No number in code (§15).
- **Track limits:** an optional `episode` key with a default of off (§15: no new required key; check that `from_dict` honours dataclass defaults), so every existing preset and model behaves exactly as before. The arena treats `track_limits` as a crash (respawn), like the other terminations.
- **Glossary:** every new config key gets a `racer/sandbox/glossary.yaml` entry (a test fails otherwise).
- **Git:** `outputs/evolve/` added to `.gitignore`.

## Build steps (tasks T92–T98)

| Step | Task | Work | Done when |
|---|---|---|---|
| S1 | T92 | Env: optional track-limits termination; obs preset `rays7_lean` | tests: the try ends when the centre crosses the edge, not on a kerb; existing presets unchanged (golden R001 test still passes); `check_env` passes for `rays7_lean`; `physics_sanity.py` passes |
| S2 | T93 | `racer/agents/neuroevolution.py`: flat weights ↔ layers, numpy forward, fitness, selection + mutation from seeds, lineage rebuild | tests: elites survive unchanged; same seed → same population; a finisher outscores every non-finisher; a faster lap outscores a slower one; a rebuilt car equals the live one bit for bit |
| S3 | T94 | Export to a model folder; `ga` algorithm entry; compat; `ga_default.yaml` + glossary; track set `train7` | test: `LoadedModel.act_batch` equals the numpy forward pass; the folder passes `tests/test_integrity.py`'s checks |
| S4 | T95 | Evolution loop: worker pool (below-normal priority), validation + stopping, checkpoints + registry rows, TensorBoard (best / mean / median fitness, validation score, map, env steps), lineage database, pause/resume; `train.py` dispatch; sandbox Build & Queue for GA; `.gitignore`; backup includes the lineage | a 20-generation scratch run goes end to end; **paused at generation 10 and resumed with a different worker count, it ends identical to an uninterrupted run**; its checkpoint races in `demo.py`; the full test suite passes |
| S5 | T96 | σ test: two 200-generation scratch runs (0.02, 0.05) | the faster learner is chosen and logged |
| S6 | T97 | Control run (Training Ground) and main run (7 maps); evaluate (v2); back up | both evaluated; the Evolution Race of the main run watched; results in PROJECT_CONTEXT §11 (E10) |
| S7 | T98 | Docs: README (commands, GA), PROJECT_CONTEXT, `log.md` | the docs match the code |
