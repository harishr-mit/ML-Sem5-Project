# GA fitness v2 plan (D31)

Goal: break the plateau seen in R025/R026 by scoring speed on a continuous, convex scale, shrinking σ, and judging validation on a smoothed mean. Opt-in: preset `configs/train/ga_v2.yaml`; every older snapshot keeps v1.

| Step | Done when |
| --- | --- |
| Pure functions in `neuroevolution.py`: `fitness_v2`, `sigma_at`, `track_validation`, `rebuild` with a σ per generation | unit tests: ordering, monotone, rebuild == in-run weights |
| `evolve.py`: params with defaults meaning v1, σ(g) in breeding / restore / rebuild, smoothing window in `State`, schedule in lineage meta | v1 `settings_key` unchanged; pause/resume identical on `ga_v2` |
| Preset `ga_v2`, glossary keys, Explorer fitness text | glossary test passes; Presets page lists it |
| Sandbox: Build & Queue GA learning preset, headless test | page test passes |
| Docs, scratch run | ~50-generation scratch run reports finishers vs fast crashers |

Open: φ (`fail_speed_frac`) and `finish_bonus` are guesses; check them on the scratch run before a real run.

## Staged fitness (D32)

`ga.fitness: 3`: stage 1 = fitness v1, stage 2 = fitness v2 (`fail_speed_frac` 0.1 in preset `ga_staged`). Switch when at least `stage_min_finishers` cars finish in at least `stage_share` of the last `stage_window` generations; one-way; recomputed from the lineage's finisher counts; the validation baseline restarts at the switch.

| Step | Done when |
| --- | --- |
| `neuroevolution.stage_ready` (pure) + tests | window, share and min-finishers cases |
| `evolve.py`: `State.stage`, per-generation fit, switch + baseline reset, restore from the lineage | pause/resume across the switch equals an uninterrupted run; restore gives the same stage |
| Preset `ga_staged`, learning pick, glossary, Explorer text | glossary and pick tests pass |
| Queue file, `--dry-run` | owner approves the start |
