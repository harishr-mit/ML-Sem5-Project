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
