# Decisions

Short log of decisions taken in sessions. The full design record is PROJECT_CONTEXT §16 (the master list, with the section each decision changed); this file is the quick view. Postponed decisions are in [REVIEW_LATER.md](REVIEW_LATER.md).

## 2026-10-28

| Decision | Why | Where |
|---|---|---|
| **Staged GA fitness** (D32): fitness v1 until 5 of 50 cars finish in 90 % of the last 100 generations, then fitness v2 with `fail_speed_frac` 0.1; the validation baseline restarts at the switch | R027 (v2 from the start) never finished a lap: failures were charged at 7.5 m/s, so slow early cars had no gradient | PROJECT_CONTEXT §8, §16; `docs/GA_FITNESS_V2_PLAN.md` |
| **GA fitness v2** (D31): continuous virtual lap time (failures charged the unfinished lap at a slow speed), convex speed bonus, finish bonus; σ shrinks with the generation; validation on 8 tracks, plateau on the mean of the last 3 | Owner: R025/R026 plateaued; reward faster driving more and more | PROJECT_CONTEXT §8, §16; `docs/GA_FITNESS_V2_PLAN.md` |
| Fixed `ref_speed` (24.95 m/s) instead of a per-track baseline lap time | Same ranking within a map; avoids driving the baseline on every generated track | `racer/agents/neuroevolution.py` |
| No separate re-check of top checkpoints | The post-run protocol-v2 evaluation of the 10 best-validated already is one | PROJECT_CONTEXT §8 |

## 2026-10-24 (later)

| Decision | Why | Where |
|---|---|---|
| **GA runs stay resumable after they finish** (D29): a finished run keeps `state.npz` (about 14 MB); `train.py --extend` continues it with a new patience or budget; runs finished before get their state rebuilt from the lineage | A plateau stop that turns out too early shouldn't cost the generations already driven; "main 2" would start from scratch | log.md T139-T146, `docs/GA_EXTEND_PLAN.md`, PROJECT_CONTEXT §8 |
| The backup copies a finished run's `state.npz` too | Owner | `racer/registry/backup.py` |
| Backup of extended GA runs: changed files are saved as `name.v2`, `name.v3`, … next to the old copy (no git in the backup folder) | Owner: keeps the never-overwrite rule, no CONFLICT | `racer/registry/backup.py` |
| Sandbox: *Continue this run* and a bulk *Save as a model* | Owner ("both") | log.md T144, T145 |

## 2026-10-24

| Decision | Why | Where |
|---|---|---|
| **Finish the demo and the analysis work first, then experiment freely** (D28) | Weeks 3 and 4 were dragging; free experimenting is easier once the tools are done | [docs/DEMO_PLAN.md](docs/DEMO_PLAN.md), PROJECT_CONTEXT §16 |
| In scope: mode menu, Evolution Race, Human vs AI, swarm, brain panel, audio, polish, sandbox Workbench, drive from the sandbox, delete rows in the rays table | Owner's list | log.md T123–T138 |
| Dropped: game-window drawing editor, video recorder, slides and highlight video | The sandbox already has *Draw a Track*; not needed for the demo | PROJECT_CONTEXT §10.1, §10.2 |
| Audio: on when watching a race, off for training, the sandbox and analysis; toggle in every game window | Owner | PROJECT_CONTEXT §10.4 |
| Sound key is `M`, minimap is `N` (the old plan had `S` and `M`) | `S` is the brake when driving with WASD | PROJECT_CONTEXT §10.3 |
| Brain panel = a live network graph (inputs, neuron grids, outputs), not just bars | Owner: "the neurons firing" | PROJECT_CONTEXT §10.2 |
| *Drive it yourself* opens a game window from the sandbox | Streamlit can't read 60 Hz keys | PROJECT_CONTEXT §10.5 |
| Workbench plots with Altair (ships with Streamlit, no new dependency) and matplotlib for PNG files | Interactive in the page, files for the report | `racer/sandbox/plots.py` |
| Skid marks are baked into the world surface | The full-screen overlay cost 21 ms per frame | log.md T133 |
| Development in a git worktree while a queue runs in `main`; merge when it ends | Real runs refuse uncommitted code | log.md T117, T123 |
| `tracks/user/wiggly_wiggly.json` copied into `main` | Owner asked | log.md T122 |
| E10 conclusion: PPO under the GA's rules beat neuroevolution (one seed each) | Registry R020–R023 | PROJECT_CONTEXT §11 |
| Build & Queue polish (D30): quick picks are real preset files that only fill the form; memory (`history`), five more inputs and two more layouts are options | Owner request 2026-10-26 | PROJECT_CONTEXT §16 D30 |
