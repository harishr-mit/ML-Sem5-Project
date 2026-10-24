# Decisions

Short log of decisions taken in sessions. The full design record is PROJECT_CONTEXT §16 (the master list, with the section each decision changed); this file is the quick view. Postponed decisions are in [REVIEW_LATER.md](REVIEW_LATER.md).

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
