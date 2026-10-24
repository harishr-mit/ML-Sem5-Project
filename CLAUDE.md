# CLAUDE.md

This is an RL racing-line project (Machine Learning, Semester 5, solo). The full context, design decisions and plan are in:

@PROJECT_CONTEXT.md

## Context and status tracking (required)

- **Before doing anything,** read till the latest record of Change log of `log.md` for the current context and task status.
- **Use `log.md` as your todo list.**
  - Add tasks as `[ ]` before touching any files.
  - Mark a task `[~]` when you start it, and `[x]` as soon as it's done. Don't batch these updates at the end.
  - If a task fails or is interrupted, mark it `[!]` and add notes.
- **Log every file change** in this directory (created, edited, deleted or moved) in the `log.md` Change Log. This lets a later agent diagnose and resume after a run fails partway through.
- **If a task is still `[~]` when you start,** the previous run stopped midway. Inspect its files (`git status` / `git diff`), then either finish it or mark it `[!]` with notes.

## Dependencies (strict)

- **Keep every dependency local to the project.**
  - Python packages go only into the project virtual environment `.venv`: activate it, or call `.venv\Scripts\python -m pip ...`. Record them in `requirements.txt`.
  - Never `pip install` into the system Python, never use `pip install --user`, and never change PATH, the registry or global config.
- **Global installs need explicit permission.** If one is truly unavoidable (a system tool, a new Python version), ask the user first and wait for an explicit yes. Log every approved global install in `log.md`.
- pip's download cache doesn't count as a global install.

## During a run

- **Handle every artifact.** Don't leave downloads, temporary installers, caches or build artifacts lying around unhandled. Delete them, or make sure they're ignored.
- **Check `.gitignore` after every install or update,** and extend it if needed.
- **Don't track or push dependencies,** such as `.venv/`, `__pycache__/` or downloaded wheels, unless that is absolutely necessary.

## Waiting on a process (periodic check-ins)

- **Never let the prompt cache go cold.** The cache expires after 60 min, so keep every gap between check-ins under 60 min (aim for at most 55) to avoid paying to re-cache the context.
- **Read only what each check-in needs:** the process's status file and the tail of its current log (e.g. `outputs/queue_logs/<queue>/state.json`). `log.md` and `CLAUDE.md` are the only exceptions. Don't re-read other files or earlier output, and keep command output short.

## After a training run or queue completes

- **Back up the models** once the run is trained and evaluated (or the whole queue has finished): `.venv\Scripts\python scripts\backup_models.py`. It copies only what is missing to the backup folder (`configs/backup/default.yaml`), never overwrites or deletes, and refuses while a training, queue or evaluation is running. Also run it after evaluating older models (it adds their new eval files). Log the result (files copied, any CONFLICT lines) in `log.md`.
- **Neuroevolution runs (D26):** the backup also copies a finished run's lineage database (`outputs/evolve/<run>/lineage.sqlite`). A paused run's `state.npz` is how it resumes, so never delete it while the run is unfinished. Pause a run with Ctrl+C or a `stop.request` file, never by killing the process.

## End of a run (before the final summary)

- **Stop everything you started:** training runs, the demo window, TensorBoard and any other servers. Leave something running only if the user explicitly asked for that.
- **Clean up:** confirm there are 0 background tasks and 0 open sockets left.
- **Update the docs:** review the docs related to the task or bug (`PROJECT_CONTEXT.md`, `README.md`, `log.md`) and update them to reflect the changes or fixes made.

## Every feature shows in the AI Sandbox (owner, 2026-10-24, D27)

- **A feature isn't done until the sandbox reflects it:** a way to see, explain or launch it there (page, glossary entry, launch option), with a headless page test. Plan the sandbox part as its own task in `log.md` when you plan the feature.

## Working with the owner

- Feel free to ask for any critical decisions midrun.
- If any bugs are found midrun, do the needful: fix them (with a test), and log them in `log.md`.
- Add appropriate tests to test what you build or change.

## Workflow for any change (agents)

1. **Orient** as in *Context and status tracking*, plus the PROJECT_CONTEXT sections the change touches. Work on a branch, not on `main`.
2. **Decide.** If the change alters a goal, a decision or a design rule, say so and get the owner's agreement first. Then record it in PROJECT_CONTEXT (the section and the §16 decisions log) before writing code.
3. **Plan** the tasks in `log.md`; split anything bigger than one sitting into steps that each have a "done when" check.
4. **Build test-first, in small steps.** Write the failing test, then the code. Put the logic in `racer/` (testable), not in `scripts/` or Streamlit pages.
5. **Check.** Run the checks for every part you touched (table below), then the full suite: `.venv\Scripts\python -m pytest -q`.
6. **Review** the diff before committing, and fix what the review finds.
7. **Sandbox and document:** the sandbox reflects the feature (D27). Then document as in *End of a run*, including the README preset list, the glossary and `docs/*.md`.
8. **Commit** with a Conventional Commits message; real training runs need a clean tree (D12).
9. **Finish** as in *After a training run* and *End of a run*.

## Architecture (follow it)

```
scripts/  (thin CLIs: parse arguments, call racer/)        racer/render  (pygame demo)   racer/sandbox  (Streamlit only in pages/)
        │                                                          │                          │
        ▼                                                          ▼                          ▼
racer/agents + racer/registry (one layer; they use each other)
               algorithms registry · loader · compat (manifest) · evaluation · baseline · neuroevolution (pure GA core) · evolve (GA runs)
               CSV registry (source of truth) · record (evaluate + register) · backup
        │
        ▼
racer/env      Gymnasium env · sensors · obs features · action layouts · rewards · track sets · race (multi-car, no pygame)
        │
        ▼
racer/track · racer/physics   spline/field/progress/generator · car + surfaces + physics versions
configs/<kind>/*.yaml         every tunable number; a glossary entry for every key
```

- **Dependencies point down.** The simulation (`physics`, `track`, `env`) never imports pygame, Streamlit, torch, `agents` or `registry`.
- **Extend, don't change.** New behaviour gets a new registered piece: an obs feature or spec version, an action layout, a physics version, a reward type, an algorithm, an evaluation protocol, a preset. Never change what an existing one means, because saved models depend on it.
- **Models are self-describing and immutable.** A model is its folder (`model.zip`, `vecnormalize.pkl`, a full `config.yaml` snapshot with its manifest). Every engine part checks compatibility before loading (`compat.py`).
- **Long jobs are pausable and resumable.** Their state lives outside model folders (`outputs/`), they are deterministic from seeds, and they have a stop file plus an exit code the queue understands.
- **One source of truth per fact.** Registry CSVs (xlsx is generated), configs for numbers, the glossary for explanations, PROJECT_CONTEXT for decisions.

## Parts to keep in mind

| Part | Where | When you change it, also … |
| --- | --- | --- |
| Physics engine | `racer/physics`, `configs/physics` | run `scripts/physics_sanity.py`; a change to the equations = a new physics version; the golden R001 test must still pass |
| Env, sensors, inputs | `racer/env`, `configs/env` | a new input = a new feature or obs_spec version; `check_env` and golden tests; ray diagrams and brain-panel labels in the sandbox/demo |
| Rewards, episode rules | `racer/env/reward.py`, `configs/reward`, `env.episode` | new keys need a dataclass default so old snapshots load (`from_dict`) |
| Tracks | `racer/track`, `tracks/` | `scripts/make_maps.py --check`; `tracks/test/` and `tracks/heldout/` are never trained on |
| Training | `scripts/train.py`, `racer/agents/algorithms.py`, `evolve.py` | the model folder format, manifest and compat; registry run rows; TensorBoard tags the sandbox curves read; `train.py --help` |
| Evaluation | `racer/agents/evaluation.py`, `configs/eval` | protocols are frozen once used: a change = a new protocol version; registry columns are only appended |
| Registry | `racer/registry` | CSV columns only appended; `tests/test_integrity.py` |
| Queue | `scripts/queue_overnight.py`, `configs/queue` | runs without Claude; `state.json`, `stop.request`, exit codes (3 = a paused GA run); `--dry-run` every new queue file |
| Backup | `scripts/backup_models.py`, `configs/backup` | additive only; run after every completed run or queue |
| AI Sandbox | `racer/sandbox` | every new feature appears here (D27); a glossary entry for every config and snapshot key (a test fails otherwise); Build & Queue for new algorithms or preset kinds; the Presets page; headless page tests |
| Demo | `racer/render`, `scripts/demo.py` | 60 FPS with 20 cars; every model must race (compat); the on-screen key list test |
| Tests | `tests/` | add tests for new behaviour; the full suite passes before a commit |
| Docs | PROJECT_CONTEXT, README, `log.md`, `docs/`, `assets/LICENSES.md` | PROJECT_CONTEXT first, then everything else that states the old behaviour |

## Working rules

- Follow PROJECT_CONTEXT.md. If a task conflicts with it, say so before coding, and update the file once the user agrees.
- **If a goal or requirement changes at any point, reflect it in every document it affects.** Update `PROJECT_CONTEXT.md` first (including the §16 decisions log), then `README.md`, `log.md`, `assets/LICENSES.md`, configs and anything else that states the old goal or requirement.
- Keep `racer/physics`, `racer/track` and `racer/env` free of pygame. Rendering only wraps simulation objects.
- Tunable numbers go in `configs/*.yaml`, not in code.
- Platform is Windows and the project path has spaces: use `pathlib`, and guard SubprocVecEnv scripts with `if __name__ == "__main__":`.
- Never change an existing `obs_spec` version or edit a saved model folder. Models must stay loadable forever. (Only exception: `evaluate.py` may add eval files once, never overwrite; see PROJECT_CONTEXT §9.)
- Build in small, testable steps. After physics changes, run `scripts/physics_sanity.py`.

## Advisor use

- For hard tasks, consult the advisor before committing to an approach: multi-file refactors, unclear root causes, architecture or data-model decisions, security-sensitive changes, and any task with several viable options.
- Before declaring a non-trivial task done, get an advisor review of the result against the original request.
- If a fix attempt fails twice, consult the advisor before trying a third.
- Skip the advisor for simple edits, lookups, formatting, and routine commands.
- When you consult the advisor, state the goal, the constraints, what you've already tried, and the specific decision you need help with.
- Briefly tell me when you consulted the advisor and why.

## Persistence across sessions

* anything you deem important to be remembered shall persist in one form or other
- plans go into <context>plan.md files
- decisions go into DECISIONS.md
- decisions to be taken later or postponed go into REVIEW_LATER.md
- user preferences for the project goes into user.md (local to project). whenever your statements are contradicted by the user, it is a sign of user preference. This is vague, but it is safe to assume and put whatever you think subsequent sessions need to know to make the quality of experience for the user into that user.md
