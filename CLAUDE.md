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

## End of a run (before the final summary)
- **Stop everything you started:** training runs, the demo window, TensorBoard and any other servers. Leave something running only if the user explicitly asked for that.
- **Clean up:** confirm there are 0 background tasks and 0 open sockets left.
- **Update the docs:** review the docs related to the task or bug (`PROJECT_CONTEXT.md`, `README.md`, `log.md`) and update them to reflect the changes or fixes made.

## Working rules
- Follow PROJECT_CONTEXT.md. If a task conflicts with it, say so before coding, and update the file once the user agrees.
- **If a goal or requirement changes at any point, reflect it in every document it affects.** Update `PROJECT_CONTEXT.md` first (including the §16 decisions log), then `README.md`, `log.md`, `assets/LICENSES.md`, configs and anything else that states the old goal or requirement.
- Keep `racer/physics`, `racer/track` and `racer/env` free of pygame. Rendering only wraps simulation objects.
- Tunable numbers go in `configs/*.yaml`, not in code.
- Platform is Windows and the project path has spaces: use `pathlib`, and guard SubprocVecEnv scripts with `if __name__ == "__main__":`.
- Never change an existing `obs_spec` version or edit a saved model folder. Models must stay loadable forever. (Only exception: `evaluate.py` may add eval files once, never overwrite; see PROJECT_CONTEXT §9.)
- Build in small, testable steps. After physics changes, run `scripts/physics_sanity.py`.
