# Plan: resume neuroevolution runs after they finish (D29)

Owner request (2026-10-24): "make runs resumable long after they finish" and "add the rebuild model option to sandbox" (both: continue a run, and improve *Save as a model*). Tasks T139-T146 in `log.md`.

## Why

A run stops when the validation score hasn't gained 1 % for `patience` (500) generations. If that plateau was noise or too short, the only option was a new run from scratch (R021 and R022 stopped near generation 900). The lineage database already holds every car, so nothing needs to be lost.

## Design

- **Keep the state.** When a run finishes, the loop breeds generation g+1 (cheap), saves `state.npz` for it, then writes `finished.json`. The kept state is then exactly where an uninterrupted run with a larger patience would have continued. About 14 MB per run (50 cars x 69k weights).
- **Finished is explicit.** `finished.json` marks a stopped run. `Evolution` refuses a finished run unless it is asked to extend (`train.py --extend`), so keeping `state.npz` can't restart a finished run by accident.
- **Only stopping rules change.** An extension may change `ga.validation.patience`, `ga.max_generations` (and the existing workers/evaluate). `min_gain` stays fixed, because the stored `last_gain` was computed with it. The settings key leaves `validation.patience` out; a run created under the old key (before D29) still resumes (the old key is accepted as well).
- **The extension persists.** A later plain resume must not fall back to patience 500, so the extended values are written into the run folder (`snapshot.yaml`, with a short history) and used on every resume.
- **Runs finished before D29 (R020–R022 and R024).** `restore_state(run_dir)` rebuilds all cars of the last generation from the lineage (`ORDER BY car`), re-breeds with the generation's own random stream, replays the `min_gain` rule over the recorded validation scores for `best_validation` and `last_gain`, and takes `env_steps` from the last generation row and `wall_s` from `finished.json`. Needs the same numpy version as the run (as `rebuild_car` already does).
- **Backup.** `backup_models.py` already copies every file of a finished run's folder, so `state.npz` is included. After an extension `lineage.sqlite` and `finished.json` grow; the backup saves the changed files next to the old copy as `name.v2`, `name.v3`, … (owner decision; compared by content, so a changed `state.npz` of the same size is caught too). Models keep the strict rule (a changed file is a CONFLICT).
- **Sandbox (D27).** Explorer → Evolution: *Continue this run* (patience, budget, workers) writes a queue entry that runs `train.py --extend`; the restore logic stays in `train.py`/`racer/agents`, so the sandbox's write scope doesn't grow. *Save as a model* gets a range mode (every N generations, the best-validated elite of each) and Launch → Evolution Race points to it.
- **Guards (advisor review).** The button exists only for finished runs, and is refused while any queue entry for the same tag (and scratch-ness) is waiting, running or interrupted, because two processes on one run folder would corrupt its lineage and state. After an extension `train.py` evaluates only the checkpoints that have no eval file for the protocol yet (the evaluation re-runs but never overwrites, D10, so repeating it would only waste about 50 s per checkpoint).

## Tests (first)

1. Finish, then extend with patience P2: weights and lineage rows equal an uninterrupted run with P2.
2. A state restored from the lineage equals the kept `state.npz`.
3. A paused run created under the old settings key still resumes; changing `min_gain` on extend is refused.
4. A plain resume after an extension keeps the extended patience.
5. Real data: a scratch copy of R021's `lineage.sqlite` restores to the weights of R021's last checkpoint (never open the live file; `Lineage()` runs DDL).

## Not done here

Merging into `main` and extending R021 for real wait for the owner: real runs need a clean tree (D12), and `main` may run something else.
