# Review later: pending owner decisions

Decisions that are waiting on the owner. Each one says what it blocks and what will inform it. When one is decided, record it in PROJECT_CONTEXT §16 (and the section it changes), then remove it from this list.

_Last updated: 2026-10-24 (after the demo merge)._

## Waiting on the GA runs (T97)

Start the queue first: `.venv\Scripts\python scripts\queue_overnight.py configs\queue\e10_ga.yaml` (control run: Training Ground, 300 generations; main run: 7 maps, until the validation plateau).

### 1. Drop PPO and keep only neuroevolution?
- **Update 2026-10-24 (E10 ran):** under identical inputs and rules PPO (R023, 5M steps) beat both GA seeds on unseen completion (100 % vs 75 / 94 %) and lap ratio (0.975 vs 1.05 / 1.07). One seed each, so this isn't conclusive. The owner's condition for dropping PPO ("if GA turns out better") is **not met**. Cheapest next test: a second seed of R023.
- **Owner's leaning (2026-10-24):** drop PPO entirely if GA turns out better.
- **Why it's open:** PPO's best result (R015) doesn't reproduce (R018: 0 % on the test maps; R019 stuck early). GA's single σ-test run (S004) reached 100 % seen / 87.5 % unseen / lap ratio 0.997, but on one seed, and its checkpoints vary a lot.
- **Decide after:** the main GA run is evaluated (protocol v2).
- **Things to settle with it:**
  - What "drop" means: stop training new PPO runs but keep R001–R019 as the E10 comparison and in the demo? (Models stay loadable either way, §9.) Deleting the PPO code would break loading the PPO models, so it isn't an option.
  - It changes a goal: PROJECT_CONTEXT §1, §3 (Algorithm), §8, §11 and §12 name PPO as the main method. They all need updating.
  - **The E10 comparison isn't like for like yet:** GA uses `rays7_lean` (10 inputs) with track limits on, while PPO R015 uses `rays7_narrow` (13 inputs) with up to 1 s on the grass allowed. A claim that "GA is better" either needs a PPO run on `rays7_lean` with track limits, or has to state these differences.
  - One GA seed isn't enough for a claim either. Should a second main-run seed be queued?

### 2. T83: PPO's run-to-run variance
- **Options:** more seeds per setting; higher entropy (`--set ppo.ent_coef=…`); a curriculum; or nothing.
- **Depends on #1.** If PPO is dropped, this goes away (state the variance in the report as a finding). If PPO stays, pick an option so the overnight queue has work.

### 3. GA inputs: bring back the dropped inputs?
- D26 dropped yaw rate, on-road fraction, previous action and sideways speed, "to be revisited if driving is worse" (T88–T90).
- **Decide after:** you've watched the main run's best checkpoints drive (twitchy steering, slides it doesn't catch). Adding them back would be a new obs preset, not a change to `rays7_lean`.

## Demo and showcase

### 4. Where do faculty-drawn tracks go?
- Currently Draw-a-Track saves to `tracks/user/`. `final_mix` trains on `tracks/user/`, but `maps7`/`train7` don't.
- **Options:** keep `tracks/user/` (safe as long as no new run uses `final_mix`); or save showcase tracks to `tracks/test/` so they are never trained on.
- **Needed before:** the final showcase.

### 5. Faculty track input: editor or a photo of paper?
- **Update 2026-10-24:** the game-window editor was dropped (D28); the sandbox's *Draw a Track* page is the editor now. A photo import stays a possible extension.
- Currently: the in-game editor (being built now, T08). Importing a photo would be an extension.

### 6. Demo physics default: each model's own, or one shared preset?
- Currently each model uses its own, with a toggle to a shared preset (§10.1). With GA, also: should GA cars race with track limits on (their own rule) while PPO cars don't?

### 7. Should rays also detect the barrier?
- Currently they detect only the road edge. This is a low priority; changing it would mean a new feature, so only new models would use it.

## Housekeeping

### 8. `final_mix` and `tracks/user/`
- `final_mix` trains on every track in `tracks/user/`, so tracks drawn in the demo become training data for any later `final_mix` run. Old snapshots are unaffected. Should `final_mix` be marked as "don't use for new runs"? Linked to #4.

### 9. Handbrake output layout (owner request 2026-10-26, D30)
- A `steer_gas_handbrake` layout (a rear-wheel lock that allows deliberate drifts) needs a new physics version (D23: new equations = new version). Not built yet: decide whether the drift behaviour is worth a physics version and what it should do to rear grip.
- Lap progress / lap time as an input was asked about: not added (it lets a car memorise one track, which hurts generalisation; time elapsed carries no driving information). Revisit if a single-track speed record is wanted.
