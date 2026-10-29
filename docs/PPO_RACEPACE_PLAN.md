# PPO race-pace runs (T164, 2026-10-29)

Goal (owner): the fastest car around any map, on a proper racing line.

## Diagnosis of the models so far
- Best of each run (protocol v2): lap ratio 0.96-1.00 against the PID baseline, which follows the centre line at a safe
  speed. Mean speed 24.6 m/s (baseline 24.95). Mean lateral g 0.55-0.7 where the road gives about 1.1. So the best models
  finish laps reliably but do not race: R023@1700k (PPO, 0.975), R031@34627k (GA, 0.964).
- Why (our reward): the progress terms add up to the lap length whatever the speed. The only pace signal was `time` 0.05/s
  (a 60 s lap = 3 reward, 5 % faster = 0.15), against a fixed crash of 20. Safe slow driving is the optimum.
- Why (our inputs): rays see the road edge only; nothing says a corner is coming past the ray range, so braking points
  are guessed. R023 had 10 inputs, R031 9.
- Exploration: R019's PPO policy std collapsed to 0.04 and never recovered (known PPO failure, PPO-CMA paper).

## What others do (sources)
- Gran Turismo Sport, Fuchs et al., arXiv 2008.07971: reward = course progress per step; a high gamma makes progress a good
  proxy for lap time; a fixed wall penalty made agents either ignore it or stand still; the fix was a speed-scaled penalty.
- Gran Turismo Sophy (Sony AI blog "Training the world's fastest GT racer", Nature 2022): progress + speed-scaled barrier and
  off-track penalties + tyre-slip penalty; the observation includes track points up to 6 s ahead; lap time is not rewarded directly.
- Head-to-head racing with curriculum RL, arXiv 2308.13491: training on many tracks in parallel (both directions) against
  overfitting; a negative reward for braking unnecessarily (speed below the target window).
- Map-free racing, arXiv 2604.09499: PPO stayed closer to the minimum-curvature line than SAC and had a physics-exploit
  reward fixed in 50k steps (SAC: 6M); the policy was within the friction circle limit on unseen tracks.
- A minimum-curvature line gives faster laps than the centre line as a reference (several papers, e.g. arXiv 2506.06077 survey hits).
- PPO-CMA (paper): PPO's exploration variance can shrink too early in continuous control.

## What we change (config only, no engine code)
1. Reward `racepace`: time 1.0/s (was 0.05). A 5 % faster 60 s lap is worth 3; a crash still costs 20, so a 15 % higher crash risk
   cancels that gain. Everything else as `progress`.
2. Inputs `rays7_pen` (17, D33, owner idea): rays7_lean's seven angles, all 200 m, each ray passing through the road edge (feature `rays2`): the
   first crossing and the second (the way back onto the road). A real driver sees the exit of a bend across the infield; a ray that
   stops at the first edge is tunnel vision. Geometry along the car's own rays only, so it needs no track look-up and works on a drawn
   track. (The curvature look-ahead `rays7_look` was dropped: the owner found it unfair to a real driver.) Display: `P` in the game draws the part beyond the first edge faintly; it doesn't change what the model sees.
3. Training `ppo_race`: ent_coef 0.005, 6M steps (R023's best checkpoint was at 1.7M).
4. Runs (queue `ppo_racepace.yaml`, 2 at a time, about 90 % CPU): C1, C2 = rays7_pen + racepace (PPO, seeds 1 and 2); A = rays7_lean + racepace (reward alone); GA = the neuroevolution twin of C (same inputs, rules and network; staged fitness as R031; `ga.normalise` = frozen input standardiser measured from the baseline, D33).

## Success
Unseen completion >= 90 % AND lap ratio < 0.90 (R023: 100 % / 0.975), with mean lateral g above 0.8. Compare A against C to
see which change did it; C seeds 1 and 2 to see the variance; GA against C for the method.

## Not tried yet (next, if these fail)
Speed-scaled crash penalty and a tyre-slip term (needs a new reward type and a `sliding` argument); a minimum-curvature
reference line as an extra reward (time-optimal line from the track, new racing-line reward); SAC (GT Sophy used QR-SAC).
