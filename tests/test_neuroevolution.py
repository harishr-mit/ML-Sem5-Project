"""Neuroevolution core (D26): selection, mutation, fitness ordering, lineage rebuild."""
import numpy as np
import pytest

from racer.agents import neuroevolution as ne

SIZES = ne.layer_sizes(10, [16, 16], 2)     # small net: same code path as 10 → 256 → 256 → 2


def evolve(n_gen: int, pop: int = 10, n_elite: int = 3, sigma: float = 0.05, run_seed: int = 7):
    """Run the GA on random scores; return every generation's offspring (the live lineage)."""
    gens = [ne.first_generation(SIZES, pop, np.random.default_rng([run_seed, 0]))]
    for g in range(1, n_gen):
        scores = np.random.default_rng([run_seed, g, 99]).random(pop)
        gens.append(ne.next_generation(gens[-1].weights, scores, n_elite, sigma, np.random.default_rng([run_seed, g])))
    return gens


def test_full_size_network_has_69122_weights():
    assert ne.n_weights(ne.layer_sizes(10, [256, 256], 2)) == 69_122


def test_elites_survive_unchanged_best_first():
    w = ne.first_generation(SIZES, 10, np.random.default_rng(0)).weights
    scores = np.array([0.1, 0.9, 0.3, 1.5, 0.2, 0.0, 0.8, 0.4, 0.5, 0.6])
    off = ne.next_generation(w, scores, 3, 0.05, np.random.default_rng(1))
    assert list(off.parents[:3]) == [3, 1, 6]
    assert np.array_equal(off.weights[:3], w[[3, 1, 6]])
    assert (off.seeds[:3] == ne.NO_NOISE).all()


def test_children_come_only_from_the_elites_and_are_mutated():
    w = ne.first_generation(SIZES, 10, np.random.default_rng(0)).weights
    off = ne.next_generation(w, np.arange(10.0), 3, 0.05, np.random.default_rng(1))
    assert set(off.parents[3:]) <= {9, 8, 7}
    for k in range(3, 10):
        diff = off.weights[k] - w[off.parents[k]]
        assert 0.03 < diff.std() < 0.07          # σ = 0.05 on every weight
        assert np.count_nonzero(diff) > 0.99 * diff.size


def test_same_seed_same_population():
    a, b = evolve(5), evolve(5)
    for x, y in zip(a, b):
        assert np.array_equal(x.weights, y.weights)
    assert not np.array_equal(evolve(5, run_seed=8)[-1].weights, a[-1].weights)


def test_ties_keep_the_earlier_car():
    scores = np.r_[np.zeros(40), np.ones(10)]          # population-sized: numpy's default sort is unstable here
    assert list(ne.rank(scores)) == list(range(40, 50)) + list(range(40))


def fit(progress, finished, time_s=100.0, length=1000.0):
    return ne.fitness(progress, length_m=length, finished=finished, time_s=time_s, speed_scale=70.0)


@pytest.mark.parametrize("almost", [0.0, 0.5, 0.99999])
def test_a_finisher_beats_every_non_finisher(almost):
    slow_lap = fit(1000.0, True, time_s=1000.0)        # 1 m/s
    assert slow_lap > fit(almost * 1000.0, False)
    assert fit(1000.0, False, time_s=50.0) < 1.0       # a full lap that then failed isn't a finish


def test_faster_lap_scores_higher_and_progress_orders_failures():
    assert fit(1000.0, True, time_s=40.0) > fit(1000.0, True, time_s=50.0)
    assert fit(1000.0, True, time_s=40.0) == pytest.approx(1.0 + 25.0 / 70.0)
    assert fit(600.0, False) > fit(300.0, False)
    assert fit(-20.0, False) == 0.0                    # reversed off the start


def test_rebuild_from_lineage_is_bit_identical():
    gens = evolve(12)

    def origin(g, c):
        return int(gens[g].parents[c]), int(gens[g].seeds[c])

    for g, c in [(0, 4), (5, 0), (5, 7), (11, 0), (11, 9)]:
        assert np.array_equal(ne.rebuild(origin, g, c, SIZES, 0.05), gens[g].weights[c])


def test_forward_batch_matches_single_and_is_clipped():
    w = ne.init_weights(SIZES, 3)
    layers = ne.unflatten(w, SIZES)
    obs = np.random.default_rng(0).uniform(-1, 1, (20, 10)).astype(np.float32)
    batch = ne.forward(layers, obs)
    assert batch.shape == (20, 2) and batch.dtype == np.float32
    assert np.allclose(batch, np.stack([ne.forward(layers, o) for o in obs]), atol=1e-6)
    assert (np.abs(batch) <= 1.0).all()


def test_layout_is_torch_linear_out_by_in():
    """Export copies these weights into SB3's Linear layers, which store W as (out, in)."""
    sizes = [2, 2, 1]
    flat = np.array([0, 1, 0, 0,  0, 0,  1, 0,  0], dtype=np.float32)   # W1 row 0 = [0, 1]: h0 = x1; out = h0
    x = np.array([0.3, 0.7], dtype=np.float32)
    assert ne.forward(ne.unflatten(flat, sizes), x)[0] == pytest.approx(np.tanh(0.7))


def test_generation_zero_cars_differ_and_drive():
    off = ne.first_generation(SIZES, 10, np.random.default_rng(0))
    obs = np.random.default_rng(1).uniform(-1, 1, 10).astype(np.float32)
    gas = [ne.forward(ne.unflatten(w, SIZES), obs)[1] for w in off.weights]
    assert np.std(gas) > 0.2
    assert len({w.tobytes() for w in off.weights}) == 10


def test_bad_inputs_raise_instead_of_corrupting_a_run():
    w = ne.first_generation(SIZES, 10, np.random.default_rng(0)).weights
    with pytest.raises(ValueError):
        ne.unflatten(np.zeros(5, dtype=np.float32), SIZES)
    with pytest.raises(ValueError):          # a worker lost a result
        ne.next_generation(w, np.zeros(9), 3, 0.05, np.random.default_rng(0))
    with pytest.raises(ValueError):
        ne.next_generation(w, np.zeros(10), 0, 0.05, np.random.default_rng(0))


# ----------------------------------------------------------------- D31: fitness v2, σ schedule, smoothing
V2 = dict(ref_speed=25.0, fail_speed_frac=0.3, speed_k=2.0, finish_bonus=0.5)
LAP = 1000.0


def f2(progress, time_s, finished):
    return ne.fitness_v2(progress, length_m=LAP, finished=finished, time_s=time_s, **V2)


def test_v2_finisher_at_reference_pace_scores_one_and_a_half():
    assert f2(LAP, LAP / 25.0, True) == pytest.approx(1.5)


def test_v2_faster_laps_always_score_more_and_each_step_more_than_the_last():
    times = [LAP / v for v in (15, 20, 25, 30, 35)]
    scores = [f2(LAP, t, True) for t in times]
    assert scores == sorted(scores)
    steps = np.diff(scores)
    assert all(steps[i + 1] > steps[i] for i in range(len(steps) - 1))


def test_v2_a_fast_late_crash_beats_a_slow_finisher_but_an_early_crash_does_not():
    slow_finisher = f2(LAP, LAP / 17.5, True)                 # 0.7 × reference pace
    late_fast_crash = f2(0.9 * LAP, 0.9 * LAP / 50.0, False)  # 90 % of the lap at 2 × reference pace
    early_crash = f2(0.5 * LAP, 0.5 * LAP / 37.5, False)
    assert late_fast_crash > slow_finisher > early_crash


def test_v2_a_finisher_gets_the_bonus_and_failures_do_not():
    assert f2(LAP, 40.0, True) - f2(LAP, 40.0, False) == pytest.approx(0.5)


def test_v2_going_backwards_or_standing_still_scores_zero_or_more():
    assert f2(-5.0, 10.0, False) == pytest.approx(f2(0.0, 10.0, False)) and f2(0.0, 0.0, False) >= 0.0


def test_sigma_is_constant_without_a_floor_and_shrinks_with_one():
    assert ne.sigma_at(500, 0.02, None, 100.0) == 0.02
    s = [ne.sigma_at(g, 0.02, 0.005, 100.0) for g in (0, 100, 1000, 3000, 9000)]
    assert s[0] == 0.02 and s == sorted(s, reverse=True) and s[-1] == 0.005 and s[1] == pytest.approx(0.02 / (1 + np.log(2)))


def test_rebuild_with_a_sigma_per_generation_is_bit_identical():
    sig = lambda g: ne.sigma_at(g, 0.05, 0.01, 3.0)       # noqa: E731
    pop, gens = 10, [ne.first_generation(SIZES, 10, np.random.default_rng([7, 0]))]
    for g in range(1, 12):
        scores = np.random.default_rng([7, g, 99]).random(pop)
        gens.append(ne.next_generation(gens[-1].weights, scores, 3, sig(g), np.random.default_rng([7, g])))

    def origin(g, c):
        return int(gens[g].parents[c]), int(gens[g].seeds[c])

    for g, c in [(5, 0), (5, 7), (11, 0), (11, 9)]:
        assert np.array_equal(ne.rebuild(origin, g, c, SIZES, sig), gens[g].weights[c])
    assert not np.array_equal(ne.rebuild(origin, 11, 9, SIZES, 0.05), gens[11].weights[9])


def test_validation_tracking_without_smoothing_is_the_old_rule():
    best, gain, recent = None, 0, []
    for g, val in [(19, 1.0), (39, 1.005), (59, 1.02), (79, 1.0)]:
        recent, best, gain = ne.track_validation(recent, val, best, gain, g, 0.01, 1)
    assert (best, gain) == (1.02, 59)


def test_validation_smoothing_ignores_one_lucky_validation():
    best, gain, recent = None, 0, []
    for g, val in [(19, 1.0), (39, 1.0), (59, 1.0), (79, 1.3), (99, 1.0), (119, 1.0)]:
        recent, best, gain = ne.track_validation(recent, val, best, gain, g, 0.01, 3)
    assert gain == 79 and best == pytest.approx(1.1) and len(recent) == 3
