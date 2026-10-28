"""Neuroevolution core (D26, docs/NEUROEVOLUTION_PLAN.md): networks as flat weight vectors,
fitness, selection and mutation. Pure numpy: no env, workers or files.

A network is one float32 vector holding, layer by layer, the weight matrix (out, in) in
row-major order and then the bias, which is the order and shape of the SB3 actor's Linear
layers, so export is a copy. Hidden layers use tanh; the output is linear and clipped to
[−1, 1], like PPO's deterministic ``predict``. Any car can be rebuilt from its lineage
(:func:`rebuild`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

NO_PARENT = -1     # parents column: generation 0
NO_NOISE = -1      # seeds column: an elite copied unchanged


def layer_sizes(obs_dim: int, net_arch: Sequence[int], act_dim: int) -> list[int]:
    return [int(obs_dim), *(int(n) for n in net_arch), int(act_dim)]


def n_weights(sizes: Sequence[int]) -> int:
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def unflatten(flat: np.ndarray, sizes: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(W, b) views per layer."""
    if flat.shape != (n_weights(sizes),):
        raise ValueError(f"expected {n_weights(sizes)} weights, got {flat.shape}")
    layers, i = [], 0
    for n_in, n_out in zip(sizes, sizes[1:]):
        w = flat[i:i + n_in * n_out].reshape(n_out, n_in)
        i += n_in * n_out
        layers.append((w, flat[i:i + n_out]))
        i += n_out
    return layers


def forward(layers: Sequence[tuple[np.ndarray, np.ndarray]], obs: np.ndarray) -> np.ndarray:
    """Actions in [−1, 1] for one observation (obs_dim,) or a batch (n, obs_dim)."""
    x = np.asarray(obs, dtype=np.float32)
    last = len(layers) - 1
    for k, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if k < last:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)


def init_weights(sizes: Sequence[int], seed: int) -> np.ndarray:
    """A random network: W ~ N(0, 1/fan_in), b = 0.

    Pre-activations stay at the input's scale, out of tanh saturation, and the output layer gets
    the same scale (not PPO's near-zero init), so generation 0 already steers and accelerates in
    varied ways instead of every car sitting still.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        parts.append(rng.standard_normal(n_in * n_out, dtype=np.float32) / np.float32(np.sqrt(n_in)))
        parts.append(np.zeros(n_out, dtype=np.float32))
    return np.concatenate(parts)


def mutate(parent: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal(parent.shape[0], dtype=np.float32)
    return parent + np.float32(sigma) * noise


def fitness(progress_m: float, *, length_m: float, finished: bool, time_s: float, speed_scale: float) -> float:
    """Lap fraction covered (0–1) if the try failed; 1 + mean speed / speed_scale if it finished a lap.

    A finisher always outranks a non-finisher, and a faster lap outranks a slower one.
    """
    if finished:
        return 1.0 + (progress_m / time_s) / speed_scale
    return float(min(max(progress_m / length_m, 0.0), np.nextafter(1.0, 0.0)))


def fitness_v2(progress_m: float, *, length_m: float, finished: bool, time_s: float, ref_speed: float,
               fail_speed_frac: float, speed_k: float, finish_bonus: float) -> float:
    """Fitness v2 (D31): a convex score of the (virtual) lap speed, plus a bonus for finishing.

    A failed try is charged its unfinished distance at ``fail_speed_frac * ref_speed``, so a fast crash
    late in the lap can beat a slow finisher but an early crash cannot. rho = lap speed / ref_speed.
    shortcut: one fixed ref_speed, not a per-track baseline lap time; same order within a map.
    """
    progress = min(max(progress_m, 0.0), length_m)
    t = time_s if finished else time_s + (length_m - progress) / (fail_speed_frac * ref_speed)
    rho = length_m / (max(t, 1e-9) * ref_speed)
    score = np.expm1(speed_k * rho) / np.expm1(speed_k)
    return float(score + (finish_bonus if finished else 0.0))


def sigma_at(generation: int, sigma: float, floor: float | None, g0: float) -> float:
    """Mutation size of the children born in ``generation``: constant without a floor (D31 schedule otherwise)."""
    if floor is None:
        return sigma
    return float(max(floor, sigma / (1.0 + np.log1p(generation / g0))))


def track_validation(recent: list, value: float, best: float | None, last_gain: int, generation: int,
                     min_gain: float, smooth: int) -> tuple[list, float | None, int]:
    """Plateau bookkeeping: the mean of the last ``smooth`` validations must beat the best by ``min_gain``.
    ``smooth`` 1 is the original rule (the value itself)."""
    recent = (list(recent) + [float(value)])[-max(1, smooth):]
    avg = sum(recent) / len(recent)
    if best is None or avg > best * (1 + min_gain):
        best, last_gain = avg, generation
    return recent, best, last_gain


def stage_ready(finished: Sequence[int], window: int, min_finishers: int, share: float) -> bool:
    """D32: have the cars completed laps consistently? ``finished`` = cars that finished a lap in each
    generation so far; true when, in at least ``share`` of the last ``window`` generations, at least
    ``min_finishers`` finished."""
    if len(finished) < window:
        return False
    return sum(c >= min_finishers for c in finished[-window:]) >= share * window


def rank(scores: np.ndarray) -> np.ndarray:
    """Car indices best first; ties keep the lower index (elites sit first, so they win ties)."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


@dataclass(frozen=True)
class Generation:
    weights: np.ndarray     # (population, n_weights)
    parents: np.ndarray     # index in the previous generation; NO_PARENT in generation 0
    seeds: np.ndarray       # init seed in generation 0, noise seed after; NO_NOISE for an unchanged elite


def next_generation(weights: np.ndarray, scores: np.ndarray, n_elite: int, sigma: float,
                    rng: np.random.Generator) -> Generation:
    """Top ``n_elite`` cars unchanged (best first), then children of those same elites only."""
    pop = weights.shape[0]
    scores = np.asarray(scores)
    if scores.shape != (pop,):
        raise ValueError(f"expected {pop} scores, got shape {scores.shape}")
    if not 0 < n_elite <= pop:
        raise ValueError(f"n_elite must be in 1..{pop}, got {n_elite}")
    order = rank(scores)
    elites = order[:n_elite]
    n_child = pop - n_elite
    child_parents = elites[rng.integers(0, n_elite, size=n_child)]
    child_seeds = rng.integers(0, 2**63 - 1, size=n_child, dtype=np.int64)
    out = np.empty_like(weights)
    out[:n_elite] = weights[elites]
    for j, (p, s) in enumerate(zip(child_parents, child_seeds)):
        out[n_elite + j] = mutate(weights[p], sigma, int(s))
    parents = np.concatenate([elites, child_parents]).astype(np.int64)
    seeds = np.concatenate([np.full(n_elite, NO_NOISE, dtype=np.int64), child_seeds])
    return Generation(out, parents, seeds)


def first_generation(sizes: Sequence[int], population: int, rng: np.random.Generator) -> Generation:
    seeds = rng.integers(0, 2**63 - 1, size=population, dtype=np.int64)
    weights = np.stack([init_weights(sizes, int(s)) for s in seeds])
    return Generation(weights, np.full(population, NO_PARENT, dtype=np.int64), seeds)


def rebuild(origin: Callable[[int, int], tuple[int, int]], generation: int, car: int,
            sizes: Sequence[int], sigma: float | Callable[[int], float]) -> np.ndarray:
    """Weights of ``car`` in ``generation`` from its lineage alone.

    ``origin(gen, car)`` returns that car's (parent index in gen − 1, seed): generation 0 cars
    come from their init seed; an elite is its parent unchanged (seed ``NO_NOISE``); a child is
    its parent + σ × N(0, 1) noise drawn from its seed. ``sigma`` is a number or a function of the
    generation the child was born in (D31 schedule).
    """
    chain = []
    g, c = generation, car
    while g > 0:
        parent, seed = origin(g, c)
        chain.append((g, seed))
        g, c = g - 1, parent
    flat = init_weights(sizes, origin(0, c)[1])
    for g, seed in reversed(chain):
        if seed != NO_NOISE:
            flat = mutate(flat, sigma(g) if callable(sigma) else sigma, int(seed))
    return flat
