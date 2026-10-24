"""Hidden-layer activations of a policy for the brain panel (T128, PROJECT_CONTEXT §10.2).

One forward pass through the real ``policy._predict`` with hooks on the activation modules of the
actor's hidden stack (PPO/GA ``mlp_extractor.policy_net``, SAC ``actor.latent_pi``), so what the
panel shows is exactly what drives the car. A policy of another shape is still run (inputs and
outputs shown) but reports ``layers == []``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Snapshot:
    inputs: np.ndarray                       # what the network saw (after the normaliser)
    layers: list[np.ndarray] = field(default_factory=list)    # hidden activations, one array per hidden layer
    action: np.ndarray = field(default_factory=lambda: np.zeros(0))


def hidden_stack(policy: Any):
    """The ``nn.Sequential`` holding the actor's hidden layers, or ``None``."""
    ex = getattr(policy, "mlp_extractor", None)
    if ex is not None and hasattr(ex, "policy_net"):
        return ex.policy_net
    actor = getattr(policy, "actor", None)
    return getattr(actor, "latent_pi", None)


class ActivationProbe:
    def __init__(self, policy: Any, normalise: Callable[[np.ndarray], np.ndarray] | None = None,
                 low: np.ndarray | None = None, high: np.ndarray | None = None) -> None:
        import torch
        from torch import nn

        self._torch = torch
        self.policy = policy
        self.normalise = normalise or (lambda x: x)
        space = policy.action_space
        self.low = space.low if low is None else low
        self.high = space.high if high is None else high
        self._acts: list[Any] = []
        stack = hidden_stack(policy)
        self.layer_sizes: list[int] = []
        if stack is not None:
            for m in stack:
                if isinstance(m, nn.Linear):
                    self.layer_sizes.append(m.out_features)
                else:
                    m.register_forward_hook(lambda _m, _i, out: self._acts.append(out.detach().cpu().numpy()[0]))
        self.supported = bool(self.layer_sizes)

    def capture(self, obs: np.ndarray) -> Snapshot:
        """Run one raw observation through the network; returns the activations and the action."""
        torch = self._torch
        x_np = np.asarray(self.normalise(np.asarray(obs, dtype=np.float32)[None, :]), dtype=np.float32)
        self._acts.clear()
        with torch.no_grad():
            a = self.policy._predict(torch.as_tensor(x_np), deterministic=True).cpu().numpy()[0]
        layers = list(self._acts) if len(self._acts) == len(self.layer_sizes) else []
        self._acts.clear()
        return Snapshot(x_np[0], layers, np.clip(a, self.low, self.high).astype(np.float32))
