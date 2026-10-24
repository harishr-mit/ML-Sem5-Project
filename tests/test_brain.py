"""Brain panel (T128): activation capture on real SB3 policies, neuron colours, the panel drawing."""
import os

import numpy as np
import pytest
from gymnasium import spaces
from torch import nn

from racer.agents.activations import ActivationProbe

OBS, ACT = spaces.Box(-1, 1, (10,), np.float32), spaces.Box(-1, 1, (2,), np.float32)


def ppo_policy(arch=(8, 6), act=nn.Tanh):
    from stable_baselines3.common.policies import ActorCriticPolicy
    return ActorCriticPolicy(OBS, ACT, lambda _: 1e-3, net_arch=list(arch), activation_fn=act)


def sac_policy(arch=(8, 6)):
    from stable_baselines3.sac.policies import SACPolicy
    return SACPolicy(OBS, ACT, lambda _: 1e-3, net_arch=list(arch))


@pytest.mark.parametrize("make", [ppo_policy, sac_policy])
def test_probe_matches_the_real_action_and_sees_every_hidden_layer(make):
    import torch
    pol = make()
    probe = ActivationProbe(pol)
    assert probe.supported and probe.layer_sizes == [8, 6]
    obs = np.linspace(-0.9, 0.9, 10).astype(np.float32)
    snap = probe.capture(obs)
    assert [len(layer) for layer in snap.layers] == [8, 6]
    with torch.no_grad():
        expected = pol._predict(torch.as_tensor(obs[None, :]), deterministic=True).numpy()[0]
    assert snap.action == pytest.approx(np.clip(expected, -1, 1), abs=1e-6)
    assert snap.inputs == pytest.approx(obs)
    again = probe.capture(obs)                                   # hooks don't pile up between calls
    assert [len(layer) for layer in again.layers] == [8, 6]


def test_tanh_activations_stay_inside_one_and_relu_is_not_negative():
    obs = np.random.default_rng(0).uniform(-1, 1, 10).astype(np.float32)
    assert all(np.abs(layer).max() <= 1.0 for layer in ActivationProbe(ppo_policy()).capture(obs).layers)
    relu = ActivationProbe(ppo_policy(act=nn.ReLU)).capture(obs)
    assert all(layer.min() >= 0.0 for layer in relu.layers)


def test_probe_uses_the_normaliser():
    probe = ActivationProbe(ppo_policy(), normalise=lambda x: x * 0.0)
    snap = probe.capture(np.ones(10, np.float32))
    assert not snap.inputs.any()


def test_unknown_policy_shape_is_reported_not_fatal():
    class Odd(nn.Module):
        action_space = ACT

        def _predict(self, x, deterministic=True):
            import torch
            return torch.zeros(x.shape[0], 2)

    probe = ActivationProbe(Odd())
    assert not probe.supported
    snap = probe.capture(np.zeros(10, np.float32))
    assert snap.layers == [] and snap.action.shape == (2,)


def test_grid_shape_and_neuron_colours():
    from racer.render.brain_panel import grid_shape, neuron_image
    assert grid_shape(256) == (16, 16) and grid_shape(10) == (3, 4) and grid_shape(1) == (1, 1)
    img = neuron_image(np.array([-1.0, 0.0, 1.0, 0.5]))
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    neg, zero, pos = img[0, 0], img[0, 1], img[1, 0]
    assert neg[2] > neg[0] and pos[0] > pos[2]                        # blue negative, orange positive
    assert zero.max() < 40 and pos.max() > 200                          # zero is dark, strong is bright
    assert neuron_image(np.array([5.0]), scale=5.0)[0, 0].max() > 200   # scaled by the layer's range


def test_brain_panel_draws_for_a_network_and_for_a_driver_without_one():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.agents.activations import Snapshot
    from racer.config import load_preset
    from racer.render.brain_panel import BrainPanel, ModelInfo

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    fonts = (pygame.font.Font(None, 22), pygame.font.Font(None, 60), pygame.font.Font(None, 16))
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    panel = BrainPanel(fonts, colours)
    info = ModelInfo("R000_x@1k", [f"in{i}" for i in range(10)], ["steer", "gas"], "10 > 8 > 6 > 2 · tanh")
    probe = ActivationProbe(ppo_policy())
    obs = np.linspace(-0.9, 0.9, 10).astype(np.float32)
    before = pygame.image.tostring(screen, "RGB")
    panel.draw(screen, (10, 10), info, probe.capture(obs), obs)
    assert pygame.image.tostring(screen, "RGB") != before
    for _ in range(3):                                                   # smoothing keeps working across frames
        panel.draw(screen, (10, 10), info, probe.capture(-obs), -obs)
    panel.draw(screen, (10, 10), info, Snapshot(obs, [], np.zeros(2, np.float32)), obs)    # unsupported shape
    panel.draw(screen, (10, 10), info, None)                             # no network at all
    panel.reset()
    pygame.quit()
