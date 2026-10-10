"""Surfaces (PROJECT_CONTEXT §5.1).

``Surface`` is the base and behaves as grass: anything that isn't road falls back to it.
``Road`` overrides it and ``Kerb`` overrides ``Road``, so ``isinstance(s, Road)`` is the
"on track" test and kerbs count as on-track (D1). The numbers come from the physics config;
a subclass with no value for a key inherits its parent's value.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Integer ids used by the track field and the numba kernels.
OUTSIDE_ID = -1   # beyond the barrier (crash); physics treats it as grass
GRASS_ID = 0
ROAD_ID = 1
KERB_ID = 2

PHYSICAL_KEYS = ("mu_static", "mu_kinetic", "rolling_res")


class Surface:
    """Base surface = grass physics."""

    name = "grass"
    surface_id = GRASS_ID
    visual = "grass"   # visual variants (sand, gravel, dirt) share these physics

    def __init__(self, mu_static: float, mu_kinetic: float, rolling_res: float) -> None:
        self.mu_static = float(mu_static)
        self.mu_kinetic = float(mu_kinetic)
        self.rolling_res = float(rolling_res)

    @property
    def is_road(self) -> bool:
        """True for road and kerb (anything that counts as on-track)."""
        return isinstance(self, Road)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(mu_static={self.mu_static}, "
                f"mu_kinetic={self.mu_kinetic}, rolling_res={self.rolling_res})")


class Road(Surface):
    name = "road"
    surface_id = ROAD_ID
    visual = "asphalt"


class Kerb(Road):
    """Thin band inside the road edge on corners."""

    name = "kerb"
    surface_id = KERB_ID
    visual = "kerb"


SURFACE_CLASSES: tuple[type[Surface], ...] = (Surface, Road, Kerb)


class SurfaceSet:
    """The three configured surfaces, plus flat arrays indexed by surface id for fast lookup."""

    def __init__(self, cfg: Mapping[str, Mapping[str, Any]]) -> None:
        unknown = set(cfg) - {cls.name for cls in SURFACE_CLASSES}
        if unknown:
            raise ValueError(f"unknown surfaces in config: {sorted(unknown)}")
        resolved: dict[type[Surface], dict[str, float]] = {}
        self.by_id: dict[int, Surface] = {}
        for cls in SURFACE_CLASSES:
            # Inherit values from the nearest configured parent class.
            parent = cls.__mro__[1]
            values = dict(resolved.get(parent, {}))
            entry = dict(cfg.get(cls.name, {}))
            bad = set(entry) - set(PHYSICAL_KEYS)
            if bad:
                raise ValueError(f"surface {cls.name!r}: unknown keys {sorted(bad)}")
            values.update(entry)
            missing = [k for k in PHYSICAL_KEYS if k not in values]
            if missing:
                raise ValueError(f"surface {cls.name!r}: missing keys {missing}")
            resolved[cls] = values
            self.by_id[cls.surface_id] = cls(**values)

        n = len(SURFACE_CLASSES)
        self.mu_static = np.array([self.by_id[i].mu_static for i in range(n)])
        self.mu_kinetic = np.array([self.by_id[i].mu_kinetic for i in range(n)])
        self.rolling_res = np.array([self.by_id[i].rolling_res for i in range(n)])
        # Plain lists are faster than numpy for scalar indexing in the Python physics loop.
        self.mu_static_l = self.mu_static.tolist()
        self.mu_kinetic_l = self.mu_kinetic.tolist()
        self.rolling_res_l = self.rolling_res.tolist()

    def get(self, surface_id: int) -> Surface:
        """Surface for an id; OUTSIDE_ID maps to grass."""
        return self.by_id[max(surface_id, GRASS_ID)]

    @property
    def grass(self) -> Surface:
        return self.by_id[GRASS_ID]

    @property
    def road(self) -> Surface:
        return self.by_id[ROAD_ID]

    @property
    def kerb(self) -> Surface:
        return self.by_id[KERB_ID]
