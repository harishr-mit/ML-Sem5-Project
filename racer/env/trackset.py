"""Track sets: which tracks an env samples at each reset (PROJECT_CONTEXT §8, D7).

A track set preset (``configs/trackset/<name>.yaml``) mixes

* **fixed** tracks: by name (``{track: technical, weight: 1}``) or a whole folder of
  ``tracks/`` (``{folder: showcase, weight: 1}``; the weight applies to each track in it).
  The track library is open (§6): a JSON dropped into a folder is picked up by name.
* a **generated** pool: tracks from the training seed stream (0), never the held-out one.
  Building a field costs ~65 ms and 3–14 MB, so each env keeps ``pool_size`` generated
  tracks and swaps the oldest for a new one every ``refresh_episodes`` resets.

:func:`resolve_trackset` expands folders into explicit names, so a model's snapshot keeps
meaning exactly what it trained on even after more tracks are added to a folder.
Old snapshots store a plain list of names; :class:`TrackSampler` accepts both forms.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Mapping, Sequence

import numpy as np

from racer.config import ConfigError, deep_merge, load_preset
from racer.track.generator import TRAIN_STREAM, generate
from racer.track.io import TRACKS_DIR, resolve_track_path
from racer.track.track import Track

HELDOUT_DIR = TRACKS_DIR / "heldout"
# Folders whose tracks are never used in training: generated held-out tracks and the
# hand-designed test maps (D19).
PROTECTED_DIRS = (HELDOUT_DIR, TRACKS_DIR / "test")
FIXED_KEYS = {"track", "folder", "weight"}
GENERATED_KEYS = {"weight", "seeds", "pool_size", "refresh_episodes", "track_overrides"}

# Process-wide LRU cache of loaded tracks (fixed tracks are immutable; fields cost MBs).
_CACHE_MAX = 24
_TRACK_CACHE: "OrderedDict[tuple[str, str], Track]" = OrderedDict()


def _cfg_key(track_cfg: Mapping[str, Any]) -> str:
    return json.dumps(track_cfg, sort_keys=True, default=str)


def get_track(name: str, track_cfg: Mapping[str, Any]) -> Track:
    """Load a track file by name (cached per name and track config)."""
    key = (name, _cfg_key(track_cfg))
    track = _TRACK_CACHE.get(key)
    if track is None:
        track = Track.load(name, track_cfg)
        _TRACK_CACHE[key] = track
        while len(_TRACK_CACHE) > _CACHE_MAX:
            _TRACK_CACHE.popitem(last=False)
    else:
        _TRACK_CACHE.move_to_end(key)
    return track


def folder_tracks(folder: str) -> list[str]:
    """Names of all track JSONs in ``tracks/<folder>/`` (sorted)."""
    d = TRACKS_DIR / folder
    if not d.is_dir():
        raise ConfigError(f"no track folder {d}")
    return sorted(p.stem for p in d.glob("*.json"))


def is_heldout(name: str) -> bool:
    """True for tracks that must never be trained on (``heldout/`` and ``test/``)."""
    try:
        parent = resolve_track_path(name).parent.resolve()
    except FileNotFoundError:
        return False
    return any(parent == d.resolve() for d in PROTECTED_DIRS)


def resolve_trackset(name: str) -> dict[str, Any]:
    """Load a track set preset and expand it to the explicit form stored in snapshots."""
    cfg = load_preset("trackset", name)
    unknown = set(cfg) - {"fixed", "generated"}
    if unknown:
        raise ConfigError(f"trackset {name}: unknown keys {sorted(unknown)}")
    fixed: list[dict[str, Any]] = []
    for item in cfg.get("fixed") or []:
        bad = set(item) - FIXED_KEYS
        if bad or ("track" in item) == ("folder" in item):
            raise ConfigError(f"trackset {name}: each fixed entry needs 'track' or 'folder' (+ weight), got {item}")
        names = [item["track"]] if "track" in item else folder_tracks(item["folder"])
        for n in names:
            resolve_track_path(n)   # must exist
            if is_heldout(n):
                raise ConfigError(f"trackset {name}: {n!r} is a held-out / test track (never used in training)")
            fixed.append({"track": n, "weight": float(item.get("weight", 1.0))})
    gen = cfg.get("generated")
    if gen is not None:
        bad = set(gen) - GENERATED_KEYS
        missing = GENERATED_KEYS - set(gen)
        if bad or missing:
            raise ConfigError(f"trackset {name}: generated keys unknown {sorted(bad)} / missing {sorted(missing)}")
        gen = dict(gen)
    if not fixed and gen is None:
        raise ConfigError(f"trackset {name} is empty")
    return {"trackset": name, "fixed": fixed, "generated": gen}


class TrackSampler:
    """Picks the track for each episode from a list of names or a resolved track set."""

    def __init__(self, tracks: Sequence[str] | Mapping[str, Any], track_cfg: Mapping[str, Any]) -> None:
        self.track_cfg = track_cfg
        if isinstance(tracks, Mapping):
            self.legacy = False
            self.fixed = [str(f["track"]) for f in tracks["fixed"]]
            weights = [float(f["weight"]) for f in tracks["fixed"]]
            self.gen = tracks.get("generated")
            if self.gen:
                weights.append(float(self.gen["weight"]))
                self.gen_cfg = deep_merge(track_cfg, self.gen.get("track_overrides") or {}, strict=True)
            w = np.asarray(weights, dtype=np.float64)
            if not len(w) or (w < 0).any() or w.sum() <= 0:
                raise ValueError("track set weights must be non-negative with a positive sum")
            self.p = w / w.sum()
        else:
            self.legacy = True
            self.fixed = [str(t) for t in tracks]
            self.gen = None
            if not self.fixed:
                raise ValueError("config['tracks'] is empty")
        self.pool: list[Track] = []
        self.episodes = 0

    @property
    def names(self) -> list[str]:
        """Fixed track names (generated tracks have no file)."""
        return list(self.fixed)

    def sample(self, rng: np.random.Generator) -> Track:
        if self.legacy:   # same RNG use as Week 1, so old configs sample identically
            return get_track(self.fixed[int(rng.integers(len(self.fixed)))], self.track_cfg)
        self.episodes += 1
        k = int(rng.choice(len(self.p), p=self.p))
        if k < len(self.fixed):
            return get_track(self.fixed[k], self.track_cfg)
        return self._generated(rng)

    def _generated(self, rng: np.random.Generator) -> Track:
        g = self.gen
        while len(self.pool) < int(g["pool_size"]):
            self.pool.append(self._new_track(rng))
        if self.episodes % int(g["refresh_episodes"]) == 0:
            self.pool.pop(0)
            self.pool.append(self._new_track(rng))
        return self.pool[int(rng.integers(len(self.pool)))]

    def _new_track(self, rng: np.random.Generator) -> Track:
        lo, hi = self.gen["seeds"]
        for _ in range(100):
            seed = int(rng.integers(lo, hi))
            try:
                track, _ = generate(seed, TRAIN_STREAM, name=f"gen_{seed}", track_cfg=self.gen_cfg)
                return track
            except RuntimeError:
                continue
        raise RuntimeError("could not generate a valid training track")
