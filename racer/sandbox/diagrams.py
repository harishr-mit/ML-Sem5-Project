"""Diagrams for the sandbox (T68): what a model senses and what its network looks like.

* :func:`ray_diagram_straight`: the car to scale on a straight road, each ray drawn to where
  it meets the road edge (or its full range), with the distance labelled.
* :func:`ray_diagram_map`: the same rays cast on a real track by the model's own sensor
  code (``racer.env.sensors``), at any point of the lap.
* :func:`network_diagram` / :func:`layer_table`: inputs → hidden layers → outputs, with
  the parameters per layer (they add up to ``compat.count_params``).

Figures are matplotlib (Agg); :func:`png` turns one into bytes for the page.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

ROAD_WIDTH_M = 12.0            # default road width (track generator default; most maps)
CAR_LENGTH_M, CAR_WIDTH_M = 4.5, 1.9
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]       # one per model when overlaid
GRASS, ROAD, EDGE = "#cfe3c0", "#b9b9b9", "#555555"


def png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def rays_of(inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """The ray list of a model's inputs (``manifest['inputs']``); [] if it has no rays."""
    for f in inputs or []:
        if f.get("feature") == "rays":
            params = f.get("params") or {}
            return [{"angle_deg": float(r["angle_deg"]), "range_m": float(r["range_m"])}
                    for r in params.get("rays") or f.get("rays") or []]
    return []


# --------------------------------------------------------------- straight road
def straight_hits(rays: Sequence[Mapping[str, float]], width_m: float = ROAD_WIDTH_M,
                  offset_m: float = 0.0) -> list[float]:
    """Distance along each ray to the road edge on an endless straight, capped at its range.

    The car points along the road, ``offset_m`` to the left of the centre; a ray at angle
    a (counter-clockwise, + = left) goes sideways at sin(a) per metre."""
    out = []
    for r in rays:
        a = math.radians(float(r["angle_deg"]))
        side = math.sin(a)
        if side > 1e-12:
            d = (width_m / 2 - offset_m) / side
        elif side < -1e-12:
            d = (width_m / 2 + offset_m) / -side
        else:
            d = math.inf                      # parallel to the edges: never meets them
        out.append(min(d, float(r["range_m"])))
    return out


def ray_diagram_straight(models: Sequence[tuple[str, Sequence[Mapping[str, float]]]],
                         width_m: float = ROAD_WIDTH_M, offset_m: float = 0.0, view_m: float | None = None):
    """Rays of one or more models on a straight road, car pointing right, to scale.

    ``models``: (label, rays) pairs, overlaid in different colours. ``view_m``: how far ahead
    to draw (default: just past the farthest edge hit, so short side rays stay readable;
    longer rays end in an arrow with their length)."""
    all_hits = [straight_hits(rays, width_m, offset_m) for _, rays in models]
    finite = [h for hits, (_, rays) in zip(all_hits, models) for h, r in zip(hits, rays) if h < float(r["range_m"])]
    ahead = view_m or max(40.0, min(1.15 * max(finite, default=60.0), 220.0))
    behind = max(15.0, 0.25 * ahead)
    side = width_m / 2 + max(6.0, 0.08 * ahead)
    fig, ax = plt.subplots(figsize=(10, max(2.2, 10 * 2 * side / (ahead + behind) + 0.6)))
    ax.set_facecolor(GRASS)
    ax.add_patch(plt.Rectangle((-behind, -width_m / 2), ahead + behind, width_m, color=ROAD, zorder=0))
    for y in (-width_m / 2, width_m / 2):
        ax.plot([-behind, ahead], [y, y], color=EDGE, lw=1.2, zorder=1)
    ax.plot([-behind, ahead], [0, 0], color="white", lw=1, ls=(0, (6, 6)), zorder=1)
    _car(ax, 0.0, offset_m, 0.0)
    for k, ((label, rays), hits) in enumerate(zip(models, all_hits)):
        c = COLOURS[k % len(COLOURS)]
        for r, d in zip(rays, hits):
            a = math.radians(float(r["angle_deg"]))
            dx, dy = math.cos(a), math.sin(a)
            reach = min(d, _clip_len(dx, dy, offset_m, ahead, behind, side))
            x1, y1 = reach * dx, offset_m + reach * dy
            ax.plot([0, x1], [offset_m, y1], color=c, lw=1.6, zorder=3, alpha=0.9)
            if reach < d:                                       # leaves the view: arrow + length
                ax.annotate("", (x1, y1), (x1 - 0.01 * dx, y1 - 0.01 * dy), zorder=4,
                            arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.2})
                ax.text(x1, y1, f" {d:.0f} m", color=c, fontsize=8, va="bottom" if dy >= 0 else "top",
                        ha="right" if dx > 0.5 else "center")
            else:
                ax.plot([x1], [y1], "o", color=c, ms=4, zorder=4)
                ax.text(x1, y1 + (0.6 if dy >= 0 else -0.6), f"{d:.0f}", color=c, fontsize=8, ha="center",
                        va="bottom" if dy >= 0 else "top")
        ax.plot([], [], color=c, lw=2, label=label)
    if len(models) > 1 or models and models[0][0]:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-behind, ahead)
    ax.set_ylim(-side, side)
    ax.set_aspect("equal")
    ax.set_xlabel("metres ahead of the car")
    ax.set_yticks([-width_m / 2, 0, width_m / 2], [f"right {width_m / 2:g} m", "centre", f"left {width_m / 2:g} m"])
    ax.set_title(f"Rays on a straight {width_m:g} m road (numbers = metres to the road edge)", fontsize=10)
    return fig


def _clip_len(dx: float, dy: float, y0: float, ahead: float, behind: float, side: float) -> float:
    """Length of a ray from (0, y0) inside the view box."""
    ts = []
    if dx > 1e-9:
        ts.append(ahead / dx)
    elif dx < -1e-9:
        ts.append(behind / -dx)
    if dy > 1e-9:
        ts.append((side - y0) / dy)
    elif dy < -1e-9:
        ts.append((side + y0) / -dy)
    return min(ts) if ts else math.inf


def _car(ax, x: float, y: float, heading: float, scale: float = 1.0) -> None:
    hl, hw = CAR_LENGTH_M / 2 * scale, CAR_WIDTH_M / 2 * scale
    c, s = math.cos(heading), math.sin(heading)
    pts = [(-hl, -hw), (hl * 0.7, -hw), (hl, 0), (hl * 0.7, hw), (-hl, hw)]
    ax.add_patch(Polygon([(x + px * c - py * s, y + px * s + py * c) for px, py in pts], closed=True,
                         facecolor="#f2b01e", edgecolor="black", lw=0.8, zorder=5))


# --------------------------------------------------------------- real map
@lru_cache(maxsize=6)
def _track(name: str):
    from racer.track.track import Track
    return Track.load(name)


def cast_on_track(name: str, s: float, lateral: float, rays: Sequence[Mapping[str, float]],
                  raycast_cfg: Mapping[str, Any], heading_offset_deg: float = 0.0):
    """(track, x, y, theta, distances): the model's rays cast by the real sensor code."""
    from racer.env.sensors import cast_rays, pack_raycast_params
    tr = _track(name)
    x, y, th = tr.pose_at(s, lateral)
    th += math.radians(heading_offset_deg)
    angles = np.radians([float(r["angle_deg"]) for r in rays])
    ranges = np.array([float(r["range_m"]) for r in rays])
    out = np.empty(len(rays))
    f = tr.field
    cast_rays(f.dist, f.idx, f.geom, x, y, th, angles, ranges, pack_raycast_params(raycast_cfg), out)
    return tr, x, y, th, out


def _road(ax, tr) -> None:
    """Grass, the road between its edges, the dashed centreline and the start line (black square)."""
    ax.set_facecolor(GRASS)
    outer, inner = tr.edges(-tr.half_width), tr.edges(tr.half_width)
    ring = np.vstack([outer, outer[:1], inner[::-1], inner[-1:]])
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1) + [MplPath.CLOSEPOLY]
             + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 1) + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(ring, codes), facecolor=ROAD, edgecolor=EDGE, lw=1.0, zorder=0))
    cl = tr.centreline.points
    ax.plot(cl[:, 0], cl[:, 1], color="white", lw=0.8, ls=(0, (6, 6)), zorder=1)
    ax.plot([cl[0, 0]], [cl[0, 1]], marker="s", color="black", ms=5, zorder=2)


def track_figure(tr, problems: Sequence[str] = ()):
    """A whole track (e.g. just drawn): road, start line, driving direction, and the two
    sections that are too close, if any, joined in red."""
    cl = tr.centreline
    fig, ax = plt.subplots(figsize=(7, 7))
    _road(ax, tr)
    p0, t0 = cl.points[0], cl.tangent[0] * max(25.0, cl.length / 40)
    ax.annotate("", xy=p0 + t0, xytext=p0, arrowprops=dict(color="black", width=2, headwidth=8), zorder=3)
    clash = tr.closest_sections()
    if clash and clash[0] < tr.spec.width_m + tr.cfg["validation"]["clearance_m"]:
        a, b = cl.points[clash[1]], cl.points[clash[2]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="red", lw=3, marker="o", zorder=4)
    k = np.abs(cl.curvature_smooth)
    if 1.0 / max(k.max(), 1e-9) < tr.cfg["validation"]["min_corner_radius_m"]:
        q = cl.points[int(np.argmax(k))]
        ax.plot([q[0]], [q[1]], "o", ms=14, mfc="none", mec="red", mew=3, zorder=4)
    pad = tr.half_width + tr.runoff
    ax.set_xlim(cl.points[:, 0].min() - pad, cl.points[:, 0].max() + pad)
    ax.set_ylim(cl.points[:, 1].min() - pad, cl.points[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"{tr.name}: {cl.length:.0f} m, width {tr.spec.width_m:g} m (arrow = start, driving direction)"
    ax.set_title("\n".join([title, *problems]), fontsize=9, color="red" if problems else "black")
    return fig


def ray_diagram_map(name: str, s: float, lateral: float,
                    models: Sequence[tuple[str, Sequence[Mapping[str, float]], Mapping[str, Any]]],
                    heading_offset_deg: float = 0.0, view_m: float | None = None):
    """Rays of one or more models (label, rays, raycast config) on a real track at arc length ``s``."""
    casts = [(label, rays, cast_on_track(name, s, lateral, rays, rc, heading_offset_deg))
             for label, rays, rc in models]
    tr, x, y, th, _ = casts[0][2]
    reach = max((float(d) for *_, (_, _, _, _, ds) in casts for d in ds), default=60.0)
    half = view_m or max(40.0, min(reach * 1.1, 220.0))
    fig, ax = plt.subplots(figsize=(7, 7))
    _road(ax, tr)
    for k, (label, rays, (_, cx, cy, cth, dists)) in enumerate(casts):
        col = COLOURS[k % len(COLOURS)]
        for r, d in zip(rays, dists):
            a = cth + math.radians(float(r["angle_deg"]))
            x1, y1 = cx + d * math.cos(a), cy + d * math.sin(a)
            ax.plot([cx, x1], [cy, y1], color=col, lw=1.5, zorder=3)
            hit = d < float(r["range_m"]) - 1e-6
            ax.plot([x1], [y1], "o" if hit else ">", color=col, ms=4, zorder=4)
            ax.text(x1, y1, f" {d:.0f}", color=col, fontsize=8, zorder=4)
        ax.plot([], [], color=col, lw=2, label=label)
    _car(ax, x, y, th, scale=max(1.0, half / 60))
    if len(casts) > 1 or casts[0][0]:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(x - half, x + half)
    ax.set_ylim(y - half, y + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}: {s:.0f} m into the lap (black square = start line); "
                 "dots = road edge, arrows = nothing within range", fontsize=9)
    return fig


def track_length(name: str) -> float:
    return float(_track(name).length)


# --------------------------------------------------------------- network
def layer_table(algorithm: str, obs_dim: int, act_dim: int, net_arch: Sequence[int]) -> list[dict[str, Any]]:
    """Parameters per layer of SB3's MlpPolicy, for the actor (drives) and the critic(s) (training only).

    Sums match :func:`racer.agents.compat.count_params`."""
    arch = [int(n) for n in net_arch]
    rows: list[dict[str, Any]] = []

    def stack(net: str, n_in: int, n_out: int | None, out_name: str) -> None:
        sizes = [n_in, *arch] + ([n_out] if n_out is not None else [])
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            last = n_out is not None and i == len(sizes) - 2
            rows.append({"network": net, "layer": out_name if last else f"hidden {i + 1}",
                         "shape": f"{a} → {b}", "params": a * b + b})

    if algorithm == "ppo":
        stack("actor", obs_dim, act_dim, "outputs (mean action)")
        rows.append({"network": "actor", "layer": "log std (exploration noise)", "shape": f"{act_dim}",
                     "params": act_dim})
        stack("critic", obs_dim, 1, "value")
    elif algorithm == "sac":
        stack("actor", obs_dim, None, "")
        h = arch[-1] if arch else obs_dim
        rows.append({"network": "actor", "layer": "mean head", "shape": f"{h} → {act_dim}",
                     "params": h * act_dim + act_dim})
        rows.append({"network": "actor", "layer": "log std head", "shape": f"{h} → {act_dim}",
                     "params": h * act_dim + act_dim})
        for q in ("critic 1", "critic 2"):
            stack(q, obs_dim + act_dim, 1, "Q value")
    elif algorithm == "ga":
        stack("actor", obs_dim, act_dim, "outputs (action)")
    else:
        raise ValueError(f"no layer table for algorithm {algorithm!r}")
    return rows


def network_diagram(obs_labels: Sequence[str], net_arch: Sequence[int], act_labels: Sequence[str],
                    activation: str = "tanh", max_io: int = 24, max_hidden: int = 12):
    """The driving network: inputs → hidden layers → outputs. Wide layers show a sample of
    nodes with '⋮'; every layer is captioned with its size."""
    layers = [list(obs_labels), *[[""] * int(n) for n in net_arch], list(act_labels)]
    n_layers = len(layers)
    limits = [max_io] + [max_hidden] * len(net_arch) + [max_io]
    fig, ax = plt.subplots(figsize=(min(3 + 2.2 * n_layers, 12), 5.5))
    xs = np.linspace(0, 1, n_layers)
    pos: list[list[tuple[float, float]]] = []
    for x, nodes, lim in zip(xs, layers, limits):
        shown = min(len(nodes), lim)
        ys = np.linspace(0.92, 0.08, shown) if shown > 1 else np.array([0.5])
        pos.append([(x, float(yy)) for yy in ys])
    for a, b in zip(pos, pos[1:]):
        for (x0, y0) in a:
            for (x1, y1) in b:
                ax.plot([x0, x1], [y0, y1], color="#999999", lw=0.3, alpha=0.35, zorder=1)
    for li, (nodes, pts, lim) in enumerate(zip(layers, pos, limits)):
        n = len(nodes)
        first, last = li == 0, li == n_layers - 1
        colour = "#4c9be8" if first else "#f2b01e" if last else "#bbbbbb"
        for j, (x, y) in enumerate(pts):
            gap = n > lim and j == lim // 2
            ax.plot([x], [y], "o", ms=7 if not gap else 0, color=colour, mec="black", mew=0.5, zorder=2)
            if gap:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=14, zorder=3)
            elif (first or last) and n <= lim and nodes[j]:
                ax.text(x - 0.02 if first else x + 0.02, y, nodes[j], ha="right" if first else "left",
                        va="center", fontsize=8)
        caption = (f"{n} inputs" if first else f"{n} outputs" if last else f"{n} neurons\n{activation}")
        ax.text(pts[0][0], 1.0, caption, ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xlim(-0.25, 1.22)
    ax.set_ylim(0, 1.08)
    ax.axis("off")
    return fig
