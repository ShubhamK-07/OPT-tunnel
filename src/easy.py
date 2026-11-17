"""EASY problem: shortest feasible route on the discrete 8-neighbour grid.

    min_{P in graph}  L(P) = sum_e w_e ,  w_e = metric edge length [m]
    s.t.              every visited node feasible:  s <= s_max, d_min <= d <= d_max
                      (and, depending on the edge rule, an additional check on
                       the geometry *between* two nodes)

Edge rules (the specification's rule is `midpoint_corner`, which is the baseline)
--------------------------------------------------------------------------------
midpoint_corner  THE BASELINE.  A diagonal step must satisfy all three tests:
          (a) both endpoints feasible (the node test);
          (b) the GEOMETRIC MIDPOINT of the segment -- the shared corner of the
              four surrounding cells, i.e. the average of their node values --
              satisfies 50 <= d <= 600 m and s <= s_max;
          (c) both ORTHOGONAL neighbours of the step are feasible, so the step
              cannot cut the corner of an infeasible quadrant.
          Tests (b) and (c) are different: (b) is a point test on the segment,
          (c) is a cell test on the two cells the segment passes between.
          Requiring only one of them leaves a hole, so the baseline requires both.
node      endpoint feasibility only (the node-only variant of the spec).
midpoint  test (a) + (b) only  -- the midpoint-only variant, as a sensitivity run.
corner    test (a) + (c) only  -- the corner-cut-only variant, as a sensitivity run.

Solved by Dijkstra on non-negative weights.  This is the global optimum for
this graph and this feasibility rule; it is *not* a claim of continuous
optimality.  A shortest path can also be written as a network-flow LP
(min cost flow), which is the "easy problem can be an LP" rubric option.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

SQ2 = np.sqrt(2.0)

# 8-neighbour offsets: (di, dj, multiplier, is_diagonal)
OFFSETS = [
    (1, 0, 1.0, False), (-1, 0, 1.0, False), (0, 1, 1.0, False), (0, -1, 1.0, False),
    (1, 1, SQ2, True), (1, -1, SQ2, True), (-1, 1, SQ2, True), (-1, -1, SQ2, True),
]


@dataclass
class EasyResult:
    path_ij: np.ndarray                  # (n, 2) array of (row, col)
    xy: np.ndarray                       # (n, 2) metric coordinates
    length_m: float
    connected: bool
    nodes_expanded: int
    runtime_s: float
    edge_rule: str
    meta: dict = field(default_factory=dict)


EDGE_RULES = ("midpoint_corner", "node", "midpoint", "corner")


def _shift(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """a(i+di, j+dj) with NaN padding, same shape as a."""
    ny, nx = a.shape
    pad = np.full((ny + 2, nx + 2), np.nan)
    pad[1:-1, 1:-1] = a
    return pad[1 + di:ny + 1 + di, 1 + dj:nx + 1 + dj]


def midpoint_feasibility(d: np.ndarray, s: np.ndarray, lim: dict) -> dict:
    """Feasibility of the geometric midpoint of a diagonal step, per direction.

    A diagonal step from node (i, j) to (i+di, j+dj) has its midpoint at the
    shared corner of four cells.  Bilinear interpolation of the node samples at
    that point is exactly the average of the four corner values, so the test is
    geometric (it evaluates the *point* halfway along the segment) rather than a
    test on the two surrounding cells.  If any of the four samples is masked
    (land or NoData) the midpoint is not verifiable and the step is rejected.
    """
    out = {}
    for di in (1, -1):
        for dj in (1, -1):
            dm = 0.25 * (d + _shift(d, di, 0) + _shift(d, 0, dj) + _shift(d, di, dj))
            sm = 0.25 * (s + _shift(s, di, 0) + _shift(s, 0, dj) + _shift(s, di, dj))
            ok = (np.isfinite(dm) & np.isfinite(sm)
                  & (dm >= lim["d_min"]) & (dm <= lim["d_max"])
                  & (sm <= lim["s_max"]))
            out[(di, dj)] = ok
    return out


def make_edge_rule(feas: np.ndarray, rule: str, d: np.ndarray | None = None,
                   s: np.ndarray | None = None, lim: dict | None = None):
    """Return a callable ok(i, j, di, dj) implementing one named edge rule."""
    if rule not in EDGE_RULES:
        raise ValueError(f"unknown edge rule {rule!r}; expected one of {EDGE_RULES}")
    ny, nx = feas.shape
    mid = None
    if rule in ("midpoint", "midpoint_corner"):
        if d is None or s is None or lim is None:
            raise ValueError("the midpoint test needs the d, s fields and the limits")
        mid = midpoint_feasibility(d, s, lim)

    def ok(i: int, j: int, di: int, dj: int) -> bool:
        ni, nj = i + di, j + dj
        if not (0 <= ni < ny and 0 <= nj < nx):
            return False
        if not feas[ni, nj]:                       # (a) far endpoint feasible
            return False
        if di == 0 or dj == 0:                     # orthogonal step: nothing else to test
            return True
        if rule == "node":
            return True
        mid_ok = bool(mid[(di, dj)][i, j]) if mid is not None else True   # (b)
        if rule == "midpoint":
            return mid_ok
        corner_ok = bool(feas[i + di, j] and feas[i, j + dj])             # (c)
        if rule == "corner":
            return corner_ok
        return mid_ok and corner_ok                                       # combined baseline

    return ok


def _edge_ok(feas, i, j, di, dj, rule, d=None, s=None, lim=None) -> bool:
    """Convenience wrapper (kept for scripts that pass one rule name)."""
    return make_edge_rule(feas, rule, d, s, lim)(i, j, di, dj)


def nearest_feasible(feas: np.ndarray, ij: tuple[int, int]) -> tuple[int, int]:
    """Nearest feasible cell to a given (row, col) -- used to snap A and B."""
    if feas[ij]:
        return ij
    idx = np.argwhere(feas)
    if idx.size == 0:
        raise RuntimeError("no feasible cells")
    d2 = (idx[:, 0] - ij[0]) ** 2 + (idx[:, 1] - ij[1]) ** 2
    k = int(np.argmin(d2))
    return int(idx[k, 0]), int(idx[k, 1])


def edge_weight(x1: float, y1: float, x2: float, y2: float) -> float:
    """Metric length of a grid edge: w = sqrt((dx)^2 + (dy)^2).

    Written from the node coordinates rather than assumed from the cell size, so
    the same code is correct on a non-square (or non-uniform) grid.
    """
    return float(np.hypot(x2 - x1, y2 - y1))


def dijkstra(feas: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
             dx: float, dy: float, edge_rule: str = "midpoint_corner",
             d: np.ndarray | None = None, s: np.ndarray | None = None,
             lim: dict | None = None, x: np.ndarray | None = None,
             y: np.ndarray | None = None) -> tuple[list, float, int]:
    """Own Dijkstra with heapq (counts nodes expanded). Returns (path, cost, n_expanded).

    Edge weights are computed from the node coordinates (see `edge_weight`); the
    dx/dy arguments are only used as a fallback when the coordinate vectors are
    not supplied.
    """
    ny, nx = feas.shape
    si, sj = start
    gi, gj = goal
    if not feas[si, sj] or not feas[gi, gj]:
        return [], np.inf, 0
    ok_edge = make_edge_rule(feas, edge_rule, d, s, lim)

    dist = np.full(feas.shape, np.inf)
    prev = np.full((*feas.shape, 2), -1, dtype=np.int32)
    dist[si, sj] = 0.0
    pq = [(0.0, si, sj)]
    done = np.zeros(feas.shape, bool)
    expanded = 0
    while pq:
        cd, i, j = heapq.heappop(pq)
        if done[i, j]:
            continue
        done[i, j] = True
        expanded += 1
        if (i, j) == (gi, gj):
            break
        for di, dj, mult, _ in OFFSETS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx) or done[ni, nj]:
                continue
            if not ok_edge(i, j, di, dj):
                continue
            if x is not None and y is not None:
                w_e = edge_weight(x[j], y[i], x[nj], y[ni])
            else:                                   # uniform-grid fallback
                w_e = edge_weight(0.0, 0.0, dj * dx, di * dy)
            nd = cd + w_e
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                prev[ni, nj] = (i, j)
                heapq.heappush(pq, (nd, ni, nj))
    if not np.isfinite(dist[gi, gj]):
        return [], np.inf, expanded

    path = []
    cur = (gi, gj)
    while cur != (-1, -1):
        path.append(cur)
        i, j = cur
        cur = (int(prev[i, j, 0]), int(prev[i, j, 1]))
    path.reverse()
    return path, float(dist[gi, gj]), expanded


def solve_easy(feas: np.ndarray, x: np.ndarray, y: np.ndarray,
               start_ij: tuple[int, int], goal_ij: tuple[int, int],
               edge_rule: str = "midpoint_corner", snap_to_nodes: bool = True,
               exact_xy: tuple[np.ndarray, np.ndarray] | None = None,
               d: np.ndarray | None = None, s: np.ndarray | None = None,
               lim: dict | None = None) -> EasyResult:
    """Run Dijkstra and return the route as a metric polyline.

    `exact_xy=(A_xy, B_xy)` appends exact endpoints outside the graph path.
    Dijkstra optimizes between selected snapped cells; connectors are not optimized.
    """
    import time
    t0 = time.perf_counter()
    path, cost, expanded = dijkstra(feas, start_ij, goal_ij,
                                    float(x[1] - x[0]), float(y[1] - y[0]), edge_rule,
                                    d=d, s=s, lim=lim, x=x, y=y)
    rt = time.perf_counter() - t0
    if not path:
        return EasyResult(np.empty((0, 2)), np.empty((0, 2)), np.inf, False, expanded, rt, edge_rule)
    ij = np.array(path)
    xy = np.column_stack([x[ij[:, 1]], y[ij[:, 0]]])
    if exact_xy is not None:
        A, B = exact_xy
        xy = np.vstack([np.asarray(A, float), xy, np.asarray(B, float)])
        # drop duplicated first/last grid nodes if A/B coincide with them
        if np.allclose(xy[0], xy[1]):
            xy = np.delete(xy, 1, axis=0)
        if np.allclose(xy[-1], xy[-2]):
            xy = np.delete(xy, -2, axis=0)
        cost = float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))
    return EasyResult(ij, xy, cost, True, expanded, rt, edge_rule,
                      meta={"edge_rule": edge_rule})


# --------------------------------------------------------------------------
# route-geometry diagnostics required by spec Sec. 7 (easy has no kappa limit)
# --------------------------------------------------------------------------
def heading_changes(xy: np.ndarray) -> dict:
    """Discrete heading changes of a polyline (degrees)."""
    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    ok = seg > 1e-9
    d = d[ok]
    hd = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dh = (np.diff(hd) + 180.0) % 360.0 - 180.0
    return {
        "n_segments": int(len(d)),
        "n_turns": int(np.sum(np.abs(dh) > 1e-6)),
        "largest_turn_deg": float(np.max(np.abs(dh))) if len(dh) else 0.0,
        "mean_abs_turn_deg": float(np.mean(np.abs(dh))) if len(dh) else 0.0,
        "total_abs_turn_deg": float(np.sum(np.abs(dh))) if len(dh) else 0.0,
        "zigzag_ratio": float(np.sum(seg) / max(np.hypot(*(xy[-1] - xy[0])), 1e-9)),
    }
