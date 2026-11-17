"""Common route evaluation: L, S, J, constraint validation, robustness metrics.

Everything here works on polylines in metric coordinates, so the easy route
(grid polyline) and the hard route (densely sampled spline) are evaluated by the
SAME code with the SAME risk field and the SAME units (spec Sec. 10).
"""

from __future__ import annotations

import numpy as np

from .hard import route_curvature


# --------------------------------------------------------------------------
# basic integrals along a polyline
# --------------------------------------------------------------------------
def polyline_terms(xy: np.ndarray, fields, M: int | None = None):
    """L [m], S [m] and the field samples along a polyline.

    If M is given the polyline is resampled at M equally spaced arc-length
    points (used for fair comparison and for constraint validation).
    """
    xy = np.asarray(xy, float)
    if M is not None and len(xy) > 1:
        seg = np.hypot(*np.diff(xy, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        tq = np.linspace(0.0, s[-1], M)
        xy = np.column_stack([np.interp(tq, s, xy[:, 0]), np.interp(tq, s, xy[:, 1])])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    L = float(seg.sum())
    mid = 0.5 * (xy[1:] + xy[:-1])
    R = fields("R", mid[:, 0], mid[:, 1])
    S = float(np.sum(R * seg))
    return L, S, xy


def smooth_fit_curvature(xy: np.ndarray, smooth_m: float = 500.0) -> dict:
    """Curvature of a smoothed spline fit of a (grid) route -- informational only.

    The easy problem has no curvature constraint; this reports what the
    curvature of its smoothed version would be.
    """
    from scipy.interpolate import splprep, splev
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        # no route geometry at all (e.g. no feasible A->B path at this setting):
        # report "no samples", never a silent pass or a crash
        return {"n_points": 0, "n_outside_box": 0, "n_on_masked_cell": 0,
                "n_on_infeasible_cell": 0, "fraction_on_feasible": float("nan"),
                "not_applicable": True,
                "definition": "no route geometry available; nothing to validate"}
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    xy, s = xy[keep], s[keep]
    if len(xy) < 8:
        return {"kappa_max": float("nan"), "min_radius_m": float("nan")}
    w = np.ones(len(xy))
    tck, _ = splprep([xy[:, 0], xy[:, 1]], u=s / s[-1], s=(smooth_m ** 2) * len(xy),
                     w=w, k=3)
    u = np.linspace(0, 1, 1000)
    d1 = np.array(splev(u, tck, der=1))
    d2 = np.array(splev(u, tck, der=2))
    k = np.abs(route_curvature(None, None, d1[0], d1[1], d2[0], d2[1]))
    return {"kappa_max": float(k.max()),
            "min_radius_m": float(1.0 / k.max()) if k.max() > 0 else float("inf")}


def naive_L_S(xy: np.ndarray, fields):
    """Independent, deliberately naive re-computation (spec Sec. 11 check)."""
    xy = np.asarray(xy, float)
    L = 0.0
    S = 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        d = float(np.hypot(*(b - a)))
        L += d
        Rm = float(fields("R", np.array([0.5 * (a[0] + b[0])]),
                          np.array([0.5 * (a[1] + b[1])]))[0])
        S += Rm * d
    return L, S


# --------------------------------------------------------------------------
# dense constraint validation (spec Sec. 7)
# --------------------------------------------------------------------------
def curvature_along(xy: np.ndarray, smooth_m: float = 250.0) -> np.ndarray:
    """Curvature of a polyline, estimated robustly.

    Finite differences of a densely resampled polyline are dominated by
    discretisation noise (a grid polylines has kinks at every node), so the
    heading angle is first smoothed over a physical window of `smooth_m` metres
    and then differentiated with respect to arc length.  This is an independent
    estimator -- it does not reuse the optimiser's spline derivatives.
    """
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        # no route geometry at all (e.g. no feasible A->B path at this setting):
        # report "no samples", never a silent pass or a crash
        return {"n_points": 0, "n_outside_box": 0, "n_on_masked_cell": 0,
                "n_on_infeasible_cell": 0, "fraction_on_feasible": float("nan"),
                "not_applicable": True,
                "definition": "no route geometry available; nothing to validate"}
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.zeros(len(xy))
    # uniform resampling at ~ smooth_m/5 spacing
    ds = max(smooth_m / 5.0, 1e-6)
    n = max(int(s[-1] / ds) + 2, 5)
    tq = np.linspace(0.0, s[-1], n)
    X = np.interp(tq, s, xy[:, 0])
    Y = np.interp(tq, s, xy[:, 1])
    dX = np.gradient(X, tq)
    dY = np.gradient(Y, tq)
    heading = np.unwrap(np.arctan2(dY, dX))
    k = max(int(round(smooth_m / (tq[1] - tq[0]))) | 1, 3)
    if k >= len(heading):
        k = max(len(heading) // 2 * 2 - 1, 3)
    ker = np.ones(k) / k
    pad = k // 2
    hs = np.convolve(np.pad(heading, pad, mode="edge"), ker, mode="valid")
    kap = np.gradient(hs, tq)
    # map back to the original sample positions
    return np.interp(s, tq, kap)


def validate_polyline(xy: np.ndarray, fields, lim: dict, n_points: int):
    """Sample a route at n_points and evaluate every constraint independently."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 3:
        # no route exists under this configuration: report NaNs, not a crash
        keys = ("L_m", "S", "depth_min", "depth_max", "slope_max", "slope_max_deg",
                "kappa_max", "min_radius_m", "violation_depth_lo", "violation_depth_hi",
                "violation_slope", "violation_kappa", "R_mean", "R_max")
        out = {"n_points": int(n_points), "no_route": True}
        out.update({k: float("nan") for k in keys})
        out.update({"n_violations_depth": 0, "n_violations_slope": 0,
                    "n_violations_kappa": 0, "n_violations": 0})
        return out
    L, S, xy_s = polyline_terms(xy, fields, M=n_points)
    X, Y = xy_s[:, 0], xy_s[:, 1]
    d = fields("d", X, Y)
    s = fields("s", X, Y)
    R = fields("R", X, Y)
    # curvature from an independent, physically smoothed heading estimator
    kap = curvature_along(xy_s, smooth_m=250.0)
    # physical validity: an interpolated depth <= 0 means "above sea level", i.e.
    # the interpolant implied land.  It is reported separately and counted as a
    # depth violation (lim["d_min"] > 0, so the two overlap by construction).
    n_negative_depth = int(np.sum(d <= 0.0))
    viol_depth_lo = np.maximum(lim["d_min"] - d, 0.0)
    viol_depth_hi = np.maximum(d - lim["d_max"], 0.0)
    viol_slope = np.maximum(s - lim["s_max"], 0.0)
    viol_kappa = np.maximum(np.abs(kap) - lim["kappa_max"], 0.0)
    return {
        "n_points": int(n_points),
        "L_m": L,
        "S": S,
        "depth_min": float(np.min(d)),
        "depth_max": float(np.max(d)),
        "slope_max": float(np.max(s)),
        "slope_max_deg": float(np.degrees(np.arctan(np.max(s)))),
        "kappa_max": float(np.nanmax(np.abs(kap))),
        "min_radius_m": float(1.0 / np.nanmax(np.abs(kap))) if np.nanmax(np.abs(kap)) > 0 else np.inf,
        "violation_depth_lo": float(viol_depth_lo.max()),
        "violation_depth_hi": float(viol_depth_hi.max()),
        "violation_slope": float(viol_slope.max()),
        "violation_kappa": float(viol_kappa.max()),
        "n_violations_depth": int((viol_depth_lo > 0).sum() + (viol_depth_hi > 0).sum()),
        "n_negative_depth": n_negative_depth,
        "min_interp_depth_m": float(np.min(d)),
        "n_violations_slope": int((viol_slope > 0).sum()),
        "n_violations_kappa": int((viol_kappa > 0).sum()),
        "n_violations": int((viol_depth_lo > 0).sum() + (viol_depth_hi > 0).sum()
                            + (viol_slope > 0).sum() + (viol_kappa > 0).sum()),
        "R_mean": float(np.mean(R)),
        "R_max": float(np.max(R)),
    }


def density_study(xy, fields, lim, point_counts=(200, 1000, 5000)):
    return [validate_polyline(xy, fields, lim, n) for n in point_counts]


# --------------------------------------------------------------------------
# domain and feasibility checks
# --------------------------------------------------------------------------
def domain_check(xy: np.ndarray, grid, feasible: np.ndarray, n_points: int = 5000,
                 lim: dict | None = None, fields=None):
    """Are all sampled points inside the grid box and on feasible cells?"""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        # no route geometry at all (e.g. no feasible A->B path at this setting):
        # report "no samples", never a silent pass or a crash
        return {"n_points": 0, "n_outside_box": 0, "n_on_masked_cell": 0,
                "n_on_infeasible_cell": 0, "fraction_on_feasible": float("nan"),
                "not_applicable": True,
                "definition": "no route geometry available; nothing to validate"}
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tq = np.linspace(0.0, s[-1], n_points)
    X = np.interp(tq, s, xy[:, 0])
    Y = np.interp(tq, s, xy[:, 1])
    inside = ((X >= grid.x[0]) & (X <= grid.x[-1]) & (Y >= grid.y[0]) & (Y <= grid.y[-1]))
    j = np.clip(np.round((X - grid.x[0]) / grid.dx).astype(int), 0, len(grid.x) - 1)
    i = np.clip(np.round((Y - grid.y[0]) / grid.dy).astype(int), 0, len(grid.y) - 1)
    feas = feasible[i, j]
    # land / NoData / outside-footprint: the mask that a route may never enter,
    # as opposed to `feasible`, which also encodes the depth and slope limits.
    masked = np.asarray(grid.mask_land)[i, j]
    out = {
        "n_points": int(n_points),
        "n_outside_box": int((~inside).sum()),
        "n_on_masked_cell": int((masked & inside).sum()),
        "n_on_infeasible_cell": int((~feas).sum()),
        "fraction_on_feasible": float(feas.mean()),
        "definition": ("n_on_masked_cell counts samples in land / NoData / "
                       "outside-footprint cells (never allowed); n_on_infeasible_cell "
                       "counts samples in cells whose node-level depth or slope breaks a "
                       "limit (a discretisation-flag test, explained where it is reported)"),
    }
    if lim is not None and fields is not None:
        # A continuous route can pass over a cell flagged infeasible while every
        # *sampled point* obeys the limits (cell test vs point test).  Classify
        # those samples so the discrepancy is explained, not hidden.
        inf = (~feas) & inside & ~masked
        d = fields("d", X, Y)
        sl = fields("s", X, Y)
        out["infeasible_reasons"] = {
            "depth_below_min": int(np.sum(inf & (d < lim["d_min"]))),
            "depth_above_max": int(np.sum(inf & (d > lim["d_max"]))),
            "slope_above_max": int(np.sum(inf & (sl > lim["s_max"]))),
        }
        if inf.any():
            sdeg = np.degrees(np.arctan(sl[inf]))
            out["worst_slope_on_infeasible_cell_deg"] = float(np.max(sdeg))
            out["worst_slope_excess_deg"] = float(np.max(sdeg) - np.degrees(np.arctan(lim["s_max"])))
            out["median_slope_on_infeasible_cell_deg"] = float(np.median(sdeg))
    return out


# --------------------------------------------------------------------------
# route-to-route comparison
# --------------------------------------------------------------------------
def _resample(xy, n):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) < 2:
        # no route geometry at all (e.g. no feasible A->B path at this setting):
        # report "no samples", never a silent pass or a crash
        return {"n_points": 0, "n_outside_box": 0, "n_on_masked_cell": 0,
                "n_on_infeasible_cell": 0, "fraction_on_feasible": float("nan"),
                "not_applicable": True,
                "definition": "no route geometry available; nothing to validate"}
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tq = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(tq, s, xy[:, 0]), np.interp(tq, s, xy[:, 1])])


def hausdorff_distance(xy1, xy2, n: int = 2000):
    """Symmetric Hausdorff distance and mean nearest-point distance [m]."""
    A = _resample(xy1, n)
    B = _resample(xy2, n)
    d = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    dmin = d.min(axis=1)
    dmin2 = d.min(axis=0)
    return {
        "hausdorff_m": float(max(dmin.max(), dmin2.max())),
        "mean_nearest_m": float(0.5 * (dmin.mean() + dmin2.mean())),
    }


def cell_overlap(xy1, xy2, grid):
    """Jaccard overlap of the sets of grid cells traversed by two polylines."""
    def cells(xy):
        xy = _resample(xy, 3000)
        j = np.clip(np.round((xy[:, 0] - grid.x[0]) / grid.dx).astype(int), 0, len(grid.x) - 1)
        i = np.clip(np.round((xy[:, 1] - grid.y[0]) / grid.dy).astype(int), 0, len(grid.y) - 1)
        return set(zip(i.tolist(), j.tolist()))
    c1, c2 = cells(xy1), cells(xy2)
    return {
        "jaccard": float(len(c1 & c2) / max(1, len(c1 | c2))),
        "shared_cells": int(len(c1 & c2)),
        "easy_cells": int(len(c1)),
        "hard_cells": int(len(c2)),
    }


def comparison_table(easy: dict, hard: dict) -> dict:
    """Sec. 10 table, computed from two dicts of evaluated metrics."""
    L_E, L_H = easy["L_m"], hard["L_m"]
    S_E, S_H = easy["S"], hard["S"]
    return {
        "L_easy_km": L_E / 1000.0,
        "L_hard_km": L_H / 1000.0,
        "S_easy": S_E,
        "S_hard": S_H,
        "J_easy_km": easy["J_km"],
        "J_hard_km": hard["J_km"],
        "price_of_realism_pct": 100.0 * (L_H - L_E) / L_E,
        "risk_reduction_pct": 100.0 * (S_E - S_H) / S_E if S_E > 0 else float("nan"),
        "easy_max_slope_deg": easy["slope_max_deg"],
        "hard_max_slope_deg": hard["slope_max_deg"],
        "easy_kappa_max": easy["kappa_max"],
        "hard_kappa_max": hard["kappa_max"],
        # The easy problem does not constrain route curvature, so its curvature
        # "violations" are informational only and are reported separately from
        # the limit violations that both problems are expected to respect.
        "easy_violations_slope_depth": easy.get("n_violations_slope", 0)
        + easy.get("n_violations_depth", 0),
        "easy_violations_kappa_info": easy.get("n_violations_kappa", 0),
        "hard_violations_slope_depth": hard.get("n_violations_slope", 0)
        + hard.get("n_violations_depth", 0),
        "hard_violations_kappa": hard.get("n_violations_kappa", 0),
        "easy_violations": easy["n_violations"],
        "hard_violations": hard["n_violations"],
    }

def smoothed_path_length(xy: np.ndarray, smooth_m: float = 250.0) -> float:
    """Arc length of a lightly smoothed version of the polyline [m].

    Used only to show how much of the easy route's length is discretisation:
    the 8-neighbour grid path is a staircase, so its node-to-node length
    overestimates the length of the corridor it represents.  This is a
    diagnostic, never used as the reported L_E.
    """
    xy = _resample(np.asarray(xy, float), max(int(len(xy)), 2000))
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    ds = float(np.median(np.diff(s)))
    k = max(3, int(round(smooth_m / max(ds, 1e-9))) | 1)
    ker = np.ones(k) / k
    pad = k // 2
    xs = np.convolve(np.pad(xy[:, 0], pad, mode="edge"), ker, mode="valid")
    ys = np.convolve(np.pad(xy[:, 1], pad, mode="edge"), ker, mode="valid")
    sm = np.column_stack([xs, ys])
    return float(np.sum(np.hypot(*np.diff(sm, axis=0).T)))
