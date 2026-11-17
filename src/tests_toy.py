"""Three toy tests (spec Sec. 11, verification checklist).

Each test uses a SYNTHETIC field with a known answer, so a failure means the
solver or a metric is wrong -- not that the bathymetry is hard.

  T1  blocked wall   : a vertical wall of infeasible cells that does not reach
                       either the top or the bottom edge -> the easy route must
                       pass through one of the two gaps, and no route may cross
                       the wall.
  T2  flat, zero risk: a flat 300 m plateau with R = 0 -> J = alpha*L, so the
                       optimum is a straight line A->B with kappa = 0.
  T3  one risk bump  : a single Gaussian risk bump on A-B -> the route must
                       detour around it symmetrically about the bump centre.
  T4  diagonal rules : the combined baseline rule (endpoints + geometric
                       midpoint + orthogonal neighbours) accepts and rejects the
                       four documented cases.
  T5  reprojection   : strict bilinear resampling never blends land/NoData into
                       an ocean value and never extrapolates outside the source
                       footprint.
  T6  interpolation  : a negative interpolated depth is flagged and classified
                       infeasible rather than silently clipped.
  T7  feasibility    : the sensitivity/pipeline feasibility classifier rejects a
                       candidate whose dense validation fails, even when the
                       solver's own collocation metadata reports success.

Run:  python -m src.tests_toy          (exit code 0 = all pass)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from . import metrics
from .data import _bilinear_strict
from .easy import make_edge_rule, nearest_feasible, solve_easy
from .hard import HardProblem, SmoothFields, Spline, make_starts, solve_hard
from .pipeline import classify_route_feasibility

DX = 450.0
NY, NX = 121, 121
X = np.arange(NX) * DX
Y = np.arange(NY) * DX


def _fields(z, s_field, R_field, mask=None):
    mask = np.zeros_like(z, bool) if mask is None else mask
    d = -z
    return mask, SmoothFields(X, Y, z, d, s_field, R_field, mask)


def _fit(xy_node, A, B, K=7):
    """Fit a spline through the grid route, keeping the exact endpoints."""
    return Spline(A=A, B=B, K=K, extent=(X[-1], Y[-1]), origin=(X[0], Y[0]), M=200)


def _offset_start(hp, A, B, offset_m: float, span_frac: float = 0.55):
    """Initial curve = A->B line displaced by `offset_m` across a mid-section.

    Used to start the two mirrored runs of the bump test from opposite sides, so
    that the two local optima are reached deliberately rather than by luck.
    """
    u = (B - A) / np.linalg.norm(B - A)
    n = np.array([-u[1], u[0]])
    t = np.linspace(0.0, 1.0, 400)
    # smooth bump profile: 0 at both ends, 1 in the middle
    w = np.clip(1.0 - np.abs(2.0 * t - 1.0) / span_frac, 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * w))
    xy = A[None, :] + t[:, None] * (B - A)[None, :] + (offset_m * w)[:, None] * n[None, :]
    return hp.spl.from_polyline(xy)


def _route_feasible(xy, fields, lim, n=2000, tol=1e-3):
    """Constraint check used by every toy test (feasibility, not just behaviour)."""
    v = metrics.validate_polyline(xy, fields, lim, n)
    worst = max(v["violation_depth_lo"], v["violation_depth_hi"],
                v["violation_slope"], v["violation_kappa"])
    return bool(worst <= tol), v


# --------------------------------------------------------------------------
def test_blocked_wall():
    """A wall with two gaps; the route must use a gap and never cross the wall."""
    z = np.full((NY, NX), -300.0)
    s = np.zeros((NY, NX))
    R = np.zeros((NY, NX))
    feasible = np.ones((NY, NX), bool)
    jwall = 60
    feasible[:, jwall] = False                       # wall at x = 27.0 km
    feasible[30:36, jwall] = True                    # lower gap  (y 13.5-15.75 km)
    feasible[85:91, jwall] = True                    # upper gap
    A = np.array([X[10], Y[60]])
    B = np.array([X[110], Y[60]])
    ia = nearest_feasible(feasible, (60, 10))
    ib = nearest_feasible(feasible, (60, 110))
    ez = solve_easy(feasible, X, Y, ia, ib, edge_rule="midpoint", exact_xy=(A, B),
                    d=-z, s=s, lim={"d_min": 50.0, "d_max": 600.0, "s_max": 0.5})

    path = ez.xy
    # Crossing test.  The route may pass THROUGH a gap cell on the wall column
    # (so segment-based crossing counts are ambiguous), therefore the test
    # samples the route densely and asks which cell it occupies inside the wall
    # band: every sample in the band must sit in a gap, i.e. a feasible cell.
    band = np.abs(path[:, 0] - X[jwall]) <= 0.5 * DX
    n_cross = int(band.sum())
    blocked_cross = 0
    for x, y in path[band]:
        i = int(np.clip(round(y / DX), 0, NY - 1))
        if not feasible[i, jwall]:
            blocked_cross += 1
    # the path must actually get from one side of the wall to the other
    j_of = [j for (_, j) in ez.path_ij]
    crossed = bool(min(j_of) <= jwall - 1 and max(j_of) >= jwall + 1)
    # the wall forces a detour: the route is longer than the straight line
    detour = ez.length_m > np.hypot(*(B - A)) * 1.05
    # every visited node must be feasible (the grid solver's own feasibility)
    on_feasible = bool(all(feasible[i, j] for i, j in ez.path_ij))
    through_gap = crossed and blocked_cross == 0
    ok = through_gap and detour and on_feasible
    return ok, {
        "test": "T1 blocked wall",
        "samples_in_wall_band": n_cross,
        "samples_in_blocked_cells": blocked_cross,
        "crossed_the_wall": crossed,
        "L_route_km": ez.length_m / 1000, "L_straight_km": float(np.hypot(*(B - A))) / 1000,
        "detour_ratio": ez.length_m / float(np.hypot(*(B - A))),
        "all_nodes_feasible": on_feasible,
        "pass": bool(ok),
    }


# --------------------------------------------------------------------------
def test_flat_zero_risk():
    """Flat, risk-free terrain: J = alpha*L, so the answer is a straight line."""
    z = np.full((NY, NX), -300.0)
    s = np.zeros((NY, NX))
    R = np.zeros((NY, NX))
    mask, fields = _fields(z, s, R)
    A = np.array([X[5], Y[15]])
    B = np.array([X[115], Y[105]])
    hp = HardProblem(fields, _fit(None, A, B), 50.0, 600.0, 0.5, 1 / 900.0,
                     alpha=1.0, beta=20.0, M=200)
    starts = [("straight_line", hp.spl.straight())]
    best, runs = solve_hard(hp, starts, strategy="penalty")
    L, S, xys = metrics.polyline_terms(best.xy, fields, M=1000)
    straight = float(np.hypot(*(B - A)))
    kap = metrics.curvature_along(xys, smooth_m=250.0)
    lim2 = {"d_min": 50.0, "d_max": 600.0, "s_max": np.tan(np.radians(30.0)),
            "kappa_max": 1 / 900.0}
    feas_ok, val = _route_feasible(best.xy, fields, lim2)
    ok = ((best.J < straight * 1.01 / 1000.0 + 1e-9) and np.max(np.abs(kap)) < 1e-3
          and feas_ok)
    return bool(ok), {
        "test": "T2 flat zero risk", "J_km": best.J, "L_km": L / 1000,
        "L_straight_km": straight / 1000, "S": S,
        "kappa_max": float(np.nanmax(np.abs(kap))),
        "max_length_excess_pct": 100.0 * (L / straight - 1.0),
        "constraints_satisfied": feas_ok,
        "dense_max_violation": float(max(val["violation_depth_lo"], val["violation_depth_hi"],
                                         val["violation_slope"], val["violation_kappa"])),
        "pass": bool(ok),
    }


# --------------------------------------------------------------------------
def test_single_bump():
    """One Gaussian risk bump on the straight line: symmetric detour, both sides.

    SYMMETRY ASSERTION.  The bump is rotationally symmetric about the A-B line,
    so going round it on the left and on the right are two equal-cost local
    optima.  Two runs are started from mirrored offset curves and the test
    requires, with explicit tolerances:

      (i)  objective symmetry   |J_left - J_right| / max(J) <= J_TOL   (2 %)
      (ii) geometric symmetry   max_t |across_left + across_right| <= OFF_TOL
           (5 % of the larger maximum offset) -- the two routes are mirror
           images of each other, not two arbitrary curves on one side
      (iii) behaviour           both routes avoid the bump (mean R below the
           straight line's) and both satisfy depth, slope and curvature limits.

    Anything weaker would pass with a single one-sided detour, which is exactly
    the failure mode the test exists to catch.
    """
    z = np.full((NY, NX), -300.0)
    s = np.zeros((NY, NX))
    Xg, Yg = np.meshgrid(X, Y)
    cy, cx, sig = 0.5 * (Y[15] + Y[105]), 0.5 * (X[5] + X[115]), 3000.0
    R = 0.8 * np.exp(-((Xg - cx) ** 2 - 0 + (Yg - cy) ** 2) / (2 * sig ** 2))
    mask, fields = _fields(z, s, R)
    A = np.array([X[5], Y[15]])
    B = np.array([X[115], Y[105]])
    kap_max = 1 / 900.0
    hp = HardProblem(fields, _fit(None, A, B), 50.0, 600.0, 0.5, kap_max,
                     alpha=1.0, beta=500.0, M=200)

    J_TOL = 0.02        # 2 % objective symmetry
    OFF_TOL = 0.05      # 5 % of the detour amplitude
    sides = {}
    for name, off in (("left", +3500.0), ("right", -3500.0)):
        v0 = _offset_start(hp, A, B, off)
        best, _ = solve_hard(hp, [("offset_" + name, v0)], strategy="penalty")
        L, S, xy = metrics.polyline_terms(best.xy, fields, M=1000)
        u = (B - A) / np.linalg.norm(B - A)
        nvec = np.array([-u[1], u[0]])
        across = (xy - A) @ nvec
        sides[name] = {"J": best.J, "L_km": L / 1000, "S": S, "xy": xy,
                       "across": across, "max_off": float(np.max(np.abs(across))),
                       "mean_R": float(np.mean(fields("R", xy[:, 0], xy[:, 1])))}

    straight_xys = np.linspace(A, B, 1000)
    R_straight = float(np.mean(fields("R", straight_xys[:, 0], straight_xys[:, 1])))
    dJ = abs(sides["left"]["J"] - sides["right"]["J"])
    J_tol_ok = dJ / max(sides["left"]["J"], sides["right"]["J"]) <= J_TOL
    # mirror comparison on a common normalised station
    acr_l = np.interp(np.linspace(0, 1, 500), np.linspace(0, 1, len(sides["left"]["across"])),
                      sides["left"]["across"])
    acr_r = np.interp(np.linspace(0, 1, 500), np.linspace(0, 1, len(sides["right"]["across"])),
                      sides["right"]["across"])
    mirror_err = float(np.max(np.abs(acr_l + acr_r)))
    off_tol_ok = mirror_err <= OFF_TOL * max(sides["left"]["max_off"], sides["right"]["max_off"])
    lim2 = {"d_min": 50.0, "d_max": 600.0, "s_max": 0.5, "kappa_max": kap_max}
    feas = {}
    for name in sides:
        ok_f, val = _route_feasible(sides[name]["xy"], fields, lim2)
        feas[name] = ok_f
    behaviour = all(feas.values()) and all(sides[k]["mean_R"] < R_straight for k in sides)
    ok = J_tol_ok and off_tol_ok and behaviour
    return bool(ok), {
        "test": "T3 single risk bump",
        "J_left_km": sides["left"]["J"], "J_right_km": sides["right"]["J"],
        "J_symmetry_error_pct": 100.0 * dJ / max(sides["left"]["J"], sides["right"]["J"]),
        "J_symmetry_tolerance_pct": 100.0 * J_TOL, "J_symmetry_ok": bool(J_tol_ok),
        "max_offset_left_m": sides["left"]["max_off"],
        "max_offset_right_m": sides["right"]["max_off"],
        "mirror_error_m": mirror_err,
        "mirror_tolerance_m": OFF_TOL * max(sides["left"]["max_off"], sides["right"]["max_off"]),
        "mirror_ok": bool(off_tol_ok),
        "mean_R_on_route": float(np.mean([sides["left"]["mean_R"], sides["right"]["mean_R"]])),
        "mean_R_on_straight_line": R_straight,
        "constraints_satisfied_left": feas["left"],
        "constraints_satisfied_right": feas["right"],
        "pass": bool(ok),
    }




# --------------------------------------------------------------------------
def test_diagonal_rule():
    """T4 -- the combined diagonal rule: four cases with known answers.

    The grid is 4x4 nodes of a 450 m lattice with depth 300 m everywhere and a
    slope of 0 (so every node is feasible), then three perturbations are applied
    one at a time:

      case A  open terrain                     -> diagonal accepted
      case B  one of the four surrounding nodes masked (land/NoData)
                                               -> midpoint not verifiable, rejected
      case C  an orthogonal neighbour infeasible (too shallow)
                                               -> rejected by the corner test
      case D  the far endpoint infeasible      -> rejected by the endpoint test

    The test also asserts the convexity property of the midpoint test: if all
    four surrounding nodes are individually feasible, their average cannot leave
    a box constraint, so the midpoint test can only reject a step whose stencil
    touches masked data.  That is a property of the formulation, and it is
    asserted here rather than left as a claim in the report.
    """
    n = 4
    d0 = np.full((n, n), 300.0)
    s0 = np.zeros((n, n))
    lim = {"d_min": 50.0, "d_max": 600.0, "s_max": 1.0}
    feas0 = np.ones((n, n), bool)

    def rule_for(feas, d, s):
        return make_edge_rule(feas, "midpoint_corner", d, s, lim)

    out = {}

    # A: open terrain
    out["A_open_terrain"] = bool(rule_for(feas0, d0, s0)(1, 1, 1, 1))

    # B: one surrounding node masked (the segment's midpoint stencil is invalid)
    dB, sB, feasB = d0.copy(), s0.copy(), feas0.copy()
    dB[1, 2] = np.nan          # masked node: not verifiable at the midpoint
    sB[1, 2] = np.nan
    feasB[1, 2] = False
    out["B_blocked_midpoint_stencil"] = bool(rule_for(feasB, dB, sB)(1, 1, 1, 1))

    # C: orthogonal neighbour (2,1) infeasible, all four corners of the step fine
    dC, sC, feasC = d0.copy(), s0.copy(), feas0.copy()
    dC[2, 1] = 10.0            # too shallow -> infeasible cell
    feasC[2, 1] = False
    out["C_orthogonal_neighbour_infeasible"] = bool(rule_for(feasC, dC, sC)(1, 1, 1, 1))

    # D: the far endpoint itself is infeasible
    dD, sD, feasD = d0.copy(), s0.copy(), feas0.copy()
    dD[2, 2] = 10.0
    feasD[2, 2] = False
    out["D_endpoint_infeasible"] = bool(rule_for(feasD, dD, sD)(1, 1, 1, 1))

    # convexity property: with four feasible corners the midpoint test never binds
    rng = np.random.default_rng(0)
    viol = 0
    for _ in range(2000):
        dd = rng.uniform(lim["d_min"], lim["d_max"], (n, n))
        ss = rng.uniform(0.0, lim["s_max"], (n, n))
        if not make_edge_rule(feas0, "midpoint", dd, ss, lim)(1, 1, 1, 1):
            viol += 1
    out["midpoint_binds_with_feasible_corners"] = int(viol)

    ok = (out["A_open_terrain"] is True
          and out["B_blocked_midpoint_stencil"] is False
          and out["C_orthogonal_neighbour_infeasible"] is False
          and out["D_endpoint_infeasible"] is False
          and out["midpoint_binds_with_feasible_corners"] == 0)
    out["test"] = "T4 diagonal rule (combined baseline)"
    out["pass"] = bool(ok)
    return bool(ok), out


# --------------------------------------------------------------------------
def test_reprojection_masking():
    """T5 -- strict bilinear resampling: no land blending, no extrapolation.

    A synthetic 6x6 source raster (15 arc-sec spacing) with one land node and one
    NoData node is queried at five points:

      i.   open ocean, far from both bad nodes          -> accepted
      ii.  a point whose stencil contains the land node -> rejected
      iii. a point whose stencil contains the NoData node -> rejected
      iv.  between the outermost centre and the cell edge -> accepted by edge
           replication (the half-cell border is kept, not thrown away)
      v.   beyond the cell extent (one full cell outside) -> outside the
           footprint, so the caller must reject it; this test asserts the
           footprint rule itself
    """
    dl = 0.00416667
    lat = 37.0 + np.arange(6) * dl
    lon = 11.0 + np.arange(6) * dl
    z = np.full((6, 6), -200.0)
    z[2, 2] = 40.0            # land (above sea level)
    z[4, 4] = -32767.0        # NoData flag
    valid = np.isfinite(z) & (z > -1000.0) & (z < 0.0)
    zs = np.where(valid, z, np.nan)

    q_lat = np.array([lat[0] + 0.2 * dl,     # i   open ocean
                      lat[1] + 0.99 * dl,    # ii  stencil corners (1,1)(1,2)(2,1)(2,2)
                      lat[4],                # iii exactly the NoData node
                      lat[0] - 0.4 * dl,     # iv  inside the cell extent
                      lat[0] - 3.0 * dl])    # v   beyond the cell extent
    q_lon = np.array([lon[0] + 0.2 * dl,
                      lon[1] + 0.99 * dl,
                      lon[4],
                      lon[0],
                      lon[0]])
    val, ok, touches = _bilinear_strict(zs, lat, lon, valid, q_lat, q_lon)

    # footprint rule of build_grid: source cell EXTENT, not centre extent
    fp_lat = (lat[0] - dl / 2, lat[-1] + dl / 2)
    fp_lon = (lon[0] - dl / 2, lon[-1] + dl / 2)
    inside = ((q_lon >= fp_lon[0]) & (q_lon <= fp_lon[1])
              & (q_lat >= fp_lat[0]) & (q_lat <= fp_lat[1]))

    out = {
        "test": "T5 strict reprojection masking",
        "i_open_ocean_ok": bool(ok[0]),
        "i_open_ocean_value_m": float(val[0]),
        "ii_land_stencil_ok": bool(ok[1]),
        "iii_nodata_cell_ok": bool(ok[2]),
        "iv_half_cell_border_ok": bool(ok[3] and inside[3]),
        "iv_value_is_edge_cell_m": float(val[3]),
        "v_outside_extent_inside_footprint": bool(inside[4]),
        "no_positive_value_among_accepted": bool(np.all(val[ok] < 0.0)),
        "n_points_touching_invalid": int(touches.sum()),
    }
    ok_all = (out["i_open_ocean_ok"] and out["i_open_ocean_value_m"] < 0
              and not out["ii_land_stencil_ok"] and not out["iii_nodata_cell_ok"]
              and out["iv_half_cell_border_ok"]
              and not out["v_outside_extent_inside_footprint"]
              and out["no_positive_value_among_accepted"])
    out["pass"] = bool(ok_all)
    return bool(ok_all), out


# --------------------------------------------------------------------------
def test_interpolation_validity():
    """T6 -- a negative interpolated depth is infeasible, never clipped.

    A steep synthetic shore goes from -40 m to -400 m across three cells; the
    cubic interpolant of depth is evaluated on a dense transect and the count of
    negative values is compared with the linear interpolant.  A route point whose
    interpolated depth is <= 0 must be classified infeasible by the shared
    classifier.
    """
    n = 40
    x = np.arange(n) * DX
    y = np.arange(n) * DX
    z = np.full((n, n), -300.0)
    z[:, :6] = -40.0
    mask = np.zeros((n, n), bool)
    d = -z
    s = np.abs(np.gradient(z, DX, axis=1))
    R = np.zeros_like(z)
    fields_c = SmoothFields(x, y, z, d, s, R, mask, method="cubic")
    Xg, Yg = np.meshgrid(np.linspace(x[3], x[8], 2000), np.full(2000, y[20]))
    d_c = fields_c("d", Xg.ravel(), Yg.ravel())
    lin = SmoothFields(x, y, z, d, s, R, mask, method="linear")
    d_l = lin("d", Xg.ravel(), Yg.ravel())

    # a candidate whose dense validation sees a non-positive depth
    dense = {"n_violations_depth": 3, "n_violations_slope": 0, "slope_max": 0.05,
             "slope_max_deg": 2.9, "depth_min": -1.98}
    lim = {"d_min": 50.0, "d_max": 600.0, "s_max": 0.1, "kappa_max": 1 / 1350.0}
    verdict = classify_route_feasibility({"x": 0.0}, dense, None,
                                         {"n_points": 2000, "n_on_masked_cell": 0, "n_outside_box": 0},
                                         True, lim)
    out = {
        "test": "T6 interpolation validity",
        "cubic_min_depth_m": float(np.min(d_c)),
        "cubic_n_negative": int(np.sum(d_c < 0)),
        "linear_min_depth_m": float(np.min(d_l)),
        "linear_n_negative": int(np.sum(d_l < 0)),
        "negative_depth_classified_infeasible": bool(not verdict["feasible"]),
        "reason_mentions_depth": any("depth" in r for r in verdict["reasons"]),
    }
    ok = (out["negative_depth_classified_infeasible"] and out["reason_mentions_depth"]
          and out["linear_n_negative"] <= out["cubic_n_negative"])
    out["pass"] = bool(ok)
    return bool(ok), out


# --------------------------------------------------------------------------
def test_feasibility_classifier():
    """T7 -- solver metadata alone can never make a candidate feasible."""
    lim = {"d_min": 50.0, "d_max": 600.0, "s_max": np.tan(np.radians(5.5)),
           "kappa_max": 7.4074e-4}
    good_dense = {"n_violations_depth": 0, "n_violations_slope": 0,
                  "slope_max": 0.09, "slope_max_deg": 5.2, "depth_min": 90.0}
    dom_ok = {"n_points": 2000, "n_on_masked_cell": 0, "n_outside_box": 0}
    kap_ok = {"kappa_max": 7.0e-4}
    solver_ok = {"depth_lo": 0.0, "depth_hi": 0.0, "slope": 0.0, "kappa": 0.0}

    cases = {}
    cases["all_pass"] = classify_route_feasibility(solver_ok, good_dense, kap_ok, dom_ok,
                                                   True, lim)
    # THE BUG THIS TEST EXISTS FOR: solver says fine, independent dense check says no
    bad = dict(good_dense, n_violations_slope=11, slope_max_deg=5.658, slope_max=0.0990)
    cases["dense_slope_fails"] = classify_route_feasibility(solver_ok, bad, kap_ok, dom_ok,
                                                            True, lim)
    cases["dense_depth_fails"] = classify_route_feasibility(
        solver_ok, dict(good_dense, n_violations_depth=2), kap_ok, dom_ok, True, lim)
    cases["kappa_fails"] = classify_route_feasibility(
        solver_ok, good_dense, {"kappa_max": 8.0e-4}, dom_ok, True, lim)
    cases["enters_mask"] = classify_route_feasibility(
        solver_ok, good_dense, kap_ok, {"n_points": 2000, "n_on_masked_cell": 0, "n_outside_box": 4}, True, lim)

    cases["empty_geometry"] = classify_route_feasibility(
        solver_ok, good_dense, kap_ok, dict(dom_ok, n_points=0), True, lim)
    cases["missing_geometry_count"] = classify_route_feasibility(
        solver_ok, good_dense, kap_ok, {"n_on_masked_cell": 0, "n_outside_box": 0}, True, lim)
    ok = (not cases["empty_geometry"]["feasible"] and not cases["missing_geometry_count"]["feasible"]
          and cases["all_pass"]["feasible"]
          and not cases["dense_slope_fails"]["feasible"]
          and not cases["dense_depth_fails"]["feasible"]
          and not cases["kappa_fails"]["feasible"]
          and not cases["enters_mask"]["feasible"])
    out = {"test": "T7 feasibility classification",
           **{k: bool(v["feasible"]) for k, v in cases.items()},
           "slope_failure_reason": cases["dense_slope_fails"]["reasons"],
           "pass": bool(ok)}
    return bool(ok), out


# --------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    results = []
    all_ok = True
    for fn in (test_blocked_wall, test_flat_zero_risk, test_single_bump,
               test_diagonal_rule, test_reprojection_masking,
               test_interpolation_validity, test_feasibility_classifier):
        try:
            ok, info = fn()
        except Exception as exc:                       # a crash is a failure
            ok, info = False, {"test": fn.__name__, "error": repr(exc), "pass": False}
        results.append(info)
        all_ok &= bool(ok)
        print(("PASS  " if ok else "FAIL  ") + info["test"])
        for k, v in info.items():
            if k in ("test", "pass"):
                continue
            print(f"        {k:36s} {v}")
    print(f"\n{'ALL TOY TESTS PASS' if all_ok else 'SOME TOY TESTS FAILED'}"
          f"  ({time.perf_counter() - t0:.1f} s)")
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "results" / "toy_tests.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"all_pass": bool(all_ok), "tests": results}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
