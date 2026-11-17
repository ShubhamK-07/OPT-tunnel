"""End-to-end baseline pipeline (spec Sec. 16, steps 1-4).

    data -> features -> risk field -> easy (Dijkstra) -> hard (SLSQP)
         -> verification -> comparison -> results/*.json|csv

Run:  python -m src.pipeline            (uses config.yaml)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import analysis, metrics
from .data import load_grid, lonlat_to_xy, to_index
from .easy import edge_weight, heading_changes, nearest_feasible, solve_easy
from .features import derive_features, feasible_mask
from .hard import (HardProblem, SmoothFields, Spline, account_runs,
                   independent_curvature, make_starts, solve_hard)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def limits(cfg: dict) -> dict:
    c = cfg["constraints"]
    return {"d_min": c["d_min_m"], "d_max": c["d_max_m"],
            "s_max": float(np.tan(np.radians(c["s_max_deg"]))),
            "s_max_deg": c["s_max_deg"],
            "kappa_max": c["kappa_max_1_per_m"]}


def baseline_route_set(cfg: dict, grid=None, verbose: bool = True,
                       strategy: str = "auto", n_random: int | None = None,
                       save: bool = True):
    """Build features, solve both problems on the current configuration."""
    grid = grid or load_grid(cfg)
    lim = limits(cfg)
    t_feat = time.perf_counter()
    F = derive_features(grid.z, grid.dx, grid.dy, grid.mask_land,
                        cfg["features"], cfg["risk"])
    F["d"] = grid.d          # depth travels with the feature stack for tables/PCA
    feat_time = time.perf_counter() - t_feat
    feas = feasible_mask(grid.d, F["s"], grid.mask_land,
                         lim["d_min"], lim["d_max"], lim["s_max"])

    A = np.array(lonlat_to_xy(cfg, cfg["data"]["A"]["lon"], cfg["data"]["A"]["lat"]))
    B = np.array(lonlat_to_xy(cfg, cfg["data"]["B"]["lon"], cfg["data"]["B"]["lat"]))
    ia = nearest_feasible(feas, to_index(grid, *A))
    ib = nearest_feasible(feas, to_index(grid, *B))

    ez = solve_easy(feas, grid.x, grid.y, ia, ib,
                    edge_rule=cfg["easy"]["edge_rule"], exact_xy=(A, B),
                    d=grid.d, s=F["s"], lim=lim)

    fields = SmoothFields(grid.x, grid.y, grid.z, grid.d, F["s"], F["R"], grid.mask_land)
    interp_check = fields.check(n_probe=20000)

    hp = HardProblem(fields, Spline(A=A, B=B, K=cfg["hard"]["n_control_points"],
                                          bc=cfg["hard"].get("spline_bc", "clamped"),
                                    extent=(grid.x[-1] - grid.x[0],
                                            grid.y[-1] - grid.y[0]),
                                    origin=(grid.x[0], grid.y[0]),
                                    M=cfg["hard"]["n_samples"]),
                     lim["d_min"], lim["d_max"], lim["s_max"], lim["kappa_max"],
                     alpha=cfg["hard"]["alpha"], beta=cfg["hard"]["beta"],
                     M=cfg["hard"]["n_samples"],
                     margin=cfg["hard"].get("constraint_margin", 0.0),
                     n_constraint=cfg["hard"].get("n_constraint_samples"))
    starts = make_starts(hp, ez.xy, cfg, n_random=n_random)
    hard_best, hard_runs = solve_hard(hp, starts, strategy=strategy)

    if verbose:
        print(f"[baseline] feasible cells {feas.mean():.3f} | easy L={ez.length_m/1000:.3f} km "
              f"| hard J={hard_best.J:.3f} km ({hard_best.solver}, {hard_best.start_label})")

    out = dict(cfg=cfg, grid=grid, F=F, feas=feas, lim=lim, A=A, B=B, ia=ia, ib=ib,
               ez=ez, fields=fields, hp=hp, interp_check=interp_check,
               hard_best=hard_best, hard_runs=hard_runs,
               feat_time=feat_time, starts=starts)
    if save:
        RESULTS.mkdir(exist_ok=True)
        json.dump({"interpolator_check": interp_check,
                   "hard_runs": [serialise_run(r) for r in hard_runs]},
                  open(RESULTS / "hard_runs.json", "w"), indent=2)
    return out


def serialise_run(r) -> dict:
    return {
        "start": r.start_label, "solver": r.solver, "status": r.status,
        "message": r.message, "nit": r.nit, "nfev": r.meta.get("nfev"),
        "J_km": r.J, "L_km": r.L / 1000.0, "S": r.S,
        "max_kappa": r.max_kappa, "max_slope": r.max_slope,
        "depth_range": list(r.depth_range), "violations": r.max_violation,
        "runtime_s": r.runtime_s, "moved": r.meta.get("moved"),
        "continuation": r.meta.get("continuation"),
        "independent_feasibility": r.meta.get("independent_feasibility"),
    }


def evaluate_routes(ctx: dict, verbose: bool = True) -> dict:
    """Evaluate both routes with the common metric code (Sec. 10)."""
    grid, fields, lim = ctx["grid"], ctx["fields"], ctx["lim"]
    n_dense = ctx["cfg"]["validation"]["dense_factor"] * ctx["cfg"]["hard"]["n_samples"]

    L_E, S_E, xyE = metrics.polyline_terms(ctx["ez"].xy, fields)
    L_H, S_H, xyH = metrics.polyline_terms(ctx["hard_best"].xy, fields)

    def pack(xy, L, S):
        lim2 = dict(lim)
        v = metrics.validate_polyline(xy, fields, lim2, n_dense)
        v["J_km"] = (L + ctx["cfg"]["hard"]["beta"] * S) / 1000.0
        v["L_m"], v["S"] = L, S
        return v

    easy_m = pack(xyE, L_E, S_E)
    hard_m = pack(xyH, L_H, S_H)

    # ---- authoritative curvature: exact B-spline derivatives (hard problem) ----
    kap_exact = independent_curvature(ctx["hp"].spl, ctx["hard_best"].v, n=2001)
    # the smoothed-polyline estimate stays, but only as a labelled diagnostic
    hard_m["kappa_max_polyline"] = hard_m["kappa_max"]
    hard_m["min_radius_polyline_m"] = hard_m["min_radius_m"]
    hard_m["kappa_max"] = kap_exact["kappa_max"]
    hard_m["min_radius_m"] = kap_exact["kappa_min_radius_m"]
    table = metrics.comparison_table(easy_m, hard_m)
    table["hausdorff_m"] = metrics.hausdorff_distance(xyE, xyH)["hausdorff_m"]
    table["mean_distance_m"] = metrics.hausdorff_distance(xyE, xyH)["mean_nearest_m"]
    table["jaccard"] = metrics.cell_overlap(xyE, xyH, grid)["jaccard"]
    table["runtime_easy_s"] = ctx["ez"].runtime_s
    table["runtime_hard_s"] = ctx["hard_best"].runtime_s
    table["nodes_expanded_easy"] = ctx["ez"].nodes_expanded
    table["hard_kappa_exact"] = kap_exact["kappa_max"]
    table["hard_kappa_polyline_diagnostic"] = hard_m["kappa_max_polyline"]
    table["hard_min_radius_exact_m"] = kap_exact["kappa_min_radius_m"]
    table["hard_min_radius_polyline_diagnostic_m"] = hard_m["min_radius_polyline_m"]
    table["J_easy_is_post_hoc"] = True
    table["J_easy_label"] = ("post-hoc common score J_E = L_E + beta*S_E; the easy "
                             "formulation minimises L only")
    table["hard_objective_label"] = "min alpha*L + beta*S (the hard formulation's own objective)"
    table["integrator_note"] = (
        f"the hard OPTIMISER integrates J with the trapezoid rule on {ctx['hp'].M} spline samples "
        "(the same active grid as the collocation constraints); "
        "the reported L, S and J of both routes are recalculated afterwards by the same "
        "polyline evaluator (segment sums, midpoint field values)")

    # ---- feasibility verdicts (the single definition, shared with sensitivity) ----
    domE = metrics.domain_check(xyE, grid, ctx["feas"], n_points=n_dense, lim=lim, fields=fields)
    domH = metrics.domain_check(xyH, grid, ctx["feas"], n_points=n_dense, lim=lim, fields=fields)
    feasE = classify_route_feasibility(None, easy_m, None, domE,
                                       bool(np.isfinite(easy_m["L_m"])), lim)
    feasH = classify_route_feasibility(
        ctx["hard_best"].meta.get("max_violation_nominal"), hard_m, kap_exact, domH,
        bool(np.isfinite(hard_m["L_m"])), lim)

    if verbose:
        print(f"[compare] L_E={table['L_easy_km']:.3f} km L_H={table['L_hard_km']:.3f} km "
              f"| price of realism {table['price_of_realism_pct']:+.2f}% "
              f"| risk reduction {table['risk_reduction_pct']:+.2f}%")

    return {"easy": easy_m, "hard": hard_m, "table": table, "xy_easy": xyE,
            "xy_hard": xyH, "L_easy_m": L_E, "L_hard_m": L_H,
            "kappa_exact": kap_exact, "domain_easy": domE, "domain_hard": domH,
            "feasibility_easy": feasE, "feasibility_hard": feasH}


# --------------------------------------------------------------------------
# Phase-1 verification helpers (interpolation, continuous domain, curvature,
# runtimes, beta = 0, edge rules)
# --------------------------------------------------------------------------
def classify_route_feasibility(solver_nominal: dict, dense: dict, kappa_exact: dict | None,
                               domain: dict, finite_fields: bool, lim: dict,
                               tol: float = 1e-3) -> dict:
    """The ONE definition of "this route is feasible" used by pipeline and sensitivity.

    A solver returning a candidate is not proof that the candidate is feasible.
    A route counts as feasible only when all of the following hold:

    1. solver-side constraints satisfied at the collocation samples, measured
       against the NOMINAL limits (not the tightened ones the solver targets);
    2. dense depth validation finds no violation;
    3. dense slope validation finds no violation;
    4. the EXACT B-spline curvature (rebuilt from the control points) is within
       the curvature limit -- the smoothed-polyline estimate is a diagnostic, not
       the constraint quantity;
    5. the route never enters land / NoData / outside the source footprint;
    6. every interpolated quantity used by the checks is finite and physically
       valid (in particular, depth is never negative).

    `kappa_exact=None` is allowed only for the easy problem, which has no
    curvature constraint.
    """
    # solver_nominal=None means "not applicable": the easy problem has no
    # collocation constraints, its feasibility is decided by the dense checks.
    solver_ok = True if solver_nominal is None else bool(
        max(solver_nominal.values()) <= tol)
    depth_ok = int(dense.get("n_violations_depth", 1)) == 0
    slope_ok = int(dense.get("n_violations_slope", 1)) == 0
    kappa_ok = True if kappa_exact is None else bool(
        kappa_exact.get("kappa_max", np.inf) <= lim["kappa_max"] * (1.0 + 1e-9))
    domain_ok = (int(domain.get("n_points", 0)) > 0
                 and int(domain.get("n_on_masked_cell", 1)) == 0
                 and int(domain.get("n_outside_box", 1)) == 0)
    finite_ok = bool(finite_fields and np.isfinite(dense.get("slope_max", np.nan))
                     and np.isfinite(dense.get("depth_min", np.nan)))
    depth_positive = bool(np.isfinite(dense.get("depth_min", np.nan))
                          and dense.get("depth_min", -1.0) > 0.0)
    ok = bool(solver_ok and depth_ok and slope_ok and kappa_ok and domain_ok
              and finite_ok and depth_positive)
    reasons = []
    if not solver_ok:
        reasons.append("solver-side constraints violated at the collocation samples")
    if not depth_ok:
        reasons.append(f"dense depth validation found {dense.get('n_violations_depth')} violations")
    if not slope_ok:
        reasons.append(f"dense slope validation found {dense.get('n_violations_slope')} violations "
                       f"(max slope {dense.get('slope_max_deg', float('nan')):.3f} deg)")
    if not kappa_ok:
        reasons.append("exact spline curvature exceeds the limit")
    if not domain_ok:
        if int(domain.get("n_points", 0)) == 0:
            reasons.append("no route geometry to validate")
        else:
            reasons.append(f"route enters a masked cell ({domain.get('n_on_masked_cell')} samples) "
                           f"or leaves the grid ({domain.get('n_outside_box')} samples)")
    if not finite_ok:
        reasons.append("a required interpolated quantity is not finite")
    if not depth_positive:
        reasons.append("interpolated depth is not strictly positive (implies land)")
    return {
        "feasible": ok,
        "criteria": {
            "solver_collocation_nominal": solver_ok,
            "dense_depth": depth_ok,
            "dense_slope": slope_ok,
            "exact_spline_curvature": kappa_ok,
            "inside_valid_domain": domain_ok,
            "finite_and_physical": bool(finite_ok and depth_positive),
        },
        "reasons": reasons,
        "definition": ("feasible = collocation constraints at nominal limits AND dense depth "
                       "AND dense slope AND exact spline curvature AND inside the valid data "
                       "domain AND finite, strictly positive interpolated depth"),
    }


def _resample(xy, n=5000):
    xy = np.asarray(xy, float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tq = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(tq, s, xy[:, 0]), np.interp(tq, s, xy[:, 1])])


def continuous_domain_check(xy, ctx, n: int = 5000) -> dict:
    """Validate a route against the ORIGINAL (unfilled) raster, point by point.

    Three different notions of "inside the domain" are separated here:

    * `on_masked_cell`  - the sample lies in a cell that is land or NoData;
    * `masked_stencil`  - at least one of the four raster nodes used by the
      bilinear interpolation at that point is masked, i.e. the interpolant had
      to use filled values;
    * `point_violation` - the interpolated depth/slope at the point breaks the
      nominal limits.

    ADOPTED DEFINITION.  A point of a continuous route is *continuously
    feasible* when (i) it is not inside a masked cell and (ii) its interpolated
    depth and slope satisfy d_min <= d <= d_max and s <= s_max.  The mesh-based
    `feasible` array is a nearest-node test used by the grid solver; a point can
    trip it while satisfying (i) and (ii), and those samples are classified and
    counted rather than hidden.
    """
    grid, fields, feas, lim = ctx["grid"], ctx["fields"], ctx["feas"], ctx["lim"]
    xy = _resample(xy, n)
    X, Y = xy[:, 0], xy[:, 1]
    d = fields("d", X, Y)
    sl = fields("s", X, Y)
    j = np.clip(np.round((X - grid.x[0]) / grid.dx).astype(int), 0, len(grid.x) - 1)
    i = np.clip(np.round((Y - grid.y[0]) / grid.dy).astype(int), 0, len(grid.y) - 1)
    on_masked = fields.inside_mask(X, Y)
    cell_feas = feas[i, j]

    # bilinear stencil: how many of the four interpolation nodes carry real data?
    j0 = np.clip(np.floor((X - grid.x[0]) / grid.dx).astype(int), 0, len(grid.x) - 2)
    i0 = np.clip(np.floor((Y - grid.y[0]) / grid.dy).astype(int), 0, len(grid.y) - 2)
    valid = ~grid.mask_land
    n_stencil_masked = np.zeros(n, int)
    for a in (0, 1):
        for b in (0, 1):
            ok = valid[i0 + a, j0 + b] & np.isfinite(fields.raw_valid["d"][i0 + a, j0 + b])
            n_stencil_masked += (~ok).astype(int)

    v_depth = np.maximum(np.maximum(lim["d_min"] - d, d - lim["d_max"]), 0.0)
    v_slope = np.maximum(sl - lim["s_max"], 0.0)
    point_bad = (v_depth > 0) | (v_slope > 0)
    infeas_cell = ~cell_feas
    out = {
        "n_points": int(n),
        "on_masked_cell": int(on_masked.sum()),
        "masked_stencil": int((n_stencil_masked > 0).sum()),
        "masked_stencil_fraction": float((n_stencil_masked > 0).mean()),
        "point_violation_depth": int((v_depth > 0).sum()),
        "point_violation_slope": int((v_slope > 0).sum()),
        "point_violation_total": int(point_bad.sum()),
        "on_infeasible_cell": int(infeas_cell.sum()),
        "cell_flag_only": int((infeas_cell & ~point_bad & ~on_masked).sum()),
        "infeasible_and_point_bad": int((infeas_cell & point_bad).sum()),
        "infeasible_and_masked": int((infeas_cell & on_masked).sum()),
        "min_clearance_to_mask_m": float(np.min(fields.distance_to_mask(X, Y))),
        "p5_clearance_to_mask_m": float(np.percentile(fields.distance_to_mask(X, Y), 5)),
        "continuously_feasible": bool((~on_masked & ~point_bad).all()),
        "worst_depth_violation_m": float(v_depth.max()),
        "worst_slope_violation_deg": float(np.degrees(np.arctan(np.max(sl)) - np.arctan(lim["s_max"]))),
        "adopted_definition": (
            "continuously feasible = not inside a land/NoData cell AND interpolated "
            "depth and slope satisfy the nominal limits at the sample point"),
    }
    if out["cell_flag_only"]:
        sel = infeas_cell & ~point_bad & ~on_masked
        out["cell_flag_only_slope_margin_deg"] = float(np.degrees(
            np.arctan(lim["s_max"]) - np.arctan(sl[sel].max())))
        out["cell_flag_only_worst_slope_deg"] = float(np.degrees(np.arctan(sl[sel].max())))
        out["cell_flag_only_explanation"] = (
            "these samples satisfy the depth and slope limits at the sample point but "
            "fall in a cell whose node-level slope violates s_max; with a 450 m cell "
            "the node value is a cell average, so a route can clip the edge of a "
            "flagged cell while staying inside the continuous limits")
    return out


def interpolation_comparison(ctx, cfg, ev) -> dict:
    """Cubic vs linear interpolators: overshoot, leakage, route, objective."""
    grid, F, lim = ctx["grid"], ctx["F"], ctx["lim"]
    A, B = ctx["A"], ctx["B"]
    out = {"checks": {}, "route_field_difference": {}}
    route_xy = _resample(ev["xy_hard"], 4000)
    base_fields = ctx["fields"]
    base_d = base_fields("d", route_xy[:, 0], route_xy[:, 1])
    base_s = base_fields("s", route_xy[:, 0], route_xy[:, 1])
    base_R = base_fields("R", route_xy[:, 0], route_xy[:, 1])
    out["route_field_difference"]["max_abs_depth_difference_m"] = 0.0
    out["route_field_difference"]["max_abs_slope_difference"] = 0.0

    for method in ("cubic", "linear"):
        f = SmoothFields(grid.x, grid.y, grid.z, grid.d, F["s"], F["R"],
                         grid.mask_land, method=method)
        chk = f.check(n_probe=20000)
        out["checks"][method] = {
            "spline_order": chk["spline_order"],
            "depth_interior_max_overshoot_m": chk["d"]["interior_max_overshoot"],
            "depth_boundary_max_overshoot_m": chk["d"]["boundary_max_overshoot"],
            "slope_interior_p99_overshoot": chk["s"]["interior_p99_overshoot"],
            "risk_interior_max_overshoot": chk["R"]["interior_max_overshoot"],
            "n_probe_implying_land": chk["depth_leakage"]["n_probe_implying_land"],
            "min_interp_depth_m": chk["depth_leakage"]["min_interp_depth_m"],
        }
        if method == "linear":
            d = f("d", route_xy[:, 0], route_xy[:, 1])
            sl = f("s", route_xy[:, 0], route_xy[:, 1])
            R = f("R", route_xy[:, 0], route_xy[:, 1])
            out["route_field_difference"]["max_abs_depth_difference_m"] = float(
                np.max(np.abs(d - base_d)))
            out["route_field_difference"]["max_abs_slope_difference"] = float(
                np.max(np.abs(sl - base_s)))
            out["route_field_difference"]["max_abs_risk_difference"] = float(
                np.max(np.abs(R - base_R)))

    # re-solve the hard problem on the linear interpolant from the same starts
    spl = Spline(A=A, B=B, K=cfg["hard"]["n_control_points"],
                  bc=cfg["hard"].get("spline_bc", "clamped"),
                 extent=(grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0]),
                 origin=(grid.x[0], grid.y[0]), M=cfg["hard"]["n_samples"])
    f_lin = SmoothFields(grid.x, grid.y, grid.z, grid.d, F["s"], F["R"],
                         grid.mask_land, method="linear")
    hp_lin = HardProblem(f_lin, spl, lim["d_min"], lim["d_max"], lim["s_max"],
                         lim["kappa_max"], alpha=cfg["hard"]["alpha"],
                         beta=cfg["hard"]["beta"], M=cfg["hard"]["n_samples"],
                         margin=cfg["hard"].get("constraint_margin", 0.0),
                         n_constraint=cfg["hard"].get("n_constraint_samples"))
    starts_lin = make_starts(hp_lin, ctx["ez"].xy, cfg, n_random=0)
    best_lin, runs_lin = solve_hard(hp_lin, starts_lin, strategy="penalty")
    v = metrics.validate_polyline(best_lin.xy, ctx["fields"], lim, 2000)
    out["linear_solution"] = {
        "J_km": best_lin.J, "L_km": best_lin.L / 1000.0, "S": best_lin.S,
        "max_kappa": best_lin.max_kappa, "min_radius_m": 1.0 / best_lin.max_kappa
        if best_lin.max_kappa else float("inf"),
        "collocation_feasible": bool(max(best_lin.max_violation.values()) <= 1e-3),
        "feasible": bool(max(best_lin.max_violation.values()) <= 1e-3 and v["n_violations"] == 0),
        "feasibility_note": "collocation uses linear fields; dense validation below uses the baseline cubic fields, and no full independent linear-field classification was performed",
        "dense_max_slope_deg": v["slope_max_deg"],
        "dense_n_violations": v["n_violations"],
        "hausdorff_vs_adopted_m": metrics.hausdorff_distance(ctx["hard_best"].xy,
                                                             best_lin.xy)["hausdorff_m"],
        "jaccard_vs_adopted": metrics.cell_overlap(ctx["hard_best"].xy, best_lin.xy, grid)["jaccard"],
        "J_difference_km": best_lin.J - ctx["hard_best"].J,
        "L_difference_km": best_lin.L / 1000.0 - ctx["hard_best"].L / 1000.0,
    }
    out["decision"] = (
        "the cubic tensor-product spline is kept: it is the smooth field the "
        "B-spline corridor is meant to follow, its interior depth overshoot stays "
        "below 10 m (1.2 % of the 800 m depth span) and the adopted route is "
        "linear candidate must also be checked against the baseline field. Linear interpolation is "
        "reported as the overshoot-free reference, not adopted, because it has "
        "piecewise-constant gradients that make the slope constraint meaningless "
        "on a 450 m grid")
    return out


def curvature_verification(ctx, ev) -> dict:
    """Independent curvature recomputation from the adopted B-spline."""
    from .hard import independent_curvature
    hp, lim = ctx["hp"], ctx["lim"]
    ind = independent_curvature(hp.spl, ctx["hard_best"].v, n=2001)
    dense = metrics.validate_polyline(ev["xy_hard"], ctx["fields"], lim, 5000)
    ind["limit_kappa_max"] = lim["kappa_max"]
    ind["limit_min_radius_m"] = 1.0 / lim["kappa_max"]
    ind["enforced_kappa_max"] = hp.kappa_max
    ind["enforced_min_radius_m"] = 1.0 / hp.kappa_max
    ind["violation_vs_nominal"] = float(max(0.0, ind["kappa_max"] - lim["kappa_max"]))
    ind["violation_vs_enforced"] = float(max(0.0, ind["kappa_max"] - hp.kappa_max))
    ind["within_nominal_limit"] = bool(ind["kappa_max"] <= lim["kappa_max"])
    ind["polyline_estimator_kappa_max"] = dense["kappa_max"]
    ind["agreement_vs_polyline_estimator"] = float(
        abs(ind["kappa_max"] - dense["kappa_max"]))
    return ind


def runtime_protocol(ctx, cfg, repeats: int = 3) -> dict:
    """Reconcile the runtime numbers: one solve, a solve budget, and spread.

    * `easy_single_dijkstra_s`      - one Dijkstra call, one run.
    * `hard_adopted_penalty_continuation_s` - original adopted penalty continuation,
      from its warm start; not a re-refinement from the adopted point.
    * `hard_full_multistart_budget_s` - wall-clock for all solver executions of the
      reported multi-start run (sum of per-run times).
    * `*_repeated`                  - median/min/max over `repeats` repetitions of
      the same call on the same machine, so the noise of a single measurement is
      visible.
    """
    import time
    from .easy import solve_easy as _se

    easy_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _se(ctx["feas"], ctx["grid"].x, ctx["grid"].y, ctx["ia"], ctx["ib"],
            edge_rule=cfg["easy"]["edge_rule"], exact_xy=(ctx["A"], ctx["B"]),
            d=ctx["grid"].d, s=ctx["F"]["s"], lim=ctx["lim"])
        easy_times.append(time.perf_counter() - t0)

    hard_times = []
    for _ in range(repeats):
        r = ctx["hp"].solve_penalty(ctx["hard_best"].v, "runtime_probe")
        hard_times.append(r.runtime_s)

    runs = [r for r in ctx["hard_runs"] if np.isfinite(r.J)]
    return {
        "easy_single_dijkstra_s": float(ctx["ez"].runtime_s),
        "easy_repeated_median_s": float(np.median(easy_times)),
        "easy_repeated_min_s": float(np.min(easy_times)),
        "easy_repeated_max_s": float(np.max(easy_times)),
        "hard_adopted_penalty_continuation_s": float(ctx["hard_best"].runtime_s),
        "hard_repeated_median_s": float(np.median(hard_times)),
        "hard_repeated_min_s": float(np.min(hard_times)),
        "hard_repeated_max_s": float(np.max(hard_times)),
        "hard_full_multistart_budget_s": float(sum(r.runtime_s for r in runs)),
        "n_executions_in_budget": int(len(runs)),
        # the ONLY ratio reported: median-vs-median of repeated runs, same machine
        "ratio_median_hard_to_easy": float(np.median(hard_times) / max(np.median(easy_times), 1e-9)),
        "measurement_basis": ("median of %d repeated calls of each operation on the same "
                              "machine; baseline Dijkstra timer covers graph search, while repeated "
                              "easy timer covers solve_easy including route assembly; hard repetitions "
                              "are penalty continuations from the adopted point" % repeats),
        "note": ("""(1) original easy timing covers the Dijkstra call; easy repeated timing also includes
route assembly. Original adopted hard timing covers the continuation from its warm start;
repeated hard timing measures a separate continuation from the adopted control points;
(2) the multi-start budget is the sum over every solver execution and is the
cost of *finding* the candidate, not of evaluating it; (3) repeated-run medians
are quoted wherever a single measurement would be presented as precise.""")
    }


def beta0_comparison(ctx, cfg) -> dict:
    """Hard problem with the risk term removed (beta = 0): geometry only.

    This separates the three reasons the hard route differs from the grid route:
    continuous geometry, the curvature constraint, and risk minimisation.  With
    beta = 0 the objective is pure length, so any difference from the beta = 1
    route is attributable to risk.
    """
    grid, lim, A, B = ctx["grid"], ctx["lim"], ctx["A"], ctx["B"]
    spl = Spline(A=A, B=B, K=cfg["hard"]["n_control_points"],
                  bc=cfg["hard"].get("spline_bc", "clamped"),
                 extent=(grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0]),
                 origin=(grid.x[0], grid.y[0]), M=cfg["hard"]["n_samples"])
    hp0 = HardProblem(ctx["fields"], spl, lim["d_min"], lim["d_max"], lim["s_max"],
                      lim["kappa_max"], alpha=cfg["hard"]["alpha"], beta=0.0,
                      M=cfg["hard"]["n_samples"],
                      margin=cfg["hard"].get("constraint_margin", 0.0),
                      n_constraint=cfg["hard"].get("n_constraint_samples"))
    starts0 = make_starts(hp0, ctx["ez"].xy, cfg, n_random=0)
    best0, runs0 = solve_hard(hp0, starts0, strategy="penalty")
    v0 = metrics.validate_polyline(best0.xy, ctx["fields"], lim, 2000)
    k0 = independent_curvature(hp0.spl, best0.v, n=2001)
    L0, S0 = metrics.polyline_terms(best0.xy, ctx["fields"])[:2]
    base = ctx["hard_best"]
    return {
        "beta0": {"L_km": L0 / 1000.0, "S": S0, "J_km": L0 / 1000.0,
                  "max_kappa": k0["kappa_max"],
                  "min_radius_m": k0["kappa_min_radius_m"],
                  "max_kappa_polyline_diag": v0["kappa_max"],
                  "max_slope_deg": v0["slope_max_deg"],
                  "n_violations": v0["n_violations"],
                  "feasible": bool(max(best0.max_violation.values()) <= 1e-3)},
        "beta1": {"L_km": base.L / 1000.0, "S": base.S, "J_km": base.J,
                  "max_kappa": base.max_kappa},
        "difference": {
            "L_km": L0 / 1000.0 - base.L / 1000.0,
            "L_pct": 100.0 * (L0 / 1000.0 - base.L / 1000.0) / (L0 / 1000.0),
            "S": S0 - base.S,
            "S_pct": 100.0 * (S0 - base.S) / S0 if S0 else float("nan"),
            "hausdorff_m": metrics.hausdorff_distance(best0.xy, base.xy)["hausdorff_m"],
            "jaccard": metrics.cell_overlap(best0.xy, base.xy, grid)["jaccard"],
        },
        "note": ("beta = 0 removes risk from the objective but keeps the curvature "
                 "and terrain constraints, so beta0 vs beta1 isolates the cost of "
                 "risk avoidance in metres of added length."),
    }


def edge_rule_comparison(ctx, cfg) -> dict:
    """Solve the easy problem under all three documented edge rules."""
    grid, F, lim = ctx["grid"], ctx["F"], ctx["lim"]
    out = {}
    base_xy = ctx["ez"].xy
    for rule in ("midpoint_corner", "node", "midpoint", "corner"):
        r = solve_easy(ctx["feas"], grid.x, grid.y, ctx["ia"], ctx["ib"], edge_rule=rule,
                       exact_xy=(ctx["A"], ctx["B"]), d=grid.d, s=F["s"], lim=lim)
        out[rule] = {
            "connected": bool(r.connected),
            "L_km": r.length_m / 1000.0,
            "nodes_expanded": r.nodes_expanded,
            "runtime_s": r.runtime_s,
            "n_nodes": int(len(r.path_ij)),
            "hausdorff_vs_baseline_m": metrics.hausdorff_distance(base_xy, r.xy)["hausdorff_m"]
            if r.connected else float("nan"),
        }
    out["baseline_rule"] = cfg["easy"]["edge_rule"]
    out["note"] = ("`midpoint_corner` is the specification's rule and the baseline: a "
                   "diagonal step needs (a) both endpoints feasible, (b) its geometric "
                   "midpoint -- the point halfway along the segment, i.e. the average of "
                   "the four surrounding node values -- inside the depth and slope limits, "
                   "and (c) both orthogonal neighbouring cells feasible. `node`, `midpoint` "
                   "and `corner` drop parts (b) / (c) and are kept only as variants.")
    return out


def verification(ctx: dict, ev: dict) -> dict:
    """Section 11 checks that can be automated."""
    cfg, grid, hp = ctx["cfg"], ctx["grid"], ctx["hp"]
    lim = ctx["lim"]
    A, B = ctx["A"], ctx["B"]
    out = {}

    # 1. endpoints
    e = ctx["ez"].xy
    h = ev["xy_hard"]
    out["endpoints"] = {
        "easy_start_err_m": float(np.hypot(*(e[0] - A))),
        "easy_end_err_m": float(np.hypot(*(e[-1] - B))),
        "hard_start_err_m": float(np.hypot(*(h[0] - A))),
        "hard_end_err_m": float(np.hypot(*(h[-1] - B))),
    }

    # 2. independent re-computation of L and S
    L1, S1 = metrics.polyline_terms(e, ctx["fields"])[:2]
    L2, S2 = metrics.naive_L_S(e, ctx["fields"])
    Lh1, Sh1 = metrics.polyline_terms(h, ctx["fields"])[:2]
    Lh2, Sh2 = metrics.naive_L_S(h, ctx["fields"])
    out["recompute"] = {
        "easy_L_diff_m": float(abs(L1 - L2)), "easy_S_diff": float(abs(S1 - S2)),
        "hard_L_diff_m": float(abs(Lh1 - Lh2)), "hard_S_diff": float(abs(Sh1 - Sh2)),
    }

    # 3. Dijkstra cross-check against scipy.sparse.csgraph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    feas = ctx["feas"]
    ny, nx = feas.shape
    rows, cols, data = [], [], []
    from .easy import OFFSETS, make_edge_rule
    idx = -np.ones(feas.shape, int)
    ii, jj = np.where(feas)
    idx[ii, jj] = np.arange(ii.size)
    ok_edge = make_edge_rule(feas, cfg["easy"]["edge_rule"], grid.d, ctx["F"]["s"], lim)
    for i, j in zip(ii, jj):
        for di, dj, mult, _ in OFFSETS:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and feas[ni, nj] and ok_edge(i, j, di, dj):
                rows.append(idx[i, j]); cols.append(idx[ni, nj])
                # same metric weight as the production Dijkstra: measured from the
                # node coordinates, not assumed from dx/dy, so the cross-check
                # tests the path search and not a different length convention
                data.append(edge_weight(grid.x[j], grid.y[i],
                                        grid.x[nj], grid.y[ni]))
    G = coo_matrix((data, (rows, cols)), shape=(ii.size, ii.size)).tocsr()
    d_sp = sp_dijkstra(G, indices=idx[ctx["ia"][0], ctx["ia"][1]])
    node_poly = np.column_stack([grid.x[ctx["ez"].path_ij[:, 1]],
                                 grid.y[ctx["ez"].path_ij[:, 0]]])
    own_node_len = float(np.sum(np.hypot(*np.diff(node_poly, axis=0).T)))
    out["dijkstra_crosscheck"] = {
        "own_length_m": own_node_len,
        "own_length_with_exact_endpoints_m": ctx["ez"].length_m,
        "scipy_length_m": float(d_sp[idx[ctx["ib"][0], ctx["ib"][1]]]),
        "difference_m": float(abs(own_node_len - d_sp[idx[ctx["ib"][0], ctx["ib"][1]]])),
        "connector_A_m": float(np.linalg.norm(ctx["A"] - node_poly[0])),
        "connector_B_m": float(np.linalg.norm(ctx["B"] - node_poly[-1])),
        "snapped_A_xy_m": node_poly[0].tolist(),
        "snapped_B_xy_m": node_poly[-1].tolist(),
        "scope": "shortest graph path between selected snapped cells only; exact endpoint connectors are appended and validated, not optimized over alternative connector choices",
    }

    # 4. dense validation + density study + domain check (both routes)
    out["density_easy"] = metrics.density_study(ev["xy_easy"], ctx["fields"], lim,
                                                cfg["validation"]["density_study_points"])
    out["density_hard"] = metrics.density_study(ev["xy_hard"], ctx["fields"], lim,
                                                cfg["validation"]["density_study_points"])
    out["domain_easy"] = metrics.domain_check(ev["xy_easy"], grid, feas, lim=ctx["lim"],
                                              fields=ctx["fields"])
    out["domain_hard"] = metrics.domain_check(ev["xy_hard"], grid, feas, lim=ctx["lim"],
                                              fields=ctx["fields"])
    # how much of the easy route's length is 8-neighbour discretisation?
    out["easy_route_discretisation"] = {
        "L_discrete_m": float(ev["L_easy_m"]),
        "L_smoothed_250m_m": float(metrics.smoothed_path_length(ev["xy_easy"])),
        "note": ("the easy route is measured on the discrete node-to-node polyline; "
                 "a light 250 m moving-average smoothing of the same corridor is "
                 "shorter, so part of any negative 'price of realism' is grid "
                 "discretisation rather than spline smoothness."),
    }
    out["easy_route_discretisation"]["inflation_pct"] = 100.0 * (
        out["easy_route_discretisation"]["L_discrete_m"]
        / out["easy_route_discretisation"]["L_smoothed_250m_m"] - 1.0)

    # 5. easy-route geometry information (no curvature limit in the easy problem)
    out["easy_geometry"] = heading_changes(ev["xy_easy"])
    out["easy_geometry"]["smoothed_fit_curvature"] = metrics.smooth_fit_curvature(ev["xy_easy"])
    out["easy_geometry"]["note"] = (
        "The easy problem does not constrain route curvature. The 8-neighbour grid "
        "route is a zig-zag polyline, so pointwise finite-difference curvature of "
        "the polyline is not meaningful; heading changes and the curvature of a "
        "smoothed spline fit are reported for information.")

    # 6. Independently classify each solver execution, not just the adopted route.
    # Keep solver-status and physical feasibility separate. No optimization is changed.
    independent = []
    for r in ctx["hard_runs"]:
        if np.isfinite(r.J) and len(r.xy) >= 2:
            dense_r = metrics.validate_polyline(r.xy, ctx["fields"], lim, 2000)
            exact_r = independent_curvature(hp.spl, r.v, n=2001)
            domain_r = metrics.domain_check(r.xy, grid, ctx["feas"], n_points=2000,
                                            lim=lim, fields=ctx["fields"])
            verdict = classify_route_feasibility(r.meta.get("max_violation_nominal"),
                                                dense_r, exact_r, domain_r, True, lim)
        else:
            verdict = {"feasible": False, "reasons": ["solver returned no finite geometry"]}
        r.meta["independent_feasibility"] = verdict
        independent.append((r, verdict))
    good = [r for r, v in independent if v["feasible"]]
    bad = [r for r, v in independent if not v["feasible"]]
    bad_j = [r.J for r in bad if np.isfinite(r.J)]
    good_j = [r.J for r in good]
    out["multistart"] = account_runs(ctx["hard_runs"], ctx["starts"])
    out["multistart"].update(
        n_feasible=len(good), n_infeasible=len(bad),
        feasible_start_labels=sorted({r.start_label for r in good}),
        feasible_solvers=sorted({r.solver for r in good}),
        J_feasible_min=float(min(good_j)) if good_j else float("nan"),
        J_feasible_max=float(max(good_j)) if good_j else float("nan"),
        J_feasible_spread=float(max(good_j)-min(good_j)) if len(good_j)>1 else 0.0,
        n_infeasible_with_J=len(bad_j),
        J_infeasible_range=[float(min(bad_j)), float(max(bad_j))] if bad_j else [],
        feasibility_basis="independent nominal, dense depth/slope, exact spline curvature and domain checks for each execution",
        note="Distinct initial curves and solver executions differ; all executions are independently classified. Feasibility is not optimality.")
    out["multistart"]["adopted"] = {
        "solver": ctx["hard_best"].solver, "start": ctx["hard_best"].start_label,
        "status": ctx["hard_best"].status, "J_km": ctx["hard_best"].J,
        "feasible": bool(ev["feasibility_hard"]["feasible"]),
    }
    out["solver_strategies"] = {}
    for label, solver in (("direct", "slsqp_direct"), ("penalty", "slsqp_penalty")):
        subset = [(r, v) for r, v in independent if r.solver == solver]
        out["solver_strategies"][label] = {
            "description": "direct constrained SLSQP" if label == "direct" else "quadratic exterior penalty continuation",
            "n_executions": len(subset),
            "n_converged_status0": sum(r.status == 0 for r, _ in subset),
            "n_feasible": sum(bool(v["feasible"]) for _, v in subset),
        }
    out["solver_strategies"]["penalty"]["starts"] = out["multistart"]["penalty_start_labels"]
    out["solver_strategies"]["feasibility_vs_optimality"] = (
        "Independent validation checks feasibility only, not optimality or a KKT certificate.")

    # 6d. interpolation choice, continuous domain, curvature, runtimes, beta = 0, edge rules
    out["interpolation"] = interpolation_comparison(ctx, cfg, ev)
    out["continuous_domain_easy"] = continuous_domain_check(ev["xy_easy"], ctx, 5000)
    out["continuous_domain_hard"] = continuous_domain_check(ev["xy_hard"], ctx, 5000)
    out["curvature"] = curvature_verification(ctx, ev)
    out["runtimes"] = runtime_protocol(ctx, cfg)
    out["beta0"] = beta0_comparison(ctx, cfg)
    out["edge_rules"] = edge_rule_comparison(ctx, cfg)

    # 6b. collocation density: is the residual violation a sampling artefact?
    # The hard problem constrains 4*M pointwise values; the route is validated at
    # 2000 points.  Re-solving with M doubled/quadrupled from the adopted
    # control points shows whether a denser collocation set closes the gap.
    coll = {}
    for factor in (2, 4):
        M2 = int(cfg["hard"]["n_samples"]) * factor
        spl2 = Spline(A=A, B=B, K=cfg["hard"]["n_control_points"],
                     bc=cfg["hard"].get("spline_bc", "clamped"),
                      extent=(grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0]),
                      origin=(grid.x[0], grid.y[0]), M=M2)
        hp2 = HardProblem(ctx["fields"], spl2, lim["d_min"], lim["d_max"], lim["s_max"],
                          lim["kappa_max"], alpha=cfg["hard"]["alpha"],
                          beta=cfg["hard"]["beta"], M=M2,
                          margin=cfg["hard"].get("constraint_margin", 0.0),
                          n_constraint=int(M2 * 2))
        r2 = hp2.solve_penalty(ctx["hard_best"].v, f"warm_restart_M{M2}",
                               mus=(1e2, 1e4, 1e6))
        v2 = metrics.validate_polyline(r2.xy, ctx["fields"], lim, 2000)
        # the constraint quantity is the exact B-spline curvature, not the
        # finite-difference estimate on the polyline (kept as a diagnostic)
        k2 = independent_curvature(hp2.spl, r2.v, n=2001)
        coll[f"M={M2}"] = {
            "configured_M": M2, "actual_integration_and_collocation_samples": hp2.M,
            "J_km": r2.J, "L_km": r2.L / 1000.0, "S": r2.S,
            "max_violation_at_collocation": float(max(r2.max_violation.values())),
            "dense_max_slope_deg": v2["slope_max_deg"],
            "dense_n_slope_violations": v2["n_violations_slope"],
            "kappa_max_exact": float(k2["kappa_max"]),
            "min_radius_exact_m": float(1.0 / k2["kappa_max"]),
            "kappa_max_polyline_diag": v2["kappa_max"],
            "n_dense_depth_violations": v2["n_violations_depth"],
            "status": int(r2.status), "runtime_s": r2.runtime_s,
        }
    out["collocation_sensitivity"] = coll

    # 6c. what does the 2 % enforcement margin cost?
    margin = float(cfg["hard"].get("constraint_margin", 0.0) or 0.0)
    if margin > 0:
        spl3 = Spline(A=A, B=B, K=cfg["hard"]["n_control_points"],
                     bc=cfg["hard"].get("spline_bc", "clamped"),
                      extent=(grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0]),
                      origin=(grid.x[0], grid.y[0]), M=cfg["hard"]["n_samples"])
        hp3 = HardProblem(ctx["fields"], spl3, lim["d_min"], lim["d_max"], lim["s_max"],
                          lim["kappa_max"], alpha=cfg["hard"]["alpha"],
                          beta=cfg["hard"]["beta"], M=cfg["hard"]["n_samples"],
                          margin=0.0, n_constraint=cfg["hard"].get("n_constraint_samples"))
        starts3 = make_starts(hp3, ctx["ez"].xy, cfg, n_random=0)
        best3, _ = solve_hard(hp3, starts3, strategy="penalty")
        v3 = metrics.validate_polyline(best3.xy, ctx["fields"], lim, 2000)
        out["margin_cost"] = {
            "margin": margin,
            "J_with_margin_km": ctx["hard_best"].J,
            "J_without_margin_km": best3.J,
            "difference_km": best3.J - ctx["hard_best"].J,
            "relative_pct": 100.0 * (best3.J - ctx["hard_best"].J) / best3.J,
            "dense_slope_violations_without_margin": v3["n_violations_slope"],
            "dense_max_slope_without_margin_deg": v3["slope_max_deg"],
            "kappa_max_exact_without_margin": float(
                independent_curvature(spl3, best3.v, n=2001)["kappa_max"]),
            "n_dense_depth_violations_without_margin": v3["n_violations_depth"],
            "note": ("a 2 % margin means the solver targets 0.98 x the slope and curvature "
                     "limits, so that the independently validated route stays inside the "
                     "nominal limits between collocation samples"),
        }

    # 7. depth meaning reminder (Sec. 3.4)
    out["depth_convention"] = (
        "d is the GEBCO SEA-FLOOR depth (deepest = largest d). d_min/d_max limit the "
        "sea-floor depth along the route; they are NOT the tunnel depth below the seabed.")
    return out


def run_baseline(cfg: dict | None = None, verbose: bool = True, strategy="auto"):
    cfg = cfg or yaml.safe_load((ROOT / "config.yaml").read_text())
    ctx = baseline_route_set(cfg, verbose=verbose, strategy=strategy)
    ev = evaluate_routes(ctx, verbose=verbose)
    ver = verification(ctx, ev)
    RESULTS.mkdir(exist_ok=True)
    # Rewrite now that every execution has an independent classification.
    json.dump({"interpolator_check": ctx["interp_check"],
               "hard_runs": [serialise_run(r) for r in ctx["hard_runs"]]},
              open(RESULTS / "hard_runs.json", "w"), indent=2, default=float)
    # Persist the exact geometry used for the reported metrics and the adopted decision vector.
    # Hard route uses the optimizer's active integration/collocation parameter grid.
    P = ctx["hp"].spl.control_points(ctx["hard_best"].v)
    easy_xy, hard_xy = ctx["ez"].xy, ctx["hard_best"].xy
    cps = [{"point": f"P{i}", "role": "A (fixed)" if i == 0 else
            ("B (fixed)" if i == len(P)-1 else "interior (optimized)"),
            "easting_m": float(x), "northing_m": float(y)}
           for i, (x, y) in enumerate(P)]
    pd.DataFrame(cps).to_csv(RESULTS / "control_points_hard.csv", index=False, float_format="%.12f")
    pd.DataFrame({"point_index": np.arange(len(easy_xy)), "easting_m": easy_xy[:, 0],
                  "northing_m": easy_xy[:, 1]}).to_csv(RESULTS / "route_easy.csv", index=False, float_format="%.12f")
    pd.DataFrame({"point_index": np.arange(len(hard_xy)), "parameter_t": ctx["hp"].spl.t,
                  "easting_m": hard_xy[:, 0], "northing_m": hard_xy[:, 1]}).to_csv(
                      RESULTS / "route_hard.csv", index=False, float_format="%.12f")
    geometry = {"crs": cfg["data"]["crs_epsg"], "control_points_hard": cps,
                "hard_normalized_decision_vector": ctx["hard_best"].v.tolist(),
                "route_easy_xy_m": easy_xy.tolist(), "route_hard_xy_m": hard_xy.tolist(),
                "hard_sample_parameter_t": ctx["hp"].spl.t.tolist(),
                "description": "Exact endpoints and ordered graph nodes; adopted spline sampled on optimizer active integration/collocation grid. Interior B-spline control points do not generally lie on the curve."}
    json.dump({"comparison": ev["table"], "verification": ver, "geometry": geometry,
               "sampling_protocol": {"configured_n_samples": cfg["hard"]["n_samples"],
                    "configured_n_constraint_samples": cfg["hard"]["n_constraint_samples"],
                    "actual_objective_and_constraint_samples": ctx["hp"].M,
                    "note": "HardProblem sets its active spline grid to n_constraint_samples; both objective integration and collocation use it."},
               "route_feasibility": {"easy": ev["feasibility_easy"],
                                     "hard": ev["feasibility_hard"]},
               "kappa_exact_hard": ev["kappa_exact"],
               "endpoints": {
                   "A_lonlat": [cfg["data"]["A"]["lon"], cfg["data"]["A"]["lat"]],
                   "B_lonlat": [cfg["data"]["B"]["lon"], cfg["data"]["B"]["lat"]],
                   "A_xy_m": [float(x) for x in ctx["A"]],
                   "B_xy_m": [float(x) for x in ctx["B"]],
                   "A_cell": [int(v) for v in ctx["ia"]],
                   "B_cell": [int(v) for v in ctx["ib"]],
                   "crs": cfg["data"]["crs_epsg"],
               },
               "interpolator_check": ctx["interp_check"],
               "data_meta": ctx["grid"].meta,
               "feature_settings": ctx["F"]["settings"],
               "norm_ranges": ctx["F"]["norm_ranges"],
               "limits": ctx["lim"],
               "beta": cfg["hard"]["beta"], "alpha": cfg["hard"]["alpha"]},
              open(RESULTS / "baseline.json", "w"), indent=2, default=float)
    csv_row = dict(ev["table"])
    csv_row["runtime_easy_baseline_graph_search_s"] = csv_row.pop("runtime_easy_s")
    csv_row["runtime_hard_adopted_penalty_continuation_s"] = csv_row.pop("runtime_hard_s")
    rt = ver["runtimes"]
    csv_row.update(runtime_easy_repeated_solve_median_s=rt["easy_repeated_median_s"],
                   runtime_hard_repeated_refinement_median_s=rt["hard_repeated_median_s"],
                   runtime_hard_full_multistart_budget_s=rt["hard_full_multistart_budget_s"],
                   runtime_hard_budget_executions=rt["n_executions_in_budget"])
    pd.DataFrame([csv_row]).to_csv(RESULTS / "comparison.csv", index=False)
    _write_comparison_md(ev["table"], RESULTS / "comparison_table.md")
    return ctx, ev, ver


def _write_comparison_md(t: dict, path: Path) -> None:
    """The §11 comparison table as Markdown, regenerated with the baseline."""
    rows = [
        ("Metric", "Easy (Dijkstra)", "Hard (clamped B-spline)"),
        ("Length L", f"{t['L_easy_km']:.3f} km", f"{t['L_hard_km']:.3f} km"),
        ("Cumulative risk S", f"{t['S_easy']:.1f}", f"{t['S_hard']:.1f}"),
        ("J = L + beta*S (post-hoc common score, easy)", f"{t['J_easy_km']:.3f} km", ""),
        ("J = alpha*L + beta*S (hard objective)", "", f"{t['J_hard_km']:.3f} km"),
        ("Max slope", f"{t['easy_max_slope_deg']:.3f} deg", f"{t['hard_max_slope_deg']:.3f} deg"),
        ("Max route curvature (exact spline, hard)",
         f"{t['easy_kappa_max']:.3e} 1/m (polyline estimate, no limit)",
         f"{t['hard_kappa_exact']:.3e} 1/m"),
        ("Min turning radius",
         f"{1/t['easy_kappa_max']:.0f} m",
         f"{t['hard_min_radius_exact_m']:.0f} m (exact) / "
         f"{t['hard_min_radius_polyline_diagnostic_m']:.0f} m (polyline diag.)"),
        ("Limit violations (dense)",
         f"{t['easy_violations_slope_depth']} slope/depth (+{t['easy_violations_kappa_info']} kappa, informational)",
         f"{t['hard_violations_slope_depth']} slope/depth, {t['hard_violations_kappa']} kappa (polyline diag.)"),
        ("Price of realism (L_H - L_E)/L_E", f"{t['price_of_realism_pct']:+.2f} %", ""),
        ("Risk reduction (S_E - S_H)/S_E", f"{t['risk_reduction_pct']:+.2f} %", ""),
        ("Hausdorff / mean distance", f"{t['hausdorff_m']:.0f} m / {t['mean_distance_m']:.0f} m", ""),
        ("Cell overlap (Jaccard)", f"{t['jaccard']:.3f}", ""),
    ]
    out = ["# Comparison table (regenerated from results/baseline.json)", "",
           "| " + " | ".join(rows[0]) + " |", "|" + "---|" * 3]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    out += ["", f"* {t['J_easy_label']}",
            f"* {t['hard_objective_label']}",
            f"* {t['integrator_note']}"]
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    ctx, ev, ver = run_baseline()
    print(f"done in {time.time() - t0:.1f} s")
