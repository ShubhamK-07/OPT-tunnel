"""Sensitivity analysis (spec Sec. 9): one factor at a time, both problems re-solved.

Every run records: the changed factor, its settings, feature-map changes,
feasible-cell count/connectivity, whether a feasible route exists, L, S,
constraint violations, route-displacement metrics, runtime and solver status.

Usage:
    python -m src.sensitivity --set score      # the S-core set (presentation)
    python -m src.sensitivity --set full       # the full factor matrix
    python -m src.sensitivity --set secondary  # beta sweep, weights, K study
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics
from .analysis import feasibility_summary
from .data import load_grid
from .hard import independent_curvature
from .pipeline import baseline_route_set, classify_route_feasibility, limits
from .features import derive_features, feasible_mask

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _run_config(cfg, grid, base_xy_easy=None, base_xy_hard=None, label="", factor="",
                value="", n_random=0, strategy="penalty", base_fields=None):
    """One full re-derivation + re-solve for a given configuration.

    Two evaluations are recorded for every route:

    * NATIVE  -- L, S, J measured on this configuration's own risk field.  These
      are the values the configuration's solver actually saw, but because the
      min-max normalisation is recomputed inside each configuration they are NOT
      comparable with another configuration's numbers.
    * FROZEN  -- L, S, J of the same route re-measured on the BASELINE risk field
      and the baseline normalisation (`base_fields`).  This is the only risk
      number that may be compared across configurations, and it is what the
      report uses for cross-configuration statements.
    """
    t0 = time.perf_counter()
    ctx = baseline_route_set(cfg, grid=grid, verbose=False, strategy=strategy,
                             n_random=n_random, save=False)
    def _xy(a):
        a = np.asarray(a, float)
        if a.ndim != 2 or len(a) < 3:
            return a
        return metrics.polyline_terms(a, ctx["fields"], M=2000)[2]

    xyE = _xy(ctx["ez"].xy)
    xyH = _xy(ctx["hard_best"].xy)

    lim = ctx["lim"]
    # L and S are always the exact segment sums of the route polyline (no
    # resampling); the 2000-point resample is used only to *validate*
    # constraints, because constraint checks need dense samples.
    def _LS(a):
        a = np.asarray(a, float)
        if a.ndim != 2 or len(a) < 2:
            return float("nan"), float("nan")
        return metrics.polyline_terms(a, ctx["fields"])[:2]

    L_E, S_E = _LS(ctx["ez"].xy)
    L_H, S_H = _LS(ctx["hard_best"].xy)
    vE = metrics.validate_polyline(xyE, ctx["fields"], lim, 2000)
    vH = metrics.validate_polyline(xyH, ctx["fields"], lim, 2000)
    beta = cfg["hard"]["beta"]

    # ---- authoritative feasibility: dense checks + exact curvature + domain ----
    kap_exact = independent_curvature(ctx["hp"].spl, ctx["hard_best"].v, n=2001)
    domH = metrics.domain_check(xyH, grid, ctx["feas"], n_points=2000, lim=lim,
                                fields=ctx["fields"])
    domE = metrics.domain_check(xyE, grid, ctx["feas"], n_points=2000, lim=lim,
                                fields=ctx["fields"])
    verH = classify_route_feasibility(
        ctx["hard_best"].meta.get("max_violation_nominal"), vH, kap_exact, domH,
        bool(np.isfinite(ctx["hard_best"].J)), lim)
    verE = classify_route_feasibility(None, vE, None, domE,
                                      bool(np.isfinite(ctx["ez"].length_m)), lim)

    # ---- frozen baseline risk field (cross-configuration comparability) ----
    if base_fields is not None:
        def _frozen(a):
            a = np.asarray(a, float)
            if a.ndim != 2 or len(a) < 2:
                return float("nan"), float("nan")
            return metrics.polyline_terms(a, base_fields)[:2]
        L_E_f, S_E_f = _frozen(ctx["ez"].xy)
        L_H_f, S_H_f = _frozen(ctx["hard_best"].xy)
    else:
        L_E_f = S_E_f = L_H_f = S_H_f = float("nan")

    row = {
        "factor": factor, "value": str(value), "label": label,
        "feasible_fraction": float(ctx["feas"].mean()),
        "feasible_cells": int(ctx["feas"].sum()),
        "n_components": feasibility_summary(ctx["feas"], grid.d, ctx["F"]["s"], lim)["n_components"],
        "L_E_km": L_E / 1000.0, "S_E": S_E, "J_E_km": (L_E + beta * S_E) / 1000.0,
        "L_H_km": L_H / 1000.0, "S_H": S_H, "J_H_km": (L_H + beta * S_H) / 1000.0,
        "hard_max_slope_deg": vH["slope_max_deg"],
        # exact B-spline curvature is the constraint quantity; the smoothed
        # polyline value is kept alongside it as a labelled diagnostic
        "hard_max_kappa": kap_exact["kappa_max"],
        "hard_min_radius_exact_m": kap_exact["kappa_min_radius_m"],
        "hard_max_kappa_polyline_diag": vH["kappa_max"],
        "hard_n_negative_depth": vH["n_negative_depth"],
        "hard_violations": vH["n_violations_slope"] + vH["n_violations_depth"] + vH["n_violations_kappa"],
        "easy_violations_slope_depth": vE["n_violations_slope"] + vE["n_violations_depth"],
        "price_of_realism_pct": 100.0 * (L_H - L_E) / L_E,
        "risk_reduction_pct": 100.0 * (S_E - S_H) / S_E if S_E else np.nan,
        "solver": ctx["hard_best"].solver, "status": ctx["hard_best"].status,
        "runtime_hard_s": ctx["hard_best"].runtime_s,
        "runtime_multistart_s": float(sum(r.runtime_s for r in ctx["hard_runs"]
                                          if np.isfinite(r.J))),
        "runtime_total_s": time.perf_counter() - t0,
        "max_violation": float(max(ctx["hard_best"].meta.get("max_violation_nominal",
                                                             ctx["hard_best"].max_violation).values()))
        if np.isfinite(ctx["hard_best"].J) else float("nan"),
        "easy_route_exists": bool(ctx["ez"].connected),
        # A solver returning a candidate is NOT proof of feasibility: the flag is
        # now decided by the shared classifier (collocation + dense depth + dense
        # slope + exact curvature + domain + physical depth), never by the
        # optimiser's own collocation metadata alone.
        "hard_feasible": bool(verH["feasible"]),
        "easy_feasible": bool(verE["feasible"]),
        "hard_feasibility_reasons": "; ".join(verH["reasons"]),
        "feasible_route_exists": bool(ctx["ez"].connected and verE["feasible"] and verH["feasible"]),
        "n_slope_violations_dense": int(vH["n_violations_slope"]),
        "n_depth_violations_dense": int(vH["n_violations_depth"]),
        "L_E_km_frozen": L_E_f / 1000.0, "S_E_frozen": S_E_f, "J_E_km_frozen": (L_E_f + beta * S_E_f) / 1000.0,
        "L_H_km_frozen": L_H_f / 1000.0, "S_H_frozen": S_H_f, "J_H_km_frozen": (L_H_f + beta * S_H_f) / 1000.0,
        "price_of_realism_pct_frozen": 100.0 * (L_H_f - L_E_f) / L_E_f if L_E_f else float("nan"),
        "risk_reduction_pct_frozen": (100.0 * (S_E_f - S_H_f) / S_E_f
                                      if (S_E_f and np.isfinite(S_E_f) and S_E_f != 0) else float("nan")),
        "mean_R": float(np.nanmean(ctx["F"]["R"])),
        "max_R": float(np.nanmax(ctx["F"]["R"])),
        "mean_slope_deg": float(np.degrees(np.arctan(np.nanmean(ctx["F"]["s"])))),
        # map-level change of the risk field itself
        "_xy_hard": ctx["hard_best"].xy,
        "_xy_easy": ctx["ez"].xy,
        "_R": ctx["F"]["R"],
    }
    # A price of realism / risk reduction is only meaningful when BOTH problems
    # produced a feasible route under this setting.  Otherwise the numbers are
    # left in the table (they are what the solver returned) but the comparison
    # columns are blanked, so no infeasible run can be read as a valid optimum.
    row["valid_solution"] = bool(row["feasible_route_exists"])
    if not row["valid_solution"]:
        row["price_of_realism_pct"] = float("nan")
        row["risk_reduction_pct"] = float("nan")
        row["price_of_realism_pct_frozen"] = float("nan")
        row["risk_reduction_pct_frozen"] = float("nan")
        why = []
        if not row["easy_route_exists"]:
            why.append("no feasible A-B route exists on the grid")
        if row["easy_route_exists"] and not row["easy_feasible"]:
            why.append("grid route failed independent feasibility validation")
        if not row["hard_feasible"]:
            why.append("hard route infeasible: " + (row["hard_feasibility_reasons"]
                                                    or "unknown reason"))
        row["invalid_reason"] = "; ".join(why)
    else:
        row["invalid_reason"] = ""
    if base_xy_hard is not None and getattr(ctx["hard_best"], "xy", np.empty(0)).shape[0] > 2:
        row["hausdorff_vs_baseline_m"] = metrics.hausdorff_distance(base_xy_hard, ctx["hard_best"].xy)["hausdorff_m"]
        row["jaccard_vs_baseline"] = metrics.cell_overlap(base_xy_hard, ctx["hard_best"].xy, grid)["jaccard"]
    if base_xy_easy is not None and getattr(ctx["ez"], "xy", np.empty(0)).shape[0] > 2:
        row["hausdorff_easy_vs_baseline_m"] = metrics.hausdorff_distance(base_xy_easy, ctx["ez"].xy)["hausdorff_m"]
    return row, ctx


def _apply(cfg, factor, value):
    c = copy.deepcopy(cfg)
    if factor == "slope_method":
        c["features"]["slope_method"] = value
    elif factor == "curvature_method":
        c["features"]["curvature_method"] = value
    elif factor == "roughness_window":
        c["features"]["roughness_window"] = int(value)
    elif factor == "roughness_field":
        c["features"]["roughness_field"] = value
    elif factor == "smooth_sigma_cells":
        c["features"]["smooth_sigma_cells"] = float(value)
    elif factor == "s_max_deg":
        c["constraints"]["s_max_deg"] = float(value)
    elif factor == "depth_range":
        lo, hi = value
        c["constraints"]["d_min_m"], c["constraints"]["d_max_m"] = float(lo), float(hi)
    elif factor == "kappa_max":
        c["constraints"]["kappa_max_1_per_m"] = float(value)
    elif factor == "edge_rule":
        c["easy"]["edge_rule"] = value
    elif factor == "beta":
        c["hard"]["beta"] = float(value)
    elif factor == "weights":
        c["risk"]["weights"] = value
    elif factor == "n_control_points":
        c["hard"]["n_control_points"] = int(value)
    elif factor == "normalization":
        c["risk"]["normalization"] = value
    else:
        raise ValueError(factor)
    return c


# --------------------------------------------------------------------------
# S-core = exactly the experiments the specification prescribes for the
# presentation (Sec. 9): slope derivation (central / Horn / Sobel), curvature
# (Laplacian / Hessian), roughness window w = 3/5/7/9 and std-of-depth,
# smoothing sigma = 0/1/2/3, s_max = P75/P90/P95, two depth ranges, three
# kappa_max values, and the edge rule.
S_CORE = [
    ("slope_method", ["central", "horn", "sobel"]),
    ("curvature_method", ["laplacian", "hessian"]),
    ("roughness_window", [3, 5, 7, 9]),
    ("roughness_field", ["slope", "depth"]),
    ("smooth_sigma_cells", [0.0, 1.0, 2.0, 3.0]),
    ("s_max_deg", [None, None, None]),     # filled from P75/P90/P95 at runtime
    ("depth_range", [(50, 600), (80, 450)]),
    ("kappa_max", None),                   # filled from the config multiples
    ("edge_rule", ["midpoint_corner", "node", "midpoint", "corner"]),
]

# Full matrix = S-core + the extra factor levels that are reported but not shown
# in the presentation (a third depth window, a fourth kappa, the normalisation
# variant).  Everything else that is reported lives in the secondary set
# (weights, K, beta).
FULL = S_CORE + [
    ("depth_range", [(50, 800)]),
    ("kappa_max_extra", [None]),
    ("normalization", ["minmax", "p99clip"]),
]


def run_factor_set(cfg, grid, spec, base_row=None, base_xy_hard=None,
                   base_xy_easy=None, tag="score", n_random=8, base_fields=None):
    rows = []
    s_pct = np.degrees(np.arctan([np.nanpercentile(_last_F["s"], p) for p in (75, 90, 95)])) \
        if _last_F else np.array([3.33, 5.48, 7.17])
    for factor, values in spec:
        if factor == "s_max_deg" and values and values[0] is None:
            values = [round(float(v), 2) for v in s_pct]
        if factor == "kappa_max" and values is None:
            base = cfg["constraints"]["kappa_max_1_per_m"]
            values = [base, base * 2.25, base * 4.0]      # min radius 3, 2, 1.5 cells
        if factor == "kappa_max_extra":
            factor = "kappa_max"
            base = cfg["constraints"]["kappa_max_1_per_m"]
            values = [base * 9.0]                          # min radius 1 cell (extra)
        for v in values:
            c = _apply(cfg, factor, v)
            if factor == "depth_range":
                lab = f"{factor}={v[0]:.0f}-{v[1]:.0f} m"
            elif factor == "kappa_max":
                lab = f"{factor}=1/{1/v:.0f} m"
            elif factor == "s_max_deg":
                pct = {0: 75, 1: 90, 2: 95}.get(values.index(v), 0)
                lab = f"{factor}={v:.2f} deg (P{pct})"
            else:
                lab = f"{factor}={v}"
            print(f"  running {lab} ...", flush=True)
            row, _ = _run_config(c, grid, base_xy_easy=base_xy_easy,
                                 base_xy_hard=base_xy_hard, label=lab,
                                 factor=factor, value=v, n_random=n_random,
                                 strategy="penalty", base_fields=base_fields)
            row["is_baseline"] = False
            rows.append(row)
    return rows


_last_F = None


def main():
    global _last_F
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="score", choices=["score", "full", "secondary", "beta"])
    ap.add_argument("--starts", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    grid = load_grid(cfg)

    # baseline first (full start budget)
    print("[sensitivity] baseline ...", flush=True)
    base_row, base_ctx = _run_config(cfg, grid, label="baseline", factor="baseline",
                                     value="baseline", n_random=8, strategy="auto")
    _last_F = base_ctx["F"]
    base_row["is_baseline"] = True
    # the baseline is its own frozen reference
    for k in ("L_E_km", "S_E", "J_E_km", "L_H_km", "S_H", "J_H_km",
              "price_of_realism_pct", "risk_reduction_pct"):
        base_row[k + "_frozen"] = base_row[k]
    base_xy_hard = base_ctx["hard_best"].xy
    base_xy_easy = base_ctx["ez"].xy
    base_fields = base_ctx["fields"]     # FROZEN baseline risk field
    rows = [base_row]

    if a.set == "beta":
        betas = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        for b in betas:
            c = _apply(cfg, "beta", b)
            print(f"  running beta={b} ...", flush=True)
            row, _ = _run_config(c, grid, base_xy_hard=base_xy_hard,
                                 base_xy_easy=base_xy_easy, label=f"beta={b}",
                                 factor="beta", value=b, n_random=a.starts,
                                 strategy="penalty", base_fields=base_fields)
            row["is_baseline"] = False
            rows.append(row)
    elif a.set == "secondary":
        for b_set in ({"slope": 0.7, "roughness": 0.2, "curvature": 0.1},
                      {"slope": 1 / 3, "roughness": 1 / 3, "curvature": 1 / 3},
                      {"slope": 0.4, "roughness": 0.2, "curvature": 0.4}):
            c = _apply(cfg, "weights", b_set)
            row, _ = _run_config(c, grid, base_xy_hard=base_xy_hard,
                                 base_xy_easy=base_xy_easy,
                                 label=f"weights={b_set}", factor="weights", value=b_set,
                                 n_random=a.starts, strategy="penalty",
                                 base_fields=base_fields)
            row["is_baseline"] = False
            rows.append(row)
        for K in (5, 7, 10):
            c = _apply(cfg, "n_control_points", K)
            row, _ = _run_config(c, grid, base_xy_hard=base_xy_hard,
                                 base_xy_easy=base_xy_easy, label=f"K={K}",
                                 factor="n_control_points", value=K,
                                 n_random=a.starts, strategy="penalty",
                                 base_fields=base_fields)
            row["is_baseline"] = False
            rows.append(row)
    else:
        spec = S_CORE if a.set == "score" else FULL
        rows += run_factor_set(cfg, grid, spec, base_row=base_row,
                               base_xy_hard=base_xy_hard, base_xy_easy=base_xy_easy,
                               tag=a.set, n_random=a.starts, base_fields=base_fields)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sensitivity_{a.set}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
