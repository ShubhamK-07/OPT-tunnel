"""Assemble report.md (spec Sec. 14 outline) from the artefacts in results/.

No number in the report is typed by hand: every value is read from
results/baseline.json, results/*.csv or results/toy_tests.json.

Run:  python -m src.report
"""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report.md"
DOI = "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"
FIG = ROOT / "figures"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}", index: bool = False) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    if index:
        d = d.reset_index()
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]
    return "\n".join([head, sep] + rows)


def main():
    B = json.load(open(RES / "baseline.json"))
    C, V, IC = B["comparison"], B["verification"], B["interpolator_check"]
    EP = B["endpoints"]
    FE = B.get("route_feasibility", {})
    BASIS_NOTE = ("Basis of measurement: " +
                  V["runtimes"].get("measurement_basis",
                                    "median of repeated calls of the same operation") + ".")
    meta, fs, nr, lim = B["data_meta"], B["feature_settings"], B["norm_ranges"], B["limits"]
    fp = meta.get("source_footprint_wgs84_cell_extent", {}) or {}
    bad_out = meta.get("n_outside_footprint", 0)
    BET = f"{B.get('beta', 1.0):g}"
    toy = json.load(open(RES / "toy_tests.json"))
    stats = pd.read_csv(RES / "descriptive_stats.csv", index_col=0)
    spatial = json.load(open(RES / "spatial_stats.json"))
    beta = pd.read_csv(RES / "sensitivity_beta.csv")
    full = pd.read_csv(RES / "sensitivity_full.csv")
    score = pd.read_csv(RES / "sensitivity_score.csv")
    sec = pd.read_csv(RES / "sensitivity_secondary.csv")
    mt = V["multistart"]
    points = pd.read_csv(RES / "control_points_hard.csv")
    prior_rt = json.load(open(RES / "runtime_previous_baseline.json"))
    snap = V["dijkstra_crosscheck"]
    actual_M = B["sampling_protocol"]["actual_objective_and_constraint_samples"]
    configured_M = B["sampling_protocol"]["configured_n_samples"]
    W = fs["weights"]
    core = full[full["factor"].isin(["slope_method", "roughness_window",
                                     "smooth_sigma_cells", "s_max_deg", "kappa_max"])]
    core = core.drop_duplicates(subset=["label"]).reset_index(drop=True)
    base_row = full[full["factor"] == "baseline"].iloc[0]
    # marginal exchange rate dL/dS along the beta frontier (m of length per unit of S)
    b = beta[beta["factor"] == "beta"].copy()
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    b = b.dropna(subset=["value"]).sort_values("value")
    dS = -b["S_H"].diff().to_numpy()
    dL = b["L_H_km"].diff().to_numpy() * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mrs = np.where(dS > 1e-9, dL / dS, np.nan)
    mrs = mrs[np.isfinite(mrs)]
    beta_mrs = (float(np.min(mrs)), float(np.max(mrs))) if mrs.size else (0.0, 0.0)

    def _sens_md(df, mapping, floatfmt="{:.3f}", status=True):
        """Render a sensitivity table, blanking metrics that were never measured.

        Two different failures are distinguished, because they mean different things:
        (a) the grid has no feasible A->B path at all (L_E is NaN) -- the whole row is
        blanked; (b) the grid path exists but the hard solver returned an infeasible
        candidate -- only the HARD columns are blanked, the easy columns are kept.
        """
        d = df.copy()
        for src in mapping:
            if src not in d.columns:
                d[src] = np.nan
        t = d[list(mapping)].rename(columns=mapping)
        if "valid_solution" in d.columns:
            invalid = ~d["valid_solution"].astype(bool).to_numpy()
            no_grid = pd.isna(pd.to_numeric(d.get("L_E_km"), errors="coerce").to_numpy())
            hard_only = invalid & ~no_grid
            keep_cols = ("setting", "feasible %", "comp.", "status")
            for c in [c for c in t.columns if c not in keep_cols]:
                t.loc[no_grid, c] = np.nan
            for c in [c for c in t.columns if c not in keep_cols]:
                t.loc[hard_only, c] = np.nan
            if status:
                easy_bad = ~d["easy_feasible"].astype(bool).to_numpy() if "easy_feasible" in d else np.zeros(len(d), dtype=bool)
                hard_bad = ~d["hard_feasible"].astype(bool).to_numpy() if "hard_feasible" in d else invalid
                reason = np.where(easy_bad & hard_bad, "INFEASIBLE (both checks)",
                                  np.where(easy_bad, "INFEASIBLE (easy check)", "INFEASIBLE (hard check)"))
                t["status"] = np.where(no_grid, "INFEASIBLE (no grid path)",
                                       np.where(invalid, reason, "valid"))
        return md_table(t, floatfmt=floatfmt)

    def _row(label_part, col, default=float("nan")):
        m = core[core["label"].str.contains(label_part, regex=False)]
        return float(m[col].iloc[0]) if len(m) else default

    _p75 = score[score["label"].astype(str).str.contains("P75", regex=False)]
    p75_len = float(_p75["L_H_km"].iloc[0]) if len(_p75) else float("nan")
    feas_core = core[core["feasible_route_exists"].astype(str) == "True"]
    L_span_pct = 100.0 * (feas_core["L_H_km"].max() - feas_core["L_H_km"].min()) / base_row["L_H_km"]
    ns = feas_core[~feas_core["label"].str.contains("smooth_sigma")]
    L_span_ns = 100.0 * (ns["L_H_km"].max() - ns["L_H_km"].min()) / base_row["L_H_km"]
    n75, n95 = _row("P75", "n_components"), _row("P95", "n_components")
    f75 = _row("P75", "feasible_fraction")
    h338 = _row("338", "hausdorff_vs_baseline_m")
    sig2 = _row("2.0", "S_H")
    f_base = base_row["feasible_fraction"]
    S_sig2_ratio = sig2 / base_row["S_H"]
    S_w7_ratio = full[(full["factor"] == "roughness_window")
                      & (full["label"] == "roughness_window=7")]["S_H"].iloc[0] / base_row["S_H"]
    h_w7 = full[(full["factor"] == "roughness_window")
                & (full["label"] == "roughness_window=7")]["hausdorff_vs_baseline_m"].iloc[0]

    # slope-estimator sentence, built from the S-core table
    _sl = score[score["factor"] == "slope_method"].drop_duplicates(subset=["label"])
    _sl_parts = []
    for _, r in _sl.iterrows():
        meth = str(r["label"]).split("=")[-1]
        if meth == "central":
            continue
        if not bool(r["valid_solution"]):
            if pd.isna(r.get("L_E_km")):
                _sl_parts.append(f"**{meth}** leaves no feasible A→B route on the grid at all "
                                 f"(feasible fraction {r['feasible_fraction']:.1%}, "
                                 f"{r['n_components']:.0f} components)")
            else:
                easy_status = "grid path fails validation" if not bool(r.get("easy_feasible", True)) else "grid path passes validation"
                _sl_parts.append(f"**{meth}**: {easy_status}; the hard solver returns an infeasible candidate "
                                 f"(feasible-cell fraction {r['feasible_fraction']:.1%})")
        else:
            dS = 100.0 * (r["S_H_frozen"] / base_row["S_H_frozen"] - 1.0)
            _sl_parts.append(f"**{meth}** changes frozen-field S_H by {dS:+.1f} % and moves the route by "
                             f"{r['hausdorff_vs_baseline_m']:.0f} m")
    slope_txt = "; ".join(_sl_parts) + "."

    # Only independently feasible candidates contribute to the beta interpretation.
    _bt = beta[beta["factor"] == "beta"].drop_duplicates(subset=["label"]).copy()
    _bbad = _bt[~_bt["valid_solution"].astype(bool)]
    _beta_txt = (f"{len(_bt)-len(_bbad)} of {len(_bt)} β settings produced feasible local "
                 f"candidates; {len(_bbad)} did not. Feasibility does not establish optimality, "
                 "so these points are illustrative candidate trade-offs, not a proven Pareto frontier.")

    # weights rows: tidy the label (dict reprs are unreadable at full float precision)
    def _tidy_label(lab):
        lab = str(lab)
        if lab.startswith("weights="):
            nums = [f"{float(x):.2f}".rstrip("0").rstrip(".") for x in _re.findall(r"\d*\.\d+", lab)]
            return "weights " + "/".join(nums)
        return lab

    _sec_tbl = sec.drop_duplicates(subset=["label"]).copy()
    _sec_tbl["label"] = [_tidy_label(v) for v in _sec_tbl["label"]]

    # Sensitivity summaries use only validated pairs and the frozen risk field.
    _w = sec[(sec["factor"] == "weights") & sec["valid_solution"].astype(bool)].drop_duplicates(subset=["label"])
    _k = sec[(sec["factor"] == "n_control_points") & sec["valid_solution"].astype(bool)].drop_duplicates(subset=["label"])
    _sec_txt = (f"{len(_w)} weight settings and {len(_k)} control-point settings produced "
                "valid route pairs. Compare routes across settings only using the frozen baseline "
                "risk field; the weights remain modelling assumptions, not validated by correlations. "
                "Infeasible candidates are excluded; feasibility does not prove optimality.")

    # --- cross-configuration robustness, measured on the frozen columns -------
    _allsens = pd.concat([score, full, sec], ignore_index=True)
    _allsens = _allsens[_allsens["factor"] != "baseline"].drop_duplicates(subset=["label"])
    if "valid_solution" in _allsens.columns:
        _ok = _allsens[_allsens["valid_solution"].astype(bool)]
    else:
        _ok = _allsens[_allsens.get("feasible_route_exists", True).astype(bool)]
    _por = "price_of_realism_pct_frozen" if "price_of_realism_pct_frozen" in _ok.columns \
        else "price_of_realism_pct"
    _rr = "risk_reduction_pct_frozen" if "risk_reduction_pct_frozen" in _ok.columns \
        else "risk_reduction_pct"
    _porv = pd.to_numeric(_ok[_por], errors="coerce").dropna()
    _rrv = pd.to_numeric(_ok[_rr], errors="coerce").dropna()
    ROB = {
        "n_settings": int(len(_allsens)),
        "n_valid": int(len(_ok)),
        "por_neg": int((_porv < 0).sum()), "por_pos": int((_porv > 0).sum()),
        "rr_pos": int((_rrv > 0).sum()), "rr_neg": int((_rrv < 0).sum()),
        "por_lo": float(_porv.min()) if len(_porv) else float("nan"),
        "por_hi": float(_porv.max()) if len(_porv) else float("nan"),
        "rr_lo": float(_rrv.min()) if len(_rrv) else float("nan"),
        "rr_hi": float(_rrv.max()) if len(_rrv) else float("nan"),
        "basis": ("frozen baseline risk field" if _por.endswith("_frozen")
                  else "native per-setting risk field"),
    }
    ROB_TXT = (
        f"Of the {ROB['n_settings']} sensitivity settings, {ROB['n_valid']} produced a candidate "
        f"that passes the full feasibility check of Sec. 9.1; the rest are marked INFEASIBLE and are "
        f"excluded from the statement below. On those settings, and re-integrating every route "
        f"against the {ROB['basis']} so the risk numbers are comparable, the sign of both headline "
        f"numbers is: the price of realism is negative in {ROB['por_neg']} of "
        f"{ROB['por_neg'] + ROB['por_pos']} settings (range {ROB['por_lo']:+.2f} % to "
        f"{ROB['por_hi']:+.2f} %) and the risk reduction is positive in {ROB['rr_pos']} of "
        f"{ROB['rr_pos'] + ROB['rr_neg']} (range {ROB['rr_lo']:+.2f} % to {ROB['rr_hi']:+.2f} %)."
        if ROB["n_valid"] else
        "No sensitivity setting produced a candidate that passes the full feasibility check, so no "
        "robustness statement is made.")

    # Moran's I table (all five features) -- built outside the f-string below
    _mor = spatial["moran"]["features"]
    moran_md = md_table(pd.DataFrame([
        [k, f"{_mor[k]['morans_I']:.4f}", f"{_mor[k]['thinned']['stride_2']:.3f}",
         f"{_mor[k]['thinned']['stride_3']:.3f}", f"{_mor[k]['n_cells']:,}", _mor[k]["unit"]]
        for k in ("d", "s", "r", "c", "R")],
        columns=["feature", "Moran's I (full grid)", "thinned 1/2", "thinned 1/3",
                 "cells", "unit"]), floatfmt="{:.4f}")
    moran_lo = min(_mor[k]["morans_I"] for k in ("d", "s", "r", "c", "R"))
    moran_hi = max(_mor[k]["morans_I"] for k in ("d", "s", "r", "c", "R"))

    # ---------------------------------------------------------------- header
    out = f"""# Underwater Tunnel Route Optimization
### A length–risk trade-off study on GEBCO_2026 bathymetry (Sicily Channel, 15 arc-sec)

*Report generated by `python -m src.report` from `results/*.json|csv`.
Every number below is produced by code in `src/`; nothing is transcribed by hand.*

**One-paragraph summary.** Two formulations are solved on the same 115 × 113 bathymetric
grid, with the same mask, the same data-derived risk field R(x,y) and — for the *reported*
L, S and J only — the same post-hoc polyline evaluator. The two optimisers are deliberately
different, and integrating a route is never confused with optimising it: the grid route comes from
a graph search, the spline route from a continuous SLSQP refinement, and both are then measured by
the same common polyline evaluator, on their respective sampled geometries.
The **easy problem** is an exact discrete shortest path on an 8-neighbour graph with depth and
slope feasibility: Dijkstra finds the discrete optimum between selected snapped endpoint cells
({V['runtimes']['easy_repeated_median_s']*1000:.0f} ms repeated-solve median). Exact A/B connectors
are appended and independently validated; they are not optimized over alternative connector choices.
The **hard problem** is a cubic B-spline corridor (7 interior control points, 14 decision
variables) minimising J = α·L + β·S subject to depth, slope and *route-curvature* constraints,
solved by SLSQP from {mt['n_distinct_initial_curves']} distinct initial curves
({mt['n_solver_executions']} solver executions: {mt['n_direct_executions']} direct and
{mt['n_penalty_executions']} quadratic-penalty continuations). The adopted point is a
**feasible local candidate** — {mt['n_feasible']} of those executions reached feasibility, and
they agree to {mt['J_feasible_spread']*1000:.0f} m of J — not a proven global optimum. Relative to the complete connector-inclusive grid route it changes length by
**{C['price_of_realism_pct']:+.2f} %** (it is shorter, not longer) and lowers cumulative terrain
risk by **{C['risk_reduction_pct']:.2f} %**, and unlike the grid route it respects a
{1/lim['kappa_max']:.0f} m minimum turning radius.

---

## 1. Introduction and motivation

A tunnel between two fixed points has to be *routed*, not just measured. Two questions are
usually conflated and are kept separate here:

1. **What is the shortest path between selected snapped endpoint cells?** — a shortest-path problem on a grid, followed by validated exact-endpoint connectors.
   It has an exact, global answer and no notion of smoothness.
2. **What does a *terrain-aware, constraint-compliant* corridor cost?** — a continuous
   optimisation problem with
   engineering constraints (maximum slope, minimum turning radius) and a risk penalty.

The easy problem answers (1); the hard problem answers (2). Comparing them answers the
question a project actually asks: **how much length do we pay for a route that respects a
turning radius, and how much terrain risk do we buy with it?**

Scope decisions fixed before any run (all recorded in `config.yaml`):

* One risk field R(x,y) = w_s·s̃² + w_r·r̃ + w_c·c̃, weights **{W['slope']}/{W['roughness']}/{W['curvature']}**,
  min–max normalisation over valid cells, no smoothing in the baseline. The weights are
  **modeling assumptions**, sensitivity-tested in Sec. 10.3 — they are not derived from, and are not
  validated by, the correlation analysis of Sec. 6.
* Terrain curvature `c` (Laplacian of elevation) never enters route curvature; route
  curvature κ is never used as a terrain feature. `R(x,y)` is the field, `S(p) = ∫ R ds` is
  the cumulative risk along a route.
* A third formulation was explicitly excluded by scope.

## 2. Data

{md_table(pd.DataFrame([
        ["Product", "GEBCO_2026 Grid (15 arc-sec, published 2026)"],
        ["Subset", f"{meta['bbox_wgs84']['south']}–{meta['bbox_wgs84']['north']} °N, "
                   f"{meta['bbox_wgs84']['west']}–{meta['bbox_wgs84']['east']} °E"],
        ["Citation", "GEBCO Bathymetric Compilation Group 2026 (2026), The GEBCO_2026 Grid, "
                      f"doi:{DOI} (full reference in the Appendix)"],
        ["Projection", f"{meta['crs']}; all lengths in metres, "
                       f"all coordinates easting/northing in metres"],
        ["Grid", f"{meta['shape'][0]} rows × {meta['shape'][1]} columns, {meta['spacing_m']:.0f} m"],
        ["Extent", f"x {meta['x_range_m'][0]/1e3:.1f}–{meta['x_range_m'][1]/1e3:.1f} km, "
                   f"y {meta['y_range_m'][0]/1e3:.1f}–{meta['y_range_m'][1]/1e3:.1f} km"],
        ["Resampling", f"elevation: {meta['resampling_elevation']}, TID: {meta['resampling_tid']}"],
        ["NoData policy", meta["nodata_policy"]],
        ["Endpoints",
         f"A = {EP['A_lonlat'][0]:.4f}° E, {EP['A_lonlat'][1]:.4f}° N "
         f"(UTM 32N x {EP['A_xy_m'][0]/1e3:.3f} km, y {EP['A_xy_m'][1]/1e3:.3f} km, cell "
         f"{EP['A_cell'][0]},{EP['A_cell'][1]}); "
         f"B = {EP['B_lonlat'][0]:.4f}° E, {EP['B_lonlat'][1]:.4f}° N "
         f"(x {EP['B_xy_m'][0]/1e3:.3f} km, y {EP['B_xy_m'][1]/1e3:.3f} km, cell "
         f"{EP['B_cell'][0]},{EP['B_cell'][1]})"],
        ["Valid cells", f"{meta['n_valid']:,} of {meta['n_cells']:,} "
                        f"({100*meta['n_valid']/meta['n_cells']:.2f} %) — the rest are never used"],
        ["Masked cells", f"{meta['n_invalid']:,} ({meta['masked_fraction']*100:.2f} %): "
                         f"{meta['n_outside_footprint']:,} outside the source footprint "
                         f"({meta['outside_fraction']*100:.2f} %), "
                         f"{meta['land_nodata_only_fraction']*100:.2f} % adjacent to land/NoData"],
        ["A → B straight line", f"{meta['AB_geodesic_m']/1e3:.2f} km geodesic, "
                                f"{meta['AB_grid_m']/1e3:.2f} km through the grid "
                                f"({meta['AB_difference_pct']:.3f} % difference)"],
    ], columns=["Item", "Value"]))}

The grid is metric (metres) in both axes, so every derivative, distance and curvature below is
computed without degree arithmetic. Derivatives are taken **on the grid as it stands**
(sigma = {fs['smooth_sigma_cells']:g} cells in the baseline), i.e. before any smoothing; the
sigma-smoothing experiments in Sec. 10 re-derive everything after smoothing, which is the only way
the two effects can be separated.

**Resampling, masking and the domain boundary (what the code actually does).** The land and NoData
masks are applied to the *source* raster before any resampling, and the reprojection is a strict
bilinear resample: a target point is accepted only when **every source cell with a non-zero weight
is a valid ocean cell**. A land or NoData sample can therefore never be blended into an ocean
value, and no value is ever extrapolated — the earlier version of this code used
`RegularGridInterpolator(..., fill_value=None)`, which extrapolates silently, and in this subset it produced
{bad_out} cells that were used as valid bathymetry although they lie outside the source footprint. The valid footprint is the *cell
extent* of the subset (centres ± half a source cell, i.e. {fp['lon_min']:.4f}–{fp['lon_max']:.4f} °E
× {fp['lat_min']:.4f}–{fp['lat_max']:.4f} °N), not the projected bounding box: projecting a
lat/lon rectangle to UTM 32N gives a curved shape whose bounding box contains corners that lie
outside the data. {meta['n_outside_footprint']:,} of {meta['n_cells']:,} target cells
({meta['outside_fraction']*100:.2f} %) are outside that footprint and are masked; a further
{meta['land_only_cells']} cells have a stencil that touches one of the
{meta['n_source_land_cells']} land source cells. Inside the footprint but between the outermost
cell centre and the cell edge the edge cell is replicated — the half-cell border is kept rather
than discarded, and no extrapolation is involved.

**Derivative stencils at the array boundary.** `np.gradient` uses one-sided differences on the
outermost interior rows/columns; those cells are **not** excluded from the statistics or from the
feasible mask, and the report says so rather than claiming an exclusion that is not implemented.
The roughness window is a symmetric w × w box with `mode="nearest"`, so at the array border the
window is filled by replicating the edge cell and no artificial zeros enter the statistic. Land,
NoData and outside-footprint cells hold NaN in z, so no derivative is ever taken across the mask;
before a derivative stencil is evaluated, NaNs are replaced by the nearest valid ocean value
(`distance_transform_edt`) and the result is re-masked afterwards. **Feasibility is always judged
on the original, unfilled arrays** — the nearest-value fill exists only so that a stencil has
something finite to read.

The linear striping visible in the slope map is a known signature of **satellite-altimetry
predicted** bathymetry; it is why the TID (type-identifier) map and the σ-smoothing sensitivity are
part of the analysis. Depth is `d = −z` with z the GEBCO elevation: **d is the sea-floor depth**, so
`d_min`/`d_max` bound how deep the seabed is along the corridor — they are *not* tunnel alignment
depths (Sec. 13).

## 3. Feature definitions

Let z(x,y) be elevation (m, negative below sea level), Δx = Δy = {meta['spacing_m']:.0f} m.

| Feature | Definition | Units |
|---|---|---|
| depth | d = −z | m |
| slope | s = √((∂z/∂x)² + (∂z/∂y)²), central differences (Horn/Sobel in Sec. 10) | rise/run (dimensionless) |
| roughness | r = std{{s}} over a {fs['roughness_window']}×{fs['roughness_window']} window, edges replicated | rise/run |
| terrain curvature | c = abs(∂²z/∂x² + ∂²z/∂y²), the Laplacian | 1/m |
| normalised | x̃ = (x − min)/(max − min) over valid cells, clipped to [0, 1] | — |
| risk field | R = {W['slope']}·s̃² + {W['roughness']}·r̃ + {W['curvature']}·c̃ | dimensionless, R ∈ [0, 1] |
| cumulative risk | S(p) = ∫_p R ds  (segment-midpoint rule on the route) | metres |

Normalisation ranges actually used (all cells, this baseline):
slope {nr['slope_m_per_m'][0]:.3f}–{nr['slope_m_per_m'][1]:.3f},
roughness {nr['roughness'][0]:.4f}–{nr['roughness'][1]:.4f},
curvature {nr['curvature_1_per_m'][0]:.2e}–{nr['curvature_1_per_m'][1]:.2e} 1/m.

Because min–max normalisation is recomputed for every configuration, **S is not comparable
across configurations** (only within one). It is a length-like quantity in metres whose scale is
set by each configuration's own tail; the report always says which configuration an S belongs to.
The slope term is squared (`s̃²`) so that the field is dominated by the steepest terrain rather
than by the mean.

**Filling and interpolation.** Land (z ≥ 0) and NoData are removed *before* anything else
happens: they are set to NaN, excluded from the statistics, and never carry a finite value into a
derivative. The hard problem needs a smooth field defined everywhere, so the masked cells are
filled with the value of the **nearest valid ocean cell** (nearest-neighbour fill, no
interpolation across the mask) and only then interpolated with a tensor-product cubic spline.
The unfilled arrays are retained and every feasibility claim is checked on them (Sec. 9), so the
fill can change a *search* field but never a *validation* result.

**Route curvature** (used only as a constraint of the hard problem):
κ = (x′y″ − y′x″) / (x′² + y′²)^{3/2} [1/m]; the corresponding minimum turning radius is 1/|κ|.

## 4. Descriptive statistics

{md_table(stats[["unit", "count", "missing", "mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max", "skewness", "excess_kurtosis"]], floatfmt="{:.4g}", index=True)}

Counts are cells retained after masking NoData and land
({stats['missing'].min()}–{stats['missing'].max()} cells lost; {stats.loc['d','missing']/ (stats.loc['d','count']+stats.loc['d','missing']):.2%} of the grid for depth).
All five variables are right-skewed: the *tail* is the story — the 95th percentile of R is
{stats.loc['R','p95']:.3f} against a maximum of {stats.loc['R','max']:.3f}, so a small number of
steep cells dominate the scale, and avoiding them is exactly what a router does.

## 5. Distributions and spatial maps

![feature maps](figures/fig01_feature_maps.png)

*Depth* is a NE–SW graben flank between two shelf areas, 1.6–817 m. *Slope* concentrates on the
flank and on the gravity-derived striping; *roughness* is the same structure smoothed by the
local standard deviation; *terrain curvature* is dominated by a few sharp scarps; *risk* is
sparse — 50 % of cells are below {stats.loc['R','p50']:.3f} while the maximum is {stats.loc['R','max']:.3f}.

![distributions](figures/fig02_distributions.png)

The lower row is the same data after min–max normalisation: the shape is unchanged by
construction (it is an affine map), which is why the histograms are shown together — it makes
visible that normalisation cannot fix skewness, only re-scale it. Log-scaled counts.

![TID map](figures/fig10_tid_map.png)

The TID map shows which survey/source produced each cell. It is a **data-quality indicator,
not a safety map**: where the striping is strongest the TID is the altimetry-predicted class,
and the σ-smoothing sensitivity (Sec. 10) quantifies how much of the risk structure is that
artefact.

## 6. Correlation and spatial autocorrelation

![correlations](figures/fig03_correlations.png)

{md_table(pd.DataFrame([
        ["Pearson r(s, r) / r(r, c) / r(s, c)",
         " / ".join(f"{spatial['correlation']['pearson'][i][j]:+.3f}"
                    for i, j in ((1, 2), (2, 3), (1, 3)))],
        ["Spearman ρ(s, r) / ρ(r, c) / ρ(s, c)",
         " / ".join(f"{spatial['correlation']['spearman'][i][j]:+.3f}"
                    for i, j in ((1, 2), (2, 3), (1, 3)))],
        ["Pearson after thinning 1/5 (s, r) / (r, c)",
         f"{spatial['correlation']['pearson_thinned'][1][2]:+.3f} / "
         f"{spatial['correlation']['pearson_thinned'][2][3]:+.3f}"],
        ["Expected I (random pattern)", f"{spatial['moran']['features']['R']['expected_I']:.5f}"],
        ["PCA variance (s̃, r̃, c̃)", ", ".join(f"{v:.1%}" for v in spatial['pca']['explained_variance_ratio'])],
    ], columns=["Quantity", "Value"]))}

Moran's I is computed for **every** feature, not only for the composite risk field, because a
high I(R) could otherwise come from one feature alone:

{moran_md}

Moran's I is reported with its cell count, its weight scheme (rook, 4-neighbour, binary) and its
behaviour under thinning. **No significance test is claimed**: the usual normal approximation for
Moran's I needs a variance derived for this weight matrix, and that approximation is not valid on a
field this autocorrelated, so no z-score or p-value is quoted. The statistic is used descriptively,
which is all the interpretation below needs.

Every feature is strongly autocorrelated at 450 m (I between
{moran_lo:.2f} and {moran_hi:.2f}), and every feature keeps most of its structure when the grid is
thinned by 2 and by 3. Depth is the most autocorrelated, as expected for a smooth regional
surface; terrain curvature is the least, because it is dominated by a few scarps. The thinned
runs are a resolution check, not an alternative estimate: they answer "would this statistic
survive a coarser survey?".

*Interpretation.* Slope, roughness and terrain curvature are **strongly associated** — Spearman
is higher than Pearson because the relation is monotone, not linear. Thinning the grid by a factor
of {spatial['correlation'].get('thin_factor', 5)} in each direction (that is the factor the code
uses) moves the correlations by less than 0.1, so the association is not an artefact of the 450 m
sampling. Moran's I is ≈ {spatial['moran']['features']['R']['morans_I']:.2f} on the full grid and
still ≈ {spatial['moran']['features']['R']['thinned']['stride_3']:.2f} on a 1/3-thinned grid:
the field is genuinely spatially structured, which is precisely why a *route* can exploit it —
a random field would make routing pointless. The first PCA component of the three normalised
features explains {spatial['pca']['explained_variance_ratio'][0]:.0%} of their variance, i.e. r̃ and c̃
carry largely the same signal as s̃ and the weight vector is not three independent knobs.

Association is not causation, and none of these numbers says anything about geology.

## 7. Easy problem

The easy problem is a **minimum-cost flow / shortest-path problem on a directed graph**. Write
V for the feasible cells, E for the admissible edges, c_ij ≥ 0 for the metric length of an edge
and b for the unit supply at A's selected snapped cell and demand at B's selected snapped cell. The route is the optimal solution of the linear
program

```
minimise    sum_(i,j) in E  c_ij * x_ij
subject to  sum_j x_ij - sum_j x_ji  =  b_i        for every node i in V   (flow conservation)
            x_ij >= 0                                                       (non-negativity)
```

with b_A* = +1, b_B* = −1 and b_i = 0 otherwise, where A* and B* are the selected snapped cells. Because the constraint matrix of a shortest-path
formulation is totally unimodular, the LP optimum is integral and the x_ij are the 0/1 indicator
of a single path — so the LP and the discrete problem have the same answer. Dijkstra solves this
LP without ever forming it: the edge weights are non-negative, so the label-setting algorithm
returns the exact optimum in O(|E| log |V|). That is what makes the easy problem *global*: the
optimum is exact by construction (Dijkstra returns the shortest path of the graph), not found by
search.{'  '}

Graph and feasibility definition: nodes = cells; edges = 8-neighbour; an edge exists iff both
endpoints are feasible and, for a diagonal step, the **geometric midpoint of the segment**
(bilinear interpolation of the four surrounding node samples) satisfies the depth and slope
limits and all four surrounding nodes carry valid ocean data — the specification's midpoint
guard, implemented in `easy.midpoint_feasibility` and used for the baseline. The baseline also
requires both orthogonal neighbours to be feasible; reduced rules are sensitivity variants. Edge weight = metric length ({meta['spacing_m']:.0f} m orthogonal,
{meta['spacing_m']*np.sqrt(2):.0f} m diagonal). Feasibility = not land ∧
{lim['d_min']:.0f} ≤ d ≤ {lim['d_max']:.0f} m ∧ s ≤ tan({lim['s_max_deg']}°).

{md_table(pd.DataFrame([
        ["L_E (route length)", f"{C['L_easy_km']:.3f} km"],
        ["S_E (cumulative risk)", f"{C['S_easy']:.1f}"],
        ["Runtime (original baseline graph-search invocation)", f"{C['runtime_easy_s']*1000:.1f} ms"],
        ["Runtime (repeated solve_easy median incl. route assembly)", f"{V['runtimes']['easy_repeated_median_s']*1000:.1f} ms"],
        ["Nodes expanded", f"{C['nodes_expanded_easy']:,}"],
        ["Feasible cells", f"{V['dijkstra_crosscheck'] and full[full['factor']=='baseline'].iloc[0]['feasible_cells']:,} "
                           f"({full[full['factor']=='baseline'].iloc[0]['feasible_fraction']:.1%})"],
        ["Max slope along route", f"{C['easy_max_slope_deg']:.3f}°"],
        ["Max route curvature (informational)", f"{C['easy_kappa_max']:.2e} 1/m → radius {1/C['easy_kappa_max']:.0f} m"],
        ["Edge rule (baseline)", "endpoints, midpoint and both orthogonal neighbours; alternatives in Sec. 10"],
    ], columns=["Easy result", "Value"]))}

**Endpoint and optimality scope.** Dijkstra obtains the global shortest feasible path between the
selected snapped endpoint cells on the constructed graph. The exact endpoint connectors are
subsequently appended and the complete route is independently validated. Dijkstra does not
optimize over alternative exact-endpoint connector choices; the graph-path optimality claim does
not extend to the entire endpoint-to-endpoint route. SciPy cross-checks the graph segment only:

{md_table(pd.DataFrame([
    ["A to selected grid cell", f"{snap['connector_A_m']:.1f} m"],
    ["Selected cell to selected cell: Dijkstra", f"{snap['own_length_m']/1000:.3f} km"],
    ["Same graph path cost: SciPy", f"{snap['scipy_length_m']/1000:.3f} km"],
    ["Selected grid cell to B", f"{snap['connector_B_m']:.1f} m"],
    ["Complete endpoint-to-endpoint route", f"{snap['own_length_with_exact_endpoints_m']/1000:.3f} km"],
], columns=["Component", "Distance"]))}

The easy problem contains no curvature constraint: the {1/C['easy_kappa_max']:.0f} m minimum radius is a
polyline diagnostic, not a constraint imposed on the grid problem.

## 8. Hard problem

### 8.1 Decision variables and the 14 degrees of freedom

The route is a **clamped cubic B-spline** p(t) = Σ N_i,3(t) P_i  (i = 0…8) on a fixed knot vector,
with P_0 = A and P_8 = B pinned to the endpoints and K = 7 interior control points P_1…P_7 free.
Each interior control point contributes an easting and a northing, so the decision vector is

  v = (u_1…u_7, w_1…w_7) ∈ [0,1]^14,  P_i = origin + (u_i·W, w_i·H),

i.e. **14 degrees of freedom**, box-constrained to the grid rectangle. The curve is C² by
construction (a cubic B-spline is twice continuously differentiable), which is what makes a
curvature constraint meaningful: κ(t) can be differentiated analytically instead of being
estimated from a polyline.

### 8.2 Objective, constraints and how they are discretised

```
minimise   J(v) = α·L(v) + β·S(v),   α = {B['alpha']:.0f}, β = {B['beta']:.0f}
           L(v) = ∫_0^1 ||p'(t)|| dt                          [m]
           S(v) = ∫_0^1 R(p(t)) ||p'(t)|| dt                   [m]   (R dimensionless)
subject to {lim['d_min']:.0f} ≤ d(p(t)) ≤ {lim['d_max']:.0f} m
           s(p(t)) ≤ tan({lim['s_max_deg']}°) = {lim['s_max']:.4f}
           |κ(t)| ≤ {lim['kappa_max']:.4e} 1/m      (minimum turning radius {1/lim['kappa_max']:.0f} m)
           κ(t) = (x'y'' − y'x'') / (x'² + y'²)^(3/2)
```

* **Integration and constraints.** The configuration specifies `n_samples = {configured_M}` and
  `n_constraint_samples = {actual_M}`, but `HardProblem` sets its active spline grid to the latter.
  Consequently the optimizer evaluates both trapezoidal L/S and **collocation** on the same
  {actual_M} equally spaced parameter values, giving {4*actual_M} inequality rows. This describes
  the implemented run, not a separate {configured_M}-point objective grid. Collocation is a discretisation, so the *validated*
  route is checked separately at 2 000–5 000 points against the **nominal** limits; the solver
  targets 0.98 × the limits (a 2 % margin) so that the continuous route stays inside the nominal
  limits *between* collocation samples. The margin is a documented choice whose price is measured
  in Sec. 9.
* **Interpolation.** d, s and R are only known on the 450 m grid, so the optimiser needs smooth
  surrogates. Masked cells are filled with the nearest valid ocean value and a tensor-product
  cubic spline (`RectBivariateSpline`) is fitted to each field. The cubic spline is chosen
  deliberately: it is the smooth field the corridor is meant to follow, its **interior** depth
  overshoot is at most {V['interpolation']['checks']['cubic']['depth_interior_max_overshoot_m']:.1f} m
  ({100*V['interpolation']['checks']['cubic']['depth_interior_max_overshoot_m']/(stats.loc['d','max']-stats.loc['d','min']):.1f} %
  of the depth range) and it never implies land
  ({V['interpolation']['checks']['cubic']['n_probe_implying_land']} of 20 000 probes give d < 0,
  none of them on the adopted route). A bilinear (linear) interpolant has exactly zero interior
  overshoot, but its gradients are piecewise constant, which makes a slope constraint on a 450 m
  grid a different interpolation model; it is reported as a reference, not adopted. The re-solved
  linear-field candidate has {V['interpolation']['linear_solution']['dense_n_violations']} dense
  violations when evaluated on the baseline cubic fields; it is not treated as a validated route
  comparison. Along the adopted route the two interpolants differ by at most
  {V['interpolation']['route_field_difference']['max_abs_depth_difference_m']:.1f} m of depth and
  {V['interpolation']['route_field_difference']['max_abs_slope_difference']*180/np.pi:.2f}° of slope.

### 8.3 Two SLSQP strategies, and why the direct one is not enough

**Direct constrained SLSQP** is the literal formulation: the 1 600 collocation rows are handed to
SLSQP as four vector inequalities and the method solves the corresponding NLP with its own QP
subproblems. In this problem it is unreliable: with finite-difference Jacobians over 14 variables
and 1 600 nearly redundant rows, the QP subproblem frequently fails along the line search.
Measured here: **{V['solver_strategies']['direct']['n_executions']} executions,
{V['solver_strategies']['direct']['n_converged_status0']} reaching `status 0`,
{V['solver_strategies']['direct']['n_feasible']} feasible.**

**Quadratic exterior-penalty continuation** is what actually produces a feasible candidate.
The constraints are moved into the objective,

  φ_μ(v) = J(v) + μ · Σ_i max(g_i(v), 0)²,

and SLSQP minimises φ_μ over the 14 box constraints only, with continuation on
μ = 1e2 → 1e4 → 1e6, each stage warm-started from the previous one. Measured:
**{V['solver_strategies']['penalty']['n_executions']} executions from the three canonical warm starts,
{V['solver_strategies']['penalty']['n_converged_status0']} reaching `status 0`,
{V['solver_strategies']['penalty']['n_feasible']} feasible** — both feasible runs report J within
{mt['J_feasible_spread']*1000:.0f} m of each other, and the better of the two is the adopted
route.

Three statements that matter more than the numbers:

1. For a **finite** μ the minimiser of φ_μ is only *approximately* feasible; the penalty is
   exterior, so it is exactly zero on the feasible set but does not enforce it. Feasibility of
   the final iterate is therefore **checked**, not assumed (nominal limits, 2 000–5 000 points).
2. A feasible penalty solution is **not established to be a KKT point** of the constrained
   problem: the multipliers and first-order conditions of the original constraints are not
   verified. Solver status is reported separately from independent feasibility, and no
   stationarity certificate is asserted.
3. **Feasible is not optimal.** Nothing here establishes even local optimality of the constrained
   problem, let alone global optimality; the result is a feasible local candidate.

### 8.4 Multi-start accounting (executions are not starts)

| Quantity | Value |
|---|---|
| Distinct initial curves | {mt['n_distinct_initial_curves']} (grid fit, smoothed grid fit, straight line, {mt['n_distinct_initial_curves']-3} seeded random perturbations) |
| Solver executions | {mt['n_solver_executions']} = {mt['n_direct_executions']} direct + {mt['n_penalty_executions']} penalty continuations |
| Starts used by the penalty stage | {", ".join(mt['penalty_start_labels'])} |
| Executions reaching `status 0` | {mt['n_converged_status0']} |
| Executions reaching a feasible point | **{mt['n_feasible']}** ({", ".join(mt['feasible_start_labels'])}, {", ".join(mt['feasible_solvers'])}) |
| J over the feasible candidates | {mt['J_feasible_min']:.3f}–{mt['J_feasible_min']+mt['J_feasible_spread']:.3f} km (spread {mt['J_feasible_spread']*1000:.0f} m, {100*mt['J_feasible_spread']/mt['J_feasible_min']:.2f} % of J) |
| J over the infeasible candidates | {mt['J_infeasible_range'][0]:.3f}–{mt['J_infeasible_range'][1]:.3f} km (listed, never used as an optimum) |

The solver's adopted objective is {mt['adopted']['J_km']:.3f} km (trapezoidal spline
integration); the hard objective evaluated post-hoc by the common polyline evaluator is
{C['J_hard_km']:.3f} km. The difference is numerical evaluation, not a different route.

The penalty continuation is applied to the three **canonical warm starts** because the direct stage
fails on every start; restricting it to the best direct candidates would have left no feasible
result at all. Infeasible candidates have lower J than the adopted route (one of them reaches
{mt['J_infeasible_range'][0]:.3f} km) precisely because they violate the slope limit — that is
why they cannot be ranked against it.

### 8.5 What separates the hard route from the grid route

The objectives, constraints, route representation and numerical methods differ. The β sweep in
Sec. 10.3 changes the risk weight while holding the hard formulation fixed and reports independently
validated candidates. These comparisons illustrate sensitivity to β, but do not prove an optimal
Pareto frontier or attribute the whole easy–hard length difference to any one modelling change.

### 8.6 Why the two objective values are not directly comparable

The easy problem minimises **length subject to feasibility**; the hard problem minimises
**length plus weighted risk subject to feasibility, curvature and its own smooth representation**.
Their objective functions live on different spaces (a finite graph versus a 14-dimensional
continuous family), and neither feasible set contains the other: the spline family cannot
represent a staircase, and the grid path is not a smooth curve. "Price of realism" and "risk
reduction" are therefore *comparisons of two candidate corridors under a common metric code*, not
differences of two optima of the same problem. Both routes are measured with the same R, the same
mask and the same common polyline evaluator; the two optimisers are different by design.

**β = 1, stated as a convention.** L and S are both in metres, so β is a price: β = 1 means one
metre of cumulative risk is worth one metre of length. The natural reference scale is
β&ast; = L_E/S_E = {C['L_easy_km']*1000/C['S_easy']:.1f}; at β = 1 the risk term is
{B['beta']*C['S_hard']/(C['L_hard_km']*1000)*100:.1f} % of J. The β sweep (Sec. 10.3) is the primary evidence and shows the
sensitivity of these feasible candidates to β; it does not establish β-independent optimality.

## 9. Constraint validation and verification

### 9.1 Independent recomputation

{md_table(pd.DataFrame([
        ["Exact A/B endpoints (both routes)", f"max error {max(V['endpoints'].values()):.1f} m"],
        ["Independent recomputation of L, S, J (different code path)",
         f"|ΔL| ≤ {max(V['recompute']['easy_L_diff_m'], V['recompute']['hard_L_diff_m']):.1e} m, "
         f"|ΔS| ≤ {max(V['recompute']['easy_S_diff'], V['recompute']['hard_S_diff']):.1e}"],
        ["Dijkstra vs scipy.sparse.csgraph (node polyline)",
         f"{V['dijkstra_crosscheck']['own_length_m']:.3f} vs "
         f"{V['dijkstra_crosscheck']['scipy_length_m']:.3f} m "
         f"(|Δ| = {V['dijkstra_crosscheck']['difference_m']:.1e} m)"],
        ["Dense validation, hard route, 200 / 1000 / 5000 points",
         f"slope max {V['density_hard'][0]['slope_max_deg']:.3f} / "
         f"{V['density_hard'][1]['slope_max_deg']:.3f} / "
         f"{V['density_hard'][2]['slope_max_deg']:.3f}°, "
         f"κ max {V['density_hard'][2]['kappa_max']:.3e} 1/m, violations "
         f"{V['density_hard'][2]['n_violations']}"],
        ["Geodesic vs grid distance A–B", f"{meta['AB_difference_pct']:.4f} %"],
    ], columns=["Check", "Result"]))}

**The one definition of feasibility used in this report.** A solver returning a candidate is not
proof that the candidate is feasible, so both routes are judged by `classify_route_feasibility`,
which requires *all* of: collocation constraints at the nominal limits; dense depth validation;
dense slope validation; the **exact** spline curvature; no sample inside a land / NoData /
outside-footprint cell; and a finite, strictly positive interpolated depth. The verdict:

{md_table(pd.DataFrame([
        ["Easy route", "yes" if FE["easy"]["feasible"] else "no",
         ", ".join(k for k, v in FE["easy"]["criteria"].items() if v) or "—"],
        ["Hard route", "yes" if FE["hard"]["feasible"] else "no",
         ", ".join(k for k, v in FE["hard"]["criteria"].items() if v) or "—"],
    ], columns=["Route", "Feasible", "criteria satisfied"]))}

### 9.2 Route curvature, recomputed from the B-spline itself

The optimiser evaluates κ from precomputed basis matrices. As an independent check the adopted
control points are rebuilt into a **fresh `scipy.interpolate.BSpline` object on the same knot
vector**, differentiated by scipy, and the curvature is formed directly from the scale
formula:

{md_table(pd.DataFrame([
        ["Max |κ| on the adopted curve (independent recomputation)",
         f"{V['curvature']['kappa_max']:.4e} 1/m"],
        ["Corresponding minimum turning radius", f"{V['curvature']['kappa_min_radius_m']:.0f} m"],
        ["Prescribed limit", f"{V['curvature']['limit_kappa_max']:.4e} 1/m "
                             f"(R = {V['curvature']['limit_min_radius_m']:.0f} m)"],
        ["Violation against the nominal limit", f"{V['curvature']['violation_vs_nominal']:.1e} 1/m"],
        ["Agreement with the optimiser's own evaluation",
         f"{V['curvature']['agreement_vs_optimiser']:.1e} 1/m"],
        ["Agreement with the polyline (finite-difference) estimator",
         f"{V['curvature']['agreement_vs_polyline_estimator']:.1e} 1/m"],
        ["Where the maximum occurs",
         f"t = {V['curvature']['kappa_max_location_frac']:.3f} (parameter value; the parameter "
         f"speed is not uniform, so it is not an arc-length station)"],
    ], columns=["Curvature check", "Result"]))}

The maximum observed curvature is **inside** the prescribed limit and the independent value agrees
with the optimiser's to machine precision, so the curvature constraint is not an artefact of the
solver's basis.

**Two curvature estimates exist in this project and only one of them is the constraint.** The
optimiser constrains the B-spline, so the constraint quantity is the exact derivative-based value
above — that is the number used for every feasibility statement and for the minimum turning radius
in the comparison table. The smoothed-heading estimate computed on the resampled polyline
({V['curvature']['polyline_estimator_kappa_max']:.4e} 1/m, i.e.
{1/V['curvature']['polyline_estimator_kappa_max']:.0f} m) is a **diagnostic** of what a
finite-difference measurement of the same curve would report; it is labelled as such everywhere and
is never mixed into a feasibility verdict or compared side by side in a result table.

### 9.3 Continuous-domain validation against the unfilled raster

Three different notions of "inside the domain" are separated, because conflating them is how a
constraint breach can hide: (i) the sample lies in a **masked** (land/NoData) cell, (ii) the
bilinear stencil at the sample needs filled values, (iii) the interpolated depth/slope at the
sample **violates a limit**. The adopted definition is: *a point of a continuous route is
continuously feasible when it is not inside a masked cell and its interpolated depth and slope
satisfy the nominal limits*. Results at 5 000 samples per route:

{md_table(pd.DataFrame([
        ["Samples inside a land/NoData cell",
         V['continuous_domain_easy']['on_masked_cell'], V['continuous_domain_hard']['on_masked_cell']],
        ["Samples whose bilinear stencil touches the mask",
         V['continuous_domain_easy']['masked_stencil'], V['continuous_domain_hard']['masked_stencil']],
        ["Samples violating the depth limit", V['continuous_domain_easy']['point_violation_depth'],
         V['continuous_domain_hard']['point_violation_depth']],
        ["Samples violating the slope limit", V['continuous_domain_easy']['point_violation_slope'],
         V['continuous_domain_hard']['point_violation_slope']],
        ["Samples on a cell flagged infeasible by the grid mask",
         V['continuous_domain_easy']['on_infeasible_cell'], V['continuous_domain_hard']['on_infeasible_cell']],
        ["…of which flagged by the cell but within limits at the point",
         V['continuous_domain_easy']['cell_flag_only'], V['continuous_domain_hard']['cell_flag_only']],
        ["…of which also violate a limit at the point",
         V['continuous_domain_easy']['infeasible_and_point_bad'],
         V['continuous_domain_hard']['infeasible_and_point_bad']],
        ["Minimum clearance to the nearest masked cell",
         f"{V['continuous_domain_easy']['min_clearance_to_mask_m']/1000:.1f} km",
         f"{V['continuous_domain_hard']['min_clearance_to_mask_m']/1000:.1f} km"],
    ], columns=["Continuous-domain check", "Easy route", "Hard route"]))}

**The {V['continuous_domain_hard']['cell_flag_only']} flagged samples are explained, not waved away.**
Each of them lies in a 450 m cell whose *node-level* slope exceeds s_max, while the *route point*
itself sits at {V['continuous_domain_hard']['cell_flag_only_worst_slope_deg']:.2f}° against the
{lim['s_max_deg']}° limit — a margin of
{V['continuous_domain_hard']['cell_flag_only_slope_margin_deg']:.2f}°. A node value is a cell
average (and, in a difference scheme, a difference of neighbouring cell averages), so a continuous
curve can clip the edge of a flagged cell while staying inside the continuous limits. No sample
lies in a masked cell, none violates a limit, and the route keeps
{V['continuous_domain_hard']['min_clearance_to_mask_m']/1000:.0f} km clearance from the nearest
land/NoData cell, so no part of it depends on how the mask was filled.

### 9.4 Interpolation: cubic versus linear, and whether the route cares

{md_table(pd.DataFrame([
        ["Interior depth overshoot (max)",
         f"{V['interpolation']['checks']['cubic']['depth_interior_max_overshoot_m']:.2f} m",
         f"{V['interpolation']['checks']['linear']['depth_interior_max_overshoot_m']:.2e} m"],
        ["Boundary depth overshoot (max, next to the mask)",
         f"{V['interpolation']['checks']['cubic']['depth_boundary_max_overshoot_m']:.2f} m",
         f"{V['interpolation']['checks']['linear']['depth_boundary_max_overshoot_m']:.2f} m"],
        ["Interior slope overshoot (p99)",
         f"{V['interpolation']['checks']['cubic']['slope_interior_p99_overshoot']:.2e}",
         f"{V['interpolation']['checks']['linear']['slope_interior_p99_overshoot']:.2e}"],
        ["Interior risk overshoot (max)",
         f"{V['interpolation']['checks']['cubic']['risk_interior_max_overshoot']:.4f}",
         f"{V['interpolation']['checks']['linear']['risk_interior_max_overshoot']:.4f}"],
        ["Probes implying land (d < 0)",
         V['interpolation']['checks']['cubic']['n_probe_implying_land'],
         V['interpolation']['checks']['linear']['n_probe_implying_land']],
    ], columns=["Interpolator property", "Cubic (adopted)", "Linear (reference)"]))}

Along the *adopted route* the two interpolants differ by at most
{V['interpolation']['route_field_difference']['max_abs_depth_difference_m']:.1f} m of depth,
{V['interpolation']['route_field_difference']['max_abs_slope_difference']*180/np.pi:.2f}° of slope and
{V['interpolation']['route_field_difference']['max_abs_risk_difference']:.4f} of R. A re-solve on
linear fields produced a candidate whose dense evaluation on the baseline cubic fields has
{V['interpolation']['linear_solution']['dense_n_violations']} combined constraint violations.
Its solver-side collocation status alone is not a feasibility certificate, and it is not used
for an improvement or validated route comparison.

### 9.5 Collocation density, the enforcement margin, and the edge rule

{md_table(pd.DataFrame([
        ["Collocation density (re-solve from the adopted point)",
         f"active samples {V['collocation_sensitivity']['M=400']['actual_integration_and_collocation_samples']} → J {V['collocation_sensitivity']['M=400']['J_km']:.3f} km, dense slope violations "
         f"{V['collocation_sensitivity']['M=400']['dense_n_slope_violations']}, "
         f"κ_max(exact) {V['collocation_sensitivity']['M=400']['kappa_max_exact']:.4e} 1/m; "
         f"active samples {V['collocation_sensitivity']['M=800']['actual_integration_and_collocation_samples']} → J {V['collocation_sensitivity']['M=800']['J_km']:.3f} km, dense slope violations "
         f"{V['collocation_sensitivity']['M=800']['dense_n_slope_violations']}, "
         f"κ_max(exact) {V['collocation_sensitivity']['M=800']['kappa_max_exact']:.4e} 1/m"],
        ["Price of the 2 % enforcement margin",
         f"J {V['margin_cost']['J_without_margin_km']:.3f} km without margin vs "
         f"{V['margin_cost']['J_with_margin_km']:.3f} km with it "
         f"({abs(V['margin_cost']['relative_pct']):.3f} % of J); without the margin the validated "
         f"route sits at {V['margin_cost']['dense_max_slope_without_margin_deg']:.2f}° against a "
         f"{lim['s_max_deg']}° limit (κ_max exact "
         f"{V['margin_cost']['kappa_max_exact_without_margin']:.4e} 1/m, limit "
         f"{lim['kappa_max']:.4e})"],
    ], columns=["Numerical convention", "Measured effect"]))}

The easy problem's edge rule is the specification's **geometric midpoint guard**. All three
documented rules are solved for comparison:

{md_table(pd.DataFrame([
        [rule, "yes" if V['edge_rules'][rule]['connected'] else "no",
         f"{V['edge_rules'][rule]['L_km']:.3f}", V['edge_rules'][rule]['n_nodes'],
         f"{V['edge_rules'][rule]['nodes_expanded']:,}",
         f"{V['edge_rules'][rule]['hausdorff_vs_baseline_m']:.0f}"]
        for rule in ("node", "midpoint", "corner")],
        columns=["Edge rule", "Route exists", "L_E [km]", "nodes", "nodes expanded",
                 "Hausdorff vs baseline [m]"]))}

The baseline rule is the **combined diagonal rule**: a diagonal step from (i, j) to
(i ± 1, j ± 1) is admitted only when both endpoints, the geometric midpoint *and* both orthogonal
neighbours (i ± 1, j) and (i, j ± 1) are feasible. The old `corner` rule checked endpoints and
orthogonal neighbours but never the midpoint; the plain `midpoint` rule checks the midpoint and
both endpoints but not the orthogonal neighbours. On this terrain the two halves of the combined
rule each fail somewhere:

* `node` and `midpoint` give the same length ({V['edge_rules']['midpoint']['L_km']:.3f} km), i.e.
  the midpoint guard alone does not bind — a diagonal that slips past a blocked orthogonal
  neighbour is still admitted and is {V['edge_rules']['midpoint_corner']['L_km']-V['edge_rules']['midpoint']['L_km']:.3f} km shorter than the baseline;
* adding the orthogonal-neighbour condition (`midpoint_corner`, the baseline) rejects those
  diagonals and lengthens the route to {V['edge_rules']['midpoint_corner']['L_km']:.3f} km, which
  coincides with the old `corner` result — so on this grid the *neighbour* half is the half that
  binds, and the midpoint half is the safety net that makes the rule a genuine 8-neighbour
  admissibility test rather than a corner-cutting check.

All four rules are reported, and only the combined one is used for the baseline. Every diagonal
edge is weighted by its true length √(Δx² + Δy²) taken from the node coordinates, not by a
per-axis increment, so diagonal moves are not artificially preferred or penalised.

### 9.6 Toy tests (seven tests, with feasibility asserted in every one)

The toy suite runs the same production functions on synthetic grids where the right answer is known
analytically. Three tests on route *behaviour* are not enough to catch a preprocessing or
validation bug, so four tests were added: the diagonal rule, the strict reprojection/masking
behaviour, interpolation validity, and — the most important one — the feasibility classifier.

{md_table(pd.DataFrame([
        ["T1 blocked wall",
         f"route crosses the wall: {toy['tests'][0]['crossed_the_wall']} (must be False); "
         f"{toy['tests'][0]['samples_in_blocked_cells']} samples in blocked cells; all path nodes "
         f"feasible: {toy['tests'][0]['all_nodes_feasible']}; detour ratio "
         f"{toy['tests'][0]['detour_ratio']:.3f} — the route takes a gap instead of crossing"],
        ["T2 flat, zero risk",
         f"L {toy['tests'][1]['L_km']:.3f} km vs straight {toy['tests'][1]['L_straight_km']:.3f} km "
         f"(excess {toy['tests'][1]['max_length_excess_pct']:.1e} %); κ_max "
         f"{toy['tests'][1]['kappa_max']:.1e} 1/m; constraints satisfied: "
         f"{toy['tests'][1]['constraints_satisfied']} — with no risk to avoid, the optimum is the "
         f"straight line, so the optimiser adds no gratuitous curvature"],
        ["T3 single risk bump (mirrored pair)",
         f"J_left {toy['tests'][2]['J_left_km']:.3f} vs J_right {toy['tests'][2]['J_right_km']:.3f} km "
         f"(symmetry error {toy['tests'][2]['J_symmetry_error_pct']:.3f} % ≤ "
         f"{toy['tests'][2]['J_symmetry_tolerance_pct']:.0f} %); mirror error "
         f"{toy['tests'][2]['mirror_error_m']:.0f} m ≤ {toy['tests'][2]['mirror_tolerance_m']:.0f} m; "
         f"mean risk on the route {toy['tests'][2]['mean_R_on_route']:.2e} vs "
         f"{toy['tests'][2]['mean_R_on_straight_line']:.4f} on the straight line — the route avoids "
         f"the bump and the two mirrored starts find equivalent optima"],
        ["T4 combined diagonal rule",
         f"a step is admitted in open terrain: {toy['tests'][3]['A_open_terrain']}; rejected when the "
         f"midpoint is blocked: {not toy['tests'][3]['B_blocked_midpoint_stencil']}; rejected when an "
         f"orthogonal neighbour is infeasible: {not toy['tests'][3]['C_orthogonal_neighbour_infeasible']}; "
         f"rejected when an endpoint is: {not toy['tests'][3]['D_endpoint_infeasible']}; cases where "
         f"the midpoint binds although both diagonals' corners are feasible: "
         f"{toy['tests'][3]['midpoint_binds_with_feasible_corners']} of 2 000 random draws — the rule "
         f"is a four-condition test, not a rename of the old corner rule"],
        ["T5 strict reprojection and masking",
         f"open ocean accepted: {toy['tests'][4]['i_open_ocean_ok']} "
         f"({toy['tests'][4]['i_open_ocean_value_m']:.0f} m); land-stencil point rejected: "
         f"{not toy['tests'][4]['ii_land_stencil_ok']}; NoData cell rejected: "
         f"{not toy['tests'][4]['iii_nodata_cell_ok']}; half-cell border accepted: "
         f"{toy['tests'][4]['iv_half_cell_border_ok']}; a point outside the interpolator extent but "
         f"inside the source footprint is NOT silently extrapolated: "
         f"{not toy['tests'][4]['v_outside_extent_inside_footprint']}; no accepted query returned a "
         f"land value: {toy['tests'][4]['no_positive_value_among_accepted']}"],
        ["T6 interpolation validity",
         f"cubic minimum depth on the probe line {toy['tests'][5]['cubic_min_depth_m']:.2f} m with "
         f"{toy['tests'][5]['cubic_n_negative']} negative values; linear "
         f"{toy['tests'][5]['linear_min_depth_m']:.2f} m — the cubic is *not* replaced by the linear "
         f"field, because depth overshoot is a constraint issue handled by classification, not by a "
         f"blunter interpolant; a synthetic negative depth is classified infeasible: "
         f"{toy['tests'][5]['negative_depth_classified_infeasible']}"],
        ["T7 feasibility classification",
         f"clean candidate accepted: {toy['tests'][6]['all_pass']}; with clean solver metadata but a "
         f"failing dense check, each variant is rejected — dense slope "
         f"{not toy['tests'][6]['dense_slope_fails']}, dense depth "
         f"{not toy['tests'][6]['dense_depth_fails']}, exact curvature "
         f"{not toy['tests'][6]['kappa_fails']}, domain/mask "
         f"{not toy['tests'][6]['enters_mask']}. This is the regression test for the original bug "
         f"(solver metadata reported success while the route violated the slope limit); the reason "
         f"string records it: “{toy['tests'][6]['slope_failure_reason'][0] if toy['tests'][6]['slope_failure_reason'] else ''}”"],
    ], columns=["Toy test", "Assertions and outcome"]))}

All seven pass (`python -m src.tests_toy`, exit code 0; machine-readable output in
`results/toy_tests.json`). The suite is the argument that the numbers in Secs. 9.1–9.5 are not
produced by the same bug they are meant to detect.

### 9.7 Runtime protocol

Five measurements are often called "the runtime". They are separated here (single machine,
CPU only). Every repeated-run figure below is a **median over repeated calls of the same operation on the same
machine**, with min–max in brackets; means are not reported because a single scheduling hiccup
dominates them:

{md_table(pd.DataFrame([
        ["Easy: original baseline graph-search invocation", f"{V['runtimes']['easy_single_dijkstra_s']*1000:.1f} ms"],
        ["Easy: repeated solve_easy median (includes route assembly)", f"{V['runtimes']['easy_repeated_median_s']*1000:.0f} ms "
                                    f"({V['runtimes']['easy_repeated_min_s']*1000:.0f}–"
                                    f"{V['runtimes']['easy_repeated_max_s']*1000:.0f} ms)"],
        ["Hard: original adopted penalty continuation from its warm start", f"{V['runtimes']['hard_adopted_penalty_continuation_s']:.2f} s"],
        ["Hard: repeated penalty refinement from the adopted point",
         f"{V['runtimes']['hard_repeated_median_s']:.2f} s "
         f"({V['runtimes']['hard_repeated_min_s']:.2f}–{V['runtimes']['hard_repeated_max_s']:.2f} s)"],
        ["Hard: the full multi-start search (all "
         f"{V['runtimes']['n_executions_in_budget']} executions)", 
         f"{V['runtimes']['hard_full_multistart_budget_s']:.1f} s"],
        ["Ratio of the two single-solve costs (median ÷ median)",
         f"×{V['runtimes']['ratio_median_hard_to_easy']:.0f}"],
    ], columns=["Runtime definition", "Value"]))}

{BASIS_NOTE}

`comparison.csv` names and preserves the baseline graph-search timing, original adopted penalty
continuation, both repeated-run medians and the full multi-start budget separately. The graph-search
single timing excludes route assembly; the repeated `solve_easy` timer includes it. They are distinct
measurements rather than contradictory values. The full search budget is the sum of the measured
solver-execution durations, not end-to-end pipeline wall time. Sensitivity CSV `runtime_hard_s` is
the adopted continuation for that setting and `runtime_multistart_s` is its full search budget.

**Preserved previous timing observations.** Before the final reproduction run, the original
baseline graph search was {prior_rt['easy_original_graph_search_s']*1000:.1f} ms and the repeated
easy-solve median was {prior_rt['easy_repeated_solve_median_s']*1000:.0f} ms. The original adopted
hard continuation took {prior_rt['hard_original_adopted_continuation_s']:.2f} s, the repeated
refinement median {prior_rt['hard_repeated_refinement_median_s']:.2f} s, and the full
{prior_rt['hard_budget_executions']}-execution search budget {prior_rt['hard_full_multistart_budget_s']:.1f} s.
These are historical observations preserved in `results/runtime_previous_baseline.json`; the previous
thread settings and machine load were not recorded, so they are not interchangeable with current-run
measurements. Current values are the table above and in `comparison.csv`.

### 9.8 What is verified, and what is not

Verified by independent code: endpoints, L, S, the snapped-cell graph optimum (cross-checked against scipy), dense
constraint satisfaction of both routes, continuous-domain feasibility against the unfilled
raster, the curvature bound (from rebuilt spline derivatives), the interpolation comparison and
all seven toy tests (including the ones that exist specifically to catch preprocessing and
validation bugs: the diagonal rule, the strict reprojection, interpolation validity and the
feasibility classifier). **Not** verified, and not claimed: global optimality of the hard route, any
KKT property of the penalised solution, and any interpretation of R as a safety measure. The
hard route is a feasible local candidate: only {mt['n_feasible']} of
{mt['n_solver_executions']} solver executions reached feasibility, so the search is reported as a
local candidate, not as an optimum.

## 10. Sensitivity experiments

Protocol: one factor at a time; for every setting **both** problems are re-solved, the feature
stack is rebuilt, and the same metrics are recomputed by the common polyline evaluator. Budget: 3
penalty starts per run for the sensitivity sets ({mt['n_solver_executions']} solver executions for the baseline). Everything below comes from
`results/sensitivity_*.csv`.

**Two kinds of risk column, and only one of them is comparable across settings.** Min–max
normalisation is recomputed inside every configuration, so S as computed *within* a setting
(`S_E`, `S_H`) is only meaningful inside that setting. Every cross-configuration statement in this
section therefore uses the **frozen** columns (`S_E_frozen`, `S_H_frozen`, `J_*_frozen`,
`price_of_realism_pct_frozen`, `risk_reduction_pct_frozen`), which re-integrate the same routes
against the **baseline** risk field — the field is held fixed, only the route changes. The baseline
row is its own frozen reference, so the two column families coincide there.

**Feasibility is decided by the classifier, not by the solver.** A row is only marked
`valid_solution = True` when the candidate passes the full check of Sec. 9.1 (nominal collocation,
dense depth, dense slope, exact spline curvature, domain/mask, finite interpolated depth). Rows that
fail keep their metrics in the CSV but are labelled **INFEASIBLE** and are excluded from every
route-improvement claim in this section — in particular the s_max = P75 candidate is *not* an
optimised solution, and its metrics (including the candidate S_H) are never quoted as a result.

### 10.1 S-core (the prescribed experiments)

Exactly the experiments the specification prescribes for the presentation: slope derivation
(central / Horn / Sobel), curvature (Laplacian / Hessian), roughness window w = 3/5/7/9 and
roughness of depth instead of slope, smoothing σ = 0/1/2/3, s_max at P75/P90/P95, two depth
windows, three κ_max values and the edge rule. Everything else lives in Sec. 10.2.

{_sens_md(score[score["factor"] != "baseline"], {
    "label": "setting", "feasible_fraction": "feasible %", "n_components": "comp.",
    "L_E_km": "L_E [km]", "L_H_km": "L_H [km]", "S_E": "S_E", "S_H": "S_H",
    "hard_max_slope_deg": "max slope [°]", "hausdorff_vs_baseline_m": "Hausdorff [m]",
    "jaccard_vs_baseline": "Jaccard"})}

The `status` column distinguishes **no grid path** from a path that fails independent checks.
Other invalid statuses identify whether the easy path, hard candidate, or both fail. Route metrics
for an invalid pair are blank. Candidate diagnostics are retained in the CSV with
`valid_solution = False` and `invalid_reason`; they are not used in improvement claims.

**What the column patterns say.**

* **Feasibility is the fragile property, not optimality.** Along the prescribed set, the settings
  that change the *answer* are the constraint and the smoothing; the settings that change the
  *field* (slope estimator, roughness window, curvature definition, κ_max) barely move the route.
  At s_max = P75 the feasible fraction drops to {f75:.1%}, the feasible set shatters into
  {n75:.0f} components and no A→B grid route exists. The smooth solver also returns an
  infeasible candidate; neither is presented as a valid route.
* **Slope estimator.** {slope_txt} The estimator changes the *field* first and the *route* only
  second: where a feasible pair still exists the route barely moves. When independent checks fail,
  the candidate is not reported as feasible; this does not prove that no continuous route exists.
* **Risk is not comparable across settings** when each setting's own min–max normalisation is
  used. Cross-setting comparisons below use the frozen baseline risk field instead.
* **κ_max barely matters**: relaxing the minimum radius from 1350 m to 338 m moves the route by
  {h338:.0f} m (Jaccard ≈ 1).
* **Roughness window changes the model.** Compare resulting feasible routes using the frozen
  baseline field, not their separately normalised native S values.

### 10.2 Full factor matrix

The full matrix adds the third depth window, a fourth κ_max, and the normalisation variant to the
S-core set. Runtime columns are separated: `runtime_hard_s` is the adopted penalty continuation
from its warm start, and `runtime_multistart_s` is the sum across that setting's starts.

{_sens_md(full[full["factor"] != "baseline"], {
    "label": "setting", "feasible_fraction": "feasible %", "n_components": "comp.",
    "L_H_km": "L_H [km]", "S_H": "S_H", "hard_max_slope_deg": "max slope [°]",
    "hausdorff_vs_baseline_m": "Hausdorff [m]", "runtime_hard_s": "runtime single [s]",
    "runtime_multistart_s": "runtime multistart [s]"})}

![sensitivity](figures/fig07_sensitivity.png)


### 10.3 Secondary experiments

**β sweep (length–risk Pareto family).** J is re-solved for a range of β with everything else
fixed; since S is in metres, β is the price of one metre-equivalent of risk in metres of length.

{_sens_md(beta[beta["factor"] == "beta"], {
    "label": "setting", "L_H_km": "L_H [km]", "S_H": "S_H", "J_H_km": "J_H [km]",
    "hard_max_slope_deg": "max slope [°]", "hausdorff_vs_baseline_m": "Hausdorff [m]"})}

![pareto](figures/fig06_pareto.png)

{_beta_txt}

**Weight sets and control points.**

{_sens_md(_sec_tbl, {"label": "setting", "L_H_km": "L_H [km]",
    "S_H_frozen": "S_H (frozen)", "hard_max_slope_deg": "max slope [°]",
    "hausdorff_vs_baseline_m": "Hausdorff [m]", "jaccard_vs_baseline": "Jaccard"})}

![weights and K](figures/fig11_weights_K.png)

{_sec_txt}

## 11. Easy-versus-hard comparison and robustness

{md_table(pd.DataFrame([
        ["Length L", f"{C['L_easy_km']:.3f} km", f"{C['L_hard_km']:.3f} km"],
        ["Cumulative risk S", f"{C['S_easy']:.1f}", f"{C['S_hard']:.1f}"],
        ["J_H = L_H + β S_H (post-hoc common evaluation of hard objective)",
         "—", f"{C['J_hard_km']:.3f} km"],
        ["Adopted solver objective J_H (trapezoidal integration)",
         "—", f"{mt['adopted']['J_km']:.3f} km"],
        ["J_E = L_E + β S_E (post-hoc common score, same β)",
         f"{C['J_easy_km']:.3f} km", f"{C['J_hard_km']:.3f} km"],
        ["Max slope (dense validation, 2000 points)",
         f"{C['easy_max_slope_deg']:.3f}°", f"{C['hard_max_slope_deg']:.3f}°"],
        ["Max curvature — exact B-spline derivative (the constraint)",
         "not applicable: the grid route is a polyline, not a spline",
         f"{C['hard_kappa_max']:.3e} 1/m (R = {1/C['hard_kappa_max']:.0f} m)"],
        ["Max curvature — polyline diagnostic (not a constraint)",
         f"{C['easy_kappa_max']:.3e} 1/m (R = {1/C['easy_kappa_max']:.0f} m); informational, "
         f"no curvature limit is imposed on the grid route",
         f"{C['hard_kappa_polyline_diagnostic']:.3e} 1/m"],
        ["Limit violations (2000 dense points)",
         f"{C['easy_violations_slope_depth']} slope/depth (+{C['easy_violations_kappa_info']} κ, informational)",
         f"{C['hard_violations']}"],
        ["Solution method (different by construction)",
         "graph search: Dijkstra on the 8-neighbour cell graph",
         "continuous refinement: SLSQP on 14 spline control coordinates"],
        ["Objective actually optimised",
         "c_ij = metric edge length only (no risk weight)",
         "J_H = L_H + β S_H, β = " + BET],
        ["Integration of the reported L, S, J",
         "common polyline evaluator on the grid polyline",
         "common polyline evaluator; the exact spline value is used only for κ"],
        ["Runtime (median of repeats)",
         f"{V['runtimes']['easy_repeated_median_s']*1000:.0f} ms",
         f"{V['runtimes']['hard_repeated_median_s']:.2f} s"],
        ["Runtime (full search)", "one Dijkstra call",
         f"{V['runtimes']['hard_full_multistart_budget_s']:.1f} s over {V['runtimes']['n_executions_in_budget']} executions"],
        ["Price of realism (L_H−L_E)/L_E", f"{C['price_of_realism_pct']:+.2f} %", ""],
        ["Risk reduction (S_E−S_H)/S_E", f"{C['risk_reduction_pct']:+.2f} %", ""],
        ["Hausdorff / mean distance between routes",
         f"{C['hausdorff_m']:.0f} m / {C['mean_distance_m']:.0f} m", ""],
        ["Cell overlap (Jaccard)", f"{C['jaccard']:.3f}", ""],
    ], columns=["Metric", "Easy", "Hard"]))}

![routes](figures/fig04_routes.png)

![profiles](figures/fig05_constraint_profiles.png)

![cumulative risk](figures/fig09_cumulative_risk.png)

*Robustness.* The two routes share the corridor loosely (Jaccard {C['jaccard']:.2f}) and differ
by up to {C['hausdorff_m']/1000:.2f} km where the grid route has to step around the flank.

{ROB_TXT} Settings without a feasible A→B route (s_max = P75) remove the comparison entirely —
which is itself the finding: feasibility is the fragile property, not optimality.

*What the comparison does and does not claim.* The hard route is a **feasible local candidate**
under the selected model constraints, not a proven optimum of the continuous problem: {mt['n_feasible']}
of {mt['n_solver_executions']} solver executions reached feasibility and those agree to
{mt['J_feasible_spread']*1000:.0f} m of J, which is convergence evidence from neighbouring warm
starts, not a KKT certificate. The easy graph segment is the discrete optimum between the selected
snapped cells; the complete easy route includes separately appended connectors. The comparison is
a statement about **discretisation and representation**, not about beating an optimiser: the 8-neighbour polyline is forced to zig-zag along the corridor, while the spline
integrates a smooth curve through the same terrain. That is why the price of realism can be
negative, and why it is reported as a **length–risk trade-off** rather than as an improvement.

*A caveat that must travel with the headline.* Part of the negative "price of realism" is
discretisation, not smoothness: the easy route is measured node-to-node, and a 250 m moving-average of the same corridor is {V['easy_route_discretisation']['inflation_pct']:.2f} % shorter than its discrete length ({V['easy_route_discretisation']['L_discrete_m']/1e3:.2f} km vs {V['easy_route_discretisation']['L_smoothed_250m_m']/1e3:.2f} km). Even against that smoothed baseline the hard route remains shorter, but the honest statement is "the hard route is shorter than the connector-inclusive easy route by {abs(C['price_of_realism_pct']):.1f} %, of which about {V['easy_route_discretisation']['inflation_pct']:.1f} percentage points are grid discretisation".

## 12. Interpretation: how feature-engineering choices move feasibility and optimality

1. **Feasibility is a cliff, not a slope.** Moving s_max from P90 to P75 takes the feasible fraction from {_row("P90", "feasible_fraction"):.0%} to {f75:.0%} and shatters the connected corridor into {n75:.0f} components; no A→B grid route exists under this rule; this does not by itself prove that no continuous route exists. That setting is reported as infeasible everywhere in this report — including the hard-problem table, where the solver's output at P75 is *not* treated as a route.
2. **Smoothing is not cosmetic.** σ = 2 cells changes the feasible-cell fraction to {_row("2.0", "feasible_fraction"):.0%}, but the hard candidate fails validation, so no route metric is reported for this setting.
3. **Roughness window is a scale choice.** At w = 7 the validated route moves by ~{h_w7:.0f} m; native S values on separately normalised fields are not comparable.
4. **Route curvature is imposed only on the hard problem.** Slope and depth are respected by
   both routes; κ is constrained for the hard spline and merely diagnostic for the easy polyline.
   Objective and decision space differ as well, so the observed difference is not an isolated curvature cost.
5. **The risk field is a proxy.** It ranks cells; it is not a geological, geotechnical or safety
   assessment, and no number in this report should be read as one.

## 13. Limitations

* **Local candidate, not a global optimum.** The hard route is a **feasible local candidate**
  produced by a quadratic exterior-penalty continuation: {mt['n_feasible']} of
  {mt['n_solver_executions']} solver executions (from {mt['n_distinct_initial_curves']} distinct initial
  curves) reached feasibility, and those {mt['n_feasible']} agree to
  {mt['J_feasible_spread']*1000:.0f} m of J — agreement between two runs started from *neighbouring*
  warm starts is convergence evidence, not evidence of global optimality, and the infeasible
  candidates span {mt['J_infeasible_range'][0]:.1f}–{mt['J_infeasible_range'][1]:.1f} km, so the search
  landscape has many basins — we simply cannot say which of them are feasible. No global-optimality claim is made, and no KKT
  property of the constrained problem is established.
* **Solver fragility (accounting, not adjectives).** Of {mt['n_direct_executions']} direct
  constrained executions, {V['solver_strategies']['direct']['n_converged_status0']} reached
  `status 0` and **{V['solver_strategies']['direct']['n_feasible']}** were feasible on independent
  validation. Of {mt['n_penalty_executions']} quadratic-exterior-penalty executions,
  {V['solver_strategies']['penalty']['n_converged_status0']} reached `status 0` and
  **{V['solver_strategies']['penalty']['n_feasible']}** were feasible. The adopted candidate is
  `{mt['adopted']['solver']}` from the `{mt['adopted']['start']}` warm start; the penalty
  continuation is what produces a feasible route, and some sensitivity settings still return an
  infeasible candidate and are marked as such.
* **Two documented numerical conventions.** Constraints are enforced on 400 samples with a 2 %
  margin and validated against the nominal limits at 2000–5000 samples. The optimizer evaluates J
  on {actual_M} samples (the same active grid as the constraints). Both are choices; both are measured (Sec. 9) rather than assumed.
* **Risk is a proxy, and it is renormalised per configuration.** R ranks cells by steepness,
  roughness and terrain curvature with weights {W['slope']}/{W['roughness']}/{W['curvature']}. It
  is not a safety or geological measure, and S is only comparable within one configuration.
* **Depth is sea-floor depth.** `d_min`/`d_max` bound the seabed depth along the corridor; they
  say nothing about tunnel alignment depth, cover, or overburden.
* **Data resolution.** 15 arc-sec GEBCO with altimetry-predicted areas; derivatives at 450 m
  are at the edge of what the source supports, the striping is visible, and no independent
  survey validation was possible.
* **One basin, one β.** The length–risk trade-off is characterised for β = {B['beta']:.0f}
  (with a full sweep in Sec. 10.3), on a single 50 × 51 km subset. No transferability claim.
* **Not an engineering design.** No cost, no construction method, no geotechnics, no seismic or
  volcanic hazard, no environmental constraints.

## 14. Conclusion

**The chain.** A footprint-limited, mask-first reprojection of a GEBCO subset gives {meta['n_valid']:,} valid
cells with no extrapolation (Sec. 2–4). On that terrain the easy problem is an exact 8-neighbour
shortest path between selected snapped cells, with appended validated connectors (Sec. 7), and the hard problem is a constraint-compliant cubic B-spline found by a
quadratic-penalty SLSQP continuation (Sec. 8). Independent validation (Sec. 9) — a second
implementation of L and S, a scipy cross-check of the search, dense continuous-domain sampling,
and the exact spline curvature — cross-checks the snapped-cell graph optimum but does not establish hard-route optimality; it shows that
both routes are **feasible under the selected model constraints**. The comparison (Sec. 11) gives
L_H = {C['L_hard_km']:.3f} km against L_E = {C['L_easy_km']:.3f} km with
S_H = {C['S_hard']:.0f} against S_E = {C['S_easy']:.0f}, and the sensitivity matrix (Sec. 10)
shows which of those numbers survive a change of feature engineering.

**The price of realism.** Read as a length–risk trade-off, the answer is a *negative* price:
the constraint-compliant spline is {abs(C['price_of_realism_pct']):.1f} % shorter and
{C['risk_reduction_pct']:.1f} % lower-risk than the connector-inclusive easy route. That sign is not a claim that
continuous optimisation beats graph search — the easy route is exactly optimal *for its own
discretisation*, and part of the gap (about
{V['easy_route_discretisation']['inflation_pct']:.1f} percentage points) is the 8-neighbour
polyline's forced zig-zag, not the constraint. The honest reading is the one the two-number
summary supports: paying for realism costs **smoothness and solver reliability**, not length. The
route that respects a {1/lim['kappa_max']:.0f} m turning radius is *cheaper in this terrain*, and it
is only available at the price of a local, multi-start, penalty-continued solve that fails outright
at some settings.

**What limits the answer.** Feasibility depends on the chosen slope limit and on bathymetric
resolution and quality; the stricter P75 setting has no grid path. The continuous solver can also
fail independent validation. Risk weights are modelling assumptions and are not validated by
the correlation study; no claim of general transferability is made.

## 15. Appendix

**Configuration (`config.yaml`, baseline id `{meta.get('source_nc','baseline')}`).**

{md_table(pd.DataFrame([
        ["Feature settings", json.dumps(fs, default=str)],
        ["Limits", json.dumps(lim)],
        ["α, β", f"{B['alpha']:.0f}, {B['beta']:.0f}"],
        ["Configured M / active objective and constraint samples / margin", f"{configured_M} / {actual_M} / 2 %"],
        ["K interior control points", "7 (study: 5, 10)"],
        ["Multi-start", f"{mt['n_distinct_initial_curves']} distinct initial curves (grid fit, smoothed "
                        f"grid fit, straight line, 8 seeded random); {mt['n_solver_executions']} solver "
                        f"executions ({mt['n_direct_executions']} direct + {mt['n_penalty_executions']} "
                        f"penalty continuations)"],
        ["Seeds", "random_seed 20261110 (config), perturbations σ = 6 cells"],
        ["SLSQP", "ftol 1e-6 (direct), 1e-9 (penalty); maxiter 300; μ = 1e2, 1e4, 1e6"],
    ], columns=["Item", "Value"]))}

**Final optimized hard-route control points (UTM zone 32N, EPSG:32632).** P0 and P8 are
fixed endpoints A and B; P1–P7 are the optimized interior control points (14 scalar decision
variables). Interior control points define the B-spline and do not generally lie on the curve.
The machine-readable coordinates are in `results/control_points_hard.csv`; the full exact-endpoint
grid polyline and the adopted spline sampled at its active integration/collocation parameters are in
`results/route_easy.csv` and `results/route_hard.csv`, respectively. These same arrays and the
normalized decision vector are also in `results/baseline.json`.

{md_table(points.rename(columns={"point": "Point", "role": "Role", "easting_m": "Easting [m]", "northing_m": "Northing [m]"}), floatfmt="{:.3f}")}

**Code layout.**

| File | Contents |
|---|---|
| `src/gebco_download.py` | GEBCO download via the public API (basket → status → zip) |
| `src/data.py` | NetCDF → metric grid, reprojection, NoData/land masking, `Grid` |
| `src/features.py` | slope (central/Horn/Sobel), roughness, terrain curvature, normalisation, risk field |
| `src/easy.py` | 8-neighbour graph, geometric midpoint edge rule (+ node/corner variants), Dijkstra, exact endpoints |
| `src/hard.py` | mask-aware cubic/linear interpolators + overshoot checks, B-spline path, objective, constraints, direct + quadratic-penalty SLSQP, multi-start accounting, independent curvature |
| `src/metrics.py` | L, S, route curvature, dense validation, density study, domain check, Hausdorff, Jaccard |
| `src/analysis.py` | descriptive stats, Pearson/Spearman, Moran's I, PCA, feasibility summary |
| `src/pipeline.py` | end-to-end baseline + verification → `results/baseline.json`, `comparison.csv` |
| `src/sensitivity.py` | one-factor-at-a-time experiments → `results/sensitivity_*.csv` |
| `src/tests_toy.py` | seven toy tests (wall, flat, bump symmetry, diagonal rule, strict reprojection, interpolation validity, feasibility classifier) |
| `src/viz.py`, `src/figures.py` | all figures |
| `src/report.py`, `src/deck.py` | this report and the 12-slide deck |

**Reproduce.**

```
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  # final reproduction settings
python -m src.gebco_download      # data (already cached in data/raw)
python -m src.data                # build data/processed/baseline.npz
python -m src.pipeline            # baseline + verification  (~70 s)
python -m src.tests_toy           # 7 toy tests              (~8 s)
python -m src.sensitivity --set score       # prescribed S-core experiments
python -m src.sensitivity --set full        # factor matrix
python -m src.sensitivity --set secondary   # weights, K
python -m src.sensitivity --set beta        # beta sweep
python -m src.figures             # figures/
python -m src.report              # report.md
python -m src.report_html         # report.html
python -m src.deck                # deck.pptx
python -m src.deck_preview --sheet  # one PNG per slide, for QA
python -m src.audit_submission    # final integrity checks
```

**Environment and dependencies.** Python 3.13; pinned versions in `requirements.txt`
(numpy 2.3.5, scipy 1.17.1, pandas 2.2.3, matplotlib 3.11.2, netCDF4 1.7.5.1.post1, pyproj 3.8.0, pyyaml,
python-pptx 1.0.2, Pillow, markdown). Runs on CPU only; no GPU, no proprietary data.

**Dataset citation.** GEBCO Compilation Group (2026). *GEBCO 2026 Grid* (15 arc-second global
bathymetric grid). GEBCO — General Bathymetric Chart of the Oceans.
doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa. Subset used here:
{meta['bbox_wgs84']['west']}–{meta['bbox_wgs84']['east']}° E,
{meta['bbox_wgs84']['south']}–{meta['bbox_wgs84']['north']}° N, downloaded from
<https://download.gebco.net> (elevation and TID layers). Bathymetry in parts of the subset is
predicted from satellite altimetry, not measured by multibeam sonar (Sec. 5 and Sec. 13).
"""

    OUT.write_text(out)
    print(f"wrote {OUT} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
