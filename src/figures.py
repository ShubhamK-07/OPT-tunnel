"""Build every figure in one pass.  Run: python -m src.figures"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import analysis, metrics, viz
from .data import load_grid
from .pipeline import baseline_route_set, evaluate_routes, limits

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _cumulative_risk(ctx, ev, n=2000):
    import matplotlib.pyplot as plt
    f = ctx["fields"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for xy, lab, col in ((ev["xy_easy"], "easy (Dijkstra)", viz.C_EASY),
                         (ev["xy_hard"], "hard (spline, SLSQP)", viz.C_HARD)):
        L, S, xys = metrics.polyline_terms(xy, f, M=n)
        seg = np.hypot(*np.diff(xys, axis=0).T)
        mid = 0.5 * (xys[1:] + xys[:-1])
        R = f("R", mid[:, 0], mid[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(R * seg)])
        ax.plot(np.linspace(0, 1, len(cum)), cum, color=col, lw=1.8,
                label=f"{lab}: S = {cum[-1]:.0f}")
    ax.set_xlabel("normalised arc length along the route")
    ax.set_ylabel("cumulative terrain risk S [m]")
    ax.set_title("Cumulative risk S(p): where the risk is accumulated")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(viz.FIG / "fig09_cumulative_risk.png")
    plt.close(fig)


def _tid_map(grid, A, B):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    tid = grid.tid
    levels = np.unique(tid[np.isfinite(tid)])
    cmap = plt.get_cmap("tab20", len(levels))
    im = ax.imshow(tid, origin="lower",
                   extent=[grid.x[0] / 1e3, grid.x[-1] / 1e3, grid.y[0] / 1e3, grid.y[-1] / 1e3],
                   cmap=cmap, aspect="equal", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("GEBCO TID (source identifier code)")
    ax.set_title("GEBCO type-identifier (TID) map: which survey/source\n"
                 "produced each cell -- a data-quality indicator, not a safety map")
    ax.set_xlabel("UTM 32N easting [km]")
    ax.set_ylabel("northing [km]")
    viz._mark_ab(ax, grid, A, B)
    fig.tight_layout()
    fig.savefig(viz.FIG / "fig10_tid_map.png")
    plt.close(fig)


def _weights_and_k(df_sec):
    """Small panels for the weight sets and the K study."""
    import matplotlib.pyplot as plt
    if df_sec is None or len(df_sec) == 0:
        return
    w = df_sec[(df_sec["factor"] == "weights") & df_sec["valid_solution"].astype(bool)]
    k = df_sec[(df_sec["factor"] == "n_control_points") & df_sec["valid_solution"].astype(bool)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sub, title, col in (
            (axes[0], w, "weight sets: L_H [km]", "L_H_km"),
            (axes[1], w, "weight sets: S_H (frozen)", "S_H_frozen"),
            (axes[2], k, "control points K: L_H and S_H", "L_H_km")):
        if len(sub) == 0:
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        ax.plot(x, sub[col], "o-", color=viz.C_HARD)
        ax.set_xticks(x, [str(v)[:28] for v in sub["value"]], rotation=20, fontsize=7)
        ax.set_title(title)
    if len(k):
        ax2 = axes[2].twinx()
        ax2.plot(np.arange(len(k)), k["S_H_frozen"], "s--", color=viz.C_EASY)
        ax2.set_ylabel("S_H (frozen field)", color=viz.C_EASY)
    fig.tight_layout()
    fig.savefig(viz.FIG / "fig11_weights_K.png")
    plt.close(fig)


def build_deck_figures(ctx, ev):
    """Presentation variants: same plots, larger type, fewer rows."""
    from . import viz
    corr = analysis.correlation_matrices(ctx["F"], ctx["grid"].mask_land)
    with viz.deck_style(10.5):
        viz.fig_maps(ctx, suffix="_deck", figsize=(8.7, 4.85))
    with viz.deck_style(11.0):
        viz.fig_deck_terrain_distributions(ctx["F"], ctx["grid"].mask_land)
        viz.fig_routes(ctx, ev, suffix="_deck", figsize=(8.7, 3.7))
        viz.fig_cover_map(ctx)
    with viz.deck_style(9.5):
        viz.fig_correlations(corr, suffix="_deck", figsize=(6.1, 2.30))
        p_full = RESULTS / "sensitivity_full.csv"
        if p_full.exists():
            d = pd.read_csv(p_full)
            core = d[d["factor"].isin(["slope_method", "smooth_sigma_cells", "s_max_deg",
                                       "kappa_max", "roughness_window"])]
            core = core[~core["label"].isin(["roughness_window=5", "roughness_window=9"])]
            with viz.deck_style(8.6):
                viz.fig_sensitivity(d, suffix="_deck", size=(7.35, 5.35))
            p_score = RESULTS / "sensitivity_score.csv"
            if p_score.exists():
                with viz.deck_style(8.8):
                    viz.fig_score_deck(pd.read_csv(p_score))
    print("deck figures written")


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    grid = load_grid(cfg)
    print("[figures] solving baseline ...", flush=True)
    ctx = baseline_route_set(cfg, verbose=True, n_random=8, save=False)
    ev = evaluate_routes(ctx, verbose=False)
    lim = ctx["lim"]

    stats = analysis.stats_table(ctx["F"], grid.mask_land)
    corr = analysis.correlation_matrices(ctx["F"], grid.mask_land)
    stats.to_csv(RESULTS / "descriptive_stats.csv")
    print("[figures] maps ...", flush=True)
    viz.fig_maps(ctx)
    viz.fig_distributions(ctx["F"], grid.mask_land)
    viz.fig_correlations(corr)
    viz.fig_routes(ctx, ev)
    viz.fig_constraint_profiles(ctx, ev)
    _cumulative_risk(ctx, ev)
    _tid_map(grid, ctx["A"], ctx["B"])

    p_beta = RESULTS / "sensitivity_beta.csv"
    if p_beta.exists():
        d = pd.read_csv(p_beta)
        d = d[(d["factor"] == "beta") & d["valid_solution"].astype(bool)].copy()
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["value"]).sort_values("value")
        viz.fig_pareto(list(zip(d["value"], d["L_H_km"], d["S_H"])), baseline=cfg["hard"]["beta"])
    p_full = RESULTS / "sensitivity_full.csv"
    p_score = RESULTS / "sensitivity_score.csv"
    df = pd.read_csv(p_full) if p_full.exists() else (
        pd.read_csv(p_score) if p_score.exists() else None)
    if df is not None:
        viz.fig_sensitivity(df)
    p_sec = RESULTS / "sensitivity_secondary.csv"
    _weights_and_k(pd.read_csv(p_sec) if p_sec.exists() else None)

    # No overlays for invalid sensitivity runs: candidate geometry is not a feasible route.

    build_deck_figures(ctx, ev)

    # Moran's I summary for the report: all five features, full grid + thinned
    mor = analysis.moran_table(ctx["F"], grid.mask_land, strides=(1, 2, 3))
    corr_out = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                for k, v in corr.items()}
    json.dump({"moran": mor, "pca": analysis.pca_features(ctx["F"], grid.mask_land),
               "correlation": corr_out},
              open(RESULTS / "spatial_stats.json", "w"), indent=2)
    print("figures written to", viz.FIG)


if __name__ == "__main__":
    main()
