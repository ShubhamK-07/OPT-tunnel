"""All figures (spec Sec. 12).  Matplotlib only; metric (UTM) axes in km."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": False,
    "figure.facecolor": "white",
})
C_EASY = "#1f6feb"
C_HARD = "#d1242f"
C_AB = "#111111"

from contextlib import contextmanager


@contextmanager
def deck_style(font: float = 11.0):
    """Presentation typography.

    Deck figures are drawn at (roughly) their final on-slide width, so a 10-11 pt
    font in the figure is a 10-11 pt font on the slide.  Report figures are drawn
    much larger and would be unreadable when scaled down.
    """
    old = dict(plt.rcParams)
    plt.rcParams.update({
        "font.size": font, "axes.titlesize": font + 1.5, "axes.labelsize": font,
        "xtick.labelsize": font - 1, "ytick.labelsize": font - 1,
        "legend.fontsize": font - 1.5, "axes.grid": False,
        "savefig.dpi": 220, "figure.dpi": 220,
    })
    try:
        yield
    finally:
        plt.rcParams.update(old)


def fig_cover_map(ctx, path=None):
    """Clean dark route map for the title slide: no axes, no titles, no legends."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    grid, F = ctx["grid"], ctx["F"]
    cmap = LinearSegmentedColormap.from_list(
        "night", ["#16283F", "#28496F", "#4E4472", "#96524C", "#D89763"])
    fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(F["R"], origin="lower", extent=_extent_km(grid), cmap=cmap, aspect="auto",
              interpolation="bilinear")
    ax.imshow(np.where(grid.mask_land, 1.0, np.nan), origin="lower", extent=_extent_km(grid),
              cmap="Greys", aspect="auto", alpha=0.5)
    easy_xy = np.asarray(ctx["easy_xy"] if "easy_xy" in ctx else ctx["ez"].xy, float)
    hard_xy = np.asarray(ctx["hard_xy"] if "hard_xy" in ctx else ctx["hard_best"].xy, float)
    for xy, col in ((easy_xy, "#9CC6FF"), (hard_xy, "#FF9A8B")):
        ax.plot(xy[:, 0] / 1e3, xy[:, 1] / 1e3, "-", color=col, lw=9.0, alpha=0.16,
                solid_capstyle="round")
        ax.plot(xy[:, 0] / 1e3, xy[:, 1] / 1e3, "-", color=col, lw=3.2, alpha=1.0,
                solid_capstyle="round")
    ax.set_xlim(_extent_km(grid)[0], _extent_km(grid)[1])
    ax.set_ylim(_extent_km(grid)[2], _extent_km(grid)[3])
    ax.axis("off")
    out = Path(path) if path else FIG / "cover_map.png"
    fig.savefig(out, facecolor="#0B1622", pad_inches=0)
    plt.close(fig)
    return out


def _extent_km(grid):
    return [grid.x[0] / 1e3, grid.x[-1] / 1e3, grid.y[0] / 1e3, grid.y[-1] / 1e3]


def _map(ax, grid, arr, title, cmap, label, vmin=None, vmax=None):
    im = ax.imshow(arr, origin="lower", extent=_extent_km(grid), cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("UTM 32N easting [km]")
    ax.set_ylabel("northing [km]")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    return im


def _mark_ab(ax, grid, A, B):
    ax.plot(A[0] / 1e3, A[1] / 1e3, "o", ms=8, mfc="white", mec=C_AB, mew=1.6, zorder=6)
    ax.plot(B[0] / 1e3, B[1] / 1e3, "s", ms=8, mfc="white", mec=C_AB, mew=1.6, zorder=6)
    ax.annotate("A", (A[0] / 1e3, A[1] / 1e3), textcoords="offset points",
                xytext=(8, 4), fontweight="bold", zorder=7)
    ax.annotate("B", (B[0] / 1e3, B[1] / 1e3), textcoords="offset points",
                xytext=(8, 4), fontweight="bold", zorder=7)


# --------------------------------------------------------------------------
def fig_maps(ctx, fields_stats=None, suffix="", figsize=None):
    grid, F, A, B = ctx["grid"], ctx["F"], ctx["A"], ctx["B"]
    specs = [
        ("depth d [m]", grid.d, "viridis_r"),
        ("slope s [deg]", F["s_deg"], "magma"),
        ("roughness r [m/m]", F["r"], "cividis"),
        ("terrain curvature |c| [1e-4 1/m]", F["c"] * 1e4, "pink_r"),
        ("terrain risk R [-]", F["R"], "inferno"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize or (13.5, 8.4))
    for ax, (title, arr, cmap) in zip(axes.ravel(), specs):
        _map(ax, grid, arr, title, cmap, title.split("[")[-1].rstrip("]"))
        _mark_ab(ax, grid, A, B)
    ax = axes.ravel()[5]
    ax.axis("off")
    feas = ctx["feas"]
    # 1 -> black (feasible), 0 -> white (masked / infeasible)
    ax.imshow((~feas).astype(float), origin="lower", extent=_extent_km(grid), cmap="Greys",
              aspect="equal", vmin=0, vmax=1)
    if suffix == "_deck":
        ax.set_title("infeasible cells")
    else:
        ax.set_title("infeasible cells (black = violates depth/slope/mask)")
        ax.text(0.02, 0.02, f"feasible fraction {feas.mean():.1%}", transform=ax.transAxes,
                fontsize=8, color="black",
                bbox=dict(fc="white", alpha=0.7, ec="none"))
    fig.suptitle("Baseline feature maps (GEBCO_2026 subset, UTM 32N, 450 m grid)"
                 if suffix != "_deck" else "Feature maps, one grid", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / f"fig01_feature_maps{suffix}.png")
    plt.close(fig)


def fig_deck_terrain_distributions(F, mask):
    """Raw distributions of all four terrain features plus risk on slide 5."""
    keys = [("d", "Depth d [m]"), ("s", "Slope s [m/m]"),
            ("r", "Roughness r [m/m]"), ("c", "Terrain curvature |c| [1/m]"),
            ("R", "Terrain risk R [–]")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.2))
    for ax, (key, label) in zip(axes.flat, keys):
        values = F[key][~mask & np.isfinite(F[key])]
        ax.hist(values, bins=50, color="#4c72b0", alpha=.88)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9.0, pad=3)
        ax.tick_params(labelsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .68, "Raw values\nLog count scale\n\nNormalized versions:\nsee report §5",
                       transform=axes.flat[-1].transAxes, fontsize=8, va="top", color="#526079")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_distributions_deck.png")
    plt.close(fig)


def fig_distributions(F, mask, suffix="", figsize=None, only=None):
    keys = [("d", "water depth d [m]", "depth"), ("s", "slope s [m/m]", "slope"),
            ("r", "roughness r [m/m]", "roughness"), ("c", "curvature |c| [1/m]", "curvature"),
            ("R", "terrain risk R [-]", "risk")]
    if only:
        keep = set(only)
        keys = [k for k in keys if k[2] in keep]
    deck = suffix == "_deck"
    fig, axes = plt.subplots(2, len(keys), figsize=figsize or (15, 6), squeeze=False)
    for ax, (k, lab, short) in zip(axes[0], keys):
        v = F[k][~mask & np.isfinite(F[k])]
        ax.hist(v, bins=60, color="#4c72b0", alpha=0.85)
        if deck:
            ax.set_title(f"{lab}\nmean {v.mean():.4g}  skew {_skew(v):.2f}", fontsize=9.5, pad=3)
        else:
            ax.set_title(f"{lab}\nmean {v.mean():.4g} | skew {_skew(v):.2f}")
        ax.set_yscale("log")
        ax.set_xlabel(lab if not deck else ("raw " + short))
    for ax, (k, lab, short) in zip(axes[1], keys):
        nk = {"d": None, "s": "s_n", "r": "r_n", "c": "c_n", "R": None}[k]
        if nk is None:
            v = F[k][~mask & np.isfinite(F[k])]
            v = (v - v.min()) / (np.ptp(v) + 1e-12)
            ttl = f"normalised {short}"
        else:
            v = F[nk][~mask & np.isfinite(F[nk])]
            ttl = f"{short} normalised (min-max)"
        ax.hist(v, bins=60, color="#dd8452", alpha=0.85)
        ax.set_title(f"normalised {short}" if deck else ttl,
                     fontsize=9.5 if deck else None, pad=3)
        ax.set_yscale("log")
    if not deck:
        fig.suptitle("Feature distributions, raw (top) and normalised (bottom)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / f"fig02_distributions{suffix}.png")
    plt.close(fig)


def _skew(v):
    m, s = v.mean(), v.std()
    return float(np.mean(((v - m) / s) ** 3)) if s > 0 else 0.0


def fig_correlations(corr, suffix="", figsize=None):
    fig, axes = plt.subplots(1, 3, figsize=figsize or (13, 3.9))
    labels = corr["labels"]
    deck = suffix == "_deck"
    for ax, key, title in ((axes[0], "pearson", "Pearson" if deck else "Pearson (all valid cells)"),
                           (axes[1], "spearman", "Spearman" if deck else "Spearman (all valid cells)"),
                           (axes[2], "pearson_thinned",
                            (f"Pearson, 1/{corr['thin_factor']}-thinned grid" if deck
                             else f"Pearson, thinned 1/{corr['thin_factor']}"))):
        M = np.asarray(corr[key], float)
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6.4 if deck else 8,
                            color="black" if abs(M[i, j]) < 0.6 else "white")
        ax.set_title(title, fontsize=10 if deck else None, pad=4 if deck else None)
        ax.tick_params(labelsize=8.5 if deck else None)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7.5 if deck else None)
    if not deck:
        fig.suptitle("Correlation of d, s, r, c  (association, not causation)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f"fig03_correlations{suffix}.png")
    plt.close(fig)


def fig_routes(ctx, ev, suffix="", figsize=None):
    grid, F, A, B = ctx["grid"], ctx["F"], ctx["A"], ctx["B"]
    fig, axes = plt.subplots(1, 2, figsize=figsize or (13.5, 5.6))
    for ax, (arr, title, cmap, lab) in zip(axes, [
            (grid.d, "routes on depth", "viridis_r", "depth [m]"),
            (F["R"], "routes on terrain risk", "inferno", "R [-]")]):
        _map(ax, grid, arr, title, cmap, lab)
        _mark_ab(ax, grid, A, B)
        ax.plot(ev["xy_easy"][:, 0] / 1e3, ev["xy_easy"][:, 1] / 1e3, "-",
                color=C_EASY, lw=1.6, label="easy (Dijkstra, min length)")
        ax.plot(ev["xy_hard"][:, 0] / 1e3, ev["xy_hard"][:, 1] / 1e3, "-",
                color=C_HARD, lw=2.2, label="hard (B-spline + SLSQP, J = L + beta S)")
        ax.legend(loc="lower left", fontsize=8 if suffix else 9, framealpha=0.9,
                  labels=["easy — Dijkstra, min length", "hard — B-spline, J = L + βS"])
    t = ev["table"]
    if suffix != "_deck":
        fig.suptitle(
            f"Easy: L={t['L_easy_km']:.2f} km, S={t['S_easy']:.0f}  |  "
            f"Hard: L={t['L_hard_km']:.2f} km, S={t['S_hard']:.0f}  |  "
            f"price of realism {t['price_of_realism_pct']:+.1f} %, "
            f"risk reduction {t['risk_reduction_pct']:.1f} %", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / f"fig04_routes{suffix}.png")
    plt.close(fig)


def fig_constraint_profiles(ctx, ev):
    """Constraint profiles along the hard route (and the easy route reference)."""
    from . import metrics
    grid, lim, fields = ctx["grid"], ctx["lim"], ctx["fields"]
    n = 2000
    out = []
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 9.5), sharex=True)
    for xy, lab, col in ((ev["xy_easy"], "easy", C_EASY), (ev["xy_hard"], "hard", C_HARD)):
        L, S, xys = metrics.polyline_terms(xy, fields, M=n)
        X, Y = xys[:, 0], xys[:, 1]
        d = fields("d", X, Y)
        s = np.degrees(np.arctan(fields("s", X, Y)))
        kap = metrics.curvature_along(xys, smooth_m=250.0)
        prog = np.linspace(0, 1, n)
        axes[0].plot(prog, d, color=col, lw=1.6, label=lab)
        axes[1].plot(prog, s, color=col, lw=1.6, label=lab)
        axes[2].plot(prog, np.abs(kap), color=col, lw=1.6, label=lab)
        axes[3].plot(prog, fields("R", X, Y), color=col, lw=1.6, label=lab)
    axes[0].axhline(lim["d_max"], color="k", ls="--", lw=1)
    axes[0].axhline(lim["d_min"], color="k", ls="--", lw=1)
    axes[0].set_ylabel("seabed depth [m]")
    axes[0].set_title("Hard-route constraint profiles (dashed = limits)")
    axes[1].axhline(lim["s_max_deg"], color="k", ls="--", lw=1)
    axes[1].set_ylabel("slope [deg]")
    axes[2].axhline(lim["kappa_max"], color="k", ls="--", lw=1)
    axes[2].set_ylabel("|kappa| [1/m]")
    # linear axis: the easy route's curvature is numerical noise on straight
    # runs and ~1e-2 at its corners, so a log axis would be pure spike noise
    axes[2].set_ylim(0, 1.15 * max(lim["kappa_max"] * 1.2, 0.011))
    axes[2].text(0.012, 0.86, "easy route is not constrained on kappa (informational)",
                 transform=axes[2].transAxes, fontsize=8, color=C_EASY)
    axes[3].set_ylabel("terrain risk R [-]")
    axes[3].set_xlabel("normalised arc length along the route")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_constraint_profiles.png")
    plt.close(fig)


def fig_pareto(points, baseline=None):
    """points: list of (beta, L_km, S) for the hard problem."""
    betas = [p[0] for p in points]
    L = [p[1] for p in points]
    S = [p[2] for p in points]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.plot(L, S, "-", color="#999999", lw=1, zorder=2)
    ax.scatter(L, S, color=C_HARD, s=45, zorder=3)
    for b, l, s_ in zip(betas, L, S):
        ax.annotate(f"beta={b:g}", (l, s_), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("route length L_H [km]")
    ax.set_ylabel("terrain risk S_H")
    ax.set_title("Length-risk trade-off: each point is one beta")
    ax = axes[1]
    ax.plot(betas, L, "o-", label="L_H [km]")
    ax.set_xscale("symlog", linthresh=1)
    ax2 = ax.twinx()
    ax2.plot(betas, S, "s--", color=C_HARD, label="S_H")
    ax.set_xlabel("beta")
    ax.set_ylabel("L_H [km]", color=C_EASY)
    ax2.set_ylabel("S_H", color=C_HARD)
    ax.set_title("Objective terms versus beta")
    if baseline is not None:
        ax.axvline(baseline, color="k", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_pareto.png")
    plt.close(fig)


def fig_sensitivity(df, baseline_row=None, suffix="", size=None):
    """Effect of each setting relative to the baseline (tornado-style).

    One row per setting, three panels: change in hard-route length, change in
    hard terrain risk, and change in the feasible fraction.  This is the honest
    way to show one-factor-at-a-time results: length barely moves, risk and
    feasibility do.
    """
    import pandas as pd
    base = df[df["factor"] == "baseline"].iloc[0]
    d = df[(df["factor"] != "baseline") & (df["factor"] != "baseline")].copy()
    d = d.drop_duplicates(subset=["label"]).reset_index(drop=True)
    d["dL"] = 100.0 * (d["L_H_km"] / base["L_H_km"] - 1.0)
    d["dS"] = 100.0 * (d["S_H_frozen"] / base["S_H_frozen"] - 1.0)
    d["dF"] = 100.0 * (d["feasible_fraction"] - base["feasible_fraction"])
    d.loc[~d["valid_solution"].astype(bool), ["dL", "dS", "dF"]] = np.nan
    d["lab"] = d["label"].str.replace("_", " ", regex=False)
    d = d.iloc[::-1].reset_index(drop=True)          # bottom-up

    figsize = size or (14.5, 0.34 * len(d) + 1.9)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    y = np.arange(len(d))
    colors = {"slope_method": "#4c72b0", "roughness_window": "#dd8452",
              "smooth_sigma_cells": "#55a868", "s_max_deg": "#c44e52",
              "kappa_max": "#8172b3", "curvature_method": "#937860",
              "roughness_field": "#da8bc3", "depth_range": "#8c8c8c",
              "edge_rule": "#ccb974", "normalization": "#64b5cd"}
    for ax, col, title, unit in (
            (axes[0], "dL", "change in hard-route length", "%"),
            (axes[1], "dS", "change in hard risk S (frozen baseline field)", "%"),
            (axes[2], "dF", "change in feasible fraction", "pp")):
        ax.barh(y, d[col], color=[colors.get(f, "#888888") for f in d["factor"]], alpha=0.9)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{title} [{unit}]", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        lo, hi = float(min(0.0, d[col].min())), float(max(0.0, d[col].max()))
        pad = max(0.16 * (hi - lo), 0.8)
        ax.set_xlim(lo - pad, hi + pad)
        for yy, v in zip(y, d[col]):
            if not np.isfinite(v):
                ax.text(0, yy, "infeasible / no valid pair", va="center", fontsize=6.5, color=C_HARD)
                continue
            ax.text(v + (0.02 if v >= 0 else -0.02) * (hi - lo + 1e-9) * 1.6, yy,
                    f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    axes[0].set_yticks(y, d["lab"], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors.keys()), loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sensitivity: each setting changed one at a time, everything else at baseline",
                 y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    fig.savefig(FIG / f"fig07_sensitivity{suffix}.png")
    plt.close(fig)


def fig_sensitivity_routes(routes, title="Sensitivity route overlays"):
    """routes: dict label -> xy array (plotted on the baseline risk map)."""
    grid, F, A, B = routes["_grid"], routes["_F"], routes["_A"], routes["_B"]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    _map(ax, grid, F["R"], title, "inferno", "R [-]")
    _mark_ab(ax, grid, A, B)
    for lab, xy in routes.items():
        if lab.startswith("_"):
            continue
        ax.plot(xy[:, 0] / 1e3, xy[:, 1] / 1e3, lw=1.7, label=lab, alpha=0.9)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_sensitivity_routes.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  deck S-core summary (purpose-built, not a scaled-down report figure)
# --------------------------------------------------------------------------- #
_SCORE_LABELS = {
    "slope_method=central": "slope: central (baseline)",
    "slope_method=horn": "slope: Horn",
    "slope_method=sobel": "slope: Sobel",
    "curvature_method=laplacian": "curvature: Laplacian (baseline)",
    "curvature_method=hessian": "curvature: Hessian",
    "roughness_window=3": "roughness window 3 (baseline)",
    "roughness_window=5": "roughness window 5",
    "roughness_window=7": "roughness window 7",
    "roughness_window=9": "roughness window 9",
    "roughness_field=depth": "roughness of depth",
    "smooth_sigma_cells=1.0": "pre-smoothing σ = 1",
    "smooth_sigma_cells=2.0": "pre-smoothing σ = 2",
    "smooth_sigma_cells=3.0": "pre-smoothing σ = 3",
    "s_max_deg=3.33 deg (P75)": "s_max = P75",
    "s_max_deg=5.48 deg (P90)": "s_max = P90",
    "s_max_deg=7.17 deg (P95)": "s_max = P95",
    "depth_range=80-450 m": "depth window 80–450 m",
    "kappa_max=1/600 m": "κ_max → R_min = 600 m",
    "kappa_max=1/338 m": "κ_max → R_min = 338 m",
    "edge_rule=node": "edge rule: node only",
    "edge_rule=midpoint": "edge rule: midpoint",
    "edge_rule=corner": "edge rule: corner",
    "curvature_method=laplacian": "curvature: Laplacian (baseline)",
}


def fig_score_deck(df, suffix="_deck"):
    """Two-panel S-core summary: hard-route length and cumulative risk vs baseline.

    Rows without a valid easy/hard pair (no feasible A->B route exists at that
    setting) are shown with a red label and no bar -- an infeasible solver output
    is not a route and must not be plotted as one.
    """
    d = df.drop_duplicates(subset=["label"]).copy()
    base = d[d["is_baseline"].astype(bool)].iloc[0]
    # Percentile thresholds come from data and may vary with preprocessing.
    for percentile in ("P75", "P90", "P95"):
        for label in d.loc[d["label"].str.contains(percentile, regex=False), "label"]:
            _SCORE_LABELS[label] = "s_max = " + percentile
    d = d[d["label"].isin(_SCORE_LABELS)].copy()
    if d.empty:
        raise ValueError("no S-core rows matched the label map")
    d["name"] = d["label"].map(_SCORE_LABELS)
    if "valid_solution" in d.columns:
        d["valid"] = d["valid_solution"].astype(bool)
    else:
        d["valid"] = True
    d["dL"] = np.where(d["valid"], 100.0 * (d["L_H_km"] - base["L_H_km"]) / base["L_H_km"], np.nan)
    d["dS"] = np.where(d["valid"], 100.0 * (d["S_H_frozen"] - base["S_H_frozen"]) / base["S_H_frozen"], np.nan)
    # valid rows first (smallest |change| at the bottom), infeasible rows last
    order = d.assign(k=np.where(d["valid"], d["dL"].abs(), np.inf),
                     k2=np.where(d["valid"], 0, 1)).sort_values(["k2", "k"])
    d = order.drop(columns=["k", "k2"]).reset_index(drop=True)

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.35, 4.62), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.08})
    y = np.arange(len(d))
    for ax, col, xlab in ((a1, "dL", "Δ hard-route length [%]"),
                          (a2, "dS", "Δ risk S_H, frozen field [%]")):
        v = d[col].to_numpy()
        ok = np.isfinite(v)
        colr = [C_EASY if x < 0 else C_HARD for x in np.nan_to_num(v)]
        ax.barh(y[ok], v[ok], height=0.62, color=np.array(colr)[ok], alpha=0.92, zorder=3)
        ax.axvline(0.0, color="#1b2733", lw=1.1, zorder=4)
        ax.grid(axis="x", color="#dfe3e8", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlab)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        finite = v[ok]
        if finite.size:
            span = max(finite.max(), 1e-6)
            neg = max(abs(min(finite.min(), 0.0)), 0.35 * span)
            ax.set_xlim(-neg * 1.35, span * 1.30)
            lim = span * 1.30
            for yi, xi in zip(y[ok], finite):
                ax.text(xi + (0.02 * lim if xi >= 0 else -0.02 * lim), yi, f"{xi:+.1f}",
                        va="center", ha="left" if xi >= 0 else "right", fontsize=8.2,
                        color="#1b2733", zorder=5)
    a1.set_yticks(y)
    a1.set_yticklabels(d["name"], fontsize=8.6)
    for yi, valid in zip(y, d["valid"].to_numpy()):
        if not valid:
            a1.get_yticklabels()[yi].set_color(C_HARD)
    fig.subplots_adjust(left=0.315, right=0.985, top=0.955, bottom=0.145)
    fig.savefig(FIG / f"fig12_score{suffix}.png")
    plt.close(fig)
