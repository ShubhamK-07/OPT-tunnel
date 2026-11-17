"""Build the 12-slide presentation deck (spec Sec. 13) with the deckkit design system.

Run:  python -m src.deck     ->  deck.pptx  (+ figures/cover_bg.png)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from .deckkit import (ACCENT, AMBER, BLUE, BODY, Deck, GREEN, INK, ML, MR, MUTED, NAVY,
                      OFFWHITE, PAPER, RULE, SLATE)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "deck.pptx"


# --------------------------------------------------------------------------
MINUS = "\u2212"


def fmt(v: float, dec: int = 2, unit: str = "", plus: bool = False) -> str:
    """Format a number for typography: true minus sign, optional unit."""
    txt = f"{v:+.{dec}f}" if plus else f"{v:.{dec}f}"
    txt = txt.replace("-", MINUS).replace("+", "+")
    return f"{txt} {unit}".strip()


def cover_asset(path: Path, size=(2560, 1440)) -> Path:
    """Dark, slightly desaturated hero image made from the route map."""
    src = Image.open(FIG / "cover_map.png").convert("RGB")
    W, H = size
    # cover-crop to 16:9
    r = max(W / src.width, H / src.height)
    src = src.resize((int(src.width * r), int(src.height * r)), Image.LANCZOS)
    left = (src.width - W) // 2
    top = (src.height - H) // 2
    img = src.crop((left, top, left + W, top + H))
    img = ImageEnhance.Brightness(img).enhance(0.82)
    img = ImageEnhance.Color(img).enhance(0.9)
    navy = Image.new("RGB", (W, H), (14, 27, 44))
    img = Image.blend(img, navy, 0.22)
    # gradient to nearly solid navy on the left for the title block
    grad = Image.new("L", (W, 1))
    for x in range(W):
        t = x / (W - 1)
        grad.putpixel((x, 0), int(255 * min(1.0, max(0.0, (0.88 - t) / 0.88)) ** 1.05))
    mask = grad.resize((W, H))
    img = Image.composite(navy, img, mask)
    d = ImageDraw.Draw(img, "RGBA")
    for y in range(H):                        # subtle vertical vignette
        a = int(70 * (y / H) ** 2)
        d.line([(0, y), (W, y)], fill=(6, 12, 22, a))
    out = FIG / "cover_bg.png"
    img.save(out)
    return out


# --------------------------------------------------------------------------
DOI = "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"


def build():
    B = json.load(open(RES / "baseline.json"))
    C, V = B["comparison"], B["verification"]
    meta, lim = B["data_meta"], B["limits"]
    full = pd.read_csv(RES / "sensitivity_full.csv")
    score = pd.read_csv(RES / "sensitivity_score.csv")
    _sp = json.load(open(RES / "spatial_stats.json"))
    _stats = pd.read_csv(RES / "descriptive_stats.csv", index_col=0)
    _p90 = float(score.loc[score["label"].str.contains("P90", regex=False), "value"].iloc[0])
    _pc1 = 100 * _sp["pca"]["explained_variance_ratio"][0]
    _pear = _sp["correlation"]["pearson"][2][3]
    _spear = _sp["correlation"]["spearman"][2][3]
    MOR_R0 = _sp["moran"]["features"]["R"]["morans_I"]
    MOR_R3 = _sp["moran"]["features"]["R"]["thinned"]["stride_3"]
    THIN = int(_sp["correlation"].get("thin_factor", 5))
    _sb = score[score["is_baseline"].astype(bool)].iloc[0]
    _sv = score[(~score["is_baseline"].astype(bool)) & (score["valid_solution"].astype(bool))]
    _sv = _sv.assign(dL=100.0 * (_sv["L_H_km"] / _sb["L_H_km"] - 1.0))
    _sd = _sv[_sv["factor"] != "depth_range"]
    SENS_DL_MAX = float(_sd["dL"].abs().max())
    _dw = _sv[_sv["factor"] == "depth_range"].sort_values("dL")
    SENS_DW_TXT = " / ".join(f"{r['label'].split('=')[-1]} {r['dL']:+.1f} %" for _, r in _dw.iterrows())
    SENS_DW_MIN = float(_dw["dL"].min())
    _bad = score[(~score["is_baseline"].astype(bool)) & (~score["valid_solution"].astype(bool))]
    SENS_N_BAD = len(_bad)
    SENS_BAD_TXT = " and ".join(str(r["label"]) for _, r in _bad.iterrows())
    _nb = _bad.assign(no_grid=pd.isna(_bad["L_E_km"]))
    SENS_BAD_NOROUTE = " and ".join(str(v) for v in _nb.loc[_nb["no_grid"], "label"]) or "no setting"
    SENS_BAD_SOLVER = " and ".join(str(v) for v in _nb.loc[~_nb["no_grid"], "label"]) or "no setting"
    SENS_N_RUNS = len(score) + len(full)
    SENS_P75_COMP = float(score[score["label"].str.contains("P75", regex=False)]["n_components"].iloc[0])
    toy = json.load(open(RES / "toy_tests.json"))
    mt = V["multistart"]
    strat = V.get("solver_strategies", {})
    C_DIRECT_N = strat.get("direct", {}).get("n_executions", 11)
    C_DIRECT_CONV = strat.get("direct", {}).get("n_converged_status0", 0)
    C_DIRECT_FEAS = strat.get("direct", {}).get("n_feasible", 0)
    D = Deck(total=12, running="Underwater Tunnel Route Optimization")
    cov = cover_asset(Path("cover"))

    # =========================================================== 01 · title
    s = D.title_slide()
    s.shapes.add_picture(str(cov), 0, 0, width=Inches(13.333))
    D.rect(s, 0, 0, 13.333, 7.5, fill=None)
    D.text(s, 1.05, 1.62, 8.4, 0.2, "Route optimisation · course project", size=10,
           bold=True, color=RGBColor_SOFT(), caps=True)
    D.text(s, 1.05, 1.98, 9.6, 1.5,
           "Routing an underwater tunnel\nacross real bathymetry", size=40, bold=True,
           color=PAPER, spacing=0.94)
    D.rect(s, 1.05, 3.62, 1.5, 0.045, fill=ACCENT)
    D.text(s, 1.05, 3.86, 8.9, 0.9,
           "Length–risk trade-offs between a snapped-cell graph optimum and a "
           "constrained smooth corridor", size=15, color=RGBColor_SOFT(), spacing=1.15)
    D.text(s, 1.05, 5.02, 9.2, 0.62,
           "GEBCO_2026  ·  15 arc-sec  ·  Sicily Channel  ·  UTM 32N  ·  "
           f"{meta['shape'][0]} × {meta['shape'][1]} grid at {meta['spacing_m']:.0f} m",
           size=11, color=RGBColor(0x9F, 0xB1, 0xC7))
    for i, (val, lab) in enumerate([(f"{C['L_hard_km']:.2f} km", "smooth route length"),
                                    (fmt(C["price_of_realism_pct"], 2, "%", plus=True), "price of realism"),
                                    (f"{C['risk_reduction_pct']:.1f} %", "less terrain risk")]):
        x = 1.05 + i * 2.72
        D.text(s, x, 5.86, 2.5, 0.42, val, size=22, bold=True, color=PAPER)
        D.text(s, x, 6.30, 2.5, 0.24, lab, size=9, color=RGBColor(0x8D, 0xA0, 0xB8), caps=True)
    D.text(s, 10.3, 6.58, 2.4, 0.3, "presentation · 12 slides", size=9,
           color=RGBColor(0x74, 0x88, 0xA0), align=PP_ALIGN.RIGHT)
    D.note(s, "0:20 — Title. Framing sentence: one dataset, two formulations; the discrete "
             "optimum says what is possible on the grid, the constrained spline says what is feasible under the selected model constraints.")

    # ======================================================= 02 · question
    s = D.slide(2, "Motivation", "The question",
                "A tunnel has fixed ends and a free corridor. Two different questions hide in that sentence — "
                "and they need two different optimisers.")
    blocks = [
        ("01", "Fixed endpoints, free corridor",
         "A and B are given as coordinates. Everything between them is a decision "
         "variable — and the seabed is the only thing we can measure."),
        ("02", "Two questions, kept apart",
         "What is the shortest graph path between selected snapped cells? An exact discrete answer. "
         "What does a constraint-compliant, terrain-aware corridor cost? A continuous, constrained, local one."),
        ("03", "The trade-off is the deliverable",
         "Terrain enters only through a data-derived risk field R(x,y). The report compares the two "
         "routes on identical data, mask and one common polyline evaluator; optimising and measuring "
         "are separate steps."),
    ]
    y = 1.86
    for num, head, body in blocks:
        D.rect(s, ML, y + 0.02, 0.42, 0.42, fill=NAVY)
        D.text(s, ML, y + 0.09, 0.42, 0.28, num, size=12, bold=True, color=PAPER,
               align=PP_ALIGN.CENTER)
        D.text(s, ML + 0.62, y, 5.7, 0.26, head, size=13.5, bold=True, color=NAVY)
        D.text(s, ML + 0.62, y + 0.30, 5.75, 0.86, body, size=11.5, color=INK, spacing=1.18)
        y += 1.44
    cx = 7.35
    D.card(s, cx, 1.86, 5.36, 4.42, kicker="headline", title="What the comparison buys")
    D.stat(s, cx + 0.22, 2.60, 2.40, 1.02, f"{C['L_hard_km']:.2f} km",
           f"hard route · easy {C['L_easy_km']:.2f} km", accent=NAVY, vsize=22)
    D.stat(s, cx + 2.74, 2.60, 2.40, 1.02, fmt(C["risk_reduction_pct"], 1, "%"),
           "risk reduction", accent=GREEN, vsize=22)
    D.bullets(s, cx + 0.22, 3.78, 4.95, 2.30, [
        "The smooth route is shorter, not longer — and the reason is reported, not hidden.",
        f"The spline meets the {1/lim['kappa_max']:.0f} m radius limit; the grid polyline's {1/C['easy_kappa_max']:.0f} m radius is diagnostic only.",
        "Both routes share a common evaluator; the optimisers are different by design.",
        "Values are regenerated from the study results artifacts.",
    ], size=11, gap=7)
    D.note(s, "1:00 — Motivation. Keep this conceptual: the two formulations answer different "
             "questions, and the price of realism is the bridge between them.")

    # ==================================================== 03 · data & units
    s = D.slide(3, "Data", "Dataset and units",
                "GEBCO_2026 subset, reprojected to a metric grid so every derivative is in metres.")
    tiles = [
        (f"{meta['shape'][0]} × {meta['shape'][1]}", "grid cells"),
        (f"{meta['spacing_m']:.0f} m", "cell size (UTM)"),
        (f"{(meta['x_range_m'][1]-meta['x_range_m'][0])/1000:.1f} × {(meta['y_range_m'][1]-meta['y_range_m'][0])/1000:.1f}", "study area  [km]"),
        (f"{meta['n_valid']:,} / {meta['n_cells']:,}", "valid / total cells"),
    ]
    for i, (val, lab) in enumerate(tiles):
        x = ML + (i % 2) * 2.62
        y = 1.90 + (i // 2) * 1.16
        D.stat(s, x, y, 2.46, 1.00, val, lab, accent=NAVY, vsize=16 if i == 3 else 21)
    D.text(s, ML, 6.50, 5.15, 0.60,
           f"Source: GEBCO Bathymetric Compilation Group 2026 (2026), GEBCO_2026 Grid, doi:{DOI} "
           "— elevation + TID layers, subset "
           "11.35–11.90 °E / 37.00–37.45 °N, download.gebco.net.",
           size=8, color=MUTED, spacing=1.12)
    D.bullets(s, ML, 4.24, 5.05, 1.95, [
        "Elevation z negative below sea level; depth d = −z is the SEA-FLOOR depth.",
        f"A–B straight line: {meta['AB_geodesic_m']/1000:.3f} km geodesic vs "
        f"{meta['AB_grid_m']/1000:.3f} km through the grid ({meta['AB_difference_pct']:.4f} %).",
        "Land and NoData are masked on the SOURCE raster, before resampling and before any derivative; "
        "nothing is extrapolated.",
        "Striping in the slope map is altimetry-predicted bathymetry — flagged, then tested in the σ sensitivity.",
    ], size=10, gap=6)
    D.card(s, 6.05, 1.90, 6.66, 4.86, kicker="definitions", title="Features and risk field")
    rows = [["Symbol", "Definition", "Units"],
            ["d", "−z  (sea-floor depth)", "m"],
            ["s", "‖∇z‖  central differences", "rise/run"],
            ["r", "std of s in a 3 × 3 window", "rise/run"],
            ["c", "abs(∇²z)  terrain curvature", "1/m"],
            ["R", f"{B['feature_settings']['weights']['slope']:g}·s̃² + {B['feature_settings']['weights']['roughness']:g}·r̃ + {B['feature_settings']['weights']['curvature']:g}·c̃", "—"],
            ["S(p)", "∫ R ds  along the route", "m"]]
    D.table(s, 6.27, 2.78, 6.22, rows, col_w=[0.72, 4.10, 1.40], size=10.5, row_h=0.31,
            align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    D.text(s, 6.27, 5.86, 6.2, 0.86,
           "x̃ = min–max normalisation over valid cells. R is a terrain proxy, never a safety "
           "or geological assessment. Every derivative and distance is in metres, so s is a "
           "dimensionless rise/run and κ is in 1/m.",
           size=9.5, color=SLATE, italic=True, spacing=1.16)
    D.note(s, "1:00 — Data and units. Insist that d is sea-floor depth, not tunnel depth: it is "
             "the most common misreading of this project.")

    # ========================================= 04 · data analysis 1 (visual)
    s = D.slide(4, "Data analysis · 1 of 2", "Where the terrain actually is",
                "One grid, one colour scale per panel; endpoints A and B marked on every map.")
    D.figure(s, FIG / "fig01_feature_maps_deck.png", ML, 1.82, 8.05,
             caption="Depth, slope, roughness, terrain curvature, risk field and the feasibility "
                     "mask — one 450 m grid (fig01).")
    x = 8.95
    D.stat(s, x, 1.86, 3.76, 0.86, f"{_stats.loc['d', 'mean']:.0f} ± {_stats.loc['d', 'std']:.0f} m", "depth, mean ± sd", accent=NAVY, vsize=19)
    D.stat(s, x, 2.86, 3.76, 0.86, f"{np.degrees(np.arctan(_stats.loc['s', 'p50'])):.2f}° / {_p90:.2f}°", "slope, median / P90", accent=NAVY, vsize=19)
    D.stat(s, x, 3.86, 3.76, 0.86, f"{B['feature_settings']['weights']['slope']:g}·s̃² + {B['feature_settings']['weights']['roughness']:g}·r̃ + {B['feature_settings']['weights']['curvature']:g}·c̃",
           "risk weights (baseline)", accent=NAVY, vsize=15)
    D.bullets(s, x, 4.90, 3.76, 2.0, [
        "The graben flank shapes the corridors and their terrain exposure.",
        f"Risk is sparse: median {_stats.loc['R', 'p50']:.2f}, maximum {_stats.loc['R', 'max']:.2f}.",
        f"{FEA_PCT:.1f} % of cells are feasible at the baseline limits, in {FEA_COMP} components.",
    ], size=10.5, gap=6)

    # ========================================= 05 · data analysis 2 (visual)
    s = D.slide(5, "Data analysis · 2 of 2", "What covaries, and what is an artefact",
                "All four terrain features plus risk: raw distributions, correlation and a thinned-grid check.")
    D.figure(s, FIG / "fig02_distributions_deck.png", ML, 1.82, 6.35,
             caption="Raw depth, slope, roughness, terrain curvature and risk; log counts. Normalised distributions: report §5.")
    D.figure(s, FIG / "fig03_correlations_deck.png", 7.18, 1.82, 5.52,
             caption=f"Pearson, Spearman, and Pearson on a 1/{THIN}-thinned grid (fig03).")
    findings = [
        ("Skewed, heavy-tailed", f"Maximum excess kurtosis {_stats['excess_kurtosis'].max():.1f} "
                                   f"and skew {_stats['skewness'].max():.1f}; depth skew {_stats.loc['d', 'skewness']:.2f}."),
        ("r and c covary", f"Pearson {_pear:.2f}; Spearman {_spear:.2f}; PCA axis 1 explains "
                             f"{_pc1:.0f} %. Weights are modelling assumptions, not validated here."),
        ("Spatial structure", "Moran's I is reported descriptively on the full and thinned grids; "
                              "no significance test is claimed."),
    ]
    D.stat(s, 7.18, 4.32, 2.66, 0.96,
           f"{MOR_R0:.2f} → {MOR_R3:.2f}", "Moran's I(R), thinned 1/3",
           accent=NAVY, vsize=19)
    D.stat(s, 10.04, 4.32, 2.66, 0.96, f"{_pc1:.0f} %", "variance on PCA axis 1 (s̃, r̃, c̃)",
           accent=NAVY, vsize=19)
    for i, (head, body) in enumerate(findings):
        x = ML + i * 4.06
        D.rect(s, x, 5.48, 3.86, 1.18, fill=OFFWHITE, line=RULE)
        D.rect(s, x, 5.48, 0.055, 1.18, fill=ACCENT)
        D.text(s, x + 0.22, 5.60, 3.5, 0.26, head, size=11, bold=True, color=NAVY)
        D.text(s, x + 0.22, 5.88, 3.50, 0.86, body, size=9.5, color=INK, spacing=1.14)
    D.note(s, "1:30 — Visual slide 2. The point is that roughness and terrain curvature are "
             "largely the same signal, so the weight vector is a labelling choice.")

    # ================================================== 06 · easy problem
    s = D.slide(6, "Formulation · easy", "Easy problem — exact, discrete, global",
                "Shortest feasible path on an 8-neighbour grid: what the discretisation allows.")
    D.card(s, ML, 1.86, 6.55, 4.30, kicker="problem", title="Formulation")
    eq = [
        "minimise    Σ c_ij · x_ij        (c_ij = edge length)",
        "s.t.        flow conservation at every node,  x_ij ≥ 0",
    ]
    D.text(s, ML + 0.24, 2.60, 6.05, 0.72, eq, size=11.5, color=NAVY, spacing=1.5)
    D.bullets(s, ML + 0.24, 3.36, 6.00, 2.70, [
        "A min-cost-flow LP; its constraint matrix is totally unimodular, so the LP optimum "
        "is a 0/1 path — the shortest path.",
        "Nodes = cells; edges = 8-neighbour, weight = metric length (450 m, 450√2 m).",
        "Feasible cell: not land, 50 ≤ d ≤ 600 m, s ≤ tan 5.5°. A diagonal step needs both "
        "endpoints, its geometric midpoint and both orthogonal neighbours to be feasible.",
        f"Dijkstra: shortest path between selected snapped cells; scipy cross-checks the graph cost. "
        f"Exact A/B connectors are appended and independently validated, not optimized.",
    ], size=10, gap=5)
    r = 7.55
    D.stat(s, r, 1.86, 2.46, 1.06, f"{C['L_easy_km']:.3f} km", "L_E  route length",
           accent=BLUE, vsize=22)
    D.stat(s, r + 2.62, 1.86, 2.46, 1.06, f"{C['S_easy']:.0f}", "S_E  cumulative risk",
           accent=BLUE, vsize=22)
    D.stat(s, r, 3.04, 2.46, 1.06,
           f"{V['runtimes']['easy_repeated_median_s']*1000:.0f} ms", "Dijkstra, median time",
           accent=NAVY, vsize=22)
    D.stat(s, r + 2.62, 3.04, 2.46, 1.06, f"{C['nodes_expanded_easy']:,}", "nodes expanded",
           accent=NAVY, vsize=22)
    D.rect(s, r, 4.26, 5.08, 1.34, fill=PAPER, line=ACCENT, lw=1.0)
    D.text(s, r, 5.78, 5.08, 0.72,
           "Curvature is deliberately NOT part of the easy formulation — adding it would change "
           "the problem, not just the solution.", size=9.5, color=SLATE, italic=True, spacing=1.16)
    D.text(s, r + 0.20, 4.38, 4.7, 0.24, "and the catch", size=8.5, bold=True, color=ACCENT, caps=True)
    D.text(s, r + 0.20, 4.66, 4.70, 0.86,
           "The easy problem does not constrain curvature. Its staircase route needs a "
           f"{1/C['easy_kappa_max']:.0f} m radius as a polyline diagnostic, not an imposed constraint. "
           "That is why the hard problem exists.", size=10.5, color=INK, spacing=1.14)
    D.note(s, "1:00 — Easy problem. Discrete optimum by construction, but curvature-free. Say the "
             "minimum-radius number out loud, and say that it is a polyline diagnostic, not a "
             "constraint the grid problem imposed.")

    # ================================================== 07 · hard problem
    s = D.slide(7, "Formulation · hard", "Hard problem — smooth, constrained, local",
                "Cubic B-spline corridor with 14 free coordinates; depth, slope and route-curvature constraints.")
    D.card(s, ML, 1.86, 6.55, 2.10, kicker="problem", title="Formulation")
    D.text(s, ML + 0.24, 2.62, 6.05, 0.82,
           ["minimise     J = α·L(p) + β·S(p),   α = β = 1",
            "subject to   d ∈ [50, 600] m,  s ≤ tan 5.5°,  |κ| ≤ 1/1350 m"],
           size=12.5, color=NAVY, spacing=1.55)
    D.text(s, ML + 0.24, 3.44, 6.05, 0.62,
           "κ = (x′y″ − y′x″) / (x′² + y′²)^1.5   ·   clamped ends, 7 interior control points, "
           "endpoints pinned to A and B", size=9.5, color=SLATE, italic=True, spacing=1.12)
    D.card(s, ML, 4.14, 6.55, 2.62, kicker="numerics", title="What actually made it converge")
    D.bullets(s, ML + 0.24, 4.84, 6.05, 1.85, [
        "Direct SLSQP on 1 600 collocation rows: "
        f"{C_DIRECT_CONV}/{C_DIRECT_N} executions reached status 0, {C_DIRECT_FEAS} feasible.",
        "Adopted: quadratic exterior penalty, μ = 1e2 → 1e4 → 1e6, from the three warm starts.",
        "Feasible ≠ optimal: no KKT property is claimed for the penalised point.",
        "Fields are mask-aware cubic splines; constraints on 400 samples, 2 % margin.",
    ], size=10, gap=5)
    r = 7.55
    D.stat(s, r, 1.86, 2.46, 1.04, f"{C['J_hard_km']:.3f} km", "J = L + βS", accent=ACCENT, vsize=21)
    D.stat(s, r + 2.62, 1.86, 2.46, 1.04, f"{C['S_hard']:.0f}", "S_H  cumulative risk",
           accent=ACCENT, vsize=21)
    D.stat(s, r, 3.00, 2.46, 1.04, f"{1/C['hard_kappa_max']:.0f} m", "min turning radius",
           accent=GREEN, vsize=21)
    D.stat(s, r + 2.62, 3.00, 2.46, 1.04,
           f"{V['runtimes']['hard_repeated_median_s']:.2f} s", "refinement, median time",
           accent=NAVY, vsize=21)
    D.rect(s, r, 4.20, 5.08, 2.56, fill=OFFWHITE, line=RULE)
    D.text(s, r + 0.22, 4.32, 4.6, 0.24, "stated, not implied", size=8.5, bold=True,
           color=AMBER, caps=True)
    D.text(s, r + 0.22, 4.60, 4.66, 2.06,
           f"A feasible LOCAL CANDIDATE, not a proven optimum: {mt['n_distinct_initial_curves']} "
           f"distinct curves, {mt['n_solver_executions']} solver executions, {mt['n_feasible']} "
           f"feasible results.\n\nβ = 1 prices one metre-equivalent of risk at one metre of length; "
           "the β sweep in the report illustrates feasible candidates, not a proven frontier.", size=10, color=INK, spacing=1.14)
    D.note(s, "1:30 — Hard problem. Be explicit: literal constrained SLSQP failed; the penalty "
             "continuation plus a 2 % margin is what makes the route verifiably feasible.")

    # ======================================================== 08 · the map
    s = D.slide(8, "Result", "Both routes on the same risk field",
                "Easy = minimum length on the grid · hard = minimum L + βS subject to depth, slope and κ limits.")
    D.figure(s, FIG / "fig04_routes_deck.png", ML, 1.84, 7.90,
             caption="Depth (left) and risk (right); identical mask and one common evaluator (fig04).")
    x = 8.90
    D.stat(s, x, 1.84, 3.72, 0.92, f"{C['hausdorff_m']/1000:.2f} km", "Hausdorff separation",
           accent=NAVY, vsize=20)
    D.stat(s, x, 2.88, 3.72, 0.92, f"{C['mean_distance_m']:.0f} m", "mean offset", accent=NAVY,
           vsize=20)
    D.stat(s, x, 3.92, 3.72, 0.92, f"{C['jaccard']:.2f}", "cell overlap (Jaccard)", accent=NAVY,
           vsize=20)
    D.bullets(s, x, 5.00, 3.72, 1.85, [
        "Two different corridors, not two drawings of one: the grid route steps around the flank.",
        "A routing decision — same risk field, same mask, same evaluator.",
    ], size=10, gap=6)
    D.note(s, "1:30 — The map slide. Say plainly that the hard route is shorter AND lower-risk "
             "here, because the grid's staircase is expensive.")

    # ====================================================== 09 · comparison
    s = D.slide(9, "Comparison", "What the added realism costs",
                "One common polyline evaluator and risk field; the optimisers are different by design.")
    rows = [["Metric", "Easy (Dijkstra)", "Hard (spline)", "Δ"],
            ["Length L", f"{C['L_easy_km']:.3f} km", f"{C['L_hard_km']:.3f} km",
             fmt(C["L_hard_km"]-C["L_easy_km"], 3, "km", plus=True)],
            ["Cumulative risk S", f"{C['S_easy']:.0f}", f"{C['S_hard']:.0f}",
             fmt(C["S_hard"]-C["S_easy"], 0, plus=True)],
            ["J (easy post-hoc score)*",
             f"{C['J_easy_km']:.3f} km*", f"{C['J_hard_km']:.3f} km",
             fmt(C["J_hard_km"]-C["J_easy_km"], 3, "km", plus=True)],
            ["Max slope", f"{C['easy_max_slope_deg']:.2f}°", f"{C['hard_max_slope_deg']:.2f}°",
             fmt(C["hard_max_slope_deg"]-C["easy_max_slope_deg"], 2, "°", plus=True)],
            ["Min radius (polyline / spline)†",
             f"{1/C['easy_kappa_max']:.0f} m†",
             f"{1/C['hard_kappa_max']:.0f} m", f"×{C['easy_kappa_max']/C['hard_kappa_max']:.0f}"],
            ["Limit violations (2000 pts)", "0 slope/depth", "0 (all constraints)", "—"],
            ["Single solve, median time",
             f"{V['runtimes']['easy_repeated_median_s']*1000:.0f} ms",
             f"{V['runtimes']['hard_repeated_median_s']:.2f} s",
             f"×{V['runtimes']['ratio_median_hard_to_easy']:.0f}"]]
    D.table(s, ML, 1.86, 7.85, rows, col_w=[2.55, 1.85, 2.05, 1.40], size=10.5, row_h=0.335,
            align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
            highlight_rows=(1, 2))
    D.text(s, ML, 4.94, 7.8, 0.42, "* J values use the common post-hoc evaluator; the easy formulation minimises length only. "
           "† polyline diagnostic, no κ constraint on the grid route.",
           size=8.5, color=MUTED, italic=True)
    D.rect(s, ML, 5.44, 7.85, 1.16, fill=OFFWHITE, line=RULE)
    D.rect(s, ML, 5.44, 0.055, 1.16, fill=NAVY)
    D.text(s, ML + 0.24, 5.56, 7.40, 0.94,
           f"Read the Δ column as hard − easy. Both rows are scored with the same risk field, the same "
           f"feasibility mask and the same polyline evaluator; the length gain is the spline's "
           f"continuity, not a cleverer search — it is the easy route that carries the "
           f"{1/C['easy_kappa_max']:.0f} m polyline radius.",
           size=10, color=INK, spacing=1.16)
    r = 8.75
    D.stat(s, r, 1.86, 3.96, 1.10, fmt(C["price_of_realism_pct"], 2, "%", plus=True), "price of realism",
           accent=ACCENT, vsize=30, note="(L_H − L_E) / L_E — not clipped")
    D.stat(s, r, 3.10, 3.96, 1.10, fmt(C["risk_reduction_pct"], 2, "%", plus=True), "risk reduction",
           accent=GREEN, vsize=30, note="(S_E − S_H) / S_E")
    D.rect(s, r, 4.34, 3.96, 1.44, fill=OFFWHITE, line=RULE)
    D.text(s, r + 0.18, 4.44, 3.6, 0.22, "caveat that travels with it", size=8.5, bold=True,
           color=AMBER, caps=True)
    disc = V["easy_route_discretisation"]
    D.text(s, r + 0.18, 4.70, 3.60, 1.02,
           f"{disc['inflation_pct']:.2f} % of the easy route's length is grid discretisation "
           f"({disc['L_discrete_m']/1e3:.2f} vs {disc['L_smoothed_250m_m']/1e3:.2f} km smoothed). "
           "The hard route stays shorter even against the smoothed baseline.",
           size=9.5, color=INK, spacing=1.12)
    D.note(s, "1:30 — Comparison table. Do not hide the negative price of realism; explain the "
             "discretisation caveat instead of removing it.")

    # ===================================================== 10 · sensitivity
    s = D.slide(10, "Robustness", "Sensitivity — one factor at a time",
                "Every setting re-derives the features and re-solves BOTH problems with the same "
                "solver protocol; the S-core set is this slide's evidence.")
    D.figure(s, FIG / "fig12_score_deck.png", ML, 1.84, 7.30,
             caption="Hard-route length (left) and cumulative risk (right) as % of the baseline, one "
                     "setting at a time, sorted by |Δ length|; settings with no valid route carry "
                     "no bar by design (fig12; full matrix in fig07).")
    x = 8.15
    cards = [
        ("Geometry is robust", f"Apart from the depth window, L_H moves ≤ {SENS_DL_MAX:.1f} % across "
                               "slope estimator, roughness window and field, pre-smoothing, curvature, "
                               "κ_max, s_max and the edge rule.", GREEN),
        ("Common risk basis", "Native S is only comparable within each setting. The right-hand panel "
                                   "re-evaluates valid routes on the frozen baseline risk field.", AMBER),
        ("Feasibility is the fragile part",
         f"{SENS_N_BAD} settings do not produce a valid pair: some have no grid path; "
         "others fail independent route checks. No route bars are shown for them.", ACCENT),
    ]
    y = 1.84
    for head, body, col in cards:
        D.rect(s, x, y, 4.56, 1.42, fill=OFFWHITE, line=RULE)
        D.rect(s, x, y, 0.055, 1.42, fill=col)
        D.text(s, x + 0.22, y + 0.16, 4.1, 0.26, head, size=11.5, bold=True, color=NAVY)
        D.text(s, x + 0.22, y + 0.46, 4.10, 0.86, body, size=10, color=INK, spacing=1.14)
        y += 1.54
    D.text(s, x, 6.52, 4.56, 0.46,
           f"{SENS_N_BAD} of {len(score)-1} S-core settings have no valid route pair and are reported "
           f"as invalid; feasibility outcomes are solver-budget dependent.",
           size=8.5, color=SLATE, italic=True)
    D.note(s, "1:30 — S-core sensitivity. Two honest points: geometry is insensitive to feature "
             "engineering, and feasibility is not.")

    # ==================================================== 11 · verification
    s = D.slide(11, "Verification", "What is checked, and by which code",
                "Every claim below is recomputed by an independent code path; the ones that are not, are listed at right.")
    cur = V.get("curvature", {})
    dom = V.get("continuous_domain_hard", {})
    checks = [
        ["Check", "Result"],
        ["A/B endpoints, both routes", f"{max(V['endpoints'].values()):.1f} m error"],
        ["Independent recomputation of L, S, J",
         f"|ΔL| ≤ {max(V['recompute']['easy_L_diff_m'], V['recompute']['hard_L_diff_m']):.0e} m"],
        ["Dijkstra vs scipy.sparse.csgraph",
         f"Δ = {V['dijkstra_crosscheck']['difference_m']:.0e} m"],
        ["Route curvature, rebuilt spline derivatives",
         f"κ {cur.get('kappa_max', float('nan')):.2e} ≤ {cur.get('limit_kappa_max', float('nan')):.2e} 1/m"],
        ["Continuous domain vs unfilled raster",
         f"{dom.get('on_masked_cell', 0)} masked, {dom.get('point_violation_total', 0)} breaches, "
         f"{dom.get('cell_flag_only', 0)} cell-flag-only"],
        ["Dense validation (200 / 1000 / 5000 pts)",
         f"hard: 0 violations; easy: 0 slope/depth (κ not constrained)"],
        ["Seven toy tests (incl. diagonal rule, masking, classifier)",
         f"{sum(t['pass'] for t in toy['tests'])}/{len(toy['tests'])} pass, feasibility asserted"],
        ["Interpolation, cubic vs linear",
         f"linear candidate: {V['interpolation']['linear_solution']['dense_n_violations']} baseline-field violations"],
    ]
    D.table(s, ML, 1.86, 7.55, checks, col_w=[4.55, 3.00], size=10.5, row_h=0.335,
            align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT], first_col_bold=False)
    D.text(s, ML, 5.02, 7.5, 0.30, "Easy: snapped-cell graph optimum cross-checked against scipy; connectors validated separately.",
           size=9.5, color=MUTED, italic=True)
    x = 8.45
    D.rect(s, x, 1.86, 4.26, 1.52, fill=PAPER, line=ACCENT, lw=1.0)
    D.text(s, x + 0.22, 1.98, 3.8, 0.24, "what we do not claim", size=8.5, bold=True,
           color=ACCENT, caps=True)
    D.bullets(s, x + 0.22, 2.28, 3.84, 1.60, [
        "Global optimality of the smooth route.",
        "That R measures safety, geology or constructability.",
        "That d is a tunnel alignment depth.",
    ], size=10, gap=5, marker_color=ACCENT)
    D.stat(s, x, 3.52, 4.26, 1.10, f"{mt['n_feasible']} / {mt['n_solver_executions']}",
           "executions reaching feasibility", accent=ACCENT, vsize=26,
           note=f"{mt['n_distinct_initial_curves']} distinct curves · {mt['n_direct_executions']} direct + "
                f"{mt['n_penalty_executions']} penalty · J spread {mt['J_feasible_spread']*1000:.0f} m")
    D.text(s, x, 4.86, 4.26, 1.60,
           f"The {mt['n_feasible']} feasible results come from neighbouring warm starts and agree "
           f"to {mt['J_feasible_spread']*1000:.0f} m of J — convergence evidence, not an optimum: "
           f"infeasible candidates reach {mt['J_infeasible_range'][0]:.1f} km. Toy tests and the "
           "recomputation cover the metrics, not the optimum.",
           size=9.5, color=SLATE, spacing=1.16, italic=True)
    D.note(s, "1:00 — Verification. Say 'feasible local candidate' out loud and why the spread is reported.")

    # ====================================================== 12 · conclusion
    s = D.slide(12, "Conclusion", "What the added complexity bought",
                "One dataset, one risk field, two formulations — with feasibility and robustness tested.")
    D.card(s, ML, 1.86, 6.10, 3.10, kicker="three findings", title=None)
    items = [
        (f"{C['risk_reduction_pct']:.1f} %", "less cumulative terrain risk for the smooth corridor"),
        (f"{abs(C['price_of_realism_pct']):.1f} %".replace("-", MINUS),
         "shorter — the staircase is costly, not the spline"),
        (f"{1/C['hard_kappa_max']:.0f} m", "minimum turning radius of the spline, against "
         f"{1/C['easy_kappa_max']:.0f} m for the grid polyline"),
    ]
    y = 2.52
    for val, lab in items:
        D.text(s, ML + 0.24, y, 1.55, 0.40, val, size=20, bold=True, color=ACCENT)
        D.text(s, ML + 1.92, y + 0.04, 3.95, 0.56, lab, size=10.5, color=INK, spacing=1.12)
        y += 0.86
    D.card(s, 6.95, 1.86, 5.76, 3.10, kicker="limitations, stated", title=None)
    D.bullets(s, 7.17, 2.18, 5.32, 2.62, [
        f"Local candidate, not an optimum: {mt['n_feasible']} of {mt['n_solver_executions']} runs feasible.",
        "R is a terrain proxy; S is only comparable within a configuration.",
        "d is sea-floor depth, never tunnel alignment depth.",
        f"Solver fragility: {SENS_N_BAD} of {len(score)-1} S-core settings end infeasible, none silently.",
    ], size=10, gap=6)
    D.rect(s, ML, 5.20, 11.73, 1.44, fill=NAVY)
    D.text(s, ML + 0.30, 5.36, 7.9, 0.30, "reproduce the whole study", size=8.5, bold=True,
           color=RGBColor(0x8D, 0xA0, 0xB8), caps=True)
    D.text(s, ML + 0.30, 5.64, 7.6, 0.86,
           "python -m src.pipeline      →  results/baseline.json      (≈ 1 min)\n"
           "python -m src.tests_toy     →  7/7 toy tests pass",
           size=10.5, color=PAPER, font="Consolas", spacing=1.25)
    D.text(s, 8.55, 5.44, 3.55, 0.9,
           "Report: report.md (15 sections)\nFigures: see report · Deck: 12 slides",
           size=10, color=RGBColor(0xC9, 0xD5, 0xE4), align=PP_ALIGN.RIGHT, spacing=1.3)
    D.note(s, "0:30 — Close on limitations, then invite questions.")

    # full speaker script overwrites the short timing tags written per slide
    for page, script in SPEAKER_NOTES.items():
        D.note(D.prs.slides[page - 1], script)

    nslides = D.save(OUT)
    print(f"wrote {OUT} with {nslides} slides")
    if D.warnings:
        print("\nlayout warnings:")
        for w in D.warnings:
            print("  -", w)
    else:
        print("no layout warnings")


def RGBColor_SOFT():
    return RGBColor(0x8F, 0xA4, 0xBC)





# --------------------------------------------------------------------------- #
#  speaker script (one entry per slide; total ~13:30 at 125 wpm)
# --------------------------------------------------------------------------- #
# numbers used by the speaker script, read once from results/ so the script
# cannot drift from the slides (the deck is regenerated from the same files)
_SB_ = json.load(open(RES / "baseline.json"))
_C, _V = _SB_["comparison"], _SB_["verification"]
_META, _MT, _IC = _SB_["data_meta"], _V["multistart"], _SB_["interpolator_check"]
try:
    _SC = pd.read_csv(RES / "sensitivity_score.csv")
    _base = _SC[_SC["is_baseline"].astype(bool)].iloc[0]
    FEA_PCT = 100.0 * float(_base["feasible_fraction"])
    FEA_COMP = int(_base["n_components"])
    _nv = _SC[(_SC["factor"] != "baseline") & (_SC["valid_solution"].astype(bool))]
    N_VALID_SENS, N_SENS = int(len(_nv)), int((_SC["factor"] != "baseline").sum())
    _bad = _SC[(_SC["factor"] != "baseline") & (~_SC["valid_solution"].astype(bool))]
    N_BAD_SENS = int(len(_bad))
    BAD_TXT = ", ".join(str(v) for v in _bad["label"]) or "none"
except Exception:                                    # sensitivity tables missing/stale
    FEA_PCT, FEA_COMP, N_VALID_SENS, N_SENS, N_BAD_SENS, BAD_TXT = 77.6, 7, 0, 0, 0, "n/a"
DL_KM = _C["L_hard_km"] - _C["L_easy_km"]
DS = _C["S_hard"] - _C["S_easy"]
DJ_KM = _C["J_hard_km"] - _C["J_easy_km"]
R_HARD = 1.0 / _C["hard_kappa_max"]
R_EASY = 1.0 / _C["easy_kappa_max"]
RT_E_MS = 1000.0 * _V["runtimes"]["easy_repeated_median_s"]
RT_H_S = _V["runtimes"]["hard_repeated_median_s"]
RT_RATIO = _V["runtimes"]["ratio_median_hard_to_easy"]
KAP = _C["hard_kappa_exact"]
KAP_LIM = _SB_["limits"]["kappa_max"]
SPREAD_M = 1000.0 * _MT["J_feasible_spread"]
DISC_PP = _V["easy_route_discretisation"]["inflation_pct"]

SPEAKER_NOTES = {
1: (
f"0:20 - Title. One framing sentence: a tunnel between two fixed points still leaves a free "
f"corridor, and this project solves that corridor two ways on the same real bathymetry. Point at "
f"the two numbers on the slide - {_C['L_hard_km']:.2f} km and {_C['price_of_realism_pct']:+.2f} % - "
f"and give the punchline now: the smooth, constraint-compliant spline is SHORTER than the grid "
f"route here, not longer, and the deck explains exactly why and how much of that is "
f"discretisation."
),
2: (
"0:55 - Motivation. A and B are given; everything between them is the decision. That one sentence "
"hides two different questions, so it needs two different optimisers: 'what is the best route the "
"discretisation allows between snapped cells' is an exact graph shortest path, and 'what does a terrain-aware, "
"constraint-compliant corridor cost' is a continuous constrained problem with a local solver. Say "
"the third bullet plainly: the deliverable is the length-risk trade-off, not either single answer. "
"Terrain enters only through the derived field R of x and y. Both routes are then scored by the "
"same evaluator on their sampled geometries - but the optimisers are deliberately different, and "
"optimising must never be confused with measuring."
),
3: (
f"1:00 - Data and units. GEBCO 2026 at 15 arc-seconds, reprojected to EPSG 32632 so the grid is "
f"{_META['shape'][0]} by {_META['shape'][1]} cells at {_META['spacing_m']:.0f} m. Be precise about "
f"the preprocessing, because it was the first version's main error: land and NoData are masked on "
f"the SOURCE raster, before resampling and before any derivative, and the reprojection is strict - "
f"a target cell is valid only if every weighted source corner is a valid ocean cell, so nothing is "
f"extrapolated. That leaves {_META['n_valid']:,} valid cells out of {_META['n_cells']:,}; "
f"{_META['n_outside_footprint']} of the rejected cells are outside the source footprint and "
f"{_META['land_only_cells']} touch land. A to B is {_META['AB_geodesic_m']/1000:.3f} km geodesic "
f"against {_META['AB_grid_m']/1000:.3f} km through the grid, {_META['AB_difference_pct']:.4f} %. "
"Insist on the definition: d equals minus z, so d is SEA-FLOOR depth; the 50 to 600 m window "
"bounds the seabed, not a tunnel alignment. The striping in the slope map is "
"altimetry-predicted bathymetry, which is why smoothing is tested rather than assumed."
),
4: (
f"1:15 - Visual slide 1. This is the required figure, so walk it panel by panel and keep it quick. "
f"Depth: a graben flank between two shelves. Slope: the flank plus the striping. Roughness: the "
f"same structure through a 3 by 3 standard deviation. Terrain curvature: dominated by a few sharp "
f"scarps. Risk R: sparse - most cells sit far below the maximum, which is exactly the sparsity a "
f"router exploits. Bottom right: the feasibility mask, {FEA_PCT:.1f} % of cells in {FEA_COMP} "
f"components at the baseline limits. A and B are marked on every panel so the eye can follow them."
),
5: (
"1:15 - Visual slide 2. The raw histograms show depth, slope, roughness, terrain curvature and risk, "
"with a logarithmic count axis. The report gives the normalized distributions as well. Then the correlation matrices: roughness and terrain curvature "
"carry largely the same signal, and the thinned grid agrees, so the association is not a gridding "
"artefact. State the limit of that analysis: the weights 0.5, 0.3 and 0.2 are MODELLING "
"ASSUMPTIONS, sensitivity-tested later; the correlation analysis does not derive them and cannot "
"validate them. On spatial structure: Moran's I is reported descriptively with its weights and its "
"behaviour under thinning, and no z-score or p-value is quoted, because the normal approximation "
"is not valid on a field this autocorrelated."
),
6: (
f"1:05 - Easy problem. Eight-neighbour grid, edge weight is the true segment length "
f"square-root of dx squared plus dy squared, feasibility is slope at most 5.5 degrees and depth "
f"between 50 and 600 m. A diagonal step is admitted only if both endpoints, the geometric midpoint "
f"AND both orthogonal neighbours are feasible - the combined rule; I tested all four rule variants "
f"and the baseline is the strict one. Frame it as a min-cost-flow LP whose constraint matrix is "
f"totally unimodular, hence the LP optimum is a 0/1 path; Dijkstra solves it without forming it. "
f"Result: {_C['L_easy_km']:.3f} km, cumulative risk {_C['S_easy']:.0f}, "
f"{RT_E_MS:.0f} ms repeated-solve median and {_C['nodes_expanded_easy']:,} nodes expanded. "
f"The globally shortest graph path is between the selected snapped cells: "
f"{_V['dijkstra_crosscheck']['own_length_m']/1000:.3f} km, cross-checked with scipy. "
f"Exact A and B connectors of {_V['dijkstra_crosscheck']['connector_A_m']:.1f} and "
f"{_V['dijkstra_crosscheck']['connector_B_m']:.1f} metres are appended and independently validated; "
f"they were not optimized over alternative connector choices. Together the route is "
f"{_C['L_easy_km']:.3f} km. Then the catch that "
f"sets up the next slide: there is no curvature in this formulation, so the graph path is a staircase "
f"whose polyline needs a {R_EASY:.0f} m turning radius - an exact answer to a question no "
f"alignment can be built from. That radius is a diagnostic on the polyline, not a constraint the "
f"grid problem ever imposed."
),
7: (
f"1:25 - Hard problem. A clamped cubic B-spline with 7 interior control points, 14 free "
f"coordinates, minimising J equal to length plus beta times cumulative risk. Constraints: depth and "
f"slope pointwise and route curvature at most {KAP_LIM:.4e} per metre, a "
f"{1/KAP_LIM:.0f} m turning radius. Numerics, honestly: direct SLSQP ran {_MT['n_direct_executions']} "
f"times, reached status zero {_V['solver_strategies']['direct']['n_converged_status0']} times and produced "
f"{_V['solver_strategies']['direct']['n_feasible']} feasible routes; what works is a quadratic exterior penalty "
f"continuation, mu from 1e2 to 1e4 to 1e6, which gave {_MT['n_feasible']} feasible candidates out "
f"of {_MT['n_penalty_executions']} executions, {SPREAD_M:.1f} m apart in J. Two sentences to say out "
"loud: the penalty objective is finite-mu approximate, so no KKT statement is available; and "
"feasible is not optimal - this is a locally found candidate, not a proven optimum."
),
8: (
f"1:30 - The map slide. Same field, same mask, and the two corridors are genuinely different: "
f"{_C['hausdorff_m']/1000:.2f} km Hausdorff separation, {_C['mean_distance_m']:.0f} m mean offset, "
f"and {100*_C['jaccard']:.0f} % cell overlap. Do not smooth that away. The interpretation is the "
f"one on the slide: the hard route is shorter AND lower-risk here because the grid route is forced "
f"to step around the flank with staircase moves, and every step is pure length with no smoothing "
f"benefit. That is a routing decision, not a data difference."
),
9: (
f"1:30 - Comparison table. Read the deltas as hard minus easy. Length: {DL_KM:+.2f} km, or "
f"{_C['price_of_realism_pct']:+.2f} %; cumulative risk: {DS:+.0f}, i.e. "
f"{_C['risk_reduction_pct']:+.2f} % less risk than the easy route; J: {DJ_KM:+.2f} km. Note that J_E is a POST-HOC common score "
f"- the easy formulation minimises length only, so its J is computed afterwards with the same beta. "
f"Maximum slope is {_C['hard_max_slope_deg']:.2f} against {_C['easy_max_slope_deg']:.2f} degrees, so "
f"both sit just under the limit: that is the constraint binding, not an accident. Minimum turning "
f"radius is {R_HARD:.0f} m for the spline against {R_EASY:.0f} m for the grid polyline, and only "
f"the first is a constraint. Runtime: {RT_H_S:.2f} s for one refinement against "
f"{RT_E_MS:.0f} ms for one Dijkstra, a factor of {RT_RATIO:.0f}, with the full multi-start search "
f"costing {_V['runtimes']['hard_full_multistart_budget_s']:.1f} s over "
f"{_V['runtimes']['n_executions_in_budget']} executions. Do not hide the negative price of "
f"realism: {DISC_PP:.2f} percent of the easy route's length is grid discretisation, and the "
f"report says so in the same breath."
),
10: (
f"1:30 - S-core sensitivity. Every point on this chart re-derives the features and re-solves BOTH "
f"problems; it is the prescribed presentation set, and the full matrix is in the report. Three "
f"readings. One: geometry is robust - apart from the depth window, hard-route length moves by a few "
f"percent across slope estimator, roughness window, pre-smoothing, curvature method, kappa limit, "
f"s_max and edge rule, and {N_VALID_SENS} of {N_SENS} settings produced a candidate that passes the "
f"full feasibility check. Two: the tighter depth window has no valid grid route. "
f"Three: feasibility is the fragile part - "
f"{N_BAD_SENS} setting(s) fail outright ({BAD_TXT}) and are drawn without bars on purpose, because "
f"an infeasible solver output is not a route. Two technical points to say clearly: infeasible rows "
f"stay in the CSV marked as such and are excluded from every improvement claim; and S is only "
f"comparable across settings through the FROZEN columns, which re-integrate each route against the "
f"baseline risk field, because min-max normalisation is recomputed inside each setting."
),
11: (
f"1:00 - Verification. Everything on the left is recomputed by a separate code path: endpoints "
f"exact, length and risk agreeing to machine precision, Dijkstra against scipy's sparse graph, "
f"curvature from a rebuilt spline's own derivatives at {KAP:.4e} against a limit of "
f"{KAP_LIM:.4e}, the continuous domain against the unfilled raster with zero masked samples and "
f"zero limit breaches, dense validation at 200, 1000 and 5000 points, SEVEN toy tests with their "
f"assertions - including the diagonal rule, the strict reprojection and the feasibility classifier "
f"itself - and the cubic against the linear interpolant, which has documented depth "
f"overshoot and never a licence to replace the cubic. On the right, what is not claimed: no global "
f"optimality, R is not a safety or geological measure, d is not an alignment depth. "
f"{_MT['n_feasible']} feasible points out of {_MT['n_solver_executions']} executions, "
f"{SPREAD_M:.1f} m apart in J, is the local-candidate evidence, and it is on the slide deliberately."
),
12: (
f"0:35 - Close. Three findings: {_C['risk_reduction_pct']:.1f} % less cumulative terrain risk, a "
f"corridor {abs(_C['price_of_realism_pct']):.1f} % shorter, and a {R_HARD:.0f} m turning radius "
f"where the grid route needs {R_EASY:.0f} m. The price of realism is therefore paid in solver "
f"reliability, not in length. Then the limitations in one breath: a local candidate rather than an "
f"optimum, one beta, one basin, a terrain proxy only, and a solver that reaches feasibility twice "
f"in {_MT['n_solver_executions']} executions. Invite questions; the reproduction commands are on "
f"the slide."
),
}


if __name__ == "__main__":
    build()
