# Underwater Tunnel Route Optimization

A length–risk trade-off study for a fixed-endpoint tunnel corridor over real GEBCO bathymetry
(Sicily Channel, 15 arc-sec, 50 × 51 km subset, 115 × 113 cells at 450 m).

Two formulations, one risk field, one comparison protocol:

| | Easy problem | Hard problem |
|---|---|---|
| Decision space | cells on an 8-neighbour grid | clamped cubic B-spline, 7 interior control points → **14 free coordinates** |
| Objective | min length (cost = metric length) | min J = α·L + β·S, α = β = 1 |
| Constraints | depth band 50–600 m, slope ≤ 5.5°, combined diagonal rule (midpoint + both orthogonal neighbours) | depth band, slope ≤ 5.5°, **route curvature** \|κ\| ≤ 7.4074e-04 1/m (R ≥ 1350 m) |
| Formulation | min-cost-flow LP (totally unimodular), solved by Dijkstra | 400-sample collocation + 2 % margin, quadratic exterior penalty continuation |
| Optimality | **discrete optimum between selected snapped cells**, cross-checked against scipy; exact A/B connectors appended and validated | **feasible local candidate** — no KKT property, no global claim |
| Result | L 53.434 km, S 3107.8 (J_E 56.542 km, post-hoc score) | L 50.991 km, S 2268.8, J_H 53.260 km |

Headline: price of realism (L_H − L_E)/L_E = **-4.571 %** (the smooth route is shorter),
risk reduction (S_E − S_H)/S_E = **+26.99 %**. The spline satisfies
the 1350 m minimum-radius constraint; the grid polyline's
102 m radius is a diagnostic, not an imposed constraint.
`report.md` §11 explains the discretisation caveat.

Solver accounting: **11 distinct initial curves**, **14 solver executions** — 11 direct SLSQP runs
(0 reached `status 0`, 0 passed independent feasibility checks) and
3 penalty continuations (3 reached `status 0`, 2 passed feasibility checks).
The feasible runs' solver objectives differ by 1.4 m; the adopted candidate
uses the smoothed-grid warm start. Feasibility is not optimality; no KKT or global claim is made.

## Deliverables

| Path | What it is |
|---|---|
| `report.md` | full 15-section report (auto-generated from `results/`, no hand-typed numbers) |
| `report.html` | the same report as a self-contained, print-ready document (cover, contents, boxed figures, every PNG embedded) |
| `deck.pptx` | 12-slide, ~13 min presentation with a per-slide speaker script in the notes |
| `deck_preview/slide_XX.png`, `sheet_N.png` | PIL renders of the deck — layout QA without PowerPoint |
| `results/baseline.json` | baseline metrics, verification, endpoints, decision vector and route-coordinate arrays |
| `results/control_points_hard.csv` | P0=A, optimized interior control points P1–P7, P8=B (UTM metres) |
| `results/route_easy.csv` | exact endpoints and ordered Dijkstra graph nodes (UTM metres) |
| `results/route_hard.csv` | adopted spline sampled at integration parameters (UTM metres) |
| `results/sensitivity_{score,full,secondary,beta}.csv` | one-factor-at-a-time experiments (`score` = the prescribed presentation set) |
| `results/reproduction.log`, `results/submission_audit.txt` | final run log and submission integrity checks |
| `results/runtime_previous_baseline.json` | prior baseline timing observations, retained separately from the final run |
| `results/toy_tests.json` | the seven toy tests (wall, flat plane, bump symmetry, diagonal rule, strict reprojection, interpolation validity, feasibility classifier) |
| `results/descriptive_stats.csv`, `results/spatial_stats.json` | statistics, correlations, Moran's I (full + thinned), PCA |
| `figures/fig01…fig12` (excluding fig08) | maps, distributions, correlations, routes, profiles, Pareto, sensitivity, TID, S-core (report-sized) |
| `figures/*_deck.png`, `cover_map.png` | presentation-sized variants drawn at final on-slide width, plus the axis-free cover map |
| `requirements.txt` | the exact versions that produced these artefacts |

## Run it

```bash
cd tunnel-opt
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  # final reproduction settings
python -m src.gebco_download               # data (already cached in data/raw)
python -m src.data                         # NetCDF -> metric grid  (data/processed/baseline.npz)
python -m src.pipeline                     # baseline + verification      (~1 min)
python -m src.tests_toy                    # 7 toy tests                  (~8 s)
python -m src.sensitivity --set score      # prescribed S-core set        (~6 min)
python -m src.sensitivity --set full       # full factor matrix           (~10 min)
python -m src.sensitivity --set secondary  # weights, control points      (~4 min)
python -m src.sensitivity --set beta       # length-risk Pareto family    (~5 min)
python -m src.figures                      # figures/                     (~45 s)
python -m src.report                       # report.md
python -m src.report_html                  # report.html (self-contained; --link to reference figures/)
python -m src.deck                         # deck.pptx (+ figures/cover_bg.png)
python -m src.deck_preview --sheet         # deck_preview/ renders for layout QA
python -m src.audit_submission             # final integrity checks
```

`src/deck.py` contains content only; every colour, type size, card, stat tile, table and figure
frame comes from `src/deckkit.py`, so the deck cannot drift out of its design system (the
generator prints *no layout warnings* when nothing overflows).

The easy Dijkstra optimum and SciPy cross-check concern the path between the selected snapped
endpoint cells. Exact A/B connectors are appended and the complete route is independently
validated; alternative connector choices are not optimized. `comparison.csv` explicitly separates
the original baseline graph-search time, repeated easy-solve median, adopted hard continuation,
repeated hard refinement median and full multi-start search budget. The baseline easy graph timer
excludes route assembly while the repeated easy-solve timer includes it.

## Method notes worth knowing before reading the numbers

* `d` is the **GEBCO sea-floor depth** (`d = −z`). `d_min`/`d_max` bound the seabed depth along
  the corridor — they are **not** tunnel alignment depths.
* `R(x,y) = 0.5·s̃² + 0.3·r̃ + 0.2·c̃` with min–max normalisation over valid cells; `S(p) = ∫ R ds`.
  Risk is a **terrain proxy**, not a safety, geological or geotechnical assessment. Because the
  normalisation is recomputed per configuration, native `S` is only comparable within one configuration.
  Cross-configuration comparisons use routes re-evaluated on the **frozen baseline risk field**.
* Terrain curvature `c` (Laplacian of elevation) is never used as route curvature, and route
  curvature κ is never used as a terrain feature.
* **Feasible ≠ optimal.** The penalised objective is a finite-μ approximation of the constrained
  one: crossing a constraint costs a finite amount, no KKT multiplier is available, and the
  result is reported as a feasible local candidate.
* Constraints are enforced on 400 collocation samples with a **2 % margin** and validated against
  the *nominal* limits at 2000–5000 samples; J is integrated on the same active 400-point grid.
  Although `n_samples` is configured as 200, the implementation sets the active grid to the
  400 constraint samples for both objective and constraints. Both choices are
  measured in `report.md` §9, not assumed.
* Everything is reproducible from `config.yaml`; the seed is 20261110.

## Layout

```
tunnel-opt/
  config.yaml          all parameters (single source of truth)
  src/                 gebco_download, data, features, easy, hard, metrics, analysis,
                       pipeline, sensitivity, tests_toy, viz, figures, report, report_html,
                       deckkit, deck, deck_preview
  data/raw             GEBCO_2026 NetCDF subsets + TID
  data/processed       baseline.npz + metadata
  results/             JSON/CSV artefacts (the report reads only these)
  figures/             PNG figures
```

Dataset: GEBCO Compilation Group (2026), *GEBCO_2026 Grid*, 15 arc-sec,
doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa, subset 11.35–11.90 °E / 37.00–37.45 °N,
<https://download.gebco.net>.
