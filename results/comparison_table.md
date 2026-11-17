# Comparison table (regenerated from results/baseline.json)

| Metric | Easy (Dijkstra) | Hard (clamped B-spline) |
|---|---|---|
| Length L | 53.434 km | 50.991 km |
| Cumulative risk S | 3107.8 | 2268.8 |
| J = L + beta*S (post-hoc common score, easy) | 56.542 km |  |
| J = alpha*L + beta*S (hard objective) |  | 53.260 km |
| Max slope | 5.408 deg | 5.390 deg |
| Max route curvature (exact spline, hard) | 9.770e-03 1/m (polyline estimate, no limit) | 7.204e-04 1/m |
| Min turning radius | 102 m | 1388 m (exact) / 1723 m (polyline diag.) |
| Limit violations (dense) | 0 slope/depth (+133 kappa, informational) | 0 slope/depth, 0 kappa (polyline diag.) |
| Price of realism (L_H - L_E)/L_E | -4.57 % |  |
| Risk reduction (S_E - S_H)/S_E | +26.99 % |  |
| Hausdorff / mean distance | 3238 m / 847 m |  |
| Cell overlap (Jaccard) | 0.108 |  |

* post-hoc common score J_E = L_E + beta*S_E; the easy formulation minimises L only
* min alpha*L + beta*S (the hard formulation's own objective)
* the hard OPTIMISER integrates J with the trapezoid rule on 400 spline samples (the same active grid as the collocation constraints); the reported L, S and J of both routes are recalculated afterwards by the same polyline evaluator (segment sums, midpoint field values)
