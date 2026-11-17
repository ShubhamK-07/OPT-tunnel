"""Deep data analysis of the derived features (spec Sec. 8).

Statistics, distributions, correlations, Moran's I and PCA on the baseline
fields.  Everything is computed on valid (unmasked) cells unless stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["d", "s", "r", "c", "R"]
UNITS = {"d": "m", "s": "m/m (rise/run)", "r": "m/m (std of slope)",
         "c": "1/m", "R": "dimensionless"}


# --------------------------------------------------------------------------
# descriptive statistics
# --------------------------------------------------------------------------
def describe(arr: np.ndarray, mask: np.ndarray) -> dict:
    v = arr[~mask & np.isfinite(arr)].astype(float)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    m, sd = v.mean(), v.std()
    skew = float(np.mean(((v - m) / sd) ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(((v - m) / sd) ** 4) - 3.0) if sd > 0 else 0.0
    return {
        "count": int(v.size),
        "missing": int(np.sum(mask | ~np.isfinite(arr))),
        "mean": float(m), "std": float(sd), "min": float(v.min()),
        "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]), "max": float(v.max()),
        "skewness": skew, "excess_kurtosis": kurt,
    }


def stats_table(fields: dict, mask: np.ndarray) -> pd.DataFrame:
    rows = {}
    for k in FEATURES:
        st = describe(fields[k], mask)
        st["unit"] = UNITS[k]
        rows[k] = st
    df = pd.DataFrame(rows).T
    return df


# --------------------------------------------------------------------------
# correlations
# --------------------------------------------------------------------------
def correlation_matrices(fields: dict, mask: np.ndarray, thin: int = 5):
    keys = ["d", "s", "r", "c"]
    data = {k: fields[k][~mask & np.isfinite(fields[k])] for k in keys}
    # a shared valid mask so the four samples line up
    valid = ~mask
    for k in keys:
        valid &= np.isfinite(fields[k])
    X = np.column_stack([fields[k][valid] for k in keys])
    pear = np.corrcoef(X, rowvar=False)
    spear = np.corrcoef(pd.DataFrame(X).rank().to_numpy(), rowvar=False)
    thin_sl = (slice(None, None, thin), slice(None, None, thin))
    Xt = np.column_stack([fields[k][valid][thin_sl] if False else
                          fields[k][thin_sl][valid[thin_sl]] for k in keys])
    pear_thin = np.corrcoef(Xt, rowvar=False) if Xt.shape[0] > 10 else np.full((4, 4), np.nan)
    return {
        "labels": keys,
        "pearson": pear, "spearman": spear, "pearson_thinned": pear_thin,
        "n_cells": int(X.shape[0]), "thin_factor": thin,
    }


# --------------------------------------------------------------------------
# spatial autocorrelation
# --------------------------------------------------------------------------
def morans_i(arr: np.ndarray, mask: np.ndarray, max_cells: int = 20000, seed: int = 0,
             grid_stride: int = 1) -> dict:
    """Moran's I with a 4-neighbour (rook) weight matrix on valid cells.

    grid_stride > 1 thins the grid itself (every n-th row and column) and reruns
    the statistic.  This is the "thinned-sample check": if I is an artefact of
    the 450 m resolution, it drops sharply when the grid is thinned; if the
    field is genuinely spatially structured, I stays high.
    """
    if grid_stride > 1:
        arr = arr[::grid_stride, ::grid_stride]
        mask = mask[::grid_stride, ::grid_stride]
    valid = ~mask & np.isfinite(arr)
    idx = np.argwhere(valid)
    vals = arr[valid]
    if idx.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        sel = rng.choice(idx.shape[0], max_cells, replace=False)
        idx, vals = idx[sel], vals[sel]
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(idx)}
    num = 0.0
    w_sum = 0.0
    z = vals - vals.mean()
    for (i, j), k in lookup.items():
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k2 = lookup.get((i + di, j + dj))
            if k2 is not None:
                num += z[k] * z[k2]
                w_sum += 1
    var = np.sum(z ** 2)
    I = (len(vals) / w_sum) * (num / var) if w_sum > 0 and var > 0 else np.nan
    # expected value for a random pattern and a normal-approximation z-score
    E = -1.0 / (len(vals) - 1)
    # NOTE: no significance test is reported.  A normal-approximation z-score
    # (I - E[I]) / sqrt(1/(n-1)) assumes independent observations with a
    # correctly derived variance for this weight matrix; for a 4-neighbour rook
    # scheme on a strongly autocorrelated field that variance is not valid, so
    # quoting a z value or a p value would overstate what has been shown.  The
    # statistic is reported with its cell count, its weight scheme and its
    # behaviour under thinning, which is what the interpretation actually needs.
    return {"morans_I": float(I), "expected_I": float(E),
            "n_cells": int(len(vals)), "n_pairs": int(w_sum / 2),
            "grid_stride": int(grid_stride),
            "weights_scheme": "rook (4-neighbour), binary, row-weighted",
            "mean": float(vals.mean()), "std": float(vals.std())}


def moran_table(fields: dict, mask: np.ndarray, strides=(1, 2, 3)) -> dict:
    """Moran's I for every feature (d, s, r, c, R), full grid plus thinned checks.

    The full-grid statistic uses every valid cell.  The thinned runs subsample the
    grid itself (every 2nd/3rd row and column) as a resolution artefact check: a
    value that survives thinning is a property of the terrain, not of the 450 m
    sampling.
    """
    out = {"features": {}, "strides": list(strides),
           "weights_scheme": "rook (4-neighbour), binary, row-weighted",
           "significance": ("not reported: the normal approximation for Moran's I is "
                            "not valid for this weight matrix on an autocorrelated "
                            "field, so no p-value or z-score is claimed")}
    for k in FEATURES:
        arr = np.asarray(fields[k], float)
        per = {}
        for st in strides:
            per[f"stride_{st}"] = morans_i(arr, mask, grid_stride=st)
        out["features"][k] = {
            "morans_I": per["stride_1"]["morans_I"],
            "expected_I": per["stride_1"]["expected_I"],
            "n_cells": per["stride_1"]["n_cells"],
            "n_pairs": per["stride_1"]["n_pairs"],
            "thinned": {st: per[st]["morans_I"] for st in per},
            "unit": UNITS[k],
        }
    return out


# --------------------------------------------------------------------------
# optional PCA on the normalised features
# --------------------------------------------------------------------------
def pca_features(fields: dict, mask: np.ndarray) -> dict:
    valid = ~mask & np.isfinite(fields["R"])
    X = np.column_stack([fields["s_n"][valid], fields["r_n"][valid], fields["c_n"][valid]])
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    return {
        "explained_variance_ratio": (w / w.sum()).tolist(),
        "components": v.tolist(),
        "feature_order": ["s_n", "r_n", "c_n"],
        "n_cells": int(valid.sum()),
    }


# --------------------------------------------------------------------------
# interpretation helpers
# --------------------------------------------------------------------------
def feasibility_summary(feasible: np.ndarray, d: np.ndarray, s: np.ndarray,
                        lim: dict) -> dict:
    from scipy import ndimage
    lab, n = ndimage.label(feasible, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    return {
        "feasible_fraction": float(feasible.mean()),
        "feasible_cells": int(feasible.sum()),
        "total_cells": int(feasible.size),
        "n_components": int(n),
        "largest_component_cells": int(sizes.max()),
        "largest_component_fraction": float(sizes.max() / feasible.size),
        "limits": lim,
    }
