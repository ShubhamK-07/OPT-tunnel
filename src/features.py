"""Feature engineering: slope s, roughness r, terrain curvature c, risk field R.

Every function takes/returns arrays shaped (ny, nx) on the metric UTM grid, so
all derivatives are in metres and all slope values are dimensionless
(rise/run, i.e. tan of the slope angle).

Sign convention: derivatives act on z = elevation (negative below sea level),
consistent with `data.py`. Curvature is reported as |Laplacian(z)| [1/m] (or a
Hessian-norm variant); it is a terrain-curvature proxy, never route curvature.

Masking policy: land / NoData cells hold NaN in z. Before any derivative we
fill them with the nearest valid value (`distance_transform_edt`), so stencils
near the coast never mix in fake elevations. The features are re-masked to NaN
afterwards, so masked cells never enter statistics or the risk field.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs with the value of the nearest valid cell (documented policy)."""
    a = np.asarray(a, dtype=float)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a.copy()
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def gaussian_smooth(z: np.ndarray, sigma_cells: float) -> np.ndarray:
    if not sigma_cells or sigma_cells <= 0:
        return z
    return ndimage.gaussian_filter(fill_nan_nearest(z), sigma=sigma_cells, mode="nearest")


# --------------------------------------------------------------------------
# slope s [rise/run]  -- central difference / Horn / Sobel
# --------------------------------------------------------------------------
def slope(z: np.ndarray, dx: float, dy: float, method: str = "central",
          sigma: float = 0.0) -> np.ndarray:
    """|grad z| in rise/run. dx, dy in metres.

    Three genuinely different estimators are provided for the sensitivity study:
      * "central" : 2nd-order central differences (the textbook baseline).
      * "horn"    : Horn (1981) 3x3 weighted plane fit.
      * "sobel"   : Sobel-Feldman with a 5x5 support (smoothing [1,4,6,4,1] x
                    derivative [-1,-2,0,2,1]).  The classic 3x3 Sobel kernel is
                    identical to Horn's kernel up to a scale factor, so the 5x5
                    variant is used here to make the comparison non-degenerate.
    """
    zz = gaussian_smooth(z, sigma)
    if method == "central":
        gy, gx = np.gradient(zz, dy, dx)
    elif method == "horn":
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / (8.0 * dx)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / (8.0 * dy)
        gx = ndimage.convolve(zz, kx, mode="nearest")
        gy = ndimage.convolve(zz, ky, mode="nearest")
    elif method == "sobel":
        sm = np.array([1, 4, 6, 4, 1], float) / 16.0
        dv = np.array([-1, -2, 0, 2, 1], float) / 8.0
        kx = np.outer(sm, dv) / dx
        ky = np.outer(dv, sm) / dy
        gx = ndimage.convolve(zz, kx, mode="nearest")
        gy = ndimage.convolve(zz, ky, mode="nearest")
    else:
        raise ValueError(f"unknown slope method {method!r}")
    return np.hypot(gx, gy)


def slope_deg(s: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(s))


# --------------------------------------------------------------------------
# roughness r  -- local std of a field in a w x w window
# --------------------------------------------------------------------------
def roughness(field: np.ndarray, w: int = 3) -> np.ndarray:
    """Local standard deviation of `field` in a w x w window (population std)."""
    if w % 2 == 0:
        raise ValueError("roughness window must be odd")
    f = fill_nan_nearest(field)
    m1 = ndimage.uniform_filter(f, size=w, mode="nearest")
    m2 = ndimage.uniform_filter(f * f, size=w, mode="nearest")
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


# --------------------------------------------------------------------------
# terrain curvature c  -- Laplacian / Hessian norm
# --------------------------------------------------------------------------
def curvature(z: np.ndarray, dx: float, dy: float, method: str = "laplacian",
              sigma: float = 0.0) -> np.ndarray:
    """Terrain curvature magnitude in 1/m.

    * "laplacian" : |z_xx + z_yy|   (the baseline)
    * "hessian"   : ||H||_F = sqrt(z_xx^2 + 2 z_xy^2 + z_yy^2)
    These are proxies for terrain irregularity, not geological curvature.
    """
    zz = gaussian_smooth(z, sigma)
    zx = np.gradient(zz, dx, axis=1)
    zy = np.gradient(zz, dy, axis=0)
    zxx = np.gradient(zx, dx, axis=1)
    zyy = np.gradient(zy, dy, axis=0)
    if method == "laplacian":
        return np.abs(zxx + zyy)
    if method == "hessian":
        zxy = np.gradient(zx, dy, axis=0)
        return np.sqrt(zxx ** 2 + 2.0 * zxy ** 2 + zyy ** 2)
    raise ValueError(f"unknown curvature method {method!r}")


# --------------------------------------------------------------------------
# normalisation and risk field
# --------------------------------------------------------------------------
def normalize(field: np.ndarray, mask: np.ndarray, how: str = "minmax",
              p: float = 99.0) -> np.ndarray:
    """Min-max (optionally 99th-percentile-clipped) normalisation to [0, 1].

    Statistics are computed on valid (unmasked) cells only. Values outside the
    range are clipped, so the output is always within [0, 1].
    """
    f = field.astype(float).copy()
    valid = ~mask & np.isfinite(f)
    v = f[valid]
    if how == "minmax":
        lo, hi = v.min(), v.max()
    elif how == "p99clip":
        lo, hi = v.min(), np.percentile(v, p)
    else:
        raise ValueError(how)
    if hi <= lo:
        return np.zeros_like(f)
    out = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    out[mask] = np.nan
    return out


def derive_features(z: np.ndarray, dx: float, dy: float, mask: np.ndarray,
                    fcfg: dict, rcfg: dict) -> dict:
    """Full feature stack + risk field for one configuration.

    Returns dict with keys: s, s_deg, r, c, s_n, r_n, c_n, R (NaN on masked cells)
    plus the raw normalisation ranges (needed to report risk numbers).
    """
    sigma = float(fcfg.get("smooth_sigma_cells", 0.0) or 0.0)
    s = slope(z, dx, dy, method=fcfg.get("slope_method", "central"), sigma=sigma)
    r_src = s if fcfg.get("roughness_field", "slope") == "slope" else (-z)
    r = roughness(r_src, int(fcfg.get("roughness_window", 3)))
    c = curvature(z, dx, dy, method=fcfg.get("curvature_method", "laplacian"), sigma=sigma)

    for name, arr in (("s", s), ("r", r), ("c", c)):
        arr[mask] = np.nan

    s_n = normalize(s, mask, rcfg.get("normalization", "minmax"))
    r_n = normalize(r, mask, rcfg.get("normalization", "minmax"))
    c_n = normalize(c, mask, rcfg.get("normalization", "minmax"))

    w = rcfg["weights"]
    s_term = s_n ** 2 if rcfg.get("slope_squared", True) else s_n
    R = w["slope"] * s_term + w["roughness"] * r_n + w["curvature"] * c_n
    R[mask] = np.nan

    return {
        "s": s, "s_deg": slope_deg(s), "r": r, "c": c,
        "s_n": s_n, "r_n": r_n, "c_n": c_n, "R": R,
        "norm_ranges": {
            "slope_m_per_m": [float(np.nanmin(s)), float(np.nanmax(s))],
            "roughness": [float(np.nanmin(r)), float(np.nanmax(r))],
            "curvature_1_per_m": [float(np.nanmin(c)), float(np.nanmax(c))],
        },
        "settings": {
            "slope_method": fcfg.get("slope_method"),
            "curvature_method": fcfg.get("curvature_method"),
            "roughness_window": fcfg.get("roughness_window"),
            "roughness_field": fcfg.get("roughness_field"),
            "smooth_sigma_cells": sigma,
            "normalization": rcfg.get("normalization"),
            "weights": rcfg["weights"],
            "slope_squared": rcfg.get("slope_squared", True),
            "spacing_m": [dx, dy],
        },
    }


def feasible_mask(d: np.ndarray, s: np.ndarray, mask_land: np.ndarray,
                  d_min: float, d_max: float, s_max: float) -> np.ndarray:
    """Feasibility rule shared by both problems (spec Sec. 5)."""
    return (~mask_land) & np.isfinite(d) & np.isfinite(s) & \
           (d >= d_min) & (d <= d_max) & (s <= s_max)
