"""Data layer: acquire the GEBCO_2026 subset, reproject to a metric grid, mask.

Output objects (all in `data/processed/baseline.npz`):
    x, y          (nx,) (ny,)  UTM 32N coordinates of grid nodes [m]
    z             (ny, nx)     elevation [m], negative below sea level, NaN where invalid
    tid           (ny, nx)     GEBCO type-identifier (source type) grid, NaN where invalid
    mask_land     (ny, nx)     True where the cell may NOT be used (land / NoData /
                               outside the source footprint).  This is the "infeasible
                               regardless" mask used everywhere downstream.
    mask_land_src (ny, nx)     True where the cell's interpolation stencil touches a
                               land / NoData source cell (a subset of mask_land)
    mask_outside  (ny, nx)     True where the cell lies outside the source footprint
    lon, lat      (ny, nx)     geographic coordinates of each node

Conventions fixed here and used everywhere downstream:
  * z  = elevation from GEBCO, negative below sea level.  Used by derivatives.
  * d  = -z, water depth, positive downward.             Used by constraints.
  * All lengths, slopes and radii are computed in METRES on the UTM grid.

Reprojection policy (this is the part that used to be wrong)
------------------------------------------------------------
1. The land and NoData masks are applied to the SOURCE raster *before* any
   resampling.  Only source cells that are valid ocean (finite, below sea
   level, not the NoData flag) may contribute to an interpolated target value.
2. A target point is accepted only if EVERY source cell with a non-zero
   bilinear weight is a valid ocean cell.  No blend of land or NoData into an
   ocean value is possible, and no value is ever extrapolated.
3. The valid geographic footprint is the cell EXTENT of the source subset
   (centres +/- half a source cell), not the centre extent and not the
   projected bounding box.  Target points outside that footprint are marked
   invalid instead of being filled by `fill_value=None` extrapolation.
   Inside the footprint but between the outermost centre and the cell edge the
   edge cell is replicated (the same policy the roughness window uses), so the
   half-cell border is not thrown away.
4. The counts of valid / invalid / outside-footprint / mask-touching cells are
   written to the metadata, and the metadata carries a fingerprint of the data
   configuration so stale caches can never be reused silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Geod, Transformer

from .gebco_download import download_subset

ROOT = Path(__file__).resolve().parents[1]

# bump when the reprojection policy changes, so old caches are rebuilt
PIPELINE_VERSION = "2"


# --------------------------------------------------------------------------
# cache validation
# --------------------------------------------------------------------------
def config_fingerprint(cfg: dict) -> str:
    """Stable hash of everything that can change the processed grid."""
    d = cfg["data"]
    payload = {
        "version": PIPELINE_VERSION,
        "bbox": {k: float(d["bbox"][k]) for k in ("north", "south", "west", "east")},
        "crs_epsg": int(d["crs_epsg"]),
        "grid_spacing_m": float(d["grid_spacing_m"]),
        "resampling": dict(d["resampling"]),
        "A": [float(d["A"]["lon"]), float(d["A"]["lat"])],
        "B": [float(d["B"]["lon"]), float(d["B"]["lat"])],
        "nodata_threshold_m": float(d["nodata_threshold_m"]),
        "product": d.get("product"),
        "resolution_arcsec": d.get("resolution_arcsec"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------
def ensure_raw(cfg: dict, raw_dir: Path | None = None) -> Path:
    """Download the configured bbox once; return the bathymetry .nc path."""
    raw_dir = Path(raw_dir or ROOT / "data" / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    b = cfg["data"]["bbox"]
    nc = sorted(raw_dir.glob("gebco_2026_n*s*w*e*.nc"))
    nc = [p for p in nc if "tid" not in p.name]
    if nc:
        return nc[0]
    meta = download_subset(b["north"], b["south"], b["west"], b["east"], raw_dir,
                           with_tid=True, tag="baseline")
    return Path([f for f in meta["files"] if "tid" not in f][0])


# --------------------------------------------------------------------------
# loading / reprojection
# --------------------------------------------------------------------------
@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tid: np.ndarray
    mask_land: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    meta: dict
    mask_outside: np.ndarray | None = None
    mask_land_src: np.ndarray | None = None

    @property
    def d(self) -> np.ndarray:
        return -self.z

    @property
    def shape(self):
        return self.z.shape

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])


def _bilinear_strict(zsrc: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     valid: np.ndarray, lat_q: np.ndarray, lon_q: np.ndarray):
    """Bilinear resampling that refuses to mix invalid source cells in.

    Returns (value, ok, touches_invalid):

    * index positions are clipped to the outer cell CENTRES, so a target point
      that lies between the outermost centre and the cell edge replicates the
      edge cell (no extrapolation, no rejection);
    * a target is `ok` only when every corner with a non-zero weight is a valid
      ocean cell -- a weight of exactly 0 does not disqualify a point;
    * `touches_invalid` is True when at least one weighted corner is invalid,
      i.e. the point is inside the mask's neighbourhood.

    Points outside the source footprint must already have been removed by the
    caller; this function never extrapolates.
    """
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    fi = np.clip((lat_q - lat[0]) / dlat, 0.0, len(lat) - 1)
    fj = np.clip((lon_q - lon[0]) / dlon, 0.0, len(lon) - 1)
    i0 = np.minimum(np.floor(fi).astype(int), len(lat) - 2)
    j0 = np.minimum(np.floor(fj).astype(int), len(lon) - 2)
    wi = fi - i0
    wj = fj - j0

    w = {(0, 0): (1 - wi) * (1 - wj), (0, 1): (1 - wi) * wj,
         (1, 0): wi * (1 - wj), (1, 1): wi * wj}
    num = np.zeros(lat_q.shape)
    wsum = np.zeros(lat_q.shape)          # weight carried by VALID corners only
    wtot = np.zeros(lat_q.shape)          # weight of all corners (=1 inside)
    for (a, b), ww in w.items():
        v = valid[i0 + a, j0 + b]
        zz = zsrc[i0 + a, j0 + b]
        num = num + np.where(v, ww * zz, 0.0)
        wsum = wsum + np.where(v, ww, 0.0)
        wtot = wtot + ww
    ok = wsum >= 1.0 - 1e-9
    val = np.where(ok, num / np.maximum(wsum, 1e-12), np.nan)
    touches = (wtot - wsum) > 1e-12
    return val, ok, touches


def build_grid(cfg: dict, raw_dir: Path | None = None, save: bool = True) -> Grid:
    """Reproject the geographic GEBCO subset onto a regular metric grid (UTM 32N)."""
    raw_dir = Path(raw_dir or ROOT / "data" / "raw")
    nc_path = ensure_raw(cfg, raw_dir)
    tid_path = next(iter(sorted(raw_dir.glob("*tid*.nc"))), None)
    ds = xr.open_dataset(nc_path)
    z_geo = ds["elevation"].values.astype("float64")
    lat = ds["lat"].values.astype("float64")
    lon = ds["lon"].values.astype("float64")
    if lat[0] > lat[-1]:
        lat, z_geo = lat[::-1], z_geo[::-1, :]
    if lon[0] > lon[-1]:
        lon, z_geo = lon[::-1], z_geo[:, ::-1]

    tid_geo = None
    if tid_path is not None:
        dsd = xr.open_dataset(tid_path)
        tname = "tid" if "tid" in dsd else list(dsd.data_vars)[0]
        tid_geo = dsd[tname].values.astype("float64")
        tlat = dsd["lat"].values.astype("float64")
        tlon = dsd["lon"].values.astype("float64")
        if tlat[0] > tlat[-1]:
            tid_geo = tid_geo[::-1, :]
        if tlon[0] > tlon[-1]:
            tid_geo = tid_geo[:, ::-1]

    # ---- 1. source masks, applied BEFORE any resampling -------------------
    nodata = float(cfg["data"]["nodata_threshold_m"])
    src_isfinite = np.isfinite(z_geo)
    src_isnodata = (~src_isfinite) | (z_geo <= nodata)
    src_island = src_isfinite & (z_geo >= 0.0)
    src_valid = src_isfinite & ~src_isnodata & ~src_island      # ocean only
    z_src = np.where(src_valid, z_geo, np.nan)

    # ---- 2. valid geographic footprint = source cell EXTENT ---------------
    dlat = abs(float(np.diff(lat).mean()))
    dlon = abs(float(np.diff(lon).mean()))
    fp = {"lon_min": float(lon[0] - dlon / 2), "lon_max": float(lon[-1] + dlon / 2),
          "lat_min": float(lat[0] - dlat / 2), "lat_max": float(lat[-1] + dlat / 2),
          "centre_lon": [float(lon[0]), float(lon[-1])],
          "centre_lat": [float(lat[0]), float(lat[-1])]}

    epsg = int(cfg["data"]["crs_epsg"])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_geo = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    b = cfg["data"]["bbox"]
    clon = np.array([b["west"], b["east"], b["west"], b["east"]])
    clat = np.array([b["south"], b["south"], b["north"], b["north"]])
    cx, cy = to_utm.transform(clon, clat)
    sp = float(cfg["data"]["grid_spacing_m"])
    x0, x1 = cx.min(), cx.max()
    y0, y1 = cy.min(), cy.max()
    nx = int(round((x1 - x0) / sp)) + 1
    ny = int(round((y1 - y0) / sp)) + 1
    x = x0 + sp * np.arange(nx)
    y = y0 + sp * np.arange(ny)
    X, Y = np.meshgrid(x, y)

    lonq, latq = to_geo.transform(X, Y)

    # ---- 3. footprint test (no extrapolation) -----------------------------
    inside = ((lonq >= fp["lon_min"]) & (lonq <= fp["lon_max"])
              & (latq >= fp["lat_min"]) & (latq <= fp["lat_max"]))

    # ---- 4. strict bilinear resampling of ocean-only source cells ---------
    z, ok_stencil, touches = _bilinear_strict(z_src, lat, lon, src_valid, latq, lonq)
    valid = inside & ok_stencil
    z = np.where(valid, z, np.nan)

    if tid_geo is not None:
        # categorical layer: nearest cell, but only inside the footprint
        jn = np.clip(np.round((lonq - tlon[0]) / float(tlon[1] - tlon[0])).astype(int),
                     0, len(tlon) - 1)
        iin = np.clip(np.round((latq - tlat[0]) / float(tlat[1] - tlat[0])).astype(int),
                      0, len(tlat) - 1)
        tid = np.where(inside, tid_geo[iin, jn], np.nan)
    else:
        tid = np.full_like(z, np.nan)

    mask_outside = ~inside
    mask_land_src = inside & (touches | ~ok_stencil)     # stencil touches land/NoData
    mask_land = (~valid) | (np.isfinite(z) & (z >= 0.0))

    n_total = int(mask_land.size)
    meta = {
        "source_nc": str(nc_path.name),
        "tid_nc": None if tid_path is None else str(tid_path.name),
        "pipeline_version": PIPELINE_VERSION,
        "config_fingerprint": config_fingerprint(cfg),
        "crs": f"EPSG:{epsg} (WGS 84 / UTM zone 32N)",
        "spacing_m": sp,
        "shape": list(z.shape),
        "bbox_wgs84": b,
        "source_footprint_wgs84_cell_extent": fp,
        "x_range_m": [float(x[0]), float(x[-1])],
        "y_range_m": [float(y[0]), float(y[-1])],
        "resampling_elevation": cfg["data"]["resampling"]["elevation"],
        "resampling_tid": cfg["data"]["resampling"]["tid"],
        "resampling_policy": ("strict bilinear: a target point is valid only if every "
                              "weighted source corner is a valid ocean cell; no "
                              "extrapolation, no blending of land/NoData"),
        "nodata_policy": (f"z <= {nodata} or non-finite -> missing; land (z >= 0) masked; "
                          "both applied to the SOURCE raster before resampling"),
        "edge_policy": ("inside the footprint but outside the outermost source cell "
                        "centre, the edge cell is replicated (no extrapolation)"),
        "n_cells": n_total,
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "n_outside_footprint": int(mask_outside.sum()),
        "n_stencil_touches_land_nodata": int(mask_land_src.sum()),
        "n_source_cells": int(z_geo.size),
        "n_source_land_cells": int(src_island.sum()),
        "n_source_nodata_cells": int(src_isnodata.sum()),
        "land_fraction": float(mask_land.mean()),
        "masked_fraction": float(mask_land.mean()),
        "land_nodata_only_fraction": float((mask_land_src & ~mask_outside).mean()),
        "land_only_cells": int((mask_land_src & ~mask_outside).sum()),
        "outside_fraction": float(mask_outside.mean()),
        "invalid_fraction": float((~valid).mean()),
        "noData_fraction_source": float(src_isnodata.mean()),
    }
    g = Grid(x=x, y=y, z=z, tid=tid, mask_land=mask_land, lon=lonq, lat=latq, meta=meta,
             mask_outside=mask_outside, mask_land_src=mask_land_src)

    # ---- 5. metric validation against the geodesic distance --------------
    A, B = cfg["data"]["A"], cfg["data"]["B"]
    geod = Geod(ellps="WGS84")
    _, _, gdist = geod.inv(A["lon"], A["lat"], B["lon"], B["lat"])
    ax, ay = to_utm.transform(A["lon"], A["lat"])
    bx, by = to_utm.transform(B["lon"], B["lat"])
    edist = float(np.hypot(bx - ax, by - ay))
    meta["endpoints_wgs84"] = {k: [float(A[k]), float(B[k])] for k in ("lon", "lat")}
    meta["endpoints_xy_m"] = {"A": [float(ax), float(ay)], "B": [float(bx), float(by)]}
    meta["AB_geodesic_m"] = float(gdist)
    meta["AB_grid_m"] = edist
    meta["AB_difference_m"] = edist - float(gdist)
    meta["AB_difference_pct"] = 100 * (edist - float(gdist)) / float(gdist)

    if save:
        out = ROOT / "data" / "processed"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "baseline.npz", x=g.x, y=g.y, z=g.z, tid=g.tid,
                            mask_land=g.mask_land, mask_land_src=g.mask_land_src,
                            mask_outside=g.mask_outside, lon=g.lon, lat=g.lat)
        (out / "baseline_meta.json").write_text(json.dumps(meta, indent=2))
    return g


def load_grid(cfg: dict, rebuild_on_mismatch: bool = True) -> Grid:
    """Load the cached grid, but never silently reuse a stale one."""
    p = ROOT / "data" / "processed" / "baseline.npz"
    mp = p.parent / "baseline_meta.json"
    if p.exists() and mp.exists():
        meta = json.loads(mp.read_text())
        want = config_fingerprint(cfg)
        if meta.get("config_fingerprint") == want:
            npz = np.load(p)
            keys = set(npz.files)
            return Grid(x=npz["x"], y=npz["y"], z=npz["z"], tid=npz["tid"],
                        mask_land=npz["mask_land"], lon=npz["lon"], lat=npz["lat"],
                        meta=meta,
                        mask_outside=npz["mask_outside"] if "mask_outside" in keys else None,
                        mask_land_src=npz["mask_land_src"] if "mask_land_src" in keys else None)
        if not rebuild_on_mismatch:
            raise RuntimeError(
                "cached grid was built from a different data configuration "
                f"(fingerprint {meta.get('config_fingerprint')} != {want}); rerun `python -m src.data`")
        print(f"[data] cached grid is stale (fingerprint {meta.get('config_fingerprint')} "
              f"!= {want}) -> rebuilding", flush=True)
    return build_grid(cfg)


def to_index(g: Grid, x: float, y: float) -> tuple[int, int]:
    """Nearest (row, col) for UTM coordinates."""
    j = int(round((x - g.x[0]) / g.dx))
    i = int(round((y - g.y[0]) / g.dy))
    return i, j


def lonlat_to_xy(cfg: dict, lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['data']['crs_epsg']}", always_xy=True)
    return t.transform(lon, lat)


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    g = build_grid(cfg)
    print(json.dumps(g.meta, indent=2))
