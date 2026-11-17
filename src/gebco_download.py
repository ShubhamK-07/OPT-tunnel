"""GEBCO_2026 subset downloader (uses the public download.gebco.net basket API).

The GEBCO web downloader is a small REST service:
    POST /api/queue                      -> {"basketId": ...}
    GET  /api/queue/status/<basketId>    -> {"status": "new"|"processing"|"finished"}
    GET  /api/queue/download/<basketId>  -> zip with the NetCDF subset(s)

This module wraps that flow so the data acquisition step is reproducible from
a bounding box, with no manual clicking.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://download.gebco.net"
GRID_GEBCO_2026 = 1          # grid_id of the GEBCO 2026 global 15" grid
SRC_BATHY = 1                # data_source_ids for the 15" bathymetry grid
SRC_TID = 3                  # data_source_ids for the TID (source type) grid
FMT_NETCDF = 1


def _poll(basket_id: str, timeout: float = 900.0, interval: float = 6.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{BASE}/api/queue/status/{basket_id}", timeout=60)
        r.raise_for_status()
        status = r.json().get("status", "unknown")
        if status in ("finished", "error", "failed"):
            return status
        time.sleep(interval)
    return "timeout"


def download_subset(
    north: float,
    south: float,
    west: float,
    east: float,
    out_dir: Path,
    with_tid: bool = True,
    tag: str = "",
) -> dict:
    """Request a GEBCO_2026 NetCDF subset for the given bbox; save the .nc files.

    Returns a metadata dict describing exactly what was fetched (product,
    version, date of download, bbox, files). This is the record the spec asks
    for in Section 3.1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [SRC_BATHY, SRC_TID] if with_tid else [SRC_BATHY]
    payload = {
        "id": "0",
        "email": None,
        "submission_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000"),
        "processing_status": "new",
        "items": [
            {
                "id": 0,
                "grid_id": GRID_GEBCO_2026,
                "data_source_ids": sources,
                "formats": [FMT_NETCDF],
                "left": west,
                "right": east,
                "top": north,
                "bottom": south,
            }
        ],
    }
    r = requests.post(f"{BASE}/api/queue", json=payload, timeout=120)
    r.raise_for_status()
    basket_id = r.json()["basketId"]

    status = _poll(basket_id)
    if status != "finished":
        raise RuntimeError(f"GEBCO queue did not finish (status={status}, basket={basket_id})")

    z = requests.get(f"{BASE}/api/queue/download/{basket_id}", timeout=600)
    z.raise_for_status()

    saved = []
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".nc"):
                target = out_dir / name
                target.write_bytes(zf.read(name))
                saved.append(str(target))

    meta = {
        "product": "GEBCO_2026 Grid",
        "publisher": "GEBCO (Nippon Foundation-GEBCO Seabed 2030 / BODC)",
        "version": "GEBCO_2026, published April 2026, 15 arc-second global grid",
        "request_url": f"{BASE}/api/queue (grid_id={GRID_GEBCO_2026}, "
                       f"data_source_ids={sources}, format=netcdf)",
        "basket_id": basket_id,
        "download_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": {"north": north, "south": south, "west": west, "east": east},
        "files": saved,
        "tag": tag,
        "citation": (
            "GEBCO Compilation Group (2026) GEBCO_2026 Grid, "
            "doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"
        ),
        "terms_of_use": "GEBCO Grid terms of use (see GEBCO_Grid_terms_of_use.pdf)",
    }
    (out_dir / "download_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":  # tiny CLI
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--south", type=float, required=True)
    p.add_argument("--west", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--out", default="data/raw")
    a = p.parse_args()
    print(json.dumps(download_subset(a.north, a.south, a.west, a.east, Path(a.out)), indent=2))
