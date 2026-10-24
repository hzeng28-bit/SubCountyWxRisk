"""Download ERA5 convective-environment fields from the Copernicus CDS (perfect-prog stand-in).

Needs ~/.cdsapirc. ERA5 0.25 deg over a CO/NM/AZ box, May-Sep, 12Z and 18Z snapshots:
  single : CAPE, CIN, T2m, Td2m, MSLP, sp, PW, 10m wind, BLH
  plev   : T, q, u, v at 500/700/850 hPa
  precip : hourly total_precipitation 07-18Z (summed to 6 h windows ending 12Z and 18Z later)
One netCDF per (kind, year) under $WXRISK_DATA/cds/.

All missing (kind, year) requests are submitted to the CDS queue at once (non-blocking), then
polled and downloaded as they finish. Jobs already on the server (accepted/running/successful)
are adopted by matching collection, year and variables, so re-running never resubmits them.
Request ids are kept in cds/manifest.json. A submit the server refuses is retried on the next poll.

    python scripts/09_fetch_era5_county_env.py --years 2000 2022
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ecmwf.datastores import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

AREA = [43, -120, 29, -99]  # N, W, S, E
MONTHS = [f"{m:02d}" for m in (5, 6, 7, 8, 9)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

SINGLE = ["convective_available_potential_energy", "convective_inhibition",
          "2m_temperature", "2m_dewpoint_temperature", "mean_sea_level_pressure",
          "surface_pressure", "total_column_water_vapour", "10m_u_component_of_wind",
          "10m_v_component_of_wind", "boundary_layer_height"]
PLEV = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
KINDS = {
    "single": ("reanalysis-era5-single-levels",
               {"variable": SINGLE, "time": ["12:00", "18:00"]}),
    "plev": ("reanalysis-era5-pressure-levels",
             {"variable": PLEV, "pressure_level": ["500", "700", "850"],
              "time": ["12:00", "18:00"]}),
    "precip": ("reanalysis-era5-single-levels",
               {"variable": ["total_precipitation"],
                "time": [f"{h:02d}:00" for h in range(7, 19)]}),
}
LIVE = ("accepted", "running", "successful")


def build_request(kind: str, year: int) -> tuple[str, dict]:
    dataset, body = KINDS[kind]
    return dataset, {"product_type": "reanalysis", "year": str(year), "month": MONTHS, "day": DAYS,
                     "area": AREA, "data_format": "netcdf", "download_format": "unarchived", **body}


def classify(collection: str, req: dict) -> tuple[str, int] | None:
    """Map a server-side job back to (kind, year); None if it is not one of ours."""
    # early submissions also asked for k_index, which ERA5 never returned; they are equivalent
    variable = [v for v in req.get("variable", []) if v != "k_index"]
    for kind, (dataset, body) in KINDS.items():
        if (collection == dataset and variable == body["variable"]
                and req.get("time") == body["time"] and req.get("month") == MONTHS
                and req.get("area") == AREA):
            return kind, int(req["year"][0] if isinstance(req["year"], list) else req["year"])
    return None


def client() -> Client:
    cfg = dict(ln.split(":", 1) for ln in Path("~/.cdsapirc").expanduser().read_text().splitlines()
               if ":" in ln)
    return Client(url=cfg["url"].strip(), key=cfg["key"].strip(), progress=False)


def adopt(c: Client) -> dict[str, str]:
    """Request ids of our jobs already on the server (newest wins on duplicates)."""
    found: dict[str, str] = {}
    jobs = c.get_jobs(limit=500, sortby="created", status=list(LIVE))
    for rid in jobs.request_ids:
        r = c.get_remote(rid)
        req = r.request.get("inputs", r.request)
        hit = classify(r.collection_id, req)
        if hit:
            found[f"{hit[0]}_{hit[1]}"] = rid
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2000, 2022])
    ap.add_argument("--poll", type=float, default=60.0)
    args = ap.parse_args()

    out_dir = wb2.cache_path("cds")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    c = client()
    jobs = [(k, y) for y in range(args.years[0], args.years[1] + 1) for k in KINDS]
    todo = [(k, y) for k, y in jobs if not (out_dir / f"{k}_{y}.nc").exists()]
    print(f"{len(jobs) - len(todo)}/{len(jobs)} files on disk, {len(todo)} to go", flush=True)

    ids = adopt(c)
    ids = {key: ids[key] for key in (f"{k}_{y}" for k, y in todo) if key in ids}
    print(f"adopted {len(ids)} jobs already on the server", flush=True)

    while todo:
        for kind, year in list(todo):
            key = f"{kind}_{year}"
            if key not in ids:
                try:
                    ids[key] = c.submit(*build_request(kind, year)).request_id
                    print(f"[submit] {key} -> {ids[key][:8]}", flush=True)
                except Exception as e:  # noqa: BLE001 - server may cap queued jobs; retry next poll
                    print(f"[submit refused] {key}: {str(e)[:100]}", flush=True)
                    continue
            r = c.get_remote(ids[key])
            if r.status == "successful":
                target = out_dir / f"{key}.nc"
                fetch.require_space(out_dir, r.get_results().content_length)  # size is known once the job is done
                r.download(str(target.with_suffix(".part")))
                os.replace(target.with_suffix(".part"), target)
                todo.remove((kind, year))
                print(f"[done] {key}  ({len(todo)} left)", flush=True)
            elif r.status in ("failed", "dismissed"):
                print(f"[{r.status}] {key}: resubmitting", flush=True)
                del ids[key]
        manifest_path.write_text(json.dumps(ids, indent=1))
        if todo:
            time.sleep(args.poll)


if __name__ == "__main__":
    main()
