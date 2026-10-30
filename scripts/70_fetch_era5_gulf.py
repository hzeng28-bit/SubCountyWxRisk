"""Queue ERA5 moisture and wind fields for the Gulf / Southeast box on the Copernicus CDS (stage 31).

Box 24-37 N, 107-79 W (Texas, Louisiana, Mississippi, Florida), 0.25 deg, 2009-2023 (2009 for antecedent fields), all months, 00/06/12/18 UTC:
  single : total_column_water_vapour                                          (precipitable water, ingredient 1)
  plev   : specific humidity, u and v at 500/700/850 hPa                       (moisture flux and steering wind, ingredients 1-2)
One netCDF per (kind, year) for single and per (kind, year, quarter) for plev (a whole year of pressure levels exceeds the CDS cost limit: HTTP 403
"cost limits exceeded") under $WXRISK_DATA/cds_gulf/. Needs ~/.cdsapirc.

Default is SUBMIT ONLY: every missing (kind, year) job is put on the CDS queue (jobs already there with the same request are adopted, not
resubmitted) and the request ids are written to cds_gulf/manifest.json. Downloading is a separate, explicit step once the jobs are done:

    python scripts/70_fetch_era5_gulf.py                 # submit / adopt, print the status of each job
    python scripts/70_fetch_era5_gulf.py --download      # download the finished jobs (space check first), leave the others
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ecmwf.datastores import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

AREA = [37, -107, 24, -79]  # N, W, S, E
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
KINDS = {
    "single": ("reanalysis-era5-single-levels", {"variable": ["total_column_water_vapour"]}),
    "plev": ("reanalysis-era5-pressure-levels",
             {"variable": ["specific_humidity", "u_component_of_wind", "v_component_of_wind"], "pressure_level": ["500", "700", "850"]}),
}
PARTS = {"single": {"": MONTHS}, "plev": {f"_q{i + 1}": MONTHS[3 * i:3 * i + 3] for i in range(4)}}
LIVE = ("accepted", "running", "successful")


def build_request(kind: str, year: int, part: str = "") -> tuple[str, dict]:
    dataset, body = KINDS[kind]
    return dataset, {"product_type": "reanalysis", "year": str(year), "month": PARTS[kind][part], "day": DAYS, "time": TIMES, "area": AREA,
                     "data_format": "netcdf", "download_format": "unarchived", **body}


def classify(collection: str, req: dict) -> str | None:
    """Manifest key of a server-side job that is one of ours, else None."""
    for kind, (dataset, body) in KINDS.items():
        if collection == dataset and req.get("variable") == body["variable"] and req.get("area") == AREA and req.get("time") == TIMES:
            for part, months in PARTS[kind].items():
                if req.get("month") == months:
                    return f"{kind}_{int(req['year'][0] if isinstance(req['year'], list) else req['year'])}{part}"
    return None


def client() -> Client:
    cfg = dict(ln.split(":", 1) for ln in Path("~/.cdsapirc").expanduser().read_text().splitlines() if ":" in ln)
    return Client(url=cfg["url"].strip(), key=cfg["key"].strip(), progress=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2009, 2023])
    ap.add_argument("--download", action="store_true", help="download the jobs that are finished instead of submitting")
    a = ap.parse_args()
    out = wb2.cache_path("cds_gulf")
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    ids = json.loads(mpath.read_text()) if mpath.exists() else {}
    c = client()
    jobs = [(k, y, part) for y in range(a.years[0], a.years[1] + 1) for k in KINDS for part in PARTS[k]]
    todo = [(k, y, part) for k, y, part in jobs if not (out / f"{k}_{y}{part}.nc").exists()]
    print(f"{len(jobs) - len(todo)}/{len(jobs)} files on disk, {len(todo)} to go", flush=True)
    for rid in c.get_jobs(limit=500, sortby="created", status=list(LIVE)).request_ids:
        r = c.get_remote(rid)
        hit = classify(r.collection_id, r.request.get("inputs", r.request))
        if hit and hit not in ids:
            ids[hit] = rid
    for kind, year, part in todo:
        key = f"{kind}_{year}{part}"
        if key not in ids and not a.download:
            try:
                ids[key] = c.submit(*build_request(kind, year, part)).request_id
                print(f"[submit] {key} -> {ids[key][:8]}", flush=True)
            except Exception as e:  # noqa: BLE001 - the server may cap queued jobs; rerun later
                print(f"[submit refused] {key}: {str(e)[:120]}", flush=True)
                continue
        if key not in ids:
            print(f"[no job] {key}")
            continue
        r = c.get_remote(ids[key])
        print(f"[{r.status}] {key}", flush=True)
        if a.download and r.status == "successful":
            fetch.require_space(out, r.get_results().content_length)
            target = out / f"{key}.nc"
            r.download(str(target.with_suffix(".part")))
            os.replace(target.with_suffix(".part"), target)
            print(f"[done] {key}", flush=True)
    mpath.write_text(json.dumps(ids, indent=1))


if __name__ == "__main__":
    main()
