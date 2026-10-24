"""Download the raw label/context sources that no numbered script fetches (plain HTTPS files).

Resumable, atomic, with one free-space check for the whole batch (see wxrisk/fetch.py). Re-running skips
files that already exist. Catalog: docs/DATA_SOURCES.md.

    python scripts/fetch/fetch_raw.py [group ...]        groups: storm_events spc rma iem nwis census
    python scripts/fetch/fetch_raw.py nfip_policies --policies-url <OpenFEMA bulk parquet URL>
    WXRISK_DATA=/path python scripts/fetch/fetch_raw.py spc iem --workers 6

Not covered here (they use a vendor library or object storage, through their own scripts): WeatherBench-2
on GCS (01, 03), ERA5 via the CDS API (09), MRMS on S3 (17, 19), terrain tiles on S3 (27), NFIP claims API (20),
gridMET/HURDAT2/TIGER (22), the Texas policy extract (24), FEMA CO/NM/AZ API tables (fetch_fema_conmaz.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

EXTRA = wb2.cache_path("labels_extra")
GROUPS = ["storm_events", "spc", "rma", "iem", "nwis", "census"]
NCEI = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"


def storm_events() -> list[fetch.Job]:
    """Details files, one per year; the c<date> suffix changes with each NCEI release, so read the listing."""
    idx = fetch.get_bytes(f"{NCEI}/").decode("utf-8", "replace")
    jobs = []
    for y in range(2000, 2024):
        names = sorted(set(re.findall(rf"StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz", idx)))
        if names:
            jobs.append(fetch.Job(wb2.cache_path("storm_events") / names[-1], url=f"{NCEI}/{names[-1]}"))
    return jobs


def spc() -> list[fetch.Job]:
    fs = ["1955-2025_hail.csv.zip", "1955-2025_wind.csv.zip", "1950-2025_torn.csv.zip"]
    return [fetch.Job(EXTRA / "spc" / f, url=f"https://www.spc.noaa.gov/wcm/data/{f}") for f in fs]


def rma() -> list[fetch.Job]:
    base = "https://pubfs-rma.fpac.usda.gov/pub/Web_Data_Files/Summary_of_Business/cause_of_loss"
    return [fetch.Job(EXTRA / "rma" / f"colsom_{y}.zip", url=f"{base}/colsom_{y}.zip") for y in range(2000, 2024)]


def iem() -> list[fetch.Job]:
    return [fetch.Job(EXTRA / "iem" / f"{y}_tsmf_sbw.zip", url=f"https://mesonet.agron.iastate.edu/pickup/wwa/{y}_tsmf_sbw.zip")
            for y in range(2002, 2024)]


def nwis() -> list[fetch.Job]:
    """Daily mean discharge (00060), CO/NM/AZ, Apr 26 - Sep 30 of each year. Generated responses: no probe."""
    d, base = EXTRA / "flood" / "nwis", "https://waterservices.usgs.gov/nwis"
    jobs = []
    for s in ("co", "nm", "az"):
        jobs.append(fetch.Job(d / f"sites_{s}.rdb", probe=False, size_hint=5 << 20,
                              url=f"{base}/site/?format=rdb&stateCd={s}&siteType=ST&hasDataTypeCd=dv&parameterCd=00060"))
        for y in range(2000, 2023):
            jobs.append(fetch.Job(d / f"dv_{s}_{y}.rdb", probe=False, size_hint=20 << 20,
                                  url=f"{base}/dv/?format=rdb&stateCd={s}&parameterCd=00060&statCd=00003&siteType=ST"
                                      f"&startDT={y}-04-26&endDT={y}-09-30"))
    return jobs


def census() -> list[fetch.Job]:
    return [fetch.Job(EXTRA / "census" / "cb_2020_us_county_20m.zip",
                      url="https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", default=GROUPS, choices=[*GROUPS, "nfip_policies"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policies-url", help="OpenFEMA NFIP policies bulk parquet (3.7 GB); not recorded in the repo")
    a = ap.parse_args()
    jobs: list[fetch.Job] = []
    for g in a.groups:
        if g == "nfip_policies":
            if not a.policies_url:
                sys.exit("nfip_policies needs --policies-url (see docs/DATA_SOURCES.md)")
            jobs.append(fetch.Job(EXTRA / "flood" / "nfip_policies_v3.parquet", url=a.policies_url))
        else:
            jobs += globals()[g]()
    res = fetch.fetch_many(jobs, workers=a.workers)
    if bad := {k: v for k, v in res.items() if v.startswith("FAIL")}:
        sys.exit(f"{len(bad)} downloads failed (re-run to resume): {bad}")
    print("done:", " ".join(a.groups))


if __name__ == "__main__":
    main()
