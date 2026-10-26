# Data source catalog

All data lives under `$WXRISK_DATA` (default `~/wxrisk_data`, read by `wxrisk.wb2.DATA_DIR`). **Data is not kept in git**; it totals about 16 GB.
A machine-readable inventory (paths, sizes, parquet row counts and columns) is in [`data_manifest.json`](data_manifest.json),
regenerated with `python scripts/31_data_manifest.py`. Every download script is idempotent: files that already exist and are non-empty are skipped.

## Fetch everything

```shell
uv sync
python scripts/fetch/fetch_raw.py                       # Storm Events / SPC / RMA / IEM warnings / NWIS / county boundaries
python scripts/20_fetch_nfip_claims.py --years 2000 2023 --workers 4   # NFIP claims (national)
python scripts/22_fetch_pilot_inputs.py              # gridMET, HURDAT2, TIGER block groups (Texas pilot)
# Policies: see "NFIP policies" in the table below (the bulk file must be downloaded by hand, then run 24)
```

## Download channel conventions (`src/wxrisk/fetch.py`)

The channel is chosen by source type; no curl or shell, so behaviour is the same on macOS, Linux and Windows:

| Source type | Channel | Used for |
|---|---|---|
| Vendor API | The vendor's own library | CDS: `ecmwf-datastores-client` (`scripts/09`; the library resumes and checks sizes itself, and we added a free-space preflight once a result is ready); WeatherBench-2: `gcsfs` + zarr (`scripts/01`, `03`, `wb2.py`) |
| Object storage | `fetch.s3` / `fetch.s3_bytes` (boto3, anonymous) | MRMS MESH/QPE (`wxrisk/mrms.py`, bucket `noaa-mrms-pds`), Terrarium elevation tiles (`scripts/27`, bucket `elevation-tiles-prod`) |
| Plain HTTP(S) file | `fetch.https` / `fetch.fetch_many` | `fetch_raw.py`, `scripts/22` (gridMET THREDDS subsets, HURDAT2, TIGER) |
| Small HTTP(S) response / JSON API | `fetch.get_bytes` / `fetch.get_json` (with retries) | OpenFEMA (`scripts/20`, `fetch_fema_conmaz.py`), county centroid table (`scripts/08`) |

What `fetch.https` / `fetch.s3` guarantee:

- **Atomic**: data is written to `<file>.part` and renamed only when complete; a failure part-way never leaves a truncated file that looks finished.
- **Resumable**: when the size is known and the server supports Range (S3 always does), progress is kept in `<file>.part.json`
  (source, ETag, size, bytes written); if the remote file changed (different ETag) the download restarts from zero instead of splicing two versions.
- **Space**: free space is checked before the transfer with a 256 MB margin, and `InsufficientSpace` states the shortfall; `fetch_many` probes all
  file sizes first and **checks the whole batch once**, so a job with dozens of files fails within seconds instead of half-way through.
- **Space reservation**: `posix_fallocate` on Linux, `F_PREALLOCATE` on macOS (verified here to reserve 3 GB), extending the end of the file on Windows (**not tested on Windows**).
  Reservation is best effort (some filesystems do not support it and it is silently skipped); the free-space preflight always runs.
- **Unknown size** (generated responses such as THREDDS, chunked transfer): no resume and no reservation; passing `size_hint` still gives a space preflight,
  and no HEAD probe is sent (the server would otherwise do the work twice).
- Vendor-library downloads (CDS) resume on their own, but only inside a single `download` call; we cannot reserve space for them, because that would break
  the library's resume logic, which relies on the file size.

Tests: `uv run python -m pytest tests/test_fetch.py` (a local HTTP server simulates Range / changed file / no Range / unknown size;
with `WXRISK_NETWORK_TESTS=1` there is one more test that downloads and resumes from a real S3 bucket anonymously).

## Integrity checks and the data lock

**Upstream sources publish almost no checksums** (measured 2026-10-24), so there are three layers:

| Source | What upstream provides | How we use it |
|---|---|---|
| S3 (MRMS, Terrarium) | The ETag is the MD5 of the content (single-part uploads; verified against our own MD5 on two objects); multipart ETags contain `-` and are not hashes | MD5 is compared automatically on download, and a mismatch discards the file and retries (`fetch.s3`) |
| OpenFEMA API | Exact row counts: `$count=true`, and `recordCount` in the catalog (`NfipPolicies` v3 is 74,739,461, matching the row count of our bulk file) | Script 20 checks the row count for every year; script 24 checks the bulk file's row count at the start |
| GCS (WeatherBench-2) | Every object has `md5Hash` and `crc32c` | We read zarr chunks and never store whole files, so this is not used |
| SPC, RMA, IEM, NCEI, NHC | **No content hash**: the ETag is "file size - modification time" (SPC: the ETag's `7e2468` is exactly the Content-Length 8,266,856), and the NCEI directory has no md5 files | Only size plus the data lock |
| Census | Not even an ETag | Same as above |
| THREDDS subsets, NWIS | Generated on request, nothing at all | Same as above |
| CDS | The library checks the byte count; whether the result carries a checksum field could not be confirmed (the job result we had access to had expired) | The library's size check |

**Byte hashes can lie.** Measured by downloading repeatedly (2026-10-24):

| Source | Byte-identical across two downloads? | What changes |
|---|---|---|
| zip/gz from SPC, RMA, IEM, NCEI, Census; HURDAT2 | Yes | — |
| gridMET (THREDDS NetCDF subset) | **No** | Only the "translation date" inside the global attribute `History`; every variable's values are identical |
| NWIS `.rdb` | **No** | Only the `# retrieved:` line (time + server name); the data lines are identical |
| `.npz` written by `np.savez` | No | The timestamps of the zip members |
| parquet we write with pandas | Possibly (different pyarrow versions, compression, row-group size) | Writer metadata and layout; also, **Arrow does not define the bytes under a null slot**: in the real claims file, `floodWaterDuration` is logically identical across row-group sizes but the residual bytes in null slots differ (32 slots) |

So the data lock (`data.lock.json`, produced by `scripts/32_data_lock.py`) records two hashes per file:
`sha256` (of the bytes) and `content_sha256` (of the data with volatile metadata removed; defined in `src/wxrisk/integrity.py`: for gz, the decompressed stream;
for zip/npz, (member name, CRC, size); for NetCDF, every variable's name/type/dims/attributes/values, ignoring global attributes; for parquet, the logical rows
re-cut into fixed windows and hashed, independent of compression, row groups and metadata, and without hashing the raw buffers under null slots: only the
"validity bitmap + array of valid values only" is hashed; for `.rdb`, the text minus the `# retrieved:` line). When bytes differ, a zip is CRC-checked member by member, because the
CRCs in a zip's directory do not change when the data is damaged. `tests/test_integrity.py` shows for each format that changing volatile metadata leaves the hash unchanged, while
changing one value, a variable attribute, the row order, a null, or dropping one row necessarily changes it. Negative tests (on copies of real files): metadata-only changes
(NetCDF `History`, the `# retrieved:` line, gz recompression, parquet with different compression/row groups) → `ok-metadata-differs`; changing one data value, flipping one byte,
dropping a row, truncating → `changed` or `corrupt`. Recomputing the whole lock gives a byte-identical `data.lock.json` (safe to commit, no meaningless diffs).

Verification order: identical bytes → `ok`; otherwise identical content → `ok-metadata-differs`; otherwise `changed` (upstream changed or the file is damaged), and `corrupt` if it cannot be opened.
Sources marked `mutable` (NWIS provisional data, SPC/IEM/RMA updated in place, OpenFEMA, gridMET) only warn on a difference and do not count as failures.

```shell
python scripts/32_data_lock.py write                                   # recompute; unchanged files are not rehashed (cached in $WXRISK_DATA/.lock_cache.json)
python scripts/32_data_lock.py verify                                  # everything
python scripts/32_data_lock.py verify --category raw --allow-missing   # a member who has only part of the data
python scripts/32_data_lock.py verify --quick                          # existence and size only
```

parquet/NetCDF files above 1 GB record only the byte hash and row count (a content hash would be too slow); the 3.7 GB policy file is checked by row count against the OpenFEMA catalog.

## Raw sources

| Source | Content | How to get it | Local path (relative to `$WXRISK_DATA`) | Size | Auth |
|---|---|---|---|---|---|
| WeatherBench-2 IFS ENS / ERA5 | Ensemble forecasts and reanalysis (64×32) | `scripts/01`, `03` | `ens_*.npz`, `seq*.npz` | ~0.5 GB | None (public GCS bucket) |
| ERA5 (CDS) | Convective environment fields, CO/NM/AZ, May–Sep | `scripts/09` | `cds/` | 1.5 GB | Needs `~/.cdsapirc` |
| NOAA Storm Events | Event reports (hail / wind / tornado / flood) | `fetch_raw.py storm_events` | `storm_events/` | 0.23 GB | None |
| SPC storm reports | Hail, wind, tornado point reports | `fetch_raw.py spc` | `labels_extra/spc/` | 21 MB | None |
| USDA RMA | Crop insurance cause-of-loss summaries | `fetch_raw.py rma` | `labels_extra/rma/` | 0.11 GB | None |
| IEM warning polygons | NWS storm-based warnings, 2002+ | `fetch_raw.py iem` | `labels_extra/iem/` | 0.11 GB | None |
| MRMS MESH (radar hail) | 24 h maximum MESH, 2021–2023 | `scripts/17` | `labels_extra/mesh/` | 0.4 GB | None (public AWS bucket) |
| MRMS MultiSensor QPE | 24 h rainfall, 2021–2023 | `scripts/19` | `labels_extra/qpe/` | 1.9 GB | None |
| MRMS MultiSensor QPE 01H Pass2 | hourly rain, ~600 km crops (4 km blocks) around 8 Texas episodes, 2021–2023 | `scripts/39` | `labels_extra/flood/mrms_hourly/` | 42 MB (450 MB transferred) | None. Hourly product starts 2020-10; -3 / -999 are no-coverage flags, not dry |
| USGS NWIS | Daily river discharge (CO/NM/AZ) | `fetch_raw.py nwis` | `labels_extra/flood/nwis/` | Small | None |
| Census | 2020 county boundaries, 2010/2020 block groups | `fetch_raw.py census`, `scripts/22` | `labels_extra/census/`, `labels_extra/flood/pilot/` | Small | None |
| ACS 5-year 2022 B25001 | Housing units per block group (take-up denominator), keyless bulk table-based summary file | `scripts/34` (downloads via `fetch.https`) | `labels_extra/census/acs5_2022_b25001.dat` | 18 MB | None; 2020 block groups, so 21% of Texas policy block groups (older vintage) do not match |
| FEMA disaster declarations, Texas | One row per declaration x county, all incident types, 2000+ (event definition for scripts 36) | `scripts/fetch/fetch_fema_declarations.py` | `labels_extra/flood/fema_declarations_tx.parquet` | Small (4,001 rows) | None (OpenFEMA v2 API) |
| Freddie Mac PMMS | 30-year mortgage rate, weekly, 1971+ (FRED's MORTGAGE30US source; the FRED CSV endpoint timed out) | `scripts/fetch/fetch_econ.py` | `labels_extra/econ/pmms_history.csv` | 0.1 MB | None; rolling, `mutable` |
| Zillow ZHVI (county) | Home value index, SFR+condo, smoothed, SA, monthly 2000+ | `scripts/fetch/fetch_econ.py` | `labels_extra/econ/zillow_zhvi_county.csv` | 14 MB | None; rolling, revised, `mutable`; small counties missing |
| Census building permits (county) | Annual new-construction permits by county, 2010-2023 | `scripts/fetch/fetch_econ.py` | `labels_extra/econ/bps_county_<year>.txt` | 0.4 MB each | None |
| Redfin county market tracker | Monthly county home sales, prices, inventory, 2012+ (Texas extract cached in `redfin_tx.parquet` by `scripts/37`) | `scripts/fetch/fetch_econ.py` | `labels_extra/econ/redfin_county_market_tracker.tsv.gz` | 241 MB | None (public S3, ETag checked); rolling, `mutable`; Texas counties without Redfin coverage are missing |
| **NFIP claims (OpenFEMA v3)** | National claims 2000–2023, with block group, water depth, loss, foundation | `scripts/20` | `labels_extra/flood/nfip_national/` | 1.56 M records | None (API, paged) |
| **NFIP policies (OpenFEMA v3)** | 74.74 M policies: effective/termination dates, block group, building attributes | Download the bulk parquet by hand and save it as `labels_extra/flood/nfip_policies_v3.parquet` (3.7 GB), then extract Texas with `scripts/24` | Same as left; Texas: `labels_extra/flood/pilot/policies_tx.parquet` | 3.7 GB | None |
| FEMA flood disaster declarations, CO/NM/AZ claims | Older small tables | `scripts/fetch/fetch_fema_conmaz.py` | `labels_extra/flood/` | Small | None |
| gridMET precipitation | 4 km daily precipitation, Texas, 2000–2023 | `scripts/22` | `labels_extra/flood/pilot/gridmet/` | 1.5 GB | None (THREDDS) |
| HURDAT2 | Tropical cyclone tracks, used for exclusion | `scripts/22` | `labels_extra/flood/pilot/hurdat2_nepac.txt` etc. | Small | None |
| AWS Terrarium elevation tiles | About 150 m elevation, z=10 | `scripts/27` | `labels_extra/flood/pilot/terrain_tiles/` | Small | None |

**The direct link to the policy bulk file is not recorded in the repo.** It is on the OpenFEMA data page ("FIMA NFIP Policies"); we verified that the API
endpoint `https://www.fema.gov/api/open/v3/NfipPolicies` works (filtering by state plus effective date is fast; an "in force during a window" filter returns 503),
but we have not verified the URL of the bulk file, so do not assume any particular fixed path works.

## Derived tables (produced by scripts, rebuildable)

| File | Script that produces it | Content |
|---|---|---|
| `labels_scs3_conm_az.parquet` | 08 | CO/NM/AZ county-day SCS3 labels |
| `scalefeat_<year>.parquet` | 11 | Scale-resolved features, one file per year |
| `warnings_county_day.parquet` | 16 | County-day warning counts |
| `mesh_county_day.parquet` / `qpe_county_day.parquet` | 17 / 19 | County-day radar hail / 24 h rainfall |
| `flood_labels_county_day.parquet` | 18 | Flood labels from all sources, merged |
| `nfip_inland_nontropical.parquet` | 21 | Inland claims not named as tropical (with claim counts, water depth, loss) |
| `labels_extra/flood/pilot/county_daily.npz` | 23 | Texas county-day precipitation (1-day, 3-day) |
| `.../pilot/expo_2010_2023.npz` | `flood_pilot.py` (via 25) | In-force policy statistics per block group × day (1.3 GB; four matrices `n_pol`, `n_sfha`, `n_single`, `cov_sum`; **no foundation / elevation / basement**, which are not yet aggregated, see `FLOOD_PRIORS.md`) |
| `.../pilot/flood_pilot_table.parquet` | 25 | Main block group × rain episode table (about 3 million rows) |
| `.../pilot/bg_terrain.parquet` / `flood_pilot_wavelet.parquet` | 27 / 28 | Block group terrain, rain-episode wavelet features |
| `audit/building_attrs.txt` | 30 | Building-attribute audit report (see `FLOOD_PRIORS.md`) |

## Data traps you must know

- **NFIP policies with effective dates before 2009 do not exist**, so exposure denominators are only available from 2010.
- **NFIP claim latitude/longitude have only one decimal place**, so spatial location uses the 12-digit block group `censusGeoid` (present for 96–99.7%).
- **The "0 = unknown" share of `foundationType` changes drastically by year** (Texas: 86% in 2010 → about 0% from 2022); this is a change in reporting rules, not in the buildings.
- **A water depth of 0 does not mean "no water"**: losses in that bin are still large, and it should be treated as missing.
- **`floodWaterDuration` is effectively unusable** (almost all zeros).
- Texas non-tropical claims are very uneven across years (about 10 k each in 2015–2016, under 500 in 2011/2022/2023), so absolute probabilities do not transfer across eras.
- **No public NFIP field gives wood / brick construction**; `construction` is only an "under construction" boolean.
- The numeric codes of `foundationType` were inferred from cross-tabs, not from an official dictionary; check them against the OpenFEMA v3 data dictionary before relying on them.
