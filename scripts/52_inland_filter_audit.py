"""How the inland-flood label is isolated (progress report): NFIP cause code, NFIP event name, and the HURDAT2 track filter, cross-tabulated.

Texas claims 2010-2023 on the clean claims table, mapped to 2010 block groups. A claim is
  inland-cause   cause code 2/3/4 (river/lake overflow, alluvial fan, rainfall ponding); 1 is tidal, 0 other
  named-TC       flood_event matches wxrisk.flood_pilot.TC_PATTERN
  track-flagged  a tropical-cyclone best-track point within 800 km of the county centroid within +-2 days of the loss date (tc_flag_800_2.npy)
The label uses cause + name; the track flag removes whole episodes (scripts/25) and is the check on claims with no event name.

Run: uv run python scripts/52_inland_filter_audit.py   ->  docs/progress_report/figs/inland_filter.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wxrisk import bg_crosswalk  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402

s = (prep.scan("claims_nfip").filter((pl.col("state") == "TX") & pl.col("geoid").is_not_null())
     .select(pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"), "cause", "flood_event").collect().to_pandas())
s["date"] = pd.to_datetime(s.date)
s = s[(s.date >= "2010-01-01") & (s.date <= "2023-12-31")].copy()
cw = bg_crosswalk.build().set_index("code")
s["bg10"] = s.bg.map(cw.bg10)
s = s[s.bg10.notna()].copy()
s["cty"] = s.bg10.str[:5]
s["named"] = s.flood_event.fillna("").str.contains(fp.TC_PATTERN.replace("(", "(?:").replace("(?:?i)", "(?i)"))
d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
ci = {c: i for i, c in enumerate(d["cty"])}
tc = np.load(fp.PILOT / "tc_flag_800_2.npy")
i = (s.date - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
j = s.cty.map(ci)
ok = j.notna().to_numpy()
s["track"] = False
s.loc[ok, "track"] = tc[i[ok], j[ok].astype(int).to_numpy()]
inl = s[s.cause.isin(["2", "3", "4"])]
out = {
    "claims_total": len(s), "by_cause": s.cause.fillna("missing").value_counts().to_dict(), "inland_cause": len(inl),
    "inland_named": int(inl.named.sum()), "inland_track": int(inl.track.sum()),
    "named_and_track": int((inl.named & inl.track).sum()), "named_not_track": int((inl.named & ~inl.track).sum()),
    "track_not_named": int((~inl.named & inl.track).sum()),
    "track_not_named_without_event_name": int((~inl.named & inl.track & inl.flood_event.isna()).sum()),
    "inland_without_event_name": int(inl.flood_event.isna().sum()),
    "kept": int((~inl.named & ~inl.track).sum()), "kept_by_name_only": int((~inl.named).sum()),
    "top_named_events": inl[inl.named].flood_event.value_counts().head(5).to_dict(),
    "tidal_track_share": float(s[s.cause == "1"].track.mean()),
    "trigger_county_days": int((d["D1"] >= 30).sum()), "trigger_county_days_tc": int(((d["D1"] >= 30) & tc).sum()),
}
p = ROOT / "docs" / "progress_report" / "figs" / "inland_filter.json"
p.write_text(json.dumps(out, indent=1, default=int))
print(json.dumps(out, indent=1, default=int))
