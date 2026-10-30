"""Put NFIP's mixed-vintage 12-digit block-group codes on one geography (2010 block groups).

NFIP records carry census block-group codes that follow the 2010 vintage in early years and the
2020 vintage later, with no vintage flag. Rule per code:
  1. the code exists as a 2010 block group and its land area mostly persisted into the 2020 block
     of the same code            -> keep it (same place)
  2. the code exists only in the 2020 vintage, or exists in both but the area moved
                                  -> map the 2020 block group to the 2010 block group holding the
                                     largest share of its land
  3. otherwise                    -> unmatched
`how` records which rule fired and `share` the land-area share behind the assignment, so callers can
filter by confidence.
"""

from __future__ import annotations

import pandas as pd

from . import wb2

REL = wb2.cache_path("labels_extra") / "flood" / "pilot" / "rel_bg20_bg10_st48.txt"
PERSIST = 0.8  # same code counts as the same place if this much of the smaller block overlaps


def load_relationship(path=None) -> pd.DataFrame:
    cols = {"GEOID_BLKGRP_20": "bg20", "GEOID_BLKGRP_10": "bg10", "AREALAND_BLKGRP_20": "a20",
            "AREALAND_BLKGRP_10": "a10", "AREALAND_PART": "part"}
    # usecols returns file order, not list order, so rename by name rather than by position
    r = pd.read_csv(path or REL, sep="|", dtype=str, encoding="utf-8-sig", usecols=list(cols)).rename(columns=cols)
    for c in ("a20", "a10", "part"):
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)
    return r[(r.part > 0)]


def build(path=None) -> pd.DataFrame:
    """`path`: the relationship file of another state (wxrisk.states); default Texas."""
    r = load_relationship(path)
    # 2020 block group -> the 2010 block group that holds most of its land
    best = r.sort_values("part", ascending=False).drop_duplicates("bg20")
    to10 = best.assign(share=best.part / best.a20.where(best.a20 > 0)).set_index("bg20")[["bg10", "share"]]
    same = r[r.bg20 == r.bg10].assign(persist=lambda x: x.part / x[["a20", "a10"]].min(axis=1).where(lambda s: s > 0))
    persist = same.set_index("bg10").persist
    codes10 = set(r.bg10)
    rows = []
    for code in sorted(codes10 | set(r.bg20)):
        if code in codes10 and code in persist.index and persist[code] >= PERSIST:
            rows.append((code, code, "same_code_same_place", float(persist[code])))
        elif code in to10.index:
            rows.append((code, to10.loc[code, "bg10"], "mapped_from_2020", float(to10.loc[code, "share"])))
        elif code in codes10:
            rows.append((code, code, "2010_only", 1.0))
    return pd.DataFrame(rows, columns=["code", "bg10", "how", "share"])
