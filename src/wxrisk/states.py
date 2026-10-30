"""Per-state constants for the state-group scale-up (TX + LA + FL, RESULTS stage 27 / docs/CNN_HYBRID_DESIGN.md 10.7).

The Texas pilot keeps its own hard-coded names (`flood_pilot.TX`, `pr_tx_*`); new states use these entries so the
Texas outputs stay byte-identical. Boxes are (south, north, west, east) in degrees and cover the state with a margin
of a few cells, so the neighbourhood features at the border have data.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import wb2

PILOT = wb2.cache_path("labels_extra") / "flood" / "pilot"
CENSUS = "https://www2.census.gov/geo"


@dataclass(frozen=True)
class State:
    abbr: str
    fips: str
    name: str
    box: tuple[float, float, float, float]  # south, north, west, east

    @property
    def low(self) -> str:
        return self.abbr.lower()

    @property
    def policies(self):
        return PILOT / f"policies_{self.low}.parquet"

    @property
    def relationship(self):
        return PILOT / f"rel_bg20_bg10_st{self.fips}.txt"

    @property
    def bg2010(self):
        return PILOT / f"bg2010_{self.low}.zip"

    @property
    def bg2020(self):
        return PILOT / f"bg2020_{self.low}.zip"

    @property
    def urls(self) -> dict:
        return {
            self.relationship.name: f"{CENSUS}/docs/maps-data/data/rel2020/blkgrp/tab20_blkgrp20_blkgrp10_st{self.fips}.txt",
            self.bg2020.name: f"{CENSUS}/tiger/GENZ2020/shp/cb_2020_{self.fips}_bg_500k.zip",
            self.bg2010.name: f"{CENSUS}/tiger/GENZ2010/gz_2010_{self.fips}_150_00_500k.zip",
        }


STATES = {s.abbr: s for s in (
    State("TX", "48", "Texas", (25.8, 36.6, -106.7, -93.4)),
    State("LA", "22", "Louisiana", (28.8, 33.2, -94.2, -88.7)),
    State("FL", "12", "Florida", (24.3, 31.1, -87.8, -79.8)),
)}


def get(abbr: str) -> State:
    return STATES[abbr.upper()]
