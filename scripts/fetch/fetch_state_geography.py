"""Block-group geography for the candidate states: TIGER cartographic block groups (2010, 2020) and the Census
2020 -> 2010 block-group relationship file (input of `bg_crosswalk`), for every state in `wxrisk.states` but Texas
(Texas is fetched by scripts/22).

NFIP block-group codes mix the 2010 and 2020 vintages with no flag, so both vintages and the relationship are kept.

    python scripts/fetch/fetch_state_geography.py LA FL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, states  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    a = ap.parse_args()
    states.PILOT.mkdir(parents=True, exist_ok=True)
    jobs = [fetch.Job(states.PILOT / name, url=url) for s in a.states for name, url in states.get(s).urls.items()]
    res = fetch.fetch_many(jobs, workers=3)
    if bad := {k: v for k, v in res.items() if v.startswith("FAIL")}:
        sys.exit(f"{len(bad)} downloads failed: {bad}")


if __name__ == "__main__":
    main()
