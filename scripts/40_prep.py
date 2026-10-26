"""Data preprocessing pipeline: raw -> clean (-> normalized). Reproducible, with a sidecar per table.

    python scripts/40_prep.py build [table ...]     # clean tables into $WXRISK_DATA/clean/ (skips up-to-date ones)
    python scripts/40_prep.py verify [table ...]    # re-derive from the raw file and compare content hashes
    python scripts/40_prep.py report [table ...]    # row accounting and null handling per column
    python scripts/40_prep.py normalize policies_tx --train-until 2016   # fit scaling on eff.year <= 2016, write params JSON
    python scripts/40_prep.py scaling-plan          # docs/SCALING_PLAN.md from the plan and the scripts/42, 43 evidence
    python scripts/40_prep.py register              # docs/CLEANING_REGISTER.md from the rules, build records and scripts/41 evidence

Normalisation parameters live in $WXRISK_DATA/clean/<table>.norm.json; they are fitted on the training period only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk.prep import build as pb  # noqa: E402
from wxrisk.prep import normalize  # noqa: E402
from wxrisk.prep.specs import SPECS  # noqa: E402

NORM = {"policies_tx": {"cov_building": "log1p_standard", "premium": "log1p_standard",
                        "replacement_cost": "log1p_standard"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "report", "normalize", "register", "scaling-plan"])
    ap.add_argument("tables", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train-until", type=int, default=2016, help="last training year (flood pilot: 2016)")
    a = ap.parse_args()
    if a.cmd == "register":
        from wxrisk.prep import register

        out = Path(__file__).resolve().parents[1] / "docs" / "CLEANING_REGISTER.md"
        register.render(out)
        print(f"wrote {out}")
        return
    if a.cmd == "scaling-plan":
        from wxrisk.prep import scaling_plan

        out = Path(__file__).resolve().parents[1] / "docs" / "SCALING_PLAN.md"
        scaling_plan.render(out)
        print(f"wrote {out}")
        return
    names = a.tables or list(SPECS)
    for n in names:
        spec = SPECS[n]
        if a.cmd == "build":
            r = pb.build(spec, a.force)
            rep = r["report"]
            print(f"{n}: {r['status']}  rows {rep['rows_in']:,} -> {rep['rows_out']:,}  dropped {rep['dropped_null_or_malformed_key']}")
            print(f"   out {r['output_content_sha256'][:16]}  in {r['input_content_sha256'][:16]}  spec {r['spec_sha256'][:16]}")
        elif a.cmd == "verify":
            print(n, json.dumps(pb.verify(spec)))
        elif a.cmd == "report":
            rep = json.loads(pb.paths(spec)[2].read_text())["report"]
            print(f"\n== {n}: {rep['rows_in']:,} rows in, {rep['rows_out']:,} out, dropped {rep['dropped_null_or_malformed_key']}")
            print(f"{'column':22s}{'src_null':>11s}{'unparsable':>12s}{'sentinel/range':>16s}{'null_out':>11s}")
            for c, s in rep["columns"].items():
                print(f"{c:22s}{s['src_null']:>11,}{s['unparsable']:>12,}{s['sentinel_or_out_of_range']:>16,}{s['null_out']:>11,}")
        else:
            df = pl.read_parquet(pb.paths(spec)[1])
            p = normalize.fit(df, NORM[n], pl.col("eff").dt.year() <= a.train_until, f"eff.year <= {a.train_until}")
            out = pb.CLEAN_DIR / f"{n}.norm.json"
            out.write_text(json.dumps(p, indent=1, sort_keys=True) + "\n")
            print(f"{n}: fitted on {p['fit_rows']:,} rows ({p['fit_where']}) -> {out.name} sha {p['sha256'][:16]}")
            for c, v in p["cols"].items():
                print(f"   {c:18s} {v['method']:15s} center {v['center']:.6g} scale {v['scale']:.6g} (n {v['n']:,}, null {v['n_null']:,})")


if __name__ == "__main__":
    main()
