"""Quality audit of flood label sources on the CO/NM/AZ county-day grid (May-Sep, 2000-2022).

Sources: Storm Events (Flash Flood / Flood), NWS flash-flood warnings (IEM, 2002+), NFIP claims
(OpenFEMA v3), USGS discharge rises, FEMA flood declarations, RMA crop "Flood" losses.
For each pair we report the share of A county-days that B also flags, and vice versa, on the exact
day and within +-1 day (timing of floods is loose). Also: NFIP take-up and claim sparsity.

    python scripts/18_flood_sources.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wxrisk import labels_flood as lf  # noqa: E402
from wxrisk import wb2  # noqa: E402


def universe() -> pd.DataFrame:
    lab = pd.read_parquet(wb2.cache_path("labels_scs3_conm_az.parquet"), columns=["cty", "date"])
    return lab.sort_values(["cty", "date"]).reset_index(drop=True)


def dilate(u: pd.DataFrame, col: str) -> np.ndarray:
    """+-1 day version of a 0/1 column within each county, aligned with the rows of `u`."""
    x = u.pivot(index="date", columns="cty", values=col).fillna(0).astype(float)
    x = x.reindex(pd.date_range(x.index.min(), x.index.max()), fill_value=0.0)  # no false adjacency across years
    d = (x + x.shift(1, fill_value=0.0) + x.shift(-1, fill_value=0.0)) > 0
    return d.stack().reindex(pd.MultiIndex.from_frame(u[["date", "cty"]])).to_numpy()


def build() -> pd.DataFrame:
    u = universe()
    se = lf.se_flood_days()
    nf = lf.nfip_days()
    g = lf.gauge_rise_days()
    fe = lf.fema_flood_days()
    w = pd.read_parquet(wb2.cache_path("warnings_county_day.parquet"))
    for t in (se, nf, g, fe, w):
        t["date"] = pd.to_datetime(t["date"])
    u = u.merge(se[["cty", "date", "f_ff", "f_fl", "f_any", "f_10k", "f_100k"]], how="left", on=["cty", "date"])
    u = u.merge(nf[["cty", "date", "n_claims", "paid"]], how="left", on=["cty", "date"])
    u = u.merge(g[["cty", "date", "g_rise"]], how="left", on=["cty", "date"])
    u = u.merge(fe, how="left", on=["cty", "date"])
    u = u.merge(w[["cty", "date", "n_ff"]], how="left", on=["cty", "date"])
    gauged = set(g.cty)
    u["has_gauge"] = u.cty.isin(gauged)
    for c in ("f_ff", "f_fl", "f_any", "f_10k", "f_100k", "g_rise", "h_decl"):
        u[c] = u[c].fillna(0).astype(np.int8)
    u["nfip"] = (u.n_claims.fillna(0) > 0).astype(np.int8)
    u["w_ff"] = (u.n_ff.fillna(0) > 0).astype(np.int8)
    u["year"] = u.date.dt.year
    return u


def agree(u: pd.DataFrame, a: str, b: str, mask=None) -> dict:
    v = u if mask is None else u[mask]
    x, y = v[a].to_numpy() > 0, v[b].to_numpy() > 0
    xd, yd = dilate(v, a), dilate(v, b)
    return {"A": a, "B": b, "n": len(x), "A_per1e4": x.mean() * 1e4, "B_per1e4": y.mean() * 1e4,
            "P(B|A)": y[x].mean() if x.any() else np.nan, "P(A|B)": x[y].mean() if y.any() else np.nan,
            "P(B|A) +-1d": yd[x].mean() if x.any() else np.nan,
            "P(A|B) +-1d": xd[y].mean() if y.any() else np.nan,
            "lift": y[x].mean() / y.mean() if x.any() and y.any() else np.nan}


def section_nfip(u: pd.DataFrame) -> None:
    print("\n== NFIP take-up and claim sparsity")
    p = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "nfip_penetration.parquet")
    p = p[p.fipsCode.isin(u.cty.unique())]
    print(f"counties matched {len(p)}; residential penetration rate: median {p.resPenetrationRate.median():.4f}, "
          f"mean {p.resPenetrationRate.mean():.4f}, share of counties < 1%: {(p.resPenetrationRate < 0.01).mean():.2f}, "
          f"max {p.resPenetrationRate.max():.3f}")
    print(f"contracts in force: {int(p.resContractsInForce.sum())} of {int(p.totalResStructures.sum())} residential structures "
          f"({p.resContractsInForce.sum() / p.totalResStructures.sum():.4f})")
    by = u.groupby("year").agg(claims=("n_claims", "sum"), claim_days=("nfip", "sum"))
    print("claims per year (May-Sep):", by.claims.astype(int).to_dict())
    print(f"claim county-days {int(u.nfip.sum())} of {len(u)} ({u.nfip.mean() * 1e4:.1f} per 1e4); "
          f"top-5 days hold {u.groupby('date').n_claims.sum().nlargest(5).sum() / u.n_claims.sum():.2f} of all claims")


def section_pairs(u: pd.DataFrame) -> None:
    print("\n== Pairwise agreement of flood sources (county-days)")
    rows = []
    in_w = (u.year >= 2002).to_numpy()
    for a, b, m in (("f_ff", "w_ff", in_w), ("f_any", "w_ff", in_w), ("f_any", "nfip", None), ("w_ff", "nfip", in_w),
                    ("f_any", "h_decl", None), ("nfip", "h_decl", None),
                    ("f_any", "g_rise", u.has_gauge.to_numpy()), ("w_ff", "g_rise", (u.has_gauge & (u.year >= 2002)).to_numpy())):
        rows.append(agree(u, a, b, m))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def section_rma(u: pd.DataFrame) -> None:
    print("\n== RMA crop 'Flood' losses vs Storm Events flood (county-months, May-Sep)")
    spec = importlib.util.spec_from_file_location("agree15", ROOT / "scripts" / "15_source_agreement.py")
    m15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m15)
    r = m15.rma_month_losses()
    print("flood-like causes:", sorted(c for c in r.cause.unique() if "flood" in c.lower()))
    r = r[r.lyear.between(2000, 2022) & r.month.between(5, 9) & (r.cause == "Flood")]
    m = r.groupby(["cty", "lyear", "month"], as_index=False).indem.sum()
    t = u.assign(lyear=u.date.dt.year, month=u.date.dt.month).groupby(["cty", "lyear", "month"], as_index=False).agg(
        f_any=("f_any", "max"), f_ff=("f_ff", "max"), nfip=("nfip", "max"), n=("f_any", "size"))
    t = t.merge(m, how="left", on=["cty", "lyear", "month"]).fillna({"indem": 0})
    pos = t.indem > 0
    print(f"county-months {len(t)}; with RMA flood indemnity {int(pos.sum())}; with SE flood/flash flood {int((t.f_any > 0).sum())}")
    print(f"P(SE flood | RMA flood) = {(t.f_any[pos] > 0).mean():.3f}; P(RMA flood | SE flood) = {pos[t.f_any > 0].mean():.3f}; "
          f"P(NFIP claim | RMA flood) = {(t.nfip[pos] > 0).mean():.3f}")


def section_trend(u: pd.DataFrame) -> None:
    print("\n== Stationarity of the labels")
    by = u.groupby("year")[["f_ff", "f_any", "w_ff", "nfip", "f_10k"]].mean() * 1e4
    print(by.round(1).T.to_string())
    for c in ("f_ff", "f_10k", "nfip"):
        s = by[c]
        print(f"{c}: 2000-2010 / 2012-2022 mean rate ratio = {s.loc[:2010].mean() / s.loc[2012:].mean():.2f}; "
              f"max-year share of all positives = {(u.groupby('year')[c].sum().max() / u[c].sum()):.2f}")


def main() -> None:
    u = build()
    print(f"universe {len(u)} county-days; positives: " + ", ".join(
        f"{c} {int(u[c].sum())}" for c in ("f_ff", "f_fl", "f_any", "f_10k", "f_100k", "w_ff", "nfip", "g_rise", "h_decl")))
    print(f"counties with a USGS gauge: {u[u.has_gauge].cty.nunique()} of {u.cty.nunique()}")
    section_pairs(u)
    section_nfip(u)
    section_rma(u)
    section_trend(u)
    u.to_parquet(wb2.cache_path("flood_labels_county_day.parquet"))


if __name__ == "__main__":
    main()
