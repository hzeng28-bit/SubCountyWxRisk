"""What happens to the flood pilot when the filters that isolate 'storm-triggered' floods are relaxed? (stage 26)

The stage 8 label keeps inland-cause claims (cause 2/3/4) without a tropical event name, and drops every episode inside a tropical-cyclone
window (800 km, +-2 days). Nothing proves that these filters separate storm-triggered floods from the rest, so this script loosens them one
at a time and measures how much the label, the positives and the model skill move.

  A  rain support of claims by class (cause x tropical name x best-track window): share of claims whose county had >= 30 / 50 mm
     (1-day gridMET maximum) in the 4 days up to the loss date
  B  scope ladder on a table built WITHOUT the tropical exclusion (scripts/25 --tc-file tc_flag_none.npy --out table_notc.parquet):
       S0 non-TC episodes, inland non-named label        (stage 8)
       S1 all episodes,    inland non-named label
       S2 all episodes,    inland label incl. named tropical claims
       S3 all episodes,    claims of every cause
     each model trained 2010-2016 on its own scope and tested 2017-2023; every model is also scored on two fixed evaluation sets
       E0 non-TC test episodes with the stage 8 label, E1 TC test episodes with the label incl. named tropical claims
  within-event AUC (pair weighted), event-clustered paired bootstrap against the S0 model.
Inputs : table_notc.parquet, clean NFIP claims, county_daily.npz, tc_flag_800_2.npy
Output : $WXRISK_DATA/audit/label_scope_relaxation.txt
Run    : python scripts/64_label_scope_relaxation.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import bg_crosswalk  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e26, e48 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py")
OUT = []
D0 = pd.Timestamp("2000-01-01")


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def all_claims() -> pd.DataFrame:
    s = (prep.scan("claims_nfip").filter((pl.col("state") == "TX") & pl.col("geoid").is_not_null())
         .select("id", pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"), "cause", "flood_event").collect().to_pandas())
    s["date"] = pd.to_datetime(s.date)
    s = s[(s.date >= "2010-01-01") & (s.date <= "2023-12-31")]
    cw = bg_crosswalk.build().set_index("code")
    s["bg10"] = s.bg.map(cw.bg10)
    s = s[s.bg10.notna()].copy()
    s["cty"] = s.bg10.str[:5]
    s["named"] = s.flood_event.fillna("").str.contains(fp.TC_PATTERN)
    s["cause"] = s.cause.fillna("missing").astype(str)
    return s


def counts(t: pd.DataFrame, cl: pd.DataFrame) -> np.ndarray:
    eps = t.drop_duplicates("ep")[["cty", "ep", "start", "end"]]
    m = cl.merge(eps, on="cty")
    m = m[(m.date >= m.start) & (m.date <= m.end + pd.Timedelta(days=2))]
    c = m.groupby(["bg10", "ep"]).size().rename("n")
    return t[["bg10", "ep"]].join(c, on=["bg10", "ep"]).n.fillna(0).to_numpy()


def main() -> None:
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    D1, cty = d["D1"], list(d["cty"])
    ci = {c: i for i, c in enumerate(cty)}
    tc = np.load(fp.PILOT / "tc_flag_800_2.npy")
    cl = all_claims()
    cl["track"] = False
    j = cl.cty.map(ci)
    ok = j.notna().to_numpy()
    i = (cl.date - D0).dt.days.to_numpy()
    cl.loc[ok, "track"] = tc[i[ok], j[ok].astype(int).to_numpy()]
    for T in (30, 50):
        sup = np.zeros(len(cl), bool)
        sup[ok] = np.max(np.stack([D1[i[ok] + o, j[ok].astype(int).to_numpy()] for o in range(-3, 1)]), 0) >= T
        cl[f"rain{T}"] = sup

    say("== A. rain support of claims (county 1-day maximum >= T mm in the 4 days up to the loss date), Texas 2010-2023")
    cl["cls"] = np.where(cl.cause.isin(["2", "3", "4"]), "inland(2/3/4)", np.where(cl.cause == "1", "tidal(1)", "other/missing(0)"))
    cl["trop"] = np.where(cl.named, "named tropical", np.where(cl.track, "unnamed, in TC window", "not tropical"))
    a = cl.groupby(["cls", "trop"]).agg(claims=("id", "size"), rain30=("rain30", "mean"), rain50=("rain50", "mean")).round(3)
    say(a.to_string())
    base = cl[(cl.cls == "inland(2/3/4)") & (cl.trop == "not tropical")]
    say(f"  stage 8 label class: {len(base):,} claims, rain support {base.rain30.mean():.3f} (>= 30 mm), {base.rain50.mean():.3f} (>= 50 mm)")
    say(f"  the share of all Texas claims that this class keeps: {len(base) / len(cl):.3f}")

    say("\n== B. scope ladder")
    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    si = ((t.start - D0).dt.days.to_numpy()).astype(int)
    ei = ((t.end - D0).dt.days.to_numpy()).astype(int)
    cj = t.cty.map(ci).to_numpy().astype(int)
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    t["tc_ep"] = (csum[ei + 3, cj] - csum[si, cj]) > 0
    say(f"  table without tropical exclusion: {len(t):,} rows, {t.ep.nunique():,} episodes, of which in a TC window "
        f"{t.drop_duplicates('ep').tc_ep.sum():,}")

    inl = cl[cl.cls == "inland(2/3/4)"]
    n_inl_nn = counts(t, inl[~inl.named])
    n_inl_all = counts(t, inl)
    n_all = counts(t, cl)
    y = {"S0": (n_inl_nn > 0), "S1": (n_inl_nn > 0), "S2": (n_inl_all > 0), "S3": (n_all > 0)}
    rows = {"S0": ~t.tc_ep.to_numpy(), "S1": np.ones(len(t), bool), "S2": np.ones(len(t), bool), "S3": np.ones(len(t), bool)}
    yr, ep = t.year.to_numpy(), t.ep.to_numpy()
    tr0, te0 = yr <= 2016, yr >= 2017
    va0 = (yr >= 2014) & tr0
    cols = e26.COUNTY + e26.EXPO + e26.CELL
    X = t[cols].to_numpy(np.float32)
    say("  positives (train 2010-2016 / test 2017-2023), share of positives in TC episodes, share from the 10 largest episodes:")
    for k in y:
        r = rows[k]
        yy = y[k].astype(float)
        tcs = yy[r & t.tc_ep.to_numpy()].sum() / yy[r].sum()
        evp = pd.Series(yy[r]).groupby(ep[r]).sum().sort_values(ascending=False)
        say(f"    {k}: {int(yy[r & tr0].sum()):6d} / {int(yy[r & te0].sum()):6d}   TC share {tcs:.3f}   top-10 episodes {evp.head(10).sum() / evp.sum():.3f}")

    mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300,
                                                  l2_regularization=1.0, early_stopping=False, random_state=0)
    models = {}
    for k in y:
        r = rows[k]
        trn, vam = r & tr0, r & va0
        yy = y[k].astype(float)
        m = mk(300).fit(X[trn & ~vam], yy[trn & ~vam])
        n = int(np.argmin([np.mean((p[:, 1] - yy[vam]) ** 2) for p in m.staged_predict_proba(X[vam])])) + 1
        models[k] = mk(n).fit(X[trn], yy[trn])
        say(f"    fitted {k} ({n} iterations)")

    E0 = rows["S0"] & te0
    E1 = t.tc_ep.to_numpy() & te0
    ev = {"E0 non-TC episodes, stage 8 label": (E0, y["S0"]), "E1 TC episodes, label incl. named tropical": (E1, y["S2"]),
          "E2 all test episodes, label incl. named tropical": (te0, y["S2"])}
    per = {}
    for k, m in models.items():
        p = m.predict_proba(X)[:, 1]
        for en, (mask, lab) in ev.items():
            per[(k, en)] = e48.per_episode(ep[mask], lab[mask].astype(float), p[mask])
    say("\n  within-event AUC of each model on fixed evaluation sets (difference to the S0 model [95%])")
    for en, (mask, lab) in ev.items():
        say(f"  {en}: {int(lab[mask].sum())} positives, {len(per[('S0', en)][0])} episodes with an AUC")
        for k in models:
            a_ = per[(k, en)]
            w = a_[1].sum() / a_[2].sum()
            dif = "" if k == "S0" else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*e48.boot_diff(per[("S0", en)], a_))
            say(f"    trained on {k}: {w:.4f}   {dif}")

    say("\n  own scope (train and test on the same label scope)")
    for k in y:
        r = rows[k]
        p = models[k].predict_proba(X[r & te0])[:, 1]
        a_ = e48.per_episode(ep[r & te0], y[k][r & te0].astype(float), p)
        say(f"    {k}: within-event AUC {a_[1].sum() / a_[2].sum():.4f} over {len(a_[0])} episodes")
    p = wb2.cache_path("audit") / "label_scope_relaxation.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
