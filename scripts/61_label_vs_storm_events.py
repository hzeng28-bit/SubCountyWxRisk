"""Does an independent flood report agree with the NFIP label at the episode level? (stage 23, part C)

Episodes of the Texas pilot (county x rain episode) against NWS Storm Events 'Flash Flood' / 'Flood' county records 2010-2023
(a report overlapping [start - 1 d, end + 2 d]). Prints P(report | NFIP positive), P(NFIP positive | report), and both by county
peak rain and by year.
Inputs : flood_pilot_table.parquet, $WXRISK_DATA/storm_events/*.csv.gz
Run    : python scripts/61_label_vs_storm_events.py
"""

from __future__ import annotations

import glob
import sys

import pandas as pd
sys.path.insert(0,"src")
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402
fs=sorted(glob.glob(str(wb2.cache_path("storm_events") / "StormEvents_details*d20[12]*")))
se=pd.concat([pd.read_csv(f,usecols=["STATE_FIPS","CZ_TYPE","CZ_FIPS","EVENT_TYPE","BEGIN_YEARMONTH","BEGIN_DAY","END_YEARMONTH","END_DAY","DAMAGE_PROPERTY"],low_memory=False) for f in fs])
se=se[(se.STATE_FIPS==48)&(se.CZ_TYPE=="C")&se.EVENT_TYPE.isin(["Flash Flood","Flood"])].copy()
se["b"]=pd.to_datetime(se.BEGIN_YEARMONTH.astype(str)+se.BEGIN_DAY.astype(str).str.zfill(2),format="%Y%m%d")
se["e"]=pd.to_datetime(se.END_YEARMONTH.astype(str)+se.END_DAY.astype(str).str.zfill(2),format="%Y%m%d")
se["cty"]="48"+se.CZ_FIPS.astype(int).astype(str).str.zfill(3)
print("SE TX flood events 2010-2023",len(se[se.b.dt.year>=2010]))
t=pd.read_parquet(fp.PILOT/"flood_pilot_table.parquet",columns=["cty","ep","start","end","y","cty_f1_max","year"])
e=t.groupby(["cty","ep"]).agg(start=("start","first"),end=("end","first"),pos=("y","sum"),rain=("cty_f1_max","first"),year=("year","first")).reset_index()
m=e.merge(se[["cty","b","e"]],on="cty",how="left")
hit=((m.b<=m.end+pd.Timedelta(days=2))&(m.e>=m.start-pd.Timedelta(days=1)))
e["se"]=e.set_index(["cty","ep"]).index.isin(m[hit].set_index(["cty","ep"]).index)
e["posep"]=e.pos>0
print("P(SE report | episode has NFIP positive) %.3f ; P(NFIP positive | SE report) %.3f ; n episodes %d"%(e[e.posep].se.mean(),e[e.se].posep.mean(),len(e)))
e["rb"]=pd.cut(e.rain,[0,50,75,100,150,1000])
print(e.groupby("rb",observed=True).agg(n=("se","size"),se=("se","mean"),nfip=("posep","mean")).round(3))
# episodes with SE report but zero NFIP positives, and high rain
print("episodes with rain>=75 & SE report & no NFIP positive: %d of SE-report episodes with rain>=75: %d"%(((e.rain>=75)&e.se&~e.posep).sum(),((e.rain>=75)&e.se).sum()))
print("year-wise P(NFIP pos | SE report):"); print(e[e.se].groupby("year").posep.agg(["size","mean"]).round(3).T.to_string())
