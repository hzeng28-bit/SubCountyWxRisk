"""Builds the self-contained Colab notebook of the county viewer (docs/GEE_PLAN.md, display layer; no Earth Engine needed).

Goal: one small notebook plus one data file. Embedding the data in a notebook cell (a 6 MB single line) froze the Colab editor, so the data now live in
`county_viewer_data.zip` (table, events, county polygons; about 4.5 MB; written next to the data, never committed). The notebook loads it from, in order, the
working directory, a Google Drive file (link-shared, downloaded with gdown, no Drive mount and no sign-in), or an upload prompt, and checks the SHA-256 that is
written into the notebook at build time. The viewer is a plotly figure with native frames (play, pause, slider, hover sheet; the animation runs in the browser,
no round trip), a ranking-curve panel against the ground truth, a county inspector and a pandas analysis section (hits and an event-clustered interval
recomputed from the raw columns).

Inputs : $WXRISK_DATA/gee_export/county_event_tx.csv, events_index.csv, county_geometry.geojson (scripts 110, 112)
Output : notebooks/colab_county_viewer.ipynb (no stored outputs) and $WXRISK_DATA/gee_export/county_viewer_data.zip
Run    : python scripts/113_build_colab_notebook.py [--drive-id <Drive file id of the uploaded zip>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "colab_county_viewer.ipynb"
cells: list[dict] = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s: str, hidden: bool = False) -> None:
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- data
d = wb2.cache_path("gee_export")
C = pd.read_csv(d / "county_event_tx.csv", dtype={"fips": str, "ev": str})
C = C[C.big == 1].drop(columns=["big", "exp_pol"]).sort_values(["ev", "fips"])
f = [c for c in C.select_dtypes("float").columns if c not in ("exp_cnn", "exp_boost", "n_pol", "paid")]
C[f] = C[f].round(3)   # scores, policy counts and payments keep full precision: rounding the scores creates ties and changes the top-K flags
E = pd.read_csv(d / "events_index.csv", dtype={"ev": str})
E = E[E.n_claimed_bg >= 10].sort_values("ev")
assert set(E.ev) == set(C.ev) and len(E) == 394, (len(E), C.ev.nunique())
G = json.loads((d / "county_geometry.geojson").read_text())
G["features"] = [x for x in G["features"] if x["properties"]["STATEFP"] == "48"]
ap = argparse.ArgumentParser()
ap.add_argument("--drive-id", default="", help="Google Drive file id of county_viewer_data.zip (link-shared); empty = the notebook asks for an upload")
args = ap.parse_args()
ZIP = d / "county_viewer_data.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:   # fixed timestamps: the same data give the same bytes and hash
    for name, raw in (("table.csv", C.to_csv(index=False, float_format="%.9g").encode()), ("events.csv", E.to_csv(index=False).encode()),
                      ("county_geo.geojson", json.dumps(G, separators=(",", ":")).encode())):
        z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), raw, zipfile.ZIP_DEFLATED, compresslevel=9)
SHA = hashlib.sha256(ZIP.read_bytes()).hexdigest()
print(f"data file {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB) sha256 {SHA[:16]}...")

# ---------------------------------------------------------------- notebook
md("""
# nanoWxRisk: Texas flood model, interactive county viewer

This notebook plus **one data file** (`county_viewer_data.zip`, about 4.5 MB: the 87,780 county-event rows, the Texas county polygons and the event metrics) is everything: the figures and the analysis are computed here from that file. Share the notebook together with the data link. The only package is `plotly`, which Colab has; the data file is fetched with `gdown` (also in Colab) from a link-shared Drive file, or uploaded by hand, and checked against a SHA-256.

**What it shows.** The path of the model from input to result for one flood event at a time (Texas, test years 2017-2023, the 394 events with at least 10 claimed block groups, observed-rain mode, scope S0), aggregated to counties because NFIP block-group codes mix 2010 and 2020 vintages (county = first 5 digits, stable across vintages; county score = sum of block-group scores):

1. **Input**: rain (gridMET, observed), exposure (policies, flood-zone share, single-family share, coverage, claim history) and the *model-ready* inputs the CNN's tabular branch actually receives.
2. **Intermediate**: summed block-group scores of the CNN, the boosting model and the policies-only baseline.
3. **Result**: each model's rank in the event and its top-K flags against the **ground truth** (NFIP claims, observed weeks later, never a model input).

**Read first.**
- Scores are shown as ranks or on a relative scale. The model's absolute probabilities are not calibrated (training down-samples unclaimed rows) and must not be read as "x %".
- This is the *observed-rain* mode, a damage-assessment view. In forecast mode the county ranking is no better than policies-only (AUC 0.82 vs 0.82).
- The CNN is a 2-seed mean; seed spread can be as large as the model differences (AGENTS.md).
- "Top-K flagged" takes K from the ground truth (the number of counties that actually had claims) to make the models comparable at equal effort. It is a diagnostic, not a rule a forecaster could apply in advance.

**Use.** Run the cells top to bottom (Runtime > Run all). In the figure: drag the slider, press a play button, hover a county for its sheet. The *Layer* dropdown of the form cell in section 3 chooses the layer (the cell re-runs when you change it). Section 5 is a county inspector (also form dropdowns), section 6 recomputes the headline numbers from the raw columns.
""")

md("## 0. Setup")
code("""
import sys, subprocess
try:
    import plotly
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'plotly'], check=True)
    import plotly
print('plotly', plotly.__version__)
""")

md("""
## 1. Get the data

The loader looks, in order, for `county_viewer_data.zip` in the working directory (in Colab: the Files panel on the left), then downloads it from the Google Drive file whose id is in `DRIVE_FILE_ID`, then (in Colab) opens an upload dialog. The file is checked against the SHA-256 written into this notebook when it was built, so a wrong or changed file is refused.
""")
code("""
#@title Get the data { display-mode: "form" }
DRIVE_FILE_ID = '__DRIVE_ID__'   #@param {type:"string"}
EXPECT_SHA256 = '__SHA__'
""".replace("__DRIVE_ID__", args.drive_id).replace("__SHA__", SHA) + r"""
import os, sys, hashlib, zipfile, subprocess
FN = 'county_viewer_data.zip'
if not os.path.exists(FN):
    if DRIVE_FILE_ID:
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True); import gdown
        gdown.download(id=DRIVE_FILE_ID, output=FN, quiet=True)
    elif 'google.colab' in sys.modules:
        from google.colab import files
        up = files.upload()                       # choose county_viewer_data.zip
        if up and FN not in up: os.replace(next(iter(up)), FN)
if not os.path.exists(FN):
    raise SystemExit('Data file not found: put county_viewer_data.zip in the working directory, or set DRIVE_FILE_ID to the link-shared Drive file.')
got = hashlib.sha256(open(FN, 'rb').read()).hexdigest()
assert got == EXPECT_SHA256, f'{FN} does not match this notebook (sha256 {got[:12]}... expected {EXPECT_SHA256[:12]}...). Re-download or rebuild it.'
Z = zipfile.ZipFile(FN)
print('data file ok:', FN, f'({os.path.getsize(FN) / 1e6:.1f} MB)')
""")

md("""
## 2. Data dictionary

Every column of the table, in the order of the path from input to result. `log` columns are coloured on a log10 scale. Block-group quantities are aggregated to the county as described in each row.
""")
code(r'''
import json
import numpy as np, pandas as pd

T = pd.read_csv(Z.open('table.csv'), dtype={'ev': str, 'fips': str})
T['exp_pol'] = T['n_pol']                       # policies-only baseline score = policies in force
EV = pd.read_csv(Z.open('events.csv'), dtype={'ev': str}).sort_values('ev').reset_index(drop=True)
EV['date'] = pd.to_datetime(EV.ev, format='%Y%m%d')
GEO = json.load(Z.open('county_geo.geojson'))
CTY = [f['properties']['GEOID'] for f in GEO['features']]
NAME = {f['properties']['GEOID']: f['properties']['NAME'] for f in GEO['features']}
T_by_ev = {ev: g.set_index('fips') for ev, g in T.groupby('ev')}
print(f"{len(T):,} county-event rows, {T.ev.nunique()} events {EV.date.min().date()} to {EV.date.max().date()}, {len(CTY)} counties")

SEQ = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
BLU = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
GRN = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c']
CLS = {0: ('neither', '#e0e0e0', '-'), 1: ('hit: flagged and had claims', '#1a9850', 'H'),
       2: ('missed: had claims, not flagged', '#f46d43', 'M'), 3: ('false alarm: flagged, no claims', '#7b3294', 'F')}

def V(t, d, pal=None, vmin=0, vmax=1, log=False, eps=1.0, cat=False):
    return dict(t=t, d=d, pal=pal, vmin=vmin, vmax=vmax, log=log, eps=eps, cat=cat)

GROUPS = {
 '1 Input: rain': {
  'rain_cty_max': V('County max rain (mm)', 'Largest block-group-pixel value, over the county, of the peak daily rain in the 2-day window S, S+1 (gridMET 4 km, observed). Boosting feature "county max".', BLU, 0, 150),
  'rain_cty_mean': V('County mean rain (mm)', 'Mean over the county of the same pixel peak. Boosting feature "county mean".', BLU, 0, 60),
  'rain_sum_mean': V('Two-day rain total, mean (mm)', 'Two-day total rain at the block-group pixels, averaged over the county. The CNN reads the 4 km peak and sum maps (log1p scaled).', BLU, 0, 100)},
 '2 Input: exposure': {
  'n_pol': V('Policies in force', 'NFIP policies in force on day S (effective date <= S < termination or cancellation), clean policy table, summed over the county.', GRN, 0, 5.5, True),
  'sfha_share': V('Share in flood hazard area', 'Share of those policies rated in a Special Flood Hazard Area (zone A* or V*), policy-weighted.', GRN, 0, 1),
  'single_share': V('Share single-family', 'Share with single-family occupancy (codes 1 and 11), policy-weighted.', GRN, 0.4, 1),
  'mean_cov': V('Mean building coverage ($ M)', 'Mean insured building coverage per policy in millions of dollars, policy-weighted. Insured amount, not property value.', GRN, 0.1, 0.35),
  'hist00': V('Claims 2000-2009 (history)', 'NFIP claims in the county in 2000-2009 (claim history prior).', GRN, 0, 3, True),
  'prior3': V('Claims in the 3 years before S', 'NFIP claims in the county in the 3 years before the event.', GRN, 0, 3.5, True),
  'in_log_pol': V('MODEL INPUT: log policies per block group', "What the CNN's tabular branch receives: log(policies in force) per block group, averaged unweighted over the county's block groups.", GRN, 0.5, 4.2),
  'in_sfha': V('MODEL INPUT: flood-zone share (block-group mean)', 'Flood-zone share per block group as the model reads it, averaged unweighted over the county.', GRN, 0, 1),
  'in_single': V('MODEL INPUT: single-family share (block-group mean)', 'Single-family share per block group as the model reads it, averaged unweighted.', GRN, 0.4, 1),
  'in_cov': V('MODEL INPUT: mean coverage (block-group mean, $ M)', 'Mean building coverage per policy per block group as the model reads it, averaged over the county.', GRN, 0.05, 0.3),
  'in_log_nbg': V('MODEL INPUT: log(1 + block groups in county)', 'Size of the county in block groups, log1p scale, as the model reads it.', GRN, 0.5, 5.5),
  'in_hist00': V('MODEL INPUT: log1p claims 2000-2009 (block-group mean)', 'Claim-history prior per block group, log1p scale, averaged over the county. The model standardises all nine inputs with training-year statistics.', GRN, 0, 1.5),
  'in_prior3': V('MODEL INPUT: log1p claims, previous 3 years (block-group mean)', 'Recent claim history per block group, log1p scale, averaged over the county. (Month enters as sin/cos, the same for all counties.)', GRN, 0, 1.2)},
 '3 Intermediate: model output': {
  'exp_cnn': V('CNN: summed block-group scores', 'Sum over block groups of the CNN claim score (mean of 2 seeds). RELATIVE scale only: not a calibrated probability or a count of claims.', SEQ, -3.5, 0, True, 1e-4),
  'exp_boost': V('Boosting: summed block-group scores', 'Same for the gradient-boosting model on the hand-built features. Relative scale only.', SEQ, -3.5, 0, True, 1e-4),
  'exp_pol': V('Policies only (baseline score)', 'Exposure-only baseline: the score is the number of policies in force. What any model has to beat.', SEQ, 0, 5.5, True)},
 '4 Result: rank and ground-truth comparison': {
  'pct_cnn': V('CNN rank (1 = highest)', 'Percentile of the county among all counties of this event by the CNN score. This is the product: an ordering of units.', SEQ, 0, 1),
  'pct_boost': V('Boosting rank', 'Same for the boosting model.', SEQ, 0, 1),
  'pct_pol': V('Policies-only rank', 'Same for the exposure-only baseline. Compare with the CNN rank to see what the model adds beyond "more policies".', SEQ, 0, 1),
  'class_cnn': V('CNN vs ground truth (top-K flagged)', 'The CNN flags its top-K counties, K = number of counties that actually had claims. Green hit, orange missed, purple false alarm.', cat=True),
  'class_boost': V('Boosting vs ground truth (top-K flagged)', 'Same comparison for the boosting model.', cat=True),
  'class_pol': V('Policies-only vs ground truth (top-K flagged)', 'Same for the exposure-only baseline: flag the K counties with the most policies. The CNN has to beat this.', cat=True)},
 '5 Ground truth: NFIP claims (observed)': {
  'n_claims': V('Claims in S to S+3', 'NFIP claims with a loss date in the window (tropical-named claims removed, scope S0). Retrospective truth; arrives weeks later; never a model input.', BLU, 0, 3, True),
  'n_cbg': V('Claimed block groups', 'Block groups in the county with at least one claim. The model ranks block groups; this map aggregates them.', BLU, 0, 15),
  'paid': V('Building amount paid (USD)', 'Building claim payments in the window, summed over the county. Basement items are covered only in part, so payments understate basement damage.', BLU, 0, 6.5, True),
  'claims_per_1k': V('Claims per 1,000 policies', 'Observed claims divided by policies in force: how hard the county was hit relative to its exposure.', BLU, -1, 1.5, True, 0.1)}}
ALLVARS = {k: dict(s, group=g) for g, vs in GROUPS.items() for k, s in vs.items()}
EXTRA = {'n_bg': 'Block groups with a policy that day', 'in_month_sin': 'MODEL INPUT: sin of the event month (same for all counties)',
         'in_month_cos': 'MODEL INPUT: cos of the event month (same for all counties)'}
DICT = pd.DataFrame([dict(column=k, group=s['group'], name=s['t'], scale='log10' if s['log'] else ('class' if s['cat'] else 'linear'), meaning=s['d']) for k, s in ALLVARS.items()]
                    + [dict(column=k, group='(other)', name=v, scale='linear', meaning=v) for k, v in EXTRA.items()])
pd.set_option('display.max_colwidth', 140)
DICT
''')

md("""
## 3. The viewer

Pick a layer, then use the figure's slider and play buttons (the animation runs in your browser, no waiting). Left: the county map of the chosen variable for the event. Right: the ranking curve against the ground truth: flag counties from the highest score down; how much of the observed claims do the first x % of counties hold? The dashed line is a random order; a curve above the policies-only curve is what the model adds.
""")
code(r'''
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML

N = len(EV)
START = int(EV.index[EV.ev == '20190604'][0]) if (EV.ev == '20190604').any() else 0   # the Houston event of the scenario report
LAB = {'cnn': 'CNN', 'boost': 'Boosting', 'pol': 'Policies only'}
COL = {'cnn': '#d1495b', 'boost': '#4c78a8', 'pol': '#e8a33d'}

def curve(score, y):
    order = np.argsort(-np.asarray(score, float), kind='stable')   # highest score first, ties by position
    tot = y.sum()
    return np.r_[0, np.cumsum(y[order]) / tot] if tot else np.zeros(len(y) + 1)

def event_payload(ev, key, spec):
    g = T_by_ev[ev].reindex(CTY)
    raw = g[key]
    if spec['cat']:
        z = raw.fillna(-1).astype(int)
        shown = raw.map(lambda v: CLS[int(v)][0] if pd.notna(v) else '')
    else:
        z = (np.log10(raw + spec['eps']) if spec['log'] else raw).round(3)
        shown = raw.map(lambda v: f'{v:.4g}' if pd.notna(v) else '')
    ints = lambda s, k=1: (s * k).round(0).fillna(0).astype(int)   # integers keep the figure small (the hover sheet is sent for every frame)
    codes = g.class_cnn.map(lambda v: CLS[int(v)][2] if pd.notna(v) else '') + g.class_boost.map(lambda v: CLS[int(v)][2] if pd.notna(v) else '') \
        + g.class_pol.map(lambda v: CLS[int(v)][2] if pd.notna(v) else '')
    cd = pd.DataFrame({'a': shown, 'b': ints(g.n_pol), 'c': ints(g.rain_cty_max), 'd': ints(g.pct_cnn, 100), 'e': ints(g.pct_boost, 100),
                       'f': ints(g.pct_pol, 100), 'g': ints(g.n_claims), 'h': codes}).to_numpy(dtype=object)
    d = T_by_ev[ev]; y = d.n_claims.to_numpy(float); n = len(y)
    curves = [(1 / n, curve(d['exp_' + m].to_numpy(), y).round(3)) for m in ('cnn', 'boost', 'pol')]   # x is implicit: x0 = 0, dx = 1 / n
    r = EV[EV.ev == ev].iloc[0]
    ga = lambda k: f'{r[k]:.2f}' if k in r and pd.notna(r[k]) else 'n/a'
    title = (f"<b>{r.date:%Y-%m-%d}</b>  (event {int(EV.index[EV.ev == ev][0]) + 1} of {N}, {int(r.n_claimed_bg)} claimed block groups, {n} counties, {int(y.sum())} claims)"
             f"<br><sup>Top-{int(r.k_claimed_cty)} counties flagged, hits: CNN {int(r.hit_cnn)} | boosting {int(r.hit_boost)} | policies only {int(r.hit_pol)}"
             f"   ·   county AUC: CNN {ga('cnn_cty_auc')} | boosting {ga('boost_cty_auc')} | policies {ga('pol_cty_auc')}</sup>")
    return z.tolist(), cd.tolist(), curves, title

def build_figure(key, speed_ms=(2500, 1200, 500)):
    spec = ALLVARS[key]
    if spec['cat']:
        c = [CLS[i][1] for i in (0, 1, 2, 3)]
        scale = [[0, c[0]], [.25, c[0]], [.25, c[1]], [.5, c[1]], [.5, c[2]], [.75, c[2]], [.75, c[3]], [1, c[3]]]
        zmin, zmax = -0.5, 3.5
        cbar = dict(title=dict(text='vs ground truth', side='right'), tickvals=[0, 1, 2, 3], ticktext=['neither', 'hit', 'missed', 'false alarm'], len=0.8)
    else:
        n = len(spec['pal']); scale = [[i / (n - 1), col] for i, col in enumerate(spec['pal'])]
        zmin, zmax = spec['vmin'], spec['vmax']
        cbar = dict(title=dict(text=spec['t'] + (' (log scale)' if spec['log'] else ''), side='right'), len=0.8)
        if spec['log']:
            tv = np.arange(np.ceil(zmin), np.floor(zmax) + 1); cbar.update(tickvals=tv.tolist(), ticktext=[f'1e{int(v)}' for v in tv])
    hover = ('<b>%{text}</b><br>' + spec['t'] + ': %{customdata[0]}<br>policies in force: %{customdata[1]:,.0f} | max rain: %{customdata[2]} mm<br>'
             'rank percentile (100 = highest): CNN %{customdata[3]} | boosting %{customdata[4]} | policies %{customdata[5]}<br>'
             'observed claims: %{customdata[6]:,.0f}<br>vs truth, CNN / boosting / policies: %{customdata[7]}'
             '<br>(H hit, M missed, F false alarm, - neither)<extra></extra>')
    names = [NAME[c] + ' County' for c in CTY]
    frames, steps, first_of_year = [], [], set()
    for i, ev in enumerate(EV.ev):
        z, cd, curves, title = event_payload(ev, key, spec)
        frames.append(go.Frame(name=ev, traces=[1, 2, 3, 4], layout=go.Layout(title_text=title),
                               data=[go.Choropleth(z=z, customdata=cd)] + [go.Scatter(y=y, x0=0, dx=dx) for dx, y in curves]))
        yr = ev[:4]; lab = yr if yr not in first_of_year else ''; first_of_year.add(yr)
        steps.append(dict(method='animate', label=lab, args=[[ev], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))
    z, cd, curves, title = event_payload(EV.ev[START], key, spec)
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'choropleth'}, {'type': 'xy'}]], column_widths=[0.6, 0.4], horizontal_spacing=0.06)
    fig.add_trace(go.Choropleth(geojson=GEO, featureidkey='properties.GEOID', locations=CTY, z=[0] * len(CTY), colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                                showscale=False, marker_line_color='#bbbbbb', marker_line_width=0.4, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Choropleth(geojson=GEO, featureidkey='properties.GEOID', locations=CTY, z=z, customdata=cd, text=names, colorscale=scale, zmin=zmin, zmax=zmax,
                                colorbar=cbar, marker_line_color='#666666', marker_line_width=0.4, hovertemplate=hover), row=1, col=1)
    for (dx, y), m in zip(curves, ('cnn', 'boost', 'pol')):
        fig.add_trace(go.Scatter(y=y, x0=0, dx=dx, mode='lines', name=LAB[m], line=dict(color=COL[m], width=3 if m == 'cnn' else 2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#888', dash='dash')), row=1, col=2)
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_xaxes(title_text='share of counties flagged, highest score first', range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text='share of observed claims captured', range=[0, 1.02], row=1, col=2)
    fig.frames = frames
    play = lambda ms: dict(frame=dict(duration=ms, redraw=True), fromcurrent=True, transition=dict(duration=0), mode='immediate')
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor='left', font=dict(size=14)), height=640, margin=dict(t=95, b=70, l=20, r=20), legend=dict(orientation='h', y=-0.12, x=0.62),
        updatemenus=[dict(type='buttons', direction='left', x=0.01, y=-0.04, xanchor='left', yanchor='top', pad=dict(t=0), buttons=[
            dict(label='Play (slow)', method='animate', args=[None, play(speed_ms[0])]), dict(label='Play (normal)', method='animate', args=[None, play(speed_ms[1])]),
            dict(label='Play (fast)', method='animate', args=[None, play(speed_ms[2])]),
            dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])],
        sliders=[dict(active=START, steps=steps, x=0.01, len=0.98, y=-0.17, pad=dict(t=0, b=0), currentvalue=dict(visible=False), ticklen=3, font=dict(size=10))])
    return fig
''')

code("__VIEWER_FORM__")


md("""
## 4. Where the numbers come from

| Step | Script | What it does |
|---|---|---|
| exposure and claims | `scripts/40_prep.py`, `src/wxrisk/flood_pilot.py` | clean NFIP policies and claims; policies in force per block group and day; claim history |
| rain | gridMET 4 km (script 84) | peak and two-day sum over S and S+1 at each block group's pixel |
| models | `scripts/85` (boosting), `scripts/89` (CNN, pooled TX+LA+FL training) | block-group claim scores for the test days 2017-2023 |
| evaluation | `scripts/94_event_hierarchy_eval.py` | per-event AUC and top-k capture (the metrics in the figure title) |
| this table | `scripts/112_gee_county_events.py` | aggregates block-group rows to counties and adds the ground truth and the top-K comparison |
| this notebook | `scripts/113_build_colab_notebook.py` | embeds the table and builds the viewer |

Caveats (see `RESULTS.md`): the county map is an aggregate of a block-group ranking; the CNN is a 2-seed mean; the test years are 2017-2023 (train 2010-2016); events within a year are dependent; absolute probabilities do not transfer across years; payments understate basement loss.
""")

md("## 5. County inspector\n\nPick an event and a county in the form (the cell re-runs when you change it) for the full sheet, grouped from input to result.")
code("__INSPECTOR__")

md("""
## 6. Recompute the headline numbers from the raw columns

K = number of counties with claims in the event; each model flags its top-K counties by score. Hit rate = hits / K. The interval is a bootstrap over events (events are the unit; they are not independent within a year, so treat the interval as a guide, not a test).
""")
code(r'''
rows = []
for ev, g in T.groupby('ev'):
    K = int((g.n_claims > 0).sum())
    hit = {m: int((g.nlargest(K, 'exp_' + m, keep='first').n_claims > 0).sum()) for m in ('cnn', 'boost', 'pol')}
    rows.append(dict(ev=ev, K=K, **hit))
H = pd.DataFrame(rows)
print(f"{len(H)} events, K total {H.K.sum()}")
for m in ('cnn', 'boost', 'pol'):
    print(f"  {LAB[m]:14s} hits {H[m].sum():5d}  hit rate {H[m].sum() / H.K.sum():.1%}")

rng = np.random.default_rng(0)
def boot(a, b, n=4000):
    d = (H[a] / H.K - H[b] / H.K).to_numpy()
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, [2.5, 97.5])
for a, b in (('cnn', 'pol'), ('cnn', 'boost'), ('boost', 'pol')):
    m, (lo, hi) = boot(a, b)
    print(f"mean per-event hit-rate difference {LAB[a]} - {LAB[b]}: {m:+.3f}  (95% over events [{lo:+.3f}, {hi:+.3f}])")
''')


import re
dict_src = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "GROUPS = {" in "".join(c["source"]))
labels, grp = [], None
for line in dict_src.splitlines():
    mg = re.match(r"^ '(\d [^']+)': \{", line)
    if mg: grp = mg.group(1); continue
    mv = re.match(r"^\s+'(\w+)': V\('([^']*)'", line)
    if mv and grp: labels.append(f"[{grp}] {mv.group(2)}")
assert len(labels) == 29, len(labels)
DEFAULT = next(x for x in labels if x.endswith("CNN vs ground truth (top-K flagged)"))

viewer = """
#@title Layer { run: "auto" }
LAYER = "__DEFAULT__"  #@param __OPTIONS__
SAVE_HTML = False  #@param {type:"boolean"}
LABEL2KEY = {f"[{g}] {s['t']}": k for g, vs in GROUPS.items() for k, s in vs.items()}
key = LABEL2KEY[LAYER]
display(HTML(f"<div style='font-size:13px;max-width:1000px'><b>{ALLVARS[key]['t']}</b>: {ALLVARS[key]['d']}</div>"))
fig = build_figure(key)
fig.show()                  # plain cell output (no widget container): Colab renders plotly here
if SAVE_HTML:               # a standalone interactive page of this layer, to open in any browser
    fig.write_html('viewer.html', include_plotlyjs='cdn')
    try:
        from google.colab import files; files.download('viewer.html')
    except ImportError:
        print('saved viewer.html')
""".replace("__DEFAULT__", DEFAULT).replace("__OPTIONS__", json.dumps(labels))
vi = next(i for i, c in enumerate(cells) if "__VIEWER_FORM__" in "".join(c["source"]))
cells[vi] = {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [], "source": viewer.strip("\n").splitlines(keepends=True)}


ev_opts = [f"{d:%Y-%m-%d} ({n} claimed BG)" for d, n in zip(pd.to_datetime(E.ev, format="%Y%m%d"), E.n_claimed_bg)]
cty_opts = sorted(f["properties"]["NAME"] for f in G["features"])

inspector = r"""
#@title County inspector { run: "auto" }
EVENT = "__EVDEF__"  #@param __EVOPTS__
COUNTY = "Harris"  #@param __CTYOPTS__
POPUP = {'INPUT rain': ['rain_cty_max', 'rain_cty_mean', 'rain_sum_mean'],
         'INPUT exposure (county totals, policy-weighted shares)': ['n_pol', 'sfha_share', 'single_share', 'mean_cov', 'n_bg', 'hist00', 'prior3'],
         'MODEL INPUT (block-group mean of what the CNN tabular branch gets)': ['in_log_pol', 'in_sfha', 'in_single', 'in_cov', 'in_log_nbg', 'in_hist00', 'in_prior3', 'in_month_sin', 'in_month_cos'],
         'INTERMEDIATE (relative scale, not probabilities)': ['exp_cnn', 'exp_boost', 'exp_pol'],
         'RESULT rank (1 = highest)': ['pct_cnn', 'pct_boost', 'pct_pol'],
         'GROUND TRUTH (observed, not an input)': ['n_claims', 'n_cbg', 'paid', 'claims_per_1k'],
         'COMPARISON (top-K flagged vs claims)': ['class_cnn', 'class_boost', 'class_pol']}
ev = EVENT.split()[0].replace('-', '')
fips = next(k for k, v in NAME.items() if v == COUNTY)
if fips not in T_by_ev[ev].index:
    print(f"{COUNTY} County has no policies in force in the event of {EVENT.split()[0]}; choose another county.")
else:
    r = T_by_ev[ev].loc[fips]
    lines = [f"{COUNTY} County (FIPS {fips}), event {ev[:4]}-{ev[4:6]}-{ev[6:]}"]
    for head, keys in POPUP.items():
        lines.append('\n' + head)
        for k in keys:
            v = r[k]
            lines.append(f"  {k:14s} " + (CLS[int(v)][0] if k.startswith('class_') else f'{v:,.0f}' if abs(v) >= 100 else f'{v:.2f}' if abs(v) >= 1 else f'{v:.3g}'))
    print('\n'.join(lines))
""".replace("__EVDEF__", ev_opts[0] if False else next(o for o in ev_opts if o.startswith("2019-06-04"))).replace("__EVOPTS__", json.dumps(ev_opts)).replace("__CTYOPTS__", json.dumps(cty_opts))
ii = next(i for i, c in enumerate(cells) if "__INSPECTOR__" in "".join(c["source"]))
cells[ii] = {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [], "source": inspector.strip("\n").splitlines(keepends=True)}

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "colab": {"name": "colab_county_viewer.ipynb", "provenance": []}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(cells)} cells)")
