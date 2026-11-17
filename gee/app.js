/**
 * nanoWxRisk county display (Earth Engine Code Editor app). Plan: docs/GEE_PLAN.md, data: scripts/110-112.
 * Paste this file into the Code Editor and Run; "Apps" then publishes it. Needs read access to the wxrisk assets.
 *
 * Texas flood model, observed-rain mode (gridMET 4 km), test years 2017-2023, events with >= 10 claimed block groups, scope S0.
 * Three layer groups follow the model from input to result: INPUT (rain, exposure parts) -> INTERMEDIATE (model outputs) -> RESULT (ranks).
 * Ranks only: the model's absolute probabilities are not calibrated (training down-samples unclaimed rows) and are never shown as "x %".
 */

var ROOT = 'projects/impact-foreacsting-flooding/assets/wxrisk/';
var counties = ee.FeatureCollection(ROOT + 'county_geometry').filter(ee.Filter.eq('STATEFP', '48'));
var index = ee.FeatureCollection(ROOT + 'events_index').filter(ee.Filter.gte('n_claimed_bg', 10)).sort('ev');

var SEQ = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000'];
var BLU = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'];
var GRN = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c'];

// key: column of county_event_tx_<year>; log: paint log10(x + eps); eps keeps zeros finite; min/max on the painted scale.
var GROUPS = {
  '1 Input: rain': {
    rain_cty_max: {t: 'County max rain (mm)', pal: BLU, min: 0, max: 150,
      d: 'Largest block-group-pixel value, over the county, of the peak daily rain in the 2-day window S, S+1 (gridMET 4 km, observed). Boosting feature "county max".'},
    rain_cty_mean: {t: 'County mean rain (mm)', pal: BLU, min: 0, max: 60,
      d: 'Mean over the county of the same pixel peak (daily maximum over S, S+1). Boosting feature "county mean".'},
    rain_sum_mean: {t: 'Two-day rain total, mean (mm)', pal: BLU, min: 0, max: 100,
      d: 'Two-day total rain at the block-group pixels, averaged over the county. The CNN reads the 4 km peak and sum maps (log1p scaled).'}
  },
  '2 Input: exposure': {
    n_pol: {t: 'Policies in force', pal: GRN, log: true, eps: 1, min: 0, max: 5.5,
      d: 'NFIP policies in force on day S (effective date <= S < termination or cancellation), clean policy table, summed over the county\'s block groups. Log colour scale.'},
    sfha_share: {t: 'Share in flood hazard area', pal: GRN, min: 0, max: 1,
      d: 'Share of those policies rated in a Special Flood Hazard Area (zone A* or V*), policy-weighted.'},
    single_share: {t: 'Share single-family', pal: GRN, min: 0.4, max: 1,
      d: 'Share with single-family occupancy (occupancy codes 1 and 11), policy-weighted.'},
    mean_cov: {t: 'Mean building coverage ($ M)', pal: GRN, min: 0.1, max: 0.35,
      d: 'Mean insured building coverage per policy in millions of dollars, policy-weighted. Insured amount, not property value.'},
    hist00: {t: 'Claims 2000-2009 (history)', pal: GRN, log: true, eps: 1, min: 0, max: 3,
      d: 'NFIP claims in the county in 2000-2009 (claim history prior). Log colour scale.'},
    prior3: {t: 'Claims in the 3 years before S', pal: GRN, log: true, eps: 1, min: 0, max: 3.5,
      d: 'NFIP claims in the county in the 3 years before the event (recent claim history). Log colour scale.'},
    in_log_pol: {t: 'MODEL INPUT: log policies per block group', pal: GRN, min: 0.5, max: 4.2,
      d: 'What the CNN\'s tabular branch receives: log(policies in force) of each block group, averaged over the county\'s block groups (unweighted). Not the same as the county total above: a county of many small block groups scores low.'},
    in_sfha: {t: 'MODEL INPUT: flood-zone share (block-group mean)', pal: GRN, min: 0, max: 1,
      d: 'Share of policies in a Special Flood Hazard Area, per block group as the model reads it, averaged unweighted over the county\'s block groups.'},
    in_single: {t: 'MODEL INPUT: single-family share (block-group mean)', pal: GRN, min: 0.4, max: 1,
      d: 'Single-family share per block group as the model reads it, averaged unweighted over the county\'s block groups.'},
    in_cov: {t: 'MODEL INPUT: mean coverage (block-group mean, $ M)', pal: GRN, min: 0.05, max: 0.3,
      d: 'Mean building coverage per policy ($ millions) per block group as the model reads it, averaged over the county\'s block groups.'},
    in_log_nbg: {t: 'MODEL INPUT: log(1 + block groups in county)', pal: GRN, min: 0.5, max: 5.5,
      d: 'Size of the county in block groups, log1p scale, as the model reads it. A context feature, the same for every block group of the county.'},
    in_hist00: {t: 'MODEL INPUT: log1p claims 2000-2009 (block-group mean)', pal: GRN, min: 0, max: 1.5,
      d: 'Claim history prior per block group, log1p scale, averaged over the county\'s block groups. The model standardises all nine inputs with training-year statistics before use.'},
    in_prior3: {t: 'MODEL INPUT: log1p claims, previous 3 years (block-group mean)', pal: GRN, min: 0, max: 1.2,
      d: 'Recent claim history per block group, log1p scale, averaged over the county\'s block groups. (Month enters the model as sin/cos of the event month and is the same for all counties.)'}
  },
  '3 Intermediate: model output': {
    exp_cnn: {t: 'CNN: summed block-group scores', pal: SEQ, log: true, eps: 1e-4, min: -3.5, max: 0,
      d: 'Sum over block groups of the CNN claim score (mean of 2 seeds). RELATIVE scale only: not a calibrated probability or a count of claims.'},
    exp_boost: {t: 'Boosting: summed block-group scores', pal: SEQ, log: true, eps: 1e-4, min: -3.5, max: 0,
      d: 'Same for the gradient-boosting model on the hand-built features (tabular exposure + rain summaries). Relative scale only.'},
    exp_pol: {t: 'Policies only (baseline score)', pal: SEQ, log: true, eps: 1, min: 0, max: 5.5,
      d: 'Exposure-only baseline: the score is the number of policies in force. What any model has to beat.'}
  },
  '4 Result: rank in this event': {
    pct_cnn: {t: 'CNN rank (1 = highest)', pal: SEQ, min: 0, max: 1,
      d: 'Percentile of the county among all counties of this event by the CNN score. This is the product: an ordering of units.'},
    pct_boost: {t: 'Boosting rank', pal: SEQ, min: 0, max: 1, d: 'Same for the boosting model.'},
    pct_pol: {t: 'Policies-only rank', pal: SEQ, min: 0, max: 1,
      d: 'Same for the exposure-only baseline. Compare with the CNN rank to see what the model adds beyond "more policies".'},
    class_cnn: {t: 'CNN vs ground truth (top-K flagged)', cat: [[1, '#1a9850', 'hit: flagged and had claims'], [2, '#f46d43', 'missed: had claims, not flagged'], [3, '#7b3294', 'false alarm: flagged, no claims'], [0, '#e0e0e0', 'neither']],
      d: 'The CNN flags its top-K counties, K = number of counties that actually had claims in this event. Green = hit, orange = missed, purple = false alarm. Hits out of K are listed under Event metrics.'},
    class_boost: {t: 'Boosting vs ground truth (top-K flagged)', cat: [[1, '#1a9850', 'hit: flagged and had claims'], [2, '#f46d43', 'missed: had claims, not flagged'], [3, '#7b3294', 'false alarm: flagged, no claims'], [0, '#e0e0e0', 'neither']],
      d: 'Same comparison for the boosting model.'},
    class_pol: {t: 'Policies-only vs ground truth (top-K flagged)', cat: [[1, '#1a9850', 'hit: flagged and had claims'], [2, '#f46d43', 'missed: had claims, not flagged'], [3, '#7b3294', 'false alarm: flagged, no claims'], [0, '#e0e0e0', 'neither']],
      d: 'Same comparison for the exposure-only baseline: flag the K counties with the most policies. The CNN has to beat this.'}
  },
  '5 Ground truth: NFIP claims (observed)': {
    n_claims: {t: 'Claims in S to S+3', pal: BLU, log: true, eps: 1, min: 0, max: 3,
      d: 'NFIP claims filed with a loss date in the window S to S+3 (tropical-named claims removed, scope S0). Retrospective truth; arrives weeks later; never a model input. Log colour scale.'},
    n_cbg: {t: 'Claimed block groups', pal: BLU, min: 0, max: 15,
      d: 'Number of block groups in the county with at least one claim. The model is trained to rank block groups; the county map aggregates them.'},
    paid: {t: 'Building amount paid (USD)', pal: BLU, log: true, eps: 1, min: 0, max: 6.5,
      d: 'Building claim payments in the window, summed over the county. Log colour scale. Basement items are covered only in part (AGENTS.md trap 7), so payments understate basement damage.'},
    claims_per_1k: {t: 'Claims per 1,000 policies', pal: BLU, log: true, eps: 0.1, min: -1, max: 1.5,
      d: 'Observed claims divided by policies in force: how hard the county was hit relative to its exposure. Log colour scale.'}
  }
};

var CLASS_NAME = {0: 'neither', 1: 'hit', 2: 'missed (had claims, not flagged)', 3: 'false alarm (flagged, no claims)'};
var VAR_ORDER = ['rain_cty_max', 'rain_cty_mean', 'rain_sum_mean', 'n_pol', 'sfha_share', 'single_share', 'mean_cov', 'hist00', 'prior3',
  'exp_cnn', 'exp_boost', 'exp_pol', 'pct_cnn', 'pct_boost', 'pct_pol', 'n_claims', 'n_cbg'];

var map = ui.Map();
map.setCenter(-99.3, 31.3, 6);
map.setOptions('SATELLITE');
map.style().set('cursor', 'crosshair');

var SPEEDS = {'Slow (3 s)': 3000, 'Normal (1.5 s)': 1500, 'Fast (0.8 s)': 800};
var state = {events: [], idx: 0, ev: null, fc: null, group: null, key: null, cache: {}, playing: false, run: 0};

// ---- panel ----
var title = ui.Label('nanoWxRisk - Texas flood, county view', {fontWeight: 'bold', fontSize: '18px'});
var banner = ui.Label('Mode: OBSERVED rain (gridMET), retrospective test, 2017-2023, scope S0. Ranks, not probabilities. ' +
  'In forecast mode the county ranking is no better than policies-only (AUC 0.82 vs 0.82).',
  {fontSize: '11px', color: '#8a4b00', whiteSpace: 'pre-wrap'});
var dateLabel = ui.Label('loading events...', {fontWeight: 'bold', fontSize: '14px'});
var slider = ui.Slider({min: 0, max: 1, value: 0, step: 1, disabled: true, style: {stretch: 'horizontal'}});
var firstBtn = ui.Button('|<', null, false, {margin: '2px'});
var prevBtn = ui.Button('<', null, false, {margin: '2px'});
var playBtn = ui.Button('Play', null, false, {margin: '2px', width: '90px'});
var nextBtn = ui.Button('>', null, false, {margin: '2px'});
var lastBtn = ui.Button('>|', null, false, {margin: '2px'});
var speedSelect = ui.Select({items: Object.keys(SPEEDS), value: 'Normal (1.5 s)', style: {margin: '2px'}});
var loopBox = ui.Checkbox('Loop', false);
var yearSelect = ui.Select({items: ['Jump to year'], value: 'Jump to year', style: {margin: '2px'}});
var timeline = ui.Panel();
var groupSelect = ui.Select({items: Object.keys(GROUPS), style: {stretch: 'horizontal'}});
var varSelect = ui.Select({items: [], style: {stretch: 'horizontal'}});
var desc = ui.Label('', {fontSize: '12px', whiteSpace: 'pre-wrap'});
var legend = ui.Panel();
var metrics = ui.Label('', {fontSize: '12px', whiteSpace: 'pre-wrap'});
var gains = ui.Panel();
var info = ui.Label('Click a county for all variables of this event.', {fontSize: '12px', whiteSpace: 'pre-wrap'});

var panel = ui.Panel({
  widgets: [title, banner, ui.Label('Event timeline', {fontWeight: 'bold'}), dateLabel,
    ui.Panel([firstBtn, prevBtn, playBtn, nextBtn, lastBtn], ui.Panel.Layout.flow('horizontal')),
    ui.Panel([speedSelect, loopBox, yearSelect], ui.Panel.Layout.flow('horizontal')),
    slider, ui.Label('Slider loads the event when you release it.', {fontSize: '10px', color: '#666'}), timeline,
    ui.Label('Layer group', {fontWeight: 'bold'}), groupSelect, ui.Label('Variable', {fontWeight: 'bold'}), varSelect, desc, legend,
    ui.Label('Event metrics vs ground truth (script 94)', {fontWeight: 'bold'}), metrics, gains, ui.Label('County', {fontWeight: 'bold'}), info],
  style: {width: '400px', padding: '8px'}
});
ui.root.clear();
ui.root.add(ui.SplitPanel(panel, map));

function fmtDate(ev) { return ev.slice(0, 4) + '-' + ev.slice(4, 6) + '-' + ev.slice(6, 8); }

function paintValue(spec, key) {
  return function (f) {
    var v = ee.Number(f.get(key));
    if (spec.log) { v = v.add(spec.eps).log10(); }
    return f.set('v', v);
  };
}

function colorbar(spec) {
  if (spec.cat) {
    legend.clear();
    spec.cat.forEach(function (c) {
      legend.add(ui.Panel([ui.Label('   ', {backgroundColor: c[1], margin: '2px 6px 2px 8px'}), ui.Label(c[2], {fontSize: '11px', margin: '4px 0'})],
        ui.Panel.Layout.flow('horizontal')));
    });
    return;
  }
  var bar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0), params: {bbox: [0, 0, 1, 0.1], dimensions: '300x14', format: 'png', min: 0, max: 1, palette: spec.pal},
    style: {stretch: 'horizontal', margin: '4px 8px'}
  });
  var lo = spec.log ? '1e' + spec.min : String(spec.min);
  var hi = spec.log ? '1e' + spec.max : String(spec.max);
  legend.clear();
  legend.add(bar);
  legend.add(ui.Panel([ui.Label(lo, {fontSize: '11px'}), ui.Label(spec.log ? 'log scale' : '', {fontSize: '11px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(hi, {fontSize: '11px'})], ui.Panel.Layout.flow('horizontal')));
}

function eventFc(ev) {
  if (!state.cache[ev]) {
    var rows = ee.FeatureCollection(ROOT + 'county_event_tx_' + ev.slice(0, 4)).filter(ee.Filter.eq('ev', ev));
    var joined = ee.Join.inner().apply(counties, rows, ee.Filter.equals({leftField: 'GEOID', rightField: 'fips'}));
    state.cache[ev] = ee.FeatureCollection(joined.map(function (j) {
      var p = ee.Feature(j.get('primary'));
      var s = ee.Feature(j.get('secondary'));
      return ee.Feature(p.geometry(), s.toDictionary()).set('NAME', p.get('NAME'));
    }));
  }
  return state.cache[ev];
}

function colourImage(fc0, spec) {
  if (spec.cat) {
    var key = state.key;
    return ee.ImageCollection(spec.cat.map(function (c) {
      return ee.Image().byte().paint(fc0.filter(ee.Filter.eq(key, c[0])), 1).visualize({palette: [c[1]]});
    })).mosaic();
  }
  var fc = fc0.map(paintValue(spec, state.key));
  return ee.Image().float().paint(fc, 'v').visualize({min: spec.min, max: spec.max, palette: spec.pal});
}

function redraw() {
  if (!state.fc || !state.key) { return; }
  var spec = GROUPS[state.group][state.key];
  var img = colourImage(state.fc, spec);
  // counties with at least one claimed block group get a blue outline (observed truth, for comparison)
  var claimed = state.fc.filter(ee.Filter.gt('n_cbg', 0));
  var outline = ee.Image().byte().paint(claimed, 1, 2).visualize({palette: ['#1a5fb4']});
  var edges = ee.Image().byte().paint(state.fc, 1, 1).visualize({palette: ['#444444'], opacity: 0.6});
  var layers = [ui.Map.Layer(img, {}, spec.t, true, 0.85), ui.Map.Layer(edges, {}, 'county borders'),
    ui.Map.Layer(outline, {}, 'counties with observed claims (blue)')];
  // preload: the next event drawn at 1 % opacity so its tiles are cached when the animation steps to it
  var nxt = state.events[state.idx + 1];
  if (nxt) { layers.push(ui.Map.Layer(colourImage(eventFc(nxt.ev), spec), {}, 'preloading next event', true, 0.01)); }
  map.layers().reset(layers);
  desc.setValue(spec.t + '\n' + spec.d);
  colorbar(spec);
}

function loadEvent(ev) {
  state.ev = ev;
  state.fc = eventFc(ev);
  var mine = ev;  // ignore metrics that arrive after the user has moved on
  index.filter(ee.Filter.eq('ev', ev)).first().toDictionary().evaluate(function (d) {
    if (state.ev !== mine) { return; }
    if (!d) { metrics.setValue('(no metrics)'); return; }
    function g(k) { return d[k] === undefined ? 'n/a' : d[k].toFixed(2); }
    metrics.setValue('County AUC   CNN ' + g('cnn_cty_auc') + ' | boosting ' + g('boost_cty_auc') + ' | policies only ' + g('pol_cty_auc') +
      '\nBlock-group AUC   CNN ' + g('cnn_bg_auc') + ' | boosting ' + g('boost_bg_auc') + ' | policies only ' + g('pol_bg_auc') +
      '\nTop-5% block groups capture   CNN ' + g('cnn_bg_top5') + ' | policies only ' + g('pol_bg_top5') +
      '\n(0.5 = chance AUC; shown for the block-group model, aggregated to counties here)' +
      '\nCounties with claims: ' + d.k_claimed_cty + '. Hits when each model flags its top-' + d.k_claimed_cty + ' counties:' +
      '\n   CNN ' + d.hit_cnn + ' | boosting ' + d.hit_boost + ' | policies only ' + d.hit_pol);
  });
  info.setValue('Click a county for all variables of this event.');
  ee.Dictionary({y: state.fc.aggregate_array('n_claims'), cnn: state.fc.aggregate_array('exp_cnn'), boost: state.fc.aggregate_array('exp_boost'),
    pol: state.fc.aggregate_array('exp_pol')}).evaluate(function (r) {
    if (state.ev !== mine || !r) { return; }
    drawGains(r);
  });
  redraw();
}

// Ground-truth comparison: flag counties from the highest score down; how much of the observed claims do the first x % of counties hold?
function drawGains(r) {
  var n = r.y.length, total = r.y.reduce(function (a, b) { return a + b; }, 0);
  function curve(score) {
    var order = r.y.map(function (_, i) { return i; }).sort(function (a, b) { return score[b] - score[a] || a - b; });
    var acc = 0, out = [0];
    order.forEach(function (i) { acc += r.y[i]; out.push(total > 0 ? acc / total : 0); });
    return out;
  }
  var c1 = curve(r.cnn), c2 = curve(r.boost), c3 = curve(r.pol);
  var rows = [['Share of counties flagged', 'CNN', 'Boosting', 'Policies only', 'Random']];
  for (var i = 0; i <= n; i++) { rows.push([i / n, c1[i], c2[i], c3[i], i / n]); }
  var chart = ui.Chart(rows, 'LineChart', {
    title: 'Share of observed claims captured (' + n + ' counties, ' + total + ' claims)', legend: {position: 'bottom'},
    colors: ['#d1495b', '#4c78a8', '#e8a33d', '#999999'], lineWidth: 2, hAxis: {title: 'share of counties flagged, highest score first', viewWindow: {min: 0, max: 1}},
    vAxis: {title: 'share of claims', viewWindow: {min: 0, max: 1}}, chartArea: {left: 44, right: 8, top: 28, bottom: 60}
  });
  chart.style().set({height: '220px', stretch: 'horizontal'});
  gains.clear();
  gains.add(chart);
}

// ---- timeline control ----
function drawTimeline() {
  var rows = [[{label: 'Date', type: 'date'}, {label: 'Claimed block groups', type: 'number'}, {label: 'Selected', type: 'number'}]];
  state.events.forEach(function (x, i) {
    rows.push(i === state.idx ? [new Date(x.t), null, x.n] : [new Date(x.t), x.n, null]);
  });
  var chart = ui.Chart(rows, 'ColumnChart', {
    legend: {position: 'none'}, isStacked: true, colors: ['#9ecae1', '#d1495b'], bar: {groupWidth: '100%'},
    hAxis: {format: 'yyyy'}, vAxis: {title: 'claimed block groups', minValue: 0}, chartArea: {left: 44, right: 8, top: 8, bottom: 28},
    title: 'Events 2017-2023 (click a bar to jump)'
  });
  chart.style().set({height: '150px', stretch: 'horizontal'});
  chart.onClick(function (xValue) {
    if (xValue === null || xValue === undefined) { return; }
    var t = (xValue instanceof Date) ? xValue.getTime() : Number(xValue);
    if (isNaN(t)) { return; }
    var best = 0, bd = Infinity;
    state.events.forEach(function (x, i) { var dd = Math.abs(x.t - t); if (dd < bd) { bd = dd; best = i; } });
    pause();
    goTo(best);
  });
  timeline.clear();
  timeline.add(chart);
}

function goTo(i) {
  var N = state.events.length;
  i = Math.max(0, Math.min(N - 1, i));
  state.idx = i;
  var x = state.events[i];
  dateLabel.setValue(fmtDate(x.ev) + '   (event ' + (i + 1) + ' of ' + N + ', ' + x.n + ' claimed block groups)');
  slider.setValue(i, false);
  drawTimeline();
  loadEvent(x.ev);
}

function pause() {
  state.playing = false;
  state.run += 1;
  playBtn.setLabel('Play');
}

function play() {
  state.playing = true;
  state.run += 1;
  var run = state.run;
  playBtn.setLabel('Pause');
  (function tick() {
    if (!state.playing || run !== state.run) { return; }
    var next = state.idx + 1;
    if (next >= state.events.length) {
      if (!loopBox.getValue()) { pause(); return; }
      next = 0;
    }
    goTo(next);
    ui.util.setTimeout(tick, SPEEDS[speedSelect.getValue()]);
  })();
}

playBtn.onClick(function () { if (state.playing) { pause(); } else { play(); } });
prevBtn.onClick(function () { pause(); goTo(state.idx - 1); });
nextBtn.onClick(function () { pause(); goTo(state.idx + 1); });
firstBtn.onClick(function () { pause(); goTo(0); });
lastBtn.onClick(function () { pause(); goTo(state.events.length - 1); });
slider.onChange(function (v) { if (v !== state.idx && state.events.length) { pause(); goTo(Math.round(v)); } });  // fires when a drag is released
yearSelect.onChange(function (y) {
  if (!y || y === 'Jump to year') { return; }
  for (var i = 0; i < state.events.length; i++) { if (state.events[i].ev.slice(0, 4) === y) { pause(); goTo(i); return; } }
});

function setGroup(g) {
  state.group = g;
  var keys = Object.keys(GROUPS[g]);
  varSelect.items().reset(keys.map(function (k) { return GROUPS[g][k].t; }));
  varSelect.setValue(GROUPS[g][keys[0]].t, false);
  state.key = keys[0];
  redraw();
}

groupSelect.onChange(setGroup);
varSelect.onChange(function (label) {
  var g = GROUPS[state.group];
  Object.keys(g).forEach(function (k) { if (g[k].t === label) { state.key = k; } });
  redraw();
});

map.onClick(function (c) {
  if (!state.fc) { return; }
  var pt = ee.Geometry.Point([c.lon, c.lat]);
  var f = state.fc.filterBounds(pt).first();
  f.toDictionary().evaluate(function (d) {
    if (!d) { info.setValue('No data here (county not in this event).'); return; }
    f.get('NAME').evaluate(function (name) {
      var lines = [name + ' County (FIPS ' + d.fips + ')'];
      var sec = {'INPUT rain': ['rain_cty_max', 'rain_cty_mean', 'rain_sum_mean'],
        'INPUT exposure (county totals and policy-weighted shares)': ['n_pol', 'sfha_share', 'single_share', 'mean_cov', 'n_bg', 'hist00', 'prior3'],
        'MODEL INPUT (block-group mean of what the CNN tabular branch gets)': ['in_log_pol', 'in_sfha', 'in_single', 'in_cov', 'in_log_nbg', 'in_hist00',
          'in_prior3', 'in_month_sin', 'in_month_cos'],
        'INTERMEDIATE (relative scale)': ['exp_cnn', 'exp_boost', 'exp_pol'],
        'RESULT rank (1 = highest)': ['pct_cnn', 'pct_boost', 'pct_pol'],
        'GROUND TRUTH (observed, not an input)': ['n_claims', 'n_cbg', 'paid', 'claims_per_1k'],
        'COMPARISON (top-K flagged vs claims)': ['class_cnn', 'class_boost', 'class_pol']};
      Object.keys(sec).forEach(function (h) {
        lines.push('\n' + h);
        sec[h].forEach(function (k) {
          var v = d[k];
          if (k.indexOf('class_') === 0) { lines.push('  ' + k + ': ' + (CLASS_NAME[v] || 'n/a')); return; }
          lines.push('  ' + k + ': ' + (v === undefined ? 'n/a' : (Math.abs(v) >= 100 ? Math.round(v) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3))));
        });
      });
      info.setValue(lines.join('\n'));
    });
  });
});

// ---- timeline: events with county data (>= 10 claimed block groups), in calendar order ----
ee.Dictionary({ev: index.aggregate_array('ev'), n: index.aggregate_array('n_claimed_bg')}).evaluate(function (r) {
  state.events = r.ev.map(function (e, i) {
    return {ev: e, n: r.n[i], t: Date.UTC(+e.slice(0, 4), +e.slice(4, 6) - 1, +e.slice(6, 8))};
  });
  var N = state.events.length;
  slider.setMax(N - 1);
  slider.setDisabled(false);
  var years = [];
  state.events.forEach(function (x) { var y = x.ev.slice(0, 4); if (years.indexOf(y) < 0) { years.push(y); } });
  yearSelect.items().reset(['Jump to year'].concat(years));
  yearSelect.setValue('Jump to year', false);
  groupSelect.setValue('4 Result: rank in this event', false);
  setGroup('4 Result: rank in this event');
  // start on the Houston event used in the scenario report, else the first one
  var start = 0;
  state.events.forEach(function (x, i) { if (x.ev === '20190604') { start = i; } });
  goTo(start);
});
