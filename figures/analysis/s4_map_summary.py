#!/usr/bin/env python3
"""Section 4 numbers: summary table (colour+gate, YOLO v2+gate), agronomic statistics, detector agreement.
-> tables/map_summary.csv, agronomy_*.csv, agreement.json, entities_<det>.csv (eps = 0.4 m)"""
import json
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from common import *
from entities import entities
from geom import *
EPS = 0.4
fr = pd.read_csv(TAB / 'frames.csv'); acc = json.load(open(TAB / 'frame_accounting.json'))
t = load_trunks(); rows = fit_rows(t); cov = Coverage('frozen'); cxy, cc = cov.cells()
obs_area = len(cxy) * cov.cell ** 2
E, summ = {}, []
for det in ('yolo', 'colour'):
    w = load_weeds(det); e, lab = entities(w, EPS); e['det'] = det
    w['eid'] = lab; w[['x_enu', 'y_enu']].to_numpy()
    d, rid, sg = dist_to_rows(e[['x', 'y']].values, rows); e['d_row'], e['row'], e['off_row'] = d, rid, sg
    e['views'] = cov.at(e[['x', 'y']].values); e.to_csv(TAB / f'entities_{det}.csv', index=False); E[det] = e
    summ.append(dict(detector=det, frames=acc['frames_used'], map_points=len(w), entities=len(e),
                     pct_seen_once_frame=100 * (e.n_frames == 1).mean(), pct_one_detection=100 * (e.n_det == 1).mean(),
                     total_area_m2=e.area_med.sum(), total_area_max_m2=e.area_max.sum(), median_area_m2=e.area_med.median(),
                     trunks=len(t), rows=len(rows), observed_ground_m2=obs_area, entities_per_100m2=100 * len(e) / obs_area,
                     cover_pct_area=100 * e.area_med.sum() / obs_area))
S = pd.DataFrame(summ); S.to_csv(TAB / 'map_summary.csv', index=False); print(S.T)

# agronomy: weeds vs distance to the nearest row line, normalised by observed ground
edges = np.arange(0, 3.01, 0.25); ag = []
dcell = dist_to_rows(cxy, rows)[0]
for det, e in E.items():
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e.d_row >= lo) & (e.d_row < hi); ca = ((dcell >= lo) & (dcell < hi)).sum() * cov.cell ** 2
        ag.append(dict(det=det, d_lo=lo, d_hi=hi, n=int(m.sum()), observed_m2=ca, per_100m2=100 * m.sum() / max(ca, 1e-9),
                       area_m2=e.area_med[m].sum()))
pd.DataFrame(ag).to_csv(TAB / 'agronomy_distance_profile.csv', index=False)
print(pd.DataFrame(ag).round(2).to_string())
# per-row: weeds within STRIP of the row line per 10 m of observed row
STRIP = 1.0; per = []
for r, R in rows.items():
    s = np.arange(R['s0'], R['s1'], 0.5); pts = R['c'] + s[:, None] * R['u']
    obs = np.any([cov.at(pts + o * R['n']) > 0 for o in (-1.0, -0.5, 0.0, 0.5, 1.0)], axis=0); L_obs = obs.sum() * 0.5
    for det, e in E.items():
        m = (e.row == r) & (e.d_row <= STRIP)
        per.append(dict(row=r, det=det, row_len=R['s1'] - R['s0'], observed_len=L_obs, n_in_strip=int(m.sum()),
                        per10m=(10 * m.sum() / L_obs if L_obs >= 10 else np.nan), n_trunks=R['n_trunks']))
P = pd.DataFrame(per); P.to_csv(TAB / 'agronomy_per_row.csv', index=False); print(P.pivot(index='row', columns='det', values='per10m').round(2).T)
# row spacing
c = np.array([R['c'] @ rows[0]['n'] for R in rows.values()]); print('row offsets', np.sort(c).round(1), 'spacing', np.diff(np.sort(c)).round(1))

# detector agreement
ey, ec = E['yolo'][['x', 'y']].values, E['colour'][['x', 'y']].values
tree = cKDTree(ec); treey = cKDTree(ey); rng = np.random.default_rng(0); agr = {}
for r in (0.5, 1.0, 1.5):
    f_y = float((tree.query(ey)[0] <= r).mean()); f_c = float((treey.query(ec)[0] <= r).mean())
    # null: YOLO entities moved to random observed cells (same count)
    nul = []
    for _ in range(300):
        k = rng.choice(len(cxy), len(ey)); q = cxy[k] + rng.uniform(-cov.cell / 2, cov.cell / 2, (len(ey), 2))
        nul.append((tree.query(q)[0] <= r).mean())
    agr[f'r{r}'] = dict(yolo_with_colour_within=f_y, colour_with_yolo_within=f_c, null_mean=float(np.mean(nul)),
                        null_p95=float(np.percentile(nul, 95)))
# density-map correlation on 5 m cells
G = 5.0; ix = lambda p: np.floor((p - [cov.x0, cov.y0]) / G).astype(int)
def grid(p):
    g = {}
    for a, b in ix(p): g[(a, b)] = g.get((a, b), 0) + 1
    return g
gy, gc, gcov = grid(ey), grid(ec), grid(cxy)
keys = [k for k, n in gcov.items() if n * cov.cell ** 2 > 0.5 * G * G]
vy, vc = np.array([gy.get(k, 0) for k in keys]), np.array([gc.get(k, 0) for k in keys])
rho, pv = spearmanr(vy, vc); agr['grid5m'] = dict(n_cells=len(keys), spearman=float(rho), p=float(pv))
json.dump(agr, open(TAB / 'agreement.json', 'w'), indent=1); print(json.dumps(agr, indent=1))

# in-row strip (<= 0.5 m from the row line) vs alley: rate per 100 m^2 of observed ground, exact binomial test
from scipy.stats import binomtest
Ad = pd.DataFrame(ag); rowsT = []
for det in ('yolo', 'colour'):
    d = Ad[Ad.det == det]; i = d[d.d_hi <= 0.5]; o = d[d.d_lo >= 0.5]
    n1, a1, n2, a2 = int(i.n.sum()), i.observed_m2.sum(), int(o.n.sum()), o.observed_m2.sum()
    rowsT.append(dict(det=det, n_inrow=n1, obs_m2_inrow=a1, rate_inrow=100 * n1 / a1, n_alley=n2, obs_m2_alley=a2, rate_alley=100 * n2 / a2,
                      ratio=(n1 / a1) / (n2 / a2), p_binom=binomtest(n1, n1 + n2, a1 / (a1 + a2)).pvalue))
pd.DataFrame(rowsT).to_csv(TAB / 'agronomy_inrow_test.csv', index=False); print(pd.DataFrame(rowsT).round(3))
