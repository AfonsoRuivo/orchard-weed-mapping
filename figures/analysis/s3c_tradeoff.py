#!/usr/bin/env python3
"""Section 3c - operational form of the gating interval.  For each detector, density class and merge radius eps:
  linking  L(eps)  = share of true re-sightings (frames DN apart) closer than eps = [pairs(eps) - null(eps)] / [pairs(RMAX) - null(RMAX)]
                     (pairs: mutual nearest neighbours; null: the later frame shifted by 2-4 m, 30 draws; RMAX = 1.5 m)
  over-merge O(eps) = share of detections that have, in the SAME frame, another detection at 0.10 m < d <= eps
                     (simultaneously detected -> distinct detections; < 0.10 m is excluded because it is the same plant boxed
                     twice (YOLO weed + weed_cluster, split colour blobs)).
  The interval 'exists' if some eps has L >= 0.9 and O <= 0.1 (thresholds stated; the curves are saved so any other can be read).
  Class = tercile of colour-entity area within 1.5 m (frozen colour map), isolated = lowest, dense = highest.
usage: s3c_tradeoff.py [DN]"""
import sys
from scipy.spatial import cKDTree
from common import *
from sigma_lib import *
dn = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EPS = np.round(np.arange(0.1, 1.51, 0.05), 2); rng = np.random.default_rng(0)
ec = pd.read_csv(TAB / 'entities_colour.csv'); tree = cKDTree(ec[['x', 'y']].values); area = ec.area_med.values
dens = lambda xy: np.array([area[i].sum() for i in tree.query_ball_point(xy, 1.5)])
allxy = np.vstack([load_weeds(x)[['x_enu', 'y_enu']].values for x in ('yolo', 'colour')]); q1, q2 = np.percentile(dens(allxy), [100 / 3, 200 / 3])
rows = []
for det in ('yolo', 'colour'):
    w, f = prep(det, 'dense'); w['dens'] = dens(w[['x_enu', 'y_enu']].values)
    P = build_pairs(w, f, 1.5, dn=dn); P['dens'] = w.dens.loc[P.ia].values
    N = [build_pairs(w, f, 1.5, shift=s, rng=rng, dn=dn) for s in (2.0, 3.0, 4.0) for _ in range(10)]
    N = [n_ if len(n_) else pd.DataFrame(columns=P.columns) for n_ in N]
    for N_ in N:
        if len(N_): N_['dens'] = w.dens.loc[N_.ia].values
    # same-frame nearest neighbours beyond duplicates
    sf = []
    for st, g in w.groupby('stem'):
        if len(g) < 2: continue
        xy = g[['x_enu', 'y_enu']].values; D = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)); np.fill_diagonal(D, np.inf)
        D[D <= 0.10] = np.inf; nn = D.min(1)
        sf += [(d_, dn_) for d_, dn_ in zip(nn, g.dens)]
    sf = pd.DataFrame(sf, columns=['nn', 'dens'])
    for cls, lo, hi in (('isolated', -1, q1), ('dense', q2, 1e9), ('all', -1, 1e9)):
        sel = lambda X: X[(X.dens > lo) & (X.dens <= hi)] if cls == 'isolated' else X[(X.dens >= lo if cls == 'dense' else X.dens > -2) & (X.dens <= hi)]
        Pc = sel(P); Nc = [sel(n_) for n_ in N]; wc = sel(w.rename(columns={})); nsf = sel(sf)
        tot = len(Pc) - np.mean([len(x) for x in Nc])
        for e in EPS:
            tr = (Pc.dist <= e).sum() - np.mean([(x.dist <= e).sum() for x in Nc])
            rows.append(dict(det=det, cls=cls, dn=dn, eps=e, link=tr / max(tot, 1e-9), over=float((nsf.nn <= e).mean()) if len(nsf) else np.nan,
                             n_pairs_class=len(Pc), true_pairs=tot, n_det_class=len(wc), n_sameframe=len(nsf)))
T = pd.DataFrame(rows); T.to_csv(TAB / f'tradeoff_dn{dn}.csv', index=False)
for (det, cls), g in T.groupby(['det', 'cls']):
    ok = g[(g.link >= 0.9) & (g.over <= 0.1)]
    print(det, cls, 'n_det', g.n_det_class.iloc[0], 'true pairs', round(g.true_pairs.iloc[0], 1), '| eps for L>=0.9:', g[g.link >= 0.9].eps.min(), '| over at 0.4:', round(g[g.eps == 0.4].over.iloc[0], 3),
          'link at 0.4:', round(g[g.eps == 0.4].link.iloc[0], 2), '| over at eps(L>=.9):', (round(g[g.eps == g[g.link >= 0.9].eps.min()].over.iloc[0], 3) if len(g[g.link >= 0.9]) else None), '| interval:', len(ok) > 0, ok.eps.min() if len(ok) else '')
