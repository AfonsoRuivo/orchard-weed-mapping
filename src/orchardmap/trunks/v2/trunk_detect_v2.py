#!/usr/bin/env python3
"""
trunk_detect_v2.py - orange-tree trunk detector for Campinas B (CB-11, session_20260706_140610).

Pipeline (each stage switchable through CFG for ablations):
  0. map      dense map = raw keyframe scans on LIO-SAM keyframe poses (build_dense_map.py),
              or the LIO-SAM GlobalMap.
  1. ground   height above local ground (5th percentile of z in 0.5 m cells).
  2. seeds    2D density of points in a LOW band (trunks fork 15-30 cm above ground), Gaussian
              smoothed, local maxima -> candidate trunk positions.
  3. gate     per candidate: n_core (band points within 0.20 m), n_up (points 0.4-1.2 m within
              0.35 m: the forks / canopy above a trunk), vocc (fraction of 5 cm height bins
              0.05-1.2 m occupied in the column). Weeds and grass have a low band but no column.
  4. nms      keep the best-scoring candidate within NMS_R (trees are >= 2.5 m apart).
  5. centre   circle fit on the low slice around the seed: RANSAC on 3 points with the centre kept
              near the seed, then geometric (orthogonal-distance) refit on the inliers.
  6. rows     row prior: RANSAC lines through the detections; detections off every row are dropped;
              in gaps along a row that are ~k x the row's spacing, look again with looser gates.
Writes trunks_v2.csv: x, y, r, support (n_core), n_up, vocc, kept_by_row_prior, row_id, source.
"""
import argparse, json, copy
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from common import DENSE, GLOBALMAP, read_pcd_xyz, local_ground, write_det, score
from features import density_peaks, candidate_features

OUT = Path(__file__).resolve().parents[1] / "v2_out"

CFG = dict(
    map='dense',                 # 'dense' | 'globalmap'
    band=(0.10, 0.40),           # seed / support band above local ground (m)
    seed_sigma=3, seed_cell=0.03, seed_nms=0.6, seed_thr=0.3,
    min_core=350, min_up=600, min_vocc=0.9,
    nms_r=1.6, nms_score='core_up',
    centre='peak',               # 'peak' (density peak = seed) | 'lowpeak' | 'circle' | 'centroid'
    fit_band=(0.08, 0.30), fit_win=0.35, r_win=0.30, seed_tol=0.15, r_min=0.04, r_max=0.25, ransac_tol=0.03,
    ransac_iters=300, refit='geometric',  # 'geometric' | 'kasa'
    lp_sigma=0.04,               # smoothing (m) of the low-slice density for centre='lowpeak'
    rows=True, row_tol=0.45, row_min_pts=5, row_angle_tol=10, gapfill=True,
    gap_relax=0.5,               # gap-fill gates = relax x normal gates
    row_model='dp',              # 'nms' (NMS + gap-fill + end slots) | 'dp' (spacing-aware chain, row_dp)
    dp_a=2.0, dp_theta=0.5, dp_sigma=0.25, dp_gamma=1.5, dp_gmin=1.6, dp_min_relax=0.4,
)


# ---------------------------------------------------------------- circle fitting
def kasa(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    s, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = s[0] / 2, s[1] / 2
    return cx, cy, np.sqrt(max(s[2] + cx ** 2 + cy ** 2, 1e-9))


def circle3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9: return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return cx, cy, np.hypot(x1 - cx, y1 - cy)


def fit_circle(xy, seed, c, rng):
    """RANSAC (centre within seed_tol of the seed) + refit. Returns (cx, cy, r, inlier_frac) or None."""
    n = len(xy)
    if n < 10: return None
    best, bn = None, 0
    for _ in range(c['ransac_iters']):
        m = circle3(xy[rng.choice(n, 3, replace=False)])
        if m is None: continue
        cx, cy, r = m
        if not (c['r_min'] <= r <= c['r_max']) or np.hypot(cx - seed[0], cy - seed[1]) > c['seed_tol']: continue
        k = (np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < c['ransac_tol']).sum()
        if k > bn: best, bn = m, k
    if best is None: return None
    inl = np.abs(np.hypot(xy[:, 0] - best[0], xy[:, 1] - best[1]) - best[2]) < c['ransac_tol']
    X = xy[inl]
    if c['refit'] == 'kasa':
        cx, cy, r = kasa(X[:, 0], X[:, 1])
    else:
        f = lambda p: np.hypot(X[:, 0] - p[0], X[:, 1] - p[1]) - p[2]
        s = least_squares(f, best, loss='soft_l1', f_scale=c['ransac_tol'])
        cx, cy, r = s.x
    if not (c['r_min'] * 0.8 <= r <= c['r_max'] * 1.2) or np.hypot(cx - seed[0], cy - seed[1]) > c['seed_tol'] * 1.5:
        return None
    return cx, cy, abs(r), inl.mean()


def radius_fixed_centre(P, h, t2, xy, c):
    """Trunk radius with the centre held fixed: robust (soft-L1) fit of r to the distances of the
    low-slice points within r_win of the centre. Not validated (no circumferences measured yet)."""
    idx = np.array(t2.query_ball_point(xy, c['r_win'] + 0.05), int); hh = h[idx]
    m = (hh > c['fit_band'][0]) & (hh < c['fit_band'][1])
    d = np.hypot(P[idx[m], 0] - xy[0], P[idx[m], 1] - xy[1]); d = d[d < c['r_win']]
    if len(d) < 10: return np.nan
    return float(least_squares(lambda r: d - r, [0.1], loss='soft_l1', f_scale=0.03).x[0])


# ---------------------------------------------------------------- rows
def fit_rows(xy, tol, min_pts, angle_tol, rng, max_rows=60, min_row_dist=2.0):
    """Sequential RANSAC of parallel lines. Returns list of (point_idx, centre, direction)."""
    rem = np.ones(len(xy), bool); rows = []; dom = None
    for _ in range(max_rows):
        cur = np.where(rem)[0]
        if len(cur) < min_pts: break
        best = None
        for _ in range(500):
            a, b = rng.choice(cur, 2, replace=False); d = xy[b] - xy[a]; L = np.linalg.norm(d)
            if L < 1.5: continue
            d /= L
            if dom is not None and np.degrees(np.arccos(min(abs(d @ dom), 1))) > angle_tol: continue
            nrm = np.array([-d[1], d[0]]); inl = cur[np.abs((xy[cur] - xy[a]) @ nrm) < tol]
            if best is None or len(inl) > len(best[0]): best = (inl, d)
        if best is None or len(best[0]) < min_pts: break
        inl, d = best
        # refine the line by PCA on its inliers, then re-collect inliers
        c0 = xy[inl].mean(0); u = np.linalg.svd(xy[inl] - c0)[2][0]
        if dom is not None and u @ dom < 0: u = -u
        nrm = np.array([-u[1], u[0]]); inl = cur[np.abs((xy[cur] - c0) @ nrm) < tol]
        rem[inl] = False
        if any(abs((c0 - r[1]) @ np.array([-r[2][1], r[2][0]])) < min_row_dist for r in rows): continue
        if dom is None: dom = u
        rows.append((inl, c0, u))
    return rows


def row_dp(s, strength, sp, c):
    """Pick the chain of candidates along one row that maximises
         sum_i u(strength_i) + sum_(i->j) [ -(g - k sp)^2 / (2 sigma^2 k) - gamma (k - 1) ],
    g = gap between consecutive picks, k = max(1, round(g / sp)) (k - 1 trees missing in the gap).
    u = clip(dp_a * (strength - dp_theta), -dp_a, dp_a): strong candidates pay for themselves,
    weak ones only enter where they fit the spacing. Gaps < dp_gmin are forbidden."""
    o = np.argsort(s); s = np.asarray(s)[o]; w = np.clip(c['dp_a'] * (np.asarray(strength)[o] - c['dp_theta']), -c['dp_a'], c['dp_a'])
    n = len(s); best = w.copy(); prev = np.full(n, -1)
    for j in range(n):
        for i in range(j):
            g = s[j] - s[i]
            if g < c['dp_gmin']: continue
            k = max(1, round(g / sp))
            v = best[i] + w[j] - (g - k * sp) ** 2 / (2 * c['dp_sigma'] ** 2 * k) - c['dp_gamma'] * (k - 1)
            if v > best[j]: best[j], prev[j] = v, i
    j = int(np.argmax(best)); chain = []
    while j >= 0: chain.append(j); j = prev[j]
    return [int(o[j]) for j in chain[::-1]]


# ---------------------------------------------------------------- main
def detect(cfg=None, P=None, h=None, verbose=True, return_offrow=False, debug=None):
    """debug: optional dict, filled with the candidates and, per row, the DP pool / chain / spacing (figures)."""
    c = copy.deepcopy(CFG); c.update(cfg or {}); rng = np.random.default_rng(0)
    if P is None:
        P, h = load_map(c['map'])
    t2 = cKDTree(P[:, :2])
    seeds, peak = density_peaks(P, h, *c['band'], cell=c['seed_cell'], sigma=c['seed_sigma'],
                                nms=c['seed_nms'], thr=c['seed_thr'])
    F = candidate_features(seeds, P, h, t2, band=c['band'])
    cand = [dict(sx=s[0], sy=s[1], peak=pk, **f) for s, pk, f in zip(seeds, peak, F)]
    if debug is not None: debug.update(cand=cand, rows=[])

    def gate(d, relax=1.0):
        return d['n_core'] >= c['min_core'] * relax and d['n_up'] >= c['min_up'] * relax and d['vocc'] >= c['min_vocc'] * (relax if relax < 1 else 1) ** 0.5

    def sc(d):
        if c['nms_score'] == 'core_up': return np.log1p(d['n_core']) + np.log1p(d['n_up'])
        if c['nms_score'] == 'peak': return d['peak']
        return d['n_core']

    def strength(d):
        """>= 1 when the candidate passes the full gate."""
        r = min(d['n_core'] / c['min_core'], d['n_up'] / c['min_up'])
        return min(r, d['vocc'] / c['min_vocc']) if c.get('strength_vocc', False) else r

    def nms(L, R, keep=()):
        L = sorted(L, key=sc, reverse=True); out = list(keep)
        for d in L:
            if all(np.hypot(d['sx'] - e['sx'], d['sy'] - e['sy']) >= R for e in out): out.append(d)
        return out[len(keep):]

    def centre(d):
        d = dict(d); d['x'], d['y'], d['r'], d['inl'] = d['sx'], d['sy'], np.nan, np.nan
        if c['centre'] == 'peak':
            d['r'] = radius_fixed_centre(P, h, t2, (d['x'], d['y']), c)
            return d
        idx = np.array(t2.query_ball_point([d['sx'], d['sy']], c['fit_win']), int)
        hh = h[idx]; m = (hh > c['fit_band'][0]) & (hh < c['fit_band'][1]); xy = P[idx[m], :2]
        if c['centre'] == 'centroid':
            if len(xy): d['x'], d['y'] = xy.mean(0)
            return d
        if c['centre'] == 'lowpeak':
            # density peak of the low slice, searched within seed_tol of the seed
            if len(xy) >= 10:
                cs, R = 0.01, c['fit_win']; nb = int(2 * R / cs)
                H, _, _ = np.histogram2d(xy[:, 0] - d['sx'], xy[:, 1] - d['sy'], bins=nb, range=[[-R, R], [-R, R]])
                S = ndimage.gaussian_filter(H, c['lp_sigma'] / cs)
                g = (np.arange(nb) + .5) * cs - R; gx, gy = np.meshgrid(g, g, indexing='ij')
                S[np.hypot(gx, gy) > c['seed_tol']] = -1
                i, j = np.unravel_index(np.argmax(S), S.shape); d['x'], d['y'] = d['sx'] + g[i], d['sy'] + g[j]
            return d
        f = fit_circle(xy, (d['sx'], d['sy']), c, rng)
        if f is not None: d['x'], d['y'], d['r'], d['inl'] = f
        return d

    gated = [d for d in cand if gate(d)]
    det = nms(gated, c['nms_r'])
    rows_out = []
    if c['rows'] and len(det) >= 2 * c['row_min_pts']:
        # rows from the strong detections; then everything is re-selected per row
        rfit = nms([d for d in cand if gate(d, c.get('rowfit_relax', 1.0))], c['nms_r'])
        xy = np.array([[d['sx'], d['sy']] for d in rfit])
        rows = fit_rows(xy, c['row_tol'], c['row_min_pts'], c['row_angle_tol'], rng)
        n_off = len(rfit) - sum(len(r[0]) for r in rows)
        cxy = np.array([[d['sx'], d['sy']] for d in cand])
        out = []; n_gap = n_end = 0
        # tree spacing: global median of the gaps between consecutive strong detections in [2, 4.5] m;
        # per row refined as median(g / round(g / sp_glob)) so that rows with sparse strong detections
        # (gaps of 2-3 trees) still get the right spacing
        gaps = []
        for inl, c0, u in rows:
            g = np.diff(np.sort((xy[inl] - c0) @ u)); gaps += list(g[(g > 2.0) & (g < 4.5)])
        sp_glob = float(np.median(gaps)) if gaps else 3.0
        def row_spacing(s):
            g = np.diff(np.sort(s)); g = g[g > 0.6 * sp_glob]
            return float(np.median(g / np.maximum(1, np.round(g / sp_glob)))) if len(g) else sp_glob
        for k, (_, c0, u) in enumerate(rows):
            nrm = np.array([-u[1], u[0]]); s_all = (cxy - c0) @ u
            onrow = np.abs((cxy - c0) @ nrm) < c['row_tol']
            sel = nms([cand[i] for i in np.where(onrow)[0] if gate(cand[i])], c['nms_r'])
            for d in sel: d.update(source='gate')
            if c['gapfill'] and len(sel) >= 2:
                s = np.sort([(np.array([d['sx'], d['sy']]) - c0) @ u for d in sel]); sp = np.median(np.diff(s))
                slots = []
                if 2.0 < sp < 4.5:
                    for a, b in zip(s[:-1], s[1:]):
                        nmiss = int(round((b - a) / sp)) - 1
                        slots += [(a + q * (b - a) / (nmiss + 1), 'gapfill') for q in range(1, nmiss + 1)]
                    for q in range(1, c.get('end_slots', 0) + 1):
                        slots += [(s[0] - q * sp, 'end'), (s[-1] + q * sp, 'end')]
                for e, src in slots:
                    relax = c['gap_relax'] if src == 'gapfill' else c.get('end_relax', 1.0)
                    L = [cand[i] for i in np.where(onrow & (np.abs(s_all - e) < 0.35 * sp))[0] if gate(cand[i], relax)]
                    L = nms(L, c['nms_r'], keep=sel)
                    if L:
                        L[0] = dict(L[0], source=src); sel.append(L[0])
                        n_gap += src == 'gapfill'; n_end += src == 'end'
            if c.get('row_model') == 'dp' and len(sel) >= 2:
                sp = row_spacing([(np.array([d['sx'], d['sy']]) - c0) @ u for d in sel])
                pool = [i for i in np.where(onrow)[0] if gate(cand[i], c['dp_min_relax'])]
                pick = row_dp(s_all[pool], [strength(cand[i]) for i in pool], sp, c)
                if debug is not None:
                    debug['rows'].append(dict(row_id=k, c0=c0.tolist(), u=u.tolist(), sp=sp, pool=[int(i) for i in pool],
                                              strength=[float(strength(cand[i])) for i in pool], pick=[int(pool[i]) for i in pick]))
                sel = [dict(cand[pool[i]], source='gate' if gate(cand[pool[i]]) else 'dp') for i in pick]
            for d in sel: d.update(row_id=k, kept_by_row_prior=1)
            out += sel
        # a candidate can sit within row_tol of two rows only if rows are < 2*row_tol apart (not the case here)
        det = [centre(d) for d in nms(out, c['nms_r'])]
        if verbose: print(f'row prior ({c["row_model"]}): {len(rows)} rows, {n_off} strong detections off-row'
                          + ('' if c['row_model'] == 'dp' else f', {n_gap} gap-filled, {n_end} row-end'))
        rows_out = [(c0.tolist(), u.tolist()) for _, c0, u in rows]
        # strong detections that lie on no row: reported with kept_by_row_prior = 0 (not in the main output)
        far = lambda d: all(abs((np.array([d['sx'], d['sy']]) - c0) @ np.array([-u[1], u[0]])) >= c['row_tol']
                            for _, c0, u in rows)
        offrow = [dict(centre(d), row_id=-1, kept_by_row_prior=0, source='off-row') for d in nms(gated, c['nms_r']) if far(d)]
    else:
        offrow = []
        det = [centre(d) for d in det]
        for d in det: d.update(row_id=-1, kept_by_row_prior=0, source='gate')
    for d in det + offrow: d['support'] = d['n_core']
    if return_offrow: return det, rows_out, offrow
    return det, rows_out


_MAP_CACHE = {}
def load_map(name):
    if name not in _MAP_CACHE:
        if name == 'dense':
            P = np.load(DENSE / 'dense_kf.npz')['xyz'].astype(float); h = np.load(DENSE / 'dense_kf_height.npy').astype(float)
        elif name == 'dense_all':   # every 2nd scan, interpolated poses, 1/3 of the points (build_dense_all.py)
            z = np.load(DENSE / 'dense_all_s2_sub3.npz'); P = z['xyz'].astype(float); h = z['h'].astype(float)
        else:
            P = read_pcd_xyz(GLOBALMAP); h = local_ground(P)
        _MAP_CACHE[name] = (P, h)
    return _MAP_CACHE[name]


COLS = ['support', 'n_up', 'vocc', 'inl', 'kept_by_row_prior', 'row_id', 'source']

def run(cfg=None, out=None, verbose=True, ladders=(None, 2, 1)):
    det, rows = detect(cfg, verbose=verbose)
    out = Path(out or OUT / 'trunks_v2.csv'); out.parent.mkdir(exist_ok=True, parents=True)
    write_det(out, det, COLS)
    res = {}
    for L in ladders:
        s, r, full = score(out, L); res[L or 'all'] = r
        if verbose: print(f"[{'all' if L is None else 'L%d' % L}] {s}")
    return det, res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cfg', default='{}', help='JSON overrides of CFG')
    ap.add_argument('--out', default=None); a = ap.parse_args()
    run(json.loads(a.cfg), a.out)
