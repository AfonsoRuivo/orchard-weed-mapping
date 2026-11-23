#!/usr/bin/env python3
"""
final_rep1.py - B2 + B5: final trunk map on the reference run (LIO-SAM rep1) and the row-coverage check.

  python3 final_rep1.py [--map dense_rep1own] [--traj liosam_rep1_keyframes.csv]

Map: rep1's OWN 649 keyframe scans on rep1's keyframe poses (exact poses, like the July map;
build_dense_traj.py --scans rep1_keyframe_scans.npz --label rep1own). Final CFG, gates unchanged.
Outputs (../v2_out/):
  trunks_final_rep1.csv      trunks kept by the row model, columns as trunks_v2_final.csv:
                             x, y        rep1's own map frame (the frame of liosam_rep1_keyframes.csv, i.e. what a
                                         weed map built on the rep1 trajectory uses)
                             r, support, n_up, vocc, kept_by_row_prior, row_id, source  (as before)
                             x_july, y_july  same trunk in the July GlobalMap / GT frame (pre-alignment + one rigid
                                         SE(2) to the hand picks, eval_align.py)
                             x_enu, y_enu    July frame -> ENU with the weed map's current SLAM->ENU alignment
                             gt_id       surveyed trunk id (0.5 m gate), empty elsewhere
  trunks_final_rep1_all.csv  + strong detections on no row (kept_by_row_prior = 0)
  final_rep1/                eval_align.py scores, row_coverage.csv, whole_map_rep1.png
"""
import argparse, csv, json, subprocess, sys
import numpy as np
from scipy.spatial import cKDTree
from trunk_detect_v2 import CFG, OUT, detect
from common import DENSE, load_gt, write_det
from build_dense_traj import TRAJ_DIR, JULY, Traj, load_traj, prealign
from eval_align import align, hmatch

# July SLAM frame -> ENU, the alignment the weed map uses now (weed_map_yolo_clusters_cached.py, as in final_output.py)
R_ENU = np.array([[-0.14508231170482752, -0.9894195888653019], [0.9894195888653021, -0.14508231170482772]])
t_ENU = np.array([-0.14110743455980668, -2.798850687960666])

COLS = ['x_july', 'y_july', 'x_enu', 'y_enu', 'support', 'n_up', 'vocc', 'kept_by_row_prior', 'row_id', 'source', 'gt_id']


def row_coverage(det, dbg, c):
    """Per row: detected trunks vs the row length / spacing, and the gaps longer than 1.5 spacings."""
    cand = dbg['cand']; cxy = np.array([[d['sx'], d['sy']] for d in cand]); out, gaps = [], []
    for r in dbg['rows']:
        c0, u = np.array(r['c0']), np.array(r['u']); nrm = np.array([-u[1], u[0]]); sp = r['sp']
        s = np.sort([(np.array([d['sx'], d['sy']]) - c0) @ u for d in det if d['row_id'] == r['row_id']])
        # row extent: every candidate on the row line with some band blob + column (strength >= 0.2),
        # so the ends of the row are seen even where no trunk was kept
        st = np.array([min(d['n_core'] / c['min_core'], d['n_up'] / c['min_up']) for d in cand])
        on = (np.abs((cxy - c0) @ nrm) < c['row_tol']) & (st >= 0.2); se = (cxy[on] - c0) @ u
        lo, hi = min(se.min(), s[0]), max(se.max(), s[-1])
        g = np.diff(s); miss = int(np.sum(np.maximum(np.round(g / sp) - 1, 0)))
        exp_det = int(round((s[-1] - s[0]) / sp)) + 1; exp_ext = int(round((hi - lo) / sp)) + 1
        out.append(dict(row_id=r['row_id'], n_trunks=len(s), length_det_m=round(s[-1] - s[0], 2), length_ext_m=round(hi - lo, 2),
                        spacing_m=round(sp, 3), expected_in_span=exp_det, expected_in_extent=exp_ext, missing_in_span=miss,
                        long_gaps=int(np.sum(g > 1.5 * sp)), max_gap_m=round(float(g.max()), 2) if len(g) else 0,
                        c0x=round(c0[0], 3), c0y=round(c0[1], 3), ux=round(u[0], 4), uy=round(u[1], 4)))
        for a, b in zip(s[:-1], s[1:]):
            if b - a > 1.5 * sp:
                m = c0 + u * (a + b) / 2
                gaps.append(dict(row_id=r['row_id'], s_from=round(a, 2), s_to=round(b, 2), gap_m=round(b - a, 2),
                                 trees_missing=int(round((b - a) / sp)) - 1, x_mid=round(m[0], 2), y_mid=round(m[1], 2)))
    return out, gaps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--map', default='dense_rep1own')
    ap.add_argument('--traj', default='liosam_rep1_keyframes.csv'); ap.add_argument('--tag', default='rep1')
    a = ap.parse_args(); O = OUT / f'final_{a.tag}'; O.mkdir(parents=True, exist_ok=True)
    P = np.load(DENSE / f'{a.map}.npz')['xyz'].astype(float); h = np.load(DENSE / f'{a.map}_height.npy').astype(float)
    dbg = {}; det, rows, off = detect(P=P, h=h, verbose=True, return_offrow=True, debug=dbg)
    # back to rep1's own frame: invert the SE(3) pre-alignment (point = trunk centre at local ground height)
    est = Traj(load_traj(TRAJ_DIR / a.traj)); ref = Traj(load_traj(JULY))
    Ra, ta, _ = prealign(est, ref, ref.t[0], ref.t[0] + 120.0)
    t2 = cKDTree(P[:, :2]); G, gxy = load_gt()
    D = np.array([[d['x'], d['y']] for d in det]); R2, t2d, info = align(D, gxy)
    gi, di, _ = hmatch(gxy, D @ R2.T + t2d, 0.5); gid = {int(j): G[i]['id'] for i, j in zip(gi, di)}
    for k, d in enumerate(det + off):
        idx = t2.query_ball_point([d['x'], d['y']], 0.2); zg = float(np.median(P[idx, 2] - h[idx])) if idx else 0.0
        pn = Ra.T @ (np.array([d['x'], d['y'], zg]) - ta)
        pj = R2 @ np.array([d['x'], d['y']]) + t2d; pe = R_ENU @ pj + t_ENU
        d.update(x=round(float(pn[0]), 4), y=round(float(pn[1]), 4), x_july=round(float(pj[0]), 4), y_july=round(float(pj[1]), 4),
                 x_enu=round(float(pe[0]), 4), y_enu=round(float(pe[1]), 4), r=round(float(d['r']), 4), vocc=round(float(d['vocc']), 3),
                 gt_id=gid.get(k, '') if k < len(det) else '')
    write_det(OUT / f'trunks_final_{a.tag}.csv', det, COLS)
    write_det(OUT / f'trunks_final_{a.tag}_all.csv', det + off, COLS)
    # score: the July-frame columns (rigid fit already applied) -> eval_align without a second fit
    tmp = O / 'det_july.csv'
    write_det(tmp, [dict(d, x=d['x_july'], y=d['y_july']) for d in det], ['row_id', 'source'])
    print(subprocess.run([sys.executable, 'eval_align.py', '--det', str(tmp), '--out', str(O), '--no-align'],
                         capture_output=True, text=True).stdout)
    S = json.load(open(O / 'summary.json')); S['align'] = dict(info, prealign_R=Ra.tolist(), prealign_t=ta.tolist())
    json.dump(S, open(O / 'summary.json', 'w'), indent=1)
    cov, gaps = row_coverage([dict(x, **{'sx': x['sx'], 'sy': x['sy']}) for x in det], dbg, CFG)
    for name, L in (('row_coverage.csv', cov), ('row_gaps.csv', gaps)):
        with open(O / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(L[0])); w.writeheader(); w.writerows(L)
    print(f'{len(det)} trunks on {len(rows)} rows (+{len(off)} off-row); rows with gaps > 1.5 sp: '
          f'{sum(1 for r in cov if r["long_gaps"])}, trees missing inside spans: {sum(r["missing_in_span"] for r in cov)}')
    plot(O, det, off, dbg, cov, gaps, a.tag)


def plot(O, det, off, dbg, cov, gaps, tag):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(14, 11))
    X = np.array([[d['sx'], d['sy']] for d in det]); ax.plot(X[:, 0], X[:, 1], 'g.', ms=4, label=f'trunks ({len(det)})')
    if off: Y = np.array([[d['sx'], d['sy']] for d in off]); ax.plot(Y[:, 0], Y[:, 1], 'x', color='gray', ms=3, label=f'off-row ({len(off)})')
    for r in cov:
        c0, u = np.array([r['c0x'], r['c0y']]), np.array([r['ux'], r['uy']])
        s = [(np.array([d['sx'], d['sy']]) - c0) @ u for d in det if d['row_id'] == r['row_id']]
        p0, p1 = c0 + u * min(s), c0 + u * max(s); ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.3)
        ax.annotate(f"{r['row_id']}:{r['n_trunks']}", p1, fontsize=6)
    for g in gaps: ax.plot(g['x_mid'], g['y_mid'], 'o', mfc='none', mec='r', ms=4 + 2 * g['trees_missing'])
    ax.plot([], [], 'o', mfc='none', mec='r', label='gap > 1.5 spacing (size = trees missing)')
    ax.set(aspect='equal', title=f'v2 trunks on the {tag} dense map (map frame, pre-aligned to July); row:count', xlabel='x [m]', ylabel='y [m]')
    ax.legend(loc='lower left', fontsize=8); f.tight_layout(); f.savefig(O / f'whole_map_{tag}.png', dpi=130); plt.close(f)


if __name__ == '__main__':
    main()
