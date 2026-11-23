#!/usr/bin/env python3
"""Step 4: CA-10 trunk map with the frozen CB-11 detector, on the final three-piece merge.

  trunk_map.py <ca10_run dir>

Dense map (CB-11 rep1own recipe, trunks.dense_piece): each piece's own keyframe scans, 1-25 m, per-point deskew
with the piece's own keyframe trajectory, moved into A's frame by its merge transform (A: identity, B-pre: T1,
B-post: T2; merge_final/T_final.json); height above local ground computed on the whole merged cloud (common.local_ground).
Detector: trunk_detect_v2.detect() with the frozen CFG (row model 'dp'), as final_rep1.py. ENU: the merged
keyframes' SE2 fit to GNSS (merge_final/align_A_to_enu.json; CB-11 used the same kind of fit for rep1).
No surveyed ground truth exists for Campinas A (j9_answers.md: 0 % overlap with the CB-11 survey), so instead:
  rows.csv / row_lines.csv   per row (PCA line of its trunks): n, length, spacing (median / mean / sd of
                             consecutive gaps), lateral residual RMS and max (row straightness), as CB-11
                             row_lines.csv + row_coverage.csv (final_rep1.row_coverage, unchanged)
Merge check (brief 3.5): trunks from an A-only map vs a B-only map (B-pre + B-post, same transforms), shared rows:
  NN pairs within 1.0 m, median offset along / across the row, and the along-row shift curve per B piece.
Outputs in trunks/.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import ca10lib as L      # noqa: E402
import gnss_fit as G     # noqa: E402
import trunks as Tr      # noqa: E402
from final_rep1 import row_coverage   # noqa: E402  (unchanged)
from trunk_detect_v2 import CFG       # noqa: E402

R = Path(sys.argv[1]); O = R / 'trunks'; O.mkdir(exist_ok=True)
t0 = time.time(); log = lambda *a: print(f'[{time.time() - t0:6.0f} s]', *a, flush=True)
F = json.load(open(R / 'merge_final/T_final.json'))
T1, T2 = np.array(F['T1_Bpre']), np.array(F['T2'])
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
PIECES = dict(A=(SA, kfA, 0.0, L.A_END, np.eye(4)), Bpre=(SB, kfB, L.B_START, 966.0, T1),
              Bpost=(SB, kfB, 975.0, 1400.0, T2))
parts = {}
for k, (S, kf, a, b, T) in PIECES.items():
    parts[k] = Tr.dense_piece(S, kf, a, b, T)[0]
    log('piece', k, len(parts[k]))
P = np.vstack(list(parts.values())); piece_id = np.concatenate([np.full(len(v), i) for i, v in enumerate(parts.values())])
from common import local_ground   # noqa: E402
h = local_ground(P)
np.savez_compressed(O / 'dense_merged_meta.npz', n_points=len(P), pieces=list(parts), counts=[len(v) for v in parts.values()])
log('dense merged', len(P))
det, rows, off, dbg = Tr.run_detect(P, h)
log('detect: trunks', len(det), 'rows', len(rows), 'off-row', len(off))

# ENU alignment of the merged trajectory (keyframes, whole run, t > 10 s), as align_rep1.json for CB-11
kfM = L.read_csv_traj(R / 'traj/merged_final/keyframes.csv')
fm = G.fit(np.column_stack([kfM[:, 0] - L.T0, kfM[:, 1:]]), 0.0, 1400.0)
RA, tA = np.array(fm['R']), np.array(fm['t'])
json.dump(dict(R=fm['R'], t=fm['t'], method='SE2 Umeyama as ate_vs_gps.py (t > 10 s), merged keyframes vs CA-10 GNSS, '
               '822.8-838.8 s excluded', n=fm['n'], ate_m=fm['ate_m'], max_m=fm['max_m'], traj='traj/merged_final/keyframes.csv',
               gps='~/ca10_gap/gnss_enu.csv (bag-ENU, origin = first CA-10 fix)'),
          open(R / 'merge_final/align_A_to_enu.json', 'w'), indent=1)

tree = cKDTree(P[:, :2])


def ground_z(x, y):
    idx = tree.query_ball_point([x, y], 0.2)
    return float(np.median(P[idx, 2] - h[idx])) if idx else 0.0


def piece_of(x, y):
    idx = tree.query_ball_point([x, y], 0.5)
    return list(parts)[int(np.bincount(piece_id[idx]).argmax())] if idx else ''


def write(path, D):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_A', 'y_A', 'z_ground_A', 'east_m', 'north_m', 'r', 'support', 'n_up', 'vocc', 'kept_by_row_prior',
                    'row_id', 'source', 'piece_majority'])
        for d in D:
            e = RA @ np.array([d['x'], d['y']]) + tA
            w.writerow([round(d['x'], 4), round(d['y'], 4), round(ground_z(d['x'], d['y']), 3), round(e[0], 4), round(e[1], 4),
                        round(float(d['r']), 4), d.get('support', ''), d.get('n_up', ''), round(float(d['vocc']), 3),
                        d.get('kept_by_row_prior', ''), d.get('row_id', ''), d.get('source', ''), piece_of(d['x'], d['y'])])


write(O / 'trunks_final_ca10.csv', det); write(O / 'trunks_final_ca10_all.csv', det + off)
# rows: straightness and spacing
cov, gaps = row_coverage(det, dbg, CFG)
for name, Lst in (('row_coverage.csv', cov), ('row_gaps.csv', gaps)):
    if Lst:
        with open(O / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(Lst[0])); w.writeheader(); w.writerows(Lst)
rl = []
for rid in sorted({d['row_id'] for d in det if d.get('row_id', '') != ''}):
    X = np.array([[d['x'], d['y']] for d in det if d['row_id'] == rid])
    c = X.mean(0); w_, V = np.linalg.eigh(np.cov((X - c).T)); u = V[:, 1]; u = u if u[0] < 0 else -u
    s = (X - c) @ u; n = np.array([-u[1], u[0]]); res = (X - c) @ n; o = np.argsort(s); g = np.diff(s[o])
    st, en = c + u * s.min(), c + u * s.max()
    se, ee = RA @ st + tA, RA @ en + tA; ce = RA @ c + tA; ue = RA @ u
    rl.append(dict(row_id=rid, n_trunks=len(X), c_x_A=c[0], c_y_A=c[1], u_x_A=u[0], u_y_A=u[1],
                   start_x_A=st[0], start_y_A=st[1], end_x_A=en[0], end_y_A=en[1],
                   c_east=ce[0], c_north=ce[1], u_east=ue[0], u_north=ue[1], start_east=se[0], start_north=se[1],
                   end_east=ee[0], end_north=ee[1], length_m=float(s.max() - s.min()),
                   mean_spacing_m=float(g.mean()) if len(g) else np.nan, median_spacing_m=float(np.median(g)) if len(g) else np.nan,
                   sd_spacing_m=float(g.std()) if len(g) else np.nan, perp_rms_cm=float(np.sqrt((res ** 2).mean()) * 100),
                   perp_max_cm=float(np.abs(res).max() * 100),
                   pieces=';'.join(sorted({piece_of(*x) for x in X}))))
with open(O / 'row_lines.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rl[0])); w.writeheader()
    w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rl])
log('rows written', len(rl))

# merge check: A-only vs B-only trunks
PAo = parts['A']; PBo = np.vstack([parts['Bpre'], parts['Bpost']])
dAo = Tr.run_detect(PAo, local_ground(PAo)); log('A-only detect', len(dAo[0]), len(dAo[2]))
dBo = Tr.run_detect(PBo, local_ground(PBo)); log('B-only detect', len(dBo[0]), len(dBo[2]))
XA = Tr.xy(dAo[0] + dAo[2]); XB = Tr.xy(dBo[0] + dBo[2])
# B trunks by piece (majority of the B-only map points around them)
tb = cKDTree(PBo[:, :2]); pidB = np.r_[np.zeros(len(parts['Bpre'])), np.ones(len(parts['Bpost']))]
pieceB = np.array(['Bpre' if np.mean(pidB[tb.query_ball_point(x, 0.5)] == 0) > 0.5 else 'Bpost' for x in XB])
rowu = {r['row_id']: (np.array([r['c_x_A'], r['c_y_A']]), np.array([r['u_x_A'], r['u_y_A']])) for r in rl if r['n_trunks'] >= 3}
# along/across axes: the nearest merged row line to each B trunk
cs = np.array([v[0] for v in rowu.values()]); us = np.array([v[1] for v in rowu.values()])
check = dict(A_only=dict(trunks=len(dAo[0]), off_row=len(dAo[2])), B_only=dict(trunks=len(dBo[0]), off_row=len(dBo[2])),
             note='trunks = row-model trunks + strong off-row detections of each map (B-pre is a single pass)')
ta_ = cKDTree(XA)
for pc in ('Bpre', 'Bpost'):
    Xb = XB[pieceB == pc]
    d, j = ta_.query(Xb, distance_upper_bound=1.0); ok = np.isfinite(d)
    if not ok.any():
        check[pc] = dict(n=len(Xb), pairs=0); continue
    k = np.argmin(np.abs(np.cross(us[None], Xb[ok][:, None] - cs[None])), axis=1)   # nearest row line
    uu = us[k]; nn = np.stack([-uu[:, 1], uu[:, 0]], 1); dv = Xb[ok] - XA[j[ok]]
    al_, ac_ = np.einsum('ij,ij->i', dv, uu), np.einsum('ij,ij->i', dv, nn)
    shifts = np.round(np.arange(-6, 6.001, 0.05), 2)
    # shift along each trunk's own row direction
    Xs = Xb[ok]
    curve = [int(np.isfinite(ta_.query(Xs + s * uu, distance_upper_bound=0.3)[0]).sum()) for s in shifts]
    check[pc] = dict(n=int(len(Xb)), pairs_within_1m=int(ok.sum()), median_along_m=float(np.median(al_)),
                     median_across_m=float(np.median(ac_)), median_abs_along_m=float(np.median(np.abs(al_))),
                     median_abs_across_m=float(np.median(np.abs(ac_))), p90_dist_m=float(np.percentile(d[ok], 90)),
                     within_0p3_at_0=curve[int(np.where(shifts == 0)[0][0])],
                     curve_best=[float(shifts[int(np.argmax(curve))]), int(max(curve))],
                     at_pm3=[curve[int(np.argmin(abs(shifts + 3.0)))], curve[int(np.argmin(abs(shifts - 3.0)))]])
    np.savetxt(O / f'merge_check_shift_curve_{pc}.csv', np.column_stack([shifts, curve]), delimiter=',',
               header='shift_along_m,n_within_0p3m', comments='', fmt='%.3f')
np.savetxt(O / 'merge_check_trunks_Aonly.csv', XA, delimiter=',', header='x_A,y_A', comments='', fmt='%.4f')
np.savetxt(O / 'merge_check_trunks_Bonly.csv', np.column_stack([XB, pieceB == 'Bpost']), delimiter=',',
           header='x_A,y_A,is_Bpost', comments='', fmt='%.4f')
summ = dict(n_points=len(P), pieces={k: len(v) for k, v in parts.items()}, trunks=len(det), rows=len(rl), off_row=len(off),
            trunks_by_source={s: sum(1 for d in det if d.get('source') == s) for s in ('gate', 'dp')},
            spacing_median_of_rows_m=float(np.nanmedian([r['median_spacing_m'] for r in rl])),
            perp_rms_cm_median_of_rows=float(np.nanmedian([r['perp_rms_cm'] for r in rl])),
            merge_check=check, ground_truth='none for Campinas A (j9_answers.md: 0 % overlap with the CB-11 survey)')
json.dump(summ, open(O / 'summary.json', 'w'), indent=1)
log(json.dumps(summ))
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # noqa: E402
fig, ax = plt.subplots(figsize=(10, 9))
sub = P[::40]; ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c='#DDD8CC')
Xd = Tr.xy(det); ax.scatter(Xd[:, 0], Xd[:, 1], s=6, c='#8B5A2B', label=f'trunks ({len(det)})')
if off: Xo = Tr.xy(off); ax.scatter(Xo[:, 0], Xo[:, 1], s=6, marker='x', c='gray', label=f'off-row ({len(off)})')
for r in rl:
    ax.plot([r['start_x_A'], r['end_x_A']], [r['start_y_A'], r['end_y_A']], 'k-', lw=0.4)
    ax.annotate(f"{r['row_id']}:{r['n_trunks']}", (r['end_x_A'], r['end_y_A']), fontsize=6)
for k in kfM[::1]: pass
ax.plot(kfM[:, 1], kfM[:, 2], color='#D4A62A', lw=0.6, label='merged keyframes')
ax.set(aspect='equal', xlabel='x (A frame) [m]', ylabel='y (A frame) [m]', title='CA-10 trunk map, frozen CB-11 detector')
ax.legend(fontsize=7, loc='lower left'); fig.tight_layout(); fig.savefig(O / 'whole_map_ca10.png', dpi=130)
log('done')
