#!/usr/bin/env python3
"""CA-10 trunk maps with the frozen CB-11 trunk detector (orchard-weed-detection/trunk/v2, unchanged; imported).

Dense map recipe = CB-11 rep1own (build_dense_traj.py): the piece's own LIO-SAM keyframe scans, points with
range >= 1 m and <= 25 m (extract_scans.py default), each point placed with the piece's keyframe trajectory
interpolated at header stamp + per-point time (linear translation, slerp rotation), no voxelisation; height above
local ground = common.local_ground (5th percentile z in 0.5 m cells). Then trunk_detect_v2.detect() with the frozen
CFG. Difference: no pre-alignment to the CB-11 July frame (it does not exist here; LIO-SAM frames are gravity-
aligned, which is what the July pre-alignment was for); pieces are put in A's frame by their merge transform.

Library: dense_piece(scans, kf, t0, t1, T) -> (xyz, h);  run_detect(xyz, h) -> (det, rows, off, dbg)
CLI:     trunks.py check <ca10_run dir>      junction trunk check (brief 3.5) for the B-pre candidates
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/afonso/thesis/orchard-weed-detection/trunk/v2')
import ca10lib as L                                   # noqa: E402
from common import local_ground                       # noqa: E402
from trunk_detect_v2 import detect, CFG               # noqa: E402


def dense_piece(S, kf, t0, t1, T=np.eye(4), min_range=1.0, max_range=25.0):
    """Dense map of the keyframe scans with bag time in [t0, t1], deskewed with kf, moved by T."""
    m = (kf[:, 0] - L.T0 >= t0 - 1.0) & (kf[:, 0] - L.T0 <= t1 + 1.0)
    k = kf[m]
    sl = Slerp(k[:, 0], Rotation.from_quat(k[:, 4:8]))
    out = []
    for r in k:
        tb = r[0] - L.T0
        if not (t0 <= tb <= t1):
            continue
        j = S.index_of(r[0])
        if j is None:
            continue
        s = S.scan(j)
        rr = np.linalg.norm(s[:, :3], axis=1)
        s = s[(rr >= min_range) & (rr <= max_range)]
        t = np.clip(r[0] + s[:, 5].astype(float), k[0, 0], k[-1, 0])
        p = np.stack([np.interp(t, k[:, 0], k[:, c]) for c in (1, 2, 3)], 1)
        out.append(sl(t).apply(s[:, :3].astype(float)) + p)
    xyz = np.vstack(out) @ T[:3, :3].T + T[:3, 3]
    return xyz, local_ground(xyz)


def run_detect(xyz, h):
    dbg = {}
    det, rows, off = detect(P=xyz, h=h, verbose=False, return_offrow=True, debug=dbg)
    return det, rows, off, dbg


def xy(det):
    return np.array([[d['x'], d['y']] for d in det]) if det else np.zeros((0, 2))


def junction_check(R):
    o = json.load(open(R / 'merge/T_A_from_B.json')); c = json.load(open(R / 'merge/candidates.json'))
    u = np.array(o['refined_minus_init']['axis_along']); lat = np.array([-u[1], u[0]])
    kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
    SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
    # A's last pass down this alley (after its last turn) and B before its U-turn
    # A-only trunks from A's whole map (row model works: every pass of A along these rows). B-pre is one pass
    # (839.5-966 s): the row model finds no rows there (needs >= 5 trunks on a fitted line), so B-pre trunks =
    # all detections that pass the frozen strong gate, on-row and off-row (det + off). Gates unchanged.
    PA, hA = dense_piece(SA, kfA, 0.0, L.A_END)
    dA, rA, offA, _ = run_detect(PA, hA)
    res = dict(A_piece=[0.0, L.A_END], B_piece=[L.B_START, 966.0], axis_along=u.tolist(), n_A=len(dA), n_A_rows=len(rA),
               n_A_offrow=len(offA), B_trunks='det + off-row strong detections', candidates={})
    XA = xy(dA)
    # junction zone: trunks within 12 m (along) of the junction, on the two rows bordering the shared alley
    pj = kfA[-1, 1:3]
    for name in ('init', 'gicp6'):
        T = np.array(c['candidates']['lidar_chain'][name]['T'])
        PB, hB = dense_piece(SB, kfB, L.B_START, 966.0, T)
        dB, rB, offB, _ = run_detect(PB, hB)
        dB = dB + offB
        XB = xy(dB)
        sa, la = (XA - pj) @ u, (XA - pj) @ lat
        sb, lb = (XB - pj) @ u, (XB - pj) @ lat
        # shared region: along range seen by both detections, |lateral| < 4 m (the two bordering rows)
        lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
        ma = (sa >= lo) & (sa <= hi) & (np.abs(la) < 4); mb = (sb >= lo) & (sb <= hi) & (np.abs(lb) < 4)
        A2, B2 = XA[ma], XB[mb]
        d, j = cKDTree(A2).query(B2, distance_upper_bound=1.5) if len(A2) else (np.array([]), None)
        ok = np.isfinite(d)
        dv = B2[ok] - A2[j[ok]] if ok.any() else np.zeros((0, 2))
        # shift curve: number of B trunks within 0.3 m of an A trunk when B is shifted along the row
        shifts = np.round(np.arange(-6, 6.001, 0.05), 2); tree = cKDTree(A2)
        curve = [int(np.isfinite(tree.query(B2 + s * u, distance_upper_bound=0.3)[0]).sum()) for s in shifts]
        curve_med = [float(np.median(tree.query(B2 + s * u)[0])) for s in shifts]
        k0 = int(np.argmax(curve))
        res['candidates'][name] = dict(
            n_B=len(dB), shared_along_range_m=[float(lo), float(hi)], n_A_shared=int(ma.sum()), n_B_shared=int(mb.sum()),
            nn_pairs_within_1p5m=int(ok.sum()),
            median_offset_along_m=float(np.median(dv @ u)) if ok.any() else None,
            median_offset_across_m=float(np.median(dv @ lat)) if ok.any() else None,
            median_abs_offset_along_m=float(np.median(np.abs(dv @ u))) if ok.any() else None,
            shift_curve_best=dict(shift_m=float(shifts[k0]), n_within_0p3=curve[k0]),
            n_within_0p3_at_0=curve[int(np.where(shifts == 0)[0][0])],
            n_within_0p3_at_pm3p2={'-3.2': curve[int(np.argmin(abs(shifts + 3.2)))], '+3.2': curve[int(np.argmin(abs(shifts - 3.2)))]})
        np.savetxt(R / f'merge/trunk_shift_curve_{name}.csv', np.column_stack([shifts, curve, curve_med]), delimiter=',',
                   header='shift_along_m,n_B_within_0p3m_of_A,median_nn_m', comments='', fmt='%.3f')
        for tag, D in (('A', XA), (f'Bpre_{name}', XB)):
            np.savetxt(R / f'merge/junction_trunks_{tag}.csv', D, delimiter=',', header='x_A,y_A', comments='', fmt='%.4f')
        print(name, json.dumps(res['candidates'][name]))
    json.dump(res, open(R / 'merge/junction_trunk_check.json', 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'check':
    junction_check(Path(sys.argv[2]))
