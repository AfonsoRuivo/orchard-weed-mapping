#!/usr/bin/env python3
"""Merge LIO-SAM halves B into A's frame (brief section 3) and check the merge.

  merge.py <ca10_run dir>

Reads  traj/A, traj/B (export_half.py outputs), scans/A_kf_scans.npz, scans/B_kf_scans.npz (keyframe scans),
       slam/{A,B}/map/GlobalMap.pcd
Writes merge/: T_A_from_B.json (initial guess, refined, profile, checks), icp_profile.csv/png,
       headland_check.json; traj/merged/: keyframes, dense (TUM + CSV), gnss json; maps/merged_GlobalMap.pcd

1. Initial guess: both halves put in GNSS ENU by their own SE2 fits (gnss_fit.py), composed:
   T_A<-B = T_ENU<-A^-1 T_ENU<-B (SE2 lifted to SE3, dz = z of A's last keyframe, since B starts at z = 0
   about 7 m further along the same alley).
2. Refinement: GICP map-to-map (numpy GICP of the gap analysis, icp_np.py; multi-scale 2.0/1.0/0.5 m as in
   icp_gap.py, 6-DOF): source = keyframe scans of B in 839.5-885 s, target = keyframe scans of A in 760-805.5 s,
   per scan z > -0.45 m in the sensor frame, 1.5-40 m, 0.15 m voxels. Fitness = inlier fraction at 0.3 m.
3. Basin check: the refined T shifted along the row (direction = PCA of A's keyframe xy in 760-805.5 s) by
   -10..+10 m in 0.1 m steps, not re-optimised (as icp_shift_profile.png); local maxima found, the +-1-tree
   peaks re-optimised with full 6-DOF GICP.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import ca10lib as L      # noqa: E402
import gnss_fit as G     # noqa: E402

R = Path(sys.argv[1])
# optional: merge.py <dir> <candidate> <init|gicp6>: use that candidate's T from merge/candidates.json
# (merge_candidates.py) instead of the GICP refined from the GNSS guess; outputs go to *_<candidate>_<kind>
CAND = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else None
TAG = f'_{CAND[0]}_{CAND[1]}' if CAND else ''
M = R / f'merge{TAG}'; M.mkdir(exist_ok=True); MER = f'traj/merged{TAG}'; (R / MER).mkdir(parents=True, exist_ok=True)
(R / 'maps').mkdir(exist_ok=True)
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
dA = L.read_csv_traj(R / 'traj/A/dense_lidar_rate.csv'); dB = L.read_csv_traj(R / 'traj/B/dense_lidar_rate.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
rel = lambda a: np.column_stack([a[:, 0] - L.T0, a[:, 1:]])
gn = G.load_gnss()
fA = G.fit(rel(kfA), 0.0, L.A_END, g=gn); fB = G.fit(rel(kfB), L.B_START, 1400.0, g=gn)
RA, tA, RB, tB = (np.array(fA['R']), np.array(fA['t']), np.array(fB['R']), np.array(fB['t']))
R2 = RA.T @ RB; t2 = RA.T @ (tB - tA)
T_init = L.se2_to_se3(R2, t2, dz=float(kfA[-1, 3]))
out = dict(candidate=CAND, gnss_fit_A=G.public(fA), gnss_fit_B=G.public(fB), T_init=T_init.tolist(), T_init_xyz_yaw=L.xyyaw(T_init))

# 2. GICP refinement
PA = L.part_map(SA, kfA, 760.0, L.A_END); PB = L.part_map(SB, kfB, L.B_START, 885.0)
tgt, src = L.prep(PA), L.prep(PB)


def gicp(T0):
    T = T0
    for d in (2.0, 1.0, 0.5):
        T = L._icp(src, tgt, T, 'gicp', d)
    f, e = L.evaluate(src, tgt, T, 0.3)
    return T, float(f), float(e)


f_init, e_init = L.evaluate(src, tgt, T_init, 0.3)
T, fit, rmse = gicp(T_init)
if CAND:
    T = np.array(json.load(open(R / 'merge/candidates.json'))['candidates'][CAND[0]][CAND[1]]['T'])
    fit, rmse = (float(v) for v in L.evaluate(src, tgt, T, 0.3))
out.update(n_target=len(PA), n_source=len(PB), fitness_init=float(f_init), rmse_init=float(e_init),
           T_refined=T.tolist(), T_refined_xyz_yaw=L.xyyaw(T), T_refined_rpy_deg=L.rpy_deg(T),
           fitness=fit, rmse=rmse)
# sensitivity (not in the brief): B's 845-960 s GNSS sits ~4 m off by a near-constant vector (RUNLOG), so the
# initial guess is also built from B's fit on 1000 s-end only, and refined the same way
fB2 = G.fit(rel(kfB), 1000.0, 1400.0, g=gn, skip=0.0)
T_init2 = L.se2_to_se3(RA.T @ np.array(fB2['R']), RA.T @ (np.array(fB2['t']) - tA), dz=float(kfA[-1, 3]))
T2, fit2, rmse2 = gicp(T_init2)
out['sensitivity_init_from_B_1000_end'] = dict(
    gnss_fit_B_1000_end=dict(ate_m=fB2['ate_m'], max_m=fB2['max_m']), T_init=T_init2.tolist(),
    T_init_xyz_yaw=L.xyyaw(T_init2), T_refined_xyz_yaw=L.xyyaw(T2), fitness=fit2, rmse=rmse2,
    refined_difference_m=float(np.linalg.norm(T2[:3, 3] - T[:3, 3])),
    refined_difference_yaw_deg=L.xyyaw(T2)[3] - L.xyyaw(T)[3])

# along-row / lateral axes of the shared alley, in A's frame
xy = kfA[(kfA[:, 0] - L.T0 >= 760.0), 1:3]
w, V = np.linalg.eigh(np.cov((xy - xy.mean(0)).T)); u = V[:, 1]
mv = xy[-1] - xy[0]; u = u if u @ mv > 0 else -u            # +along = A's travel direction (towards B)
lat = np.array([-u[1], u[0]])
# refined vs initial: where each puts B's first 20 m of keyframes
pb = kfB[(kfB[:, 0] - L.T0 <= 885.0), 1:4]
d = (pb @ T[:3, :3].T + T[:3, 3]) - (pb @ T_init[:3, :3].T + T_init[:3, 3])
out['refined_minus_init'] = dict(along_m=float((d[:, :2] @ u).mean()), lateral_m=float((d[:, :2] @ lat).mean()),
                                 dz_m=float(d[:, 2].mean()), yaw_deg=L.xyyaw(T)[3] - L.xyyaw(T_init)[3],
                                 axis_along=u.tolist())
# B's origin advance along the row from A's last keyframe (cf. gap ICP 6.54 m from 805.555 to 825.463)
pA_end = kfA[-1, 1:3]; pB0 = (T @ np.r_[kfB[0, 1:4], 1])[:2]
out['B_first_kf_vs_A_last_kf'] = dict(along_m=float((pB0 - pA_end) @ u), lateral_m=float((pB0 - pA_end) @ lat),
                                      A_last_bag_s=float(kfA[-1, 0] - L.T0), B_first_bag_s=float(kfB[0, 0] - L.T0))

# 3. basin profile
shifts = np.round(np.arange(-10.0, 10.0001, 0.1), 2)
prof = []
for s in shifts:
    Ts = T.copy(); Ts[:2, 3] += s * u
    prof.append(L.evaluate(src, tgt, Ts, 0.3))
prof = np.array(prof)
np.savetxt(M / 'icp_profile.csv', np.column_stack([shifts, prof]), delimiter=',', header='shift_along_m,fitness,rmse_m',
           comments='', fmt='%.4f')
f = prof[:, 0]
peaks = [int(i) for i in argrelextrema(np.r_[-1, f, -1], np.greater_equal, order=8)[0] - 1 if 0 <= i - 1 < len(f)]
peaks = sorted(set(peaks))
pk = [dict(shift_m=float(shifts[i]), fitness=float(f[i])) for i in peaks]
reopt = []
for p in pk:
    if 1.5 <= abs(p['shift_m']) <= 4.5 or 4.5 < abs(p['shift_m']) <= 7.5:
        Ts = T.copy(); Ts[:2, 3] += p['shift_m'] * u
        Tr, fr, er = gicp(Ts)
        reopt.append(dict(start_shift_m=p['shift_m'], fitness=fr, rmse=er,
                          settled_along_m=float((Tr[:2, 3] - T[:2, 3]) @ u), settled_xyz_yaw=L.xyyaw(Tr)))
out['profile'] = dict(peaks=pk, reoptimised=reopt, fitness_at_0=float(f[shifts == 0][0]))

# 4. apply T to all of B, merged trajectories
kfBm, dBm = L.transform_traj(kfB, T), L.transform_traj(dB, T)
kfM, dM = np.vstack([kfA, kfBm]), np.vstack([dA, dBm])
hdr = ['CA-10 merged LIO-SAM: A (bag <= 805.5 s) in its own frame, B (>= 839.5 s) moved by T_A<-B (merge/T_A_from_B.json)',
       't = header stamp unix s; 805.5-839.5 s has no poses (gap)']
L.write_tum(R / MER / 'keyframes.tum', kfM, hdr); L.write_csv_traj(R / MER / 'keyframes.csv', kfM)
L.write_tum(R / MER / 'dense_lidar_rate.tum', dM, hdr); L.write_csv_traj(R / MER / 'dense_lidar_rate.csv', dM)
L.write_tum(R / MER / 'B_keyframes_in_A_frame.tum', kfBm, hdr); L.write_tum(R / MER / 'B_dense_in_A_frame.tum', dBm, hdr)

# GNSS of the merged trajectory: one SE2 for the whole run, residuals reported per half
gm = {}
for name, a in (('keyframes', kfM), ('dense', dM)):
    r = G.fit(rel(a), 0.0, 1400.0, g=gn)
    res = dict(all=G.public(r))
    for h, (lo_, hi_) in (('A', (0, L.A_END)), ('B', (L.B_START, 1400))):
        m = (r['_t'] >= lo_) & (r['_t'] <= hi_)
        e = r['_err'][m]
        res[h] = dict(n=int(m.sum()), ate_m=float(np.sqrt((e ** 2).mean())), max_m=float(e.max()),
                      median_m=float(np.median(e)))
    # B drift against GNSS once in A's frame: residuals of B under A's OWN fit (no refit), per minute
    a2 = rel(a); b = a2[a2[:, 0] >= L.B_START]
    rb = G.fit(b, L.B_START, 1400.0, g=gn, R=fA['R'], t=fA['t'])
    mins = np.floor((rb['_t'] - L.B_START) / 60).astype(int)
    res['B_under_A_fit'] = dict(ate_m=rb['ate_m'], max_m=rb['max_m'], per_minute_rmse_m=[
        float(np.sqrt((rb['_err'][mins == k] ** 2).mean())) for k in range(mins.max() + 1)])
    gm[name] = res
    np.savetxt(R / MER / f'gnss_err_{name}.csv', np.column_stack([r['_t'], r['_A'], r['_G'], r['_err']]),
               delimiter=',', header='t_bag_s,east_slam,north_slam,east_gnss,north_gnss,err_m', comments='', fmt='%.4f')
    json.dump(res, open(R / MER / f'gnss_fit_{name}.json', 'w'), indent=1)
out['merged_gnss'] = gm

# baseline: old through-the-gap LIO-SAM run (orchard-slam/data/campinas_a/lio_sam/trajectory.csv), same window
old = rel(L.read_csv_traj('/home/afonso/thesis/orchard-slam/data/campinas_a/lio_sam/trajectory.csv'))
w0, w1 = float(old[0, 0]), float(old[-1, 0])
ro = G.fit(old, w0, w1, g=gn, skip=0.0)
rm = G.fit(rel(kfM), w0, w1, g=gn, skip=0.0)
out['baseline_old_run'] = dict(window_bag_s=[w0, w1], old=G.public(ro), merged_same_window=G.public(rm))

# 5b. north headland: non-ground points of B 1111-1156 s against A 243-303 and 723-783 s
HB = L.part_map(SB, kfB, 1111.0, 1156.0, vox=0.1); HB = HB @ T[:3, :3].T + T[:3, 3]
hl = {}
for tag, (a0, a1) in (('A_243_303', (243.0, 303.0)), ('A_723_783', (723.0, 783.0))):
    HA = L.part_map(SA, kfA, a0, a1, vox=0.1)
    dd, _ = cKDTree(HA).query(HB)
    occ = {tuple(c) for c in np.floor(HA[:, :2]).astype(int)}          # B points over ground A saw (1 m cells)
    m = np.array([tuple(c) in occ for c in np.floor(HB[:, :2]).astype(int)])
    hl[tag] = dict(n_A=len(HA), n_B=len(HB), all=dict(median_m=float(np.median(dd)), p90_m=float(np.percentile(dd, 90))),
                   overlap_1m_cells=dict(n=int(m.sum()), median_m=float(np.median(dd[m])) if m.any() else None,
                                         p90_m=float(np.percentile(dd[m], 90)) if m.any() else None))
HA2 = np.vstack([L.part_map(SA, kfA, 243.0, 303.0, vox=0.1), L.part_map(SA, kfA, 723.0, 783.0, vox=0.1)])
dd, _ = cKDTree(HA2).query(HB)
occ = {tuple(c) for c in np.floor(HA2[:, :2]).astype(int)}
m = np.array([tuple(c) in occ for c in np.floor(HB[:, :2]).astype(int)])
hl['A_both'] = dict(n_A=len(HA2), n_B=len(HB), all=dict(median_m=float(np.median(dd)), p90_m=float(np.percentile(dd, 90))),
                    overlap_1m_cells=dict(n=int(m.sum()), median_m=float(np.median(dd[m])), p90_m=float(np.percentile(dd[m], 90))))
# the same with the GNSS initial guess, to show what the refinement changed
HBi = L.part_map(SB, kfB, 1111.0, 1156.0, vox=0.1) @ T_init[:3, :3].T + T_init[:3, 3]
di, _ = cKDTree(HA2).query(HBi)
mi = np.array([tuple(c) in occ for c in np.floor(HBi[:, :2]).astype(int)])
hl['A_both_with_T_init'] = dict(all=dict(median_m=float(np.median(di)), p90_m=float(np.percentile(di, 90))),
                                overlap_1m_cells=dict(n=int(mi.sum()), median_m=float(np.median(di[mi])),
                                                      p90_m=float(np.percentile(di[mi], 90))))
hl['method'] = ('keyframe scans, z > -0.45 m sensor frame, 1.5-40 m, 0.1 m voxels; NN distance from each B point '
                'to the A points; "overlap" = B points in 1 m xy cells that contain A points')
json.dump(hl, open(M / 'headland_check.json', 'w'), indent=1)
out['headland'] = hl

# 6. merged map PCD: A's GlobalMap + B's GlobalMap moved by T
ga, fa = L.read_pcd(R / 'slam/A/map/GlobalMap.pcd'); gb, _ = L.read_pcd(R / 'slam/B/map/GlobalMap.pcd')
gb2 = gb.copy(); gb2[:, :3] = gb[:, :3] @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
L.write_pcd(R / f'maps/B_GlobalMap_in_A_frame{TAG}.pcd', gb2)
L.write_pcd(R / f'maps/merged_GlobalMap{TAG}.pcd', np.vstack([ga, gb2]))
out['maps'] = dict(A_points=len(ga), B_points=len(gb), merged_points=len(ga) + len(gb))
json.dump(out, open(M / 'T_A_from_B.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # noqa: E402
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(shifts, f, color='#334155', lw=1.2)
for p in pk:
    ax.axvline(p['shift_m'], color='#D4A62A', lw=0.6, ls=':')
ax.axvline(0, color='#CC79A7', lw=1)
ax.set_xlabel('shift of refined T_A<-B along the row [m]'); ax.set_ylabel('fitness (inliers at 0.3 m)')
ax.set_title(f'CA-10 merge: along-row profile (refined fitness {fit:.3f})', fontsize=9)
fig.tight_layout(); fig.savefig(M / 'icp_shift_profile_merge.png', dpi=130)
print(json.dumps({k: v for k, v in out.items() if k not in ('gnss_fit_A', 'gnss_fit_B', 'headland', 'merged_gnss')}, indent=1))
