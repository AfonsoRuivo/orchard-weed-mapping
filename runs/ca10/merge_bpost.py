#!/usr/bin/env python3
"""T_A<-B for B-post (975 s - end, after B's U-turn fault at 966-975 s), and the final three-piece merge.

  merge_bpost.py <ca10_run dir>

B-pre (839.5-966 s): T1 = candidates.json 'chain_gicp6_plus3p15' (LiDAR chain refined by GICP, moved one tree
  along the row by the junction trunk check; RUNLOG section 3).
B-post (975 s - end): GNSS-initialised GICP against A's map (user decision 2026-10-27):
  initial guess: B-post's keyframes placed by T1, then moved by the SE2 that takes them onto their GNSS-composed
  positions (A's SE2 fit to GNSS, inverted, after B-post's own SE2 fit on its fixes 975 s - end), so xy/yaw come
  from GNSS and z/roll/pitch from T1 (same B frame);
  GICP: same recipe as the junction (keyframe scans, z > -0.45 m sensor frame, 1.5-40 m, 0.15 m voxels, numpy GICP
  multi-scale 2/1/0.5 m, 6-DOF); source = B-post scans EXCLUDING 1100-1170 s, target = A's scans EXCLUDING 243-303
  and 723-783 s and cropped to 15 m around the source, so the north-headland check is independent of this GICP;
  basin: fitness grid +-10 m along x +-7 m across (here the parts overlap in space, so fitness is meaningful),
  re-optimised from the best +-1 tree / +-1 alley cells;
  north headland: B 1111-1156 s vs A 243-303 + 723-783 s, NN distances (as merge.py).
Final merge: A (<= 805.5 s) + T1 B-pre (839.5-966 s) + T2 B-post (975 s - end); 805.5-839.5 and 966-975 s
  without poses. Writes merge_final/, traj/merged_final/, maps/merged_final_GlobalMap.pcd (A's GlobalMap + the
  B pieces rebuilt from their keyframe scans: B's own GlobalMap cannot be split by time).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import ca10lib as L      # noqa: E402
import gnss_fit as G     # noqa: E402

R = Path(sys.argv[1]); M = R / 'merge_final'; M.mkdir(exist_ok=True); MER = R / 'traj/merged_final'; MER.mkdir(exist_ok=True)
o = json.load(open(R / 'merge/T_A_from_B.json')); c = json.load(open(R / 'merge/candidates.json'))
u = np.array(o['refined_minus_init']['axis_along']); lat = np.array([-u[1], u[0]])
T1 = np.array(c['candidates']['chain_gicp6_plus3p15']['gicp6']['T'])
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
dA = L.read_csv_traj(R / 'traj/A/dense_lidar_rate.csv'); dB = L.read_csv_traj(R / 'traj/B/dense_lidar_rate.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
rel = lambda a: np.column_stack([a[:, 0] - L.T0, a[:, 1:]])
gn = G.load_gnss()
PRE, POST = (L.B_START, 966.0), (975.0, 1400.0)
cut = lambda a, w: a[(a[:, 0] - L.T0 >= w[0]) & (a[:, 0] - L.T0 <= w[1])]
kpre, kpost = cut(kfB, PRE), cut(kfB, POST)

# initial guess for B-post
fA = G.fit(rel(kfA), 0.0, L.A_END, g=gn); RA, tA = np.array(fA['R']), np.array(fA['t'])
fP = G.fit(rel(kpost), POST[0], POST[1], g=gn, skip=0.0)
q_gnss = (kpost[:, 1:3] @ np.array(fP['R']).T + fP['t'] - tA) @ RA          # ENU -> A frame
q_T1 = kpost[:, 1:4] @ T1[:3, :3].T + T1[:3, 3]
R2, t2 = G.se2(q_T1[:, :2], q_gnss)
D = L.se2_to_se3(R2, t2); T_init = D @ T1

src_raw = np.vstack([L.part_map(SB, kpost, a, b) for a, b in ((POST[0], 1100.0), (1170.0, POST[1]))])
tgt_raw = np.vstack([L.part_map(SA, kfA, a, b) for a, b in ((0.0, 243.0), (303.0, 723.0), (783.0, L.A_END))])
tgt_raw = L.voxel(tgt_raw, 0.15)
srcA = src_raw @ T_init[:3, :3].T + T_init[:3, 3]
near = cKDTree(srcA[:, :2]).query(tgt_raw[:, :2], distance_upper_bound=15.0)[0] < np.inf
tgt, src = L.prep(tgt_raw[near]), L.prep(src_raw)


def gicp(T0):
    T = T0
    for d in (2.0, 1.0, 0.5):
        T = L._icp(src, tgt, T, 'gicp', d)
    f, e = L.evaluate(src, tgt, T, 0.3)
    return T, float(f), float(e)


import time; _t0 = time.time()
log = lambda *a: print(f'[{time.time() - _t0:6.0f} s]', *a, flush=True)
log('source', len(src_raw), 'target', int(near.sum()))
f0, e0 = L.evaluate(src, tgt, T_init, 0.3)
T2, fit, rmse = gicp(T_init)
log('GICP done, fitness', round(fit, 3))
out = dict(T1_Bpre=T1.tolist(), T1_Bpre_rpy_deg=L.rpy_deg(T1),
           gnss_fit_Bpost_own=G.public(fP), T2_init=T_init.tolist(), T2_init_rpy_deg=L.rpy_deg(T_init),
           n_source=len(src_raw), n_target=int(near.sum()), fitness_init=float(f0), rmse_init=float(e0),
           T2=T2.tolist(), T2_rpy_deg=L.rpy_deg(T2), fitness=fit, rmse=rmse)
d = (kpost[:, 1:4] @ T2[:3, :3].T + T2[:3, 3]) - (kpost[:, 1:4] @ T_init[:3, :3].T + T_init[:3, 3])
out['T2_minus_init'] = dict(along_m=float((d[:, :2] @ u).mean()), across_m=float((d[:, :2] @ lat).mean()),
                            dz_m=float(d[:, 2].mean()), yaw_deg=L.xyyaw(T2)[3] - L.xyyaw(T_init)[3])
d1 = (kpost[:, 1:4] @ T2[:3, :3].T + T2[:3, 3]) - q_T1
out['T2_minus_T1_on_Bpost'] = dict(along_m=float((d1[:, :2] @ u).mean()), across_m=float((d1[:, :2] @ lat).mean()),
                                   dz_m=float(d1[:, 2].mean()), yaw_deg=L.xyyaw(T2)[3] - L.xyyaw(T1)[3],
                                   note='how far B-post sits from where B-pre\'s transform would put it (the U-turn fault)')
# basin grid (fitness on every 4th source point; 0.5 m steps)
al, la = np.round(np.arange(-10, 10.0001, 0.5), 2), np.round(np.arange(-7, 7.0001, 0.5), 2)
srcg = dict(P=src['P'][::4])
rows = []
for a in al:
    for b in la:
        Ts = T2.copy(); Ts[:2, 3] += a * u + b * lat
        rows.append((a, b, *L.evaluate(srcg, tgt, Ts, 0.3)))
log('grid done')
grid = np.array(rows)
np.savetxt(M / 'Bpost_basin_grid.csv', grid, delimiter=',', header='shift_along_m,shift_across_m,fitness,rmse_m', comments='', fmt='%.4f')
f00 = float(grid[(grid[:, 0] == 0) & (grid[:, 1] == 0), 2][0])
reopt = []
for name, sel in (('-1 tree', (grid[:, 0] <= -2) & (grid[:, 0] >= -4.5) & (grid[:, 1] == 0)),
                  ('+1 tree', (grid[:, 0] >= 2) & (grid[:, 0] <= 4.5) & (grid[:, 1] == 0)),
                  ('-1 alley', (grid[:, 1] <= -3.5) & (grid[:, 1] >= -6.5)), ('+1 alley', (grid[:, 1] >= 3.5) & (grid[:, 1] <= 6.5))):
    g = grid[sel]; k = int(np.argmax(g[:, 2]))
    Ts = T2.copy(); Ts[:2, 3] += g[k, 0] * u + g[k, 1] * lat
    Tr, fr, er = gicp(Ts)
    dd = Tr[:2, 3] - T2[:2, 3]
    reopt.append(dict(region=name, start_along=float(g[k, 0]), start_across=float(g[k, 1]), start_fitness=float(g[k, 2]),
                      fitness=fr, rmse=er, settled_along_m=float(dd @ u), settled_across_m=float(dd @ lat)))
log('re-optimisations done')
out['basin'] = dict(fitness_at_T2=f00, best_cell=dict(zip(['along', 'across', 'fitness', 'rmse'], grid[np.argmax(grid[:, 2])].tolist())),
                    reoptimised=reopt)

# north headland (independent of the GICP above: those windows were excluded from both parts)
HA = np.vstack([L.part_map(SA, kfA, 243.0, 303.0, vox=0.1), L.part_map(SA, kfA, 723.0, 783.0, vox=0.1)])
HB0 = L.part_map(SB, kpost, 1111.0, 1156.0, vox=0.1)
tree = cKDTree(HA); occ = {tuple(v) for v in np.floor(HA[:, :2]).astype(int)}
hl = {}
for k, T in (('T2', T2), ('T2_init_gnss', T_init), ('T1_on_Bpost', T1)):
    HB = HB0 @ T[:3, :3].T + T[:3, 3]
    dd, _ = tree.query(HB); m = np.array([tuple(v) in occ for v in np.floor(HB[:, :2]).astype(int)])
    hl[k] = dict(all=dict(median_m=float(np.median(dd)), p90_m=float(np.percentile(dd, 90))),
                 overlap=dict(n=int(m.sum()), median_m=float(np.median(dd[m])), p90_m=float(np.percentile(dd[m], 90))))
out['headland'] = hl

log('headland done')
# final merge
Bpre_k, Bpost_k = L.transform_traj(kpre, T1), L.transform_traj(kpost, T2)
Bpre_d, Bpost_d = L.transform_traj(cut(dB, PRE), T1), L.transform_traj(cut(dB, POST), T2)
kfM, dM = np.vstack([kfA, Bpre_k, Bpost_k]), np.vstack([dA, Bpre_d, Bpost_d])
hdr = ['CA-10 merged LIO-SAM, three pieces in A\'s frame: A (bag <= 805.5 s), B-pre (839.5-966 s, T1), B-post (975 s-end, T2)',
       'T1, T2 in merge_final/T_final.json; pieces cut at 805.5 s (sensor gap) and 966/975 s (B U-turn fault); actual empty spans: keyframes 802.23-845.40 and 964.52-975.51 s, dense 804.14-845.40 and 965.93-975.01 s',
       't = header stamp, unix s; pose = velodyne in A\'s map frame']
L.write_tum(MER / 'keyframes.tum', kfM, hdr); L.write_csv_traj(MER / 'keyframes.csv', kfM)
L.write_tum(MER / 'dense_lidar_rate.tum', dM, hdr); L.write_csv_traj(MER / 'dense_lidar_rate.csv', dM)
pieces = dict(A=(0.0, L.A_END), Bpre=PRE, Bpost=POST)
gm = {}
for name, a in (('keyframes', kfM), ('dense', dM)):
    r = G.fit(rel(a), 0.0, 1400.0, g=gn)
    res = dict(all=G.public(r))
    for p, (lo_, hi_) in pieces.items():
        m = (r['_t'] >= lo_) & (r['_t'] <= hi_); e = r['_err'][m]
        res[p] = dict(n=int(m.sum()), ate_m=float(np.sqrt((e ** 2).mean())), max_m=float(e.max()), median_m=float(np.median(e)))
    rA = G.fit(rel(a), 0.0, 1400.0, g=gn, R=fA['R'], t=fA['t'])      # B pieces under A's own fit (no refit)
    for p, (lo_, hi_) in pieces.items():
        m = (rA['_t'] >= lo_) & (rA['_t'] <= hi_); e = rA['_err'][m]
        res[p + '_under_A_fit'] = dict(ate_m=float(np.sqrt((e ** 2).mean())), max_m=float(e.max()))
    mins = np.floor((rA['_t'] - L.B_START) / 60).astype(int); mb = rA['_t'] >= L.B_START
    res['B_under_A_fit_per_minute_rmse_m'] = {int(k): float(np.sqrt((rA['_err'][mb & (mins == k)] ** 2).mean()))
                                              for k in np.unique(mins[mb])}
    gm[name] = res
    np.savetxt(MER / f'gnss_err_{name}.csv', np.column_stack([r['_t'], r['_A'], r['_G'], r['_err']]), delimiter=',',
               header='t_bag_s,east_slam,north_slam,east_gnss,north_gnss,err_m', comments='', fmt='%.4f')
    json.dump(res, open(MER / f'gnss_fit_{name}.json', 'w'), indent=1)
out['merged_gnss'] = gm
old = rel(L.read_csv_traj('/home/afonso/thesis/orchard-slam/data/campinas_a/lio_sam/trajectory.csv'))
w0, w1 = float(old[0, 0]), float(old[-1, 0])
out['baseline_old_run'] = dict(window_bag_s=[w0, w1], old=G.public(G.fit(old, w0, w1, g=gn, skip=0.0)),
                               merged_same_window=G.public(G.fit(rel(kfM), w0, w1, g=gn, skip=0.0)))
# merged map: A's GlobalMap + B pieces from their keyframe scans (LIO-SAM save_map resolution 0.2 m)
ga, _ = L.read_pcd(R / 'slam/A/map/GlobalMap.pcd')
bp = []
for k, T, w in ((kpre, T1, PRE), (kpost, T2, POST)):
    for r_ in k:
        j = SB.index_of(r_[0]); s = SB.scan(j); s = s[np.linalg.norm(s[:, :3], axis=1) <= 40]
        Mx = T @ L.to_T(r_[None])[0]
        bp.append(np.column_stack([s[:, :3] @ Mx[:3, :3].T + Mx[:3, 3], s[:, 3]]))
bp = L.voxel_first(np.vstack(bp).astype(np.float32), 0.2)
L.write_pcd(R / 'maps/B_pieces_in_A_frame_final.pcd', bp)
L.write_pcd(R / 'maps/merged_final_GlobalMap.pcd', np.vstack([ga, bp]))
out['maps'] = dict(A_GlobalMap_points=len(ga), B_pieces_points=len(bp), note='B pieces: keyframe scans <= 40 m, 0.2 m voxels')
json.dump(out, open(M / 'T_final.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('T1_Bpre', 'T2_init', 'T2', 'gnss_fit_Bpost_own', 'merged_gnss', 'baseline_old_run')}, indent=1))
print('merged keyframes GNSS:', json.dumps({k: v for k, v in gm['keyframes'].items() if k != 'all'}))
print('baseline:', out['baseline_old_run']['old']['ate_m'], out['baseline_old_run']['merged_same_window']['ate_m'])
