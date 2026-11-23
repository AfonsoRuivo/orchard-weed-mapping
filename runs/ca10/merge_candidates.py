#!/usr/bin/env python3
"""Merge candidates for T_A<-B (the GNSS-initialised GICP of merge.py landed one alley over; see RUNLOG).

  merge_candidates.py <ca10_run dir>

Candidates, each refined with the same 6-DOF GICP on the same map parts as merge.py (A 760-805.5 s, B 839.5-885 s,
z > -0.45 m sensor frame, 1.5-40 m, 0.15 m voxels, multi-scale 2/1/0.5 m), plus a planar GICP as a diagnostic:
  gnss            merge.py's GNSS initial guess (brief section 3.1)
  lidar_chain     GNSS-free: P_A(805.555) . T_gap . O(825.463 -> t_b) . P_B(t_b)^-1, t_b = B's first keyframe.
                  P_A(805.555) = A's last odometry pose (804.14 s) extended with the gap analysis's pre-gap scan
                  chain (lidar_odom_before_gap.csv); T_gap = the gap GICP (REPORT section 3, scan 805.555 <- 825.463);
                  O = post-gap scan chain (lidar_odom_after_gap.csv, 2D x, y, yaw, origin = scan 825.463).
  refined_m1/p1   merge.py's refined T moved -1 / +1 alley (-5 / +5 m across the row).
Reports per candidate: fitness/RMSE at 0.3 m, roll/pitch, and B's first keyframe relative to A's last keyframe
(along / lateral of A's alley axis). Also a fitness grid of the best candidate shifted along (-10..10 m, 0.2 m) and
across (-7..7 m, 0.5 m) the row, not re-optimised. Writes merge/candidates.json, merge/basin_grid.csv/.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ca10lib as L      # noqa: E402

R = Path(sys.argv[1]); M = R / 'merge'
o = json.load(open(M / 'T_A_from_B.json'))
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
odA = L.read_csv_traj(R / 'traj/A/dense_lidar_rate.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
PA = L.part_map(SA, kfA, 760.0, L.A_END); PB = L.part_map(SB, kfB, L.B_START, 885.0)
tgt, src = L.prep(PA), L.prep(PB)
u = np.array(o['refined_minus_init']['axis_along']); lat = np.array([-u[1], u[0]])


def gicp(T0, planar=False):
    T = T0
    for d in (2.0, 1.0, 0.5):
        T = L._icp(src, tgt, T, 'gicp', d, planar=planar)
    f, e = L.evaluate(src, tgt, T, 0.3)
    return T, float(f), float(e)


def describe(T):
    p = (T @ np.r_[kfB[0, 1:4], 1])[:2] - kfA[-1, 1:3]
    rp = L.rpy_deg(T)
    return dict(along_m=float(p @ u), lateral_m=float(p @ lat), roll_deg=rp[0], pitch_deg=rp[1], yaw_deg=rp[2],
                xyz=T[:3, 3].tolist())


def chain_pose(df, t):
    """2D cumulative pose of a gap scan chain at time t (x, y interpolated, yaw = cumulative dyaw)."""
    yaw = np.radians(np.cumsum(df.dyaw.values))
    X, Y, tt = df.X.values, df.Y.values, df.t.values
    x, y, th = np.interp(t, tt, X), np.interp(t, tt, Y), np.interp(t, tt, yaw)
    return L.se2_to_se3(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), [x, y])


G = '/home/afonso/ca10_gap/'
cA, cB = pd.read_csv(G + 'lidar_odom_before_gap.csv'), pd.read_csv(G + 'lidar_odom_after_gap.csv')
# pre-gap chain: origin = scan 151 (790.17 s); add the origin row so interpolation covers the start
cA = pd.concat([pd.DataFrame([dict(k=0, t=790.0, dx=0, dy=0, dyaw=0, speed=0, X=0, Y=0, fit=1)]), cA])
cB = pd.concat([pd.DataFrame([dict(k=303, t=825.463, dx=0, dy=0, dyaw=0, speed=0, X=0, Y=0, fit=1)]), cB])
tA_end = odA[-1, 0] - L.T0                                    # 804.14
C_end = chain_pose(cA, tA_end); C_gap = chain_pose(cA, 805.555)
P_A_805 = L.to_T(odA[-1:])[0] @ np.linalg.inv(C_end) @ C_gap  # A pose at the last pre-gap scan
T_gap = np.eye(4); T_gap[:3, :] = np.array([[0.99957, 0.02940, 0.00145, -6.51149],
                                            [-0.02944, 0.99835, 0.04936, 0.61684],
                                            [0.0, -0.04938, 0.99878, 0.24206]])
tb = kfB[0, 0] - L.T0
O = chain_pose(cB, tb)
T_chain = P_A_805 @ T_gap @ O @ np.linalg.inv(L.to_T(kfB[:1])[0])
T_ref = np.array(o['T_refined'])
cands = dict(gnss=np.array(o['T_init']), lidar_chain=T_chain)
for k, s in (('refined_m1', -5.0), ('refined_p1', 5.0)):
    Tm = T_ref.copy(); Tm[:2, 3] += s * lat; cands[k] = Tm
res = dict(note='fitness compared between candidates; map parts do not overlap along the path (~7 m apart), '
                'so absolute fitness is lower than scan-to-scan', chain_inputs=dict(
    A_last_odom_bag_s=float(tA_end), chain_end_A_bag_s=805.555, t_b_bag_s=float(tb),
    chain_after_last_bag_s=float(cB.t.iloc[-1])), candidates={})
for k, T0 in cands.items():
    f0, e0 = L.evaluate(src, tgt, T0, 0.3)
    T6, f6, e6 = gicp(T0); Tp, fp, ep = gicp(T0, planar=True)
    res['candidates'][k] = dict(init=dict(fitness=float(f0), rmse=float(e0), T=T0.tolist(), **describe(T0)),
                                gicp6=dict(fitness=f6, rmse=e6, T=T6.tolist(), **describe(T6)),
                                planar=dict(fitness=fp, rmse=ep, **describe(Tp)))
    print(f"{k:12s} init f {f0:.3f} -> 6DOF f {f6:.3f} rmse {e6:.3f} {describe(T6)} | planar f {fp:.3f}")
best = max(res['candidates'], key=lambda k: res['candidates'][k]['gicp6']['fitness'])
Tb = np.array(res['candidates'][best]['gicp6']['T'])
res['best'] = best
# 2D basin grid around the best solution
al, la = np.round(np.arange(-10, 10.0001, 0.2), 2), np.round(np.arange(-7, 7.0001, 0.5), 2)
rows = []
for a in al:
    for b in la:
        Ts = Tb.copy(); Ts[:2, 3] += a * u + b * lat
        f, e = L.evaluate(src, tgt, Ts, 0.3); rows.append((a, b, f, e))
g = pd.DataFrame(rows, columns=['shift_along_m', 'shift_lateral_m', 'fitness', 'rmse_m'])
g.to_csv(M / 'basin_grid.csv', index=False)
top = g.sort_values('fitness', ascending=False)
# best fitness per lateral offset, and per along offset at lateral 0
res['grid_best_per_lateral'] = {float(b): dict(fitness=float(d.fitness.max()), along_at_max=float(d.shift_along_m[d.fitness.idxmax()]))
                                for b, d in g.groupby('shift_lateral_m')}
res['grid_top5'] = top.head(5).to_dict('records')
json.dump(res, open(M / 'candidates.json', 'w'), indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # noqa: E402
Z = g.pivot(index='shift_lateral_m', columns='shift_along_m', values='fitness')
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(Z.values, origin='lower', aspect='auto', extent=[al[0], al[-1], la[0], la[-1]], cmap='cividis')
fig.colorbar(im, label='fitness (0.3 m)'); ax.set_xlabel('shift along row [m]'); ax.set_ylabel('shift across row [m]')
ax.set_title(f'CA-10 merge basin around candidate "{best}"', fontsize=9); fig.tight_layout()
fig.savefig(M / 'basin_grid.png', dpi=130)
print('best', best, json.dumps(res['grid_best_per_lateral']))
