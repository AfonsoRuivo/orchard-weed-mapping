#!/usr/bin/env python3
"""North-headland check for several T_A<-B candidates (brief 3.5): non-ground points of B 1111-1156 s against A
243-303 s + 723-783 s (keyframe scans, z > -0.45 m sensor frame, 1.5-40 m, 0.1 m voxels); NN distance from each B
point to A, all points and only B points in 1 m xy cells A also covers. Also the planar offset of B's headland
points that minimises the median NN distance (grid +-6 m, 0.25 m), i.e. how far each candidate is from agreeing
there. Usage: headland_score.py <ca10_run dir>; writes merge/headland_candidates.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent)); import ca10lib as L
R = Path(sys.argv[1]); M = R / 'merge'
o = json.load(open(M / 'T_A_from_B.json')); c = json.load(open(M / 'candidates.json'))
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
HA = np.vstack([L.part_map(SA, kfA, 243.0, 303.0, vox=0.1), L.part_map(SA, kfA, 723.0, 783.0, vox=0.1)])
HB0 = L.part_map(SB, kfB, 1111.0, 1156.0, vox=0.1)
tree = cKDTree(HA); occ = {tuple(v) for v in np.floor(HA[:, :2]).astype(int)}
u = np.array(o['refined_minus_init']['axis_along']); lat = np.array([-u[1], u[0]])
Ts = {'gnss_init': o['T_init'], 'gnss_refined_gicp': o['T_refined'],
      'lidar_chain_init': c['candidates']['lidar_chain']['init']['T'], 'lidar_chain_refined_gicp': c['candidates']['lidar_chain']['gicp6']['T']}
# rebuild the chain init from candidates.json xyz/yaw is lossy; recompute it the same way as merge_candidates.py

out = {}
for k, T in Ts.items():
    if T is None:
        continue
    T = np.array(T); HB = HB0 @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(HB); m = np.array([tuple(v) in occ for v in np.floor(HB[:, :2]).astype(int)])
    best = (1e9, 0, 0)
    sub = HB[m] if m.sum() > 1000 else HB
    sub = sub[::4]
    for a in np.arange(-6, 6.01, 0.25):
        for b in np.arange(-6, 6.01, 0.25):
            dd, _ = tree.query(sub + np.r_[a * u + b * lat, 0])
            med = np.median(dd)
            if med < best[0]: best = (med, a, b)
    out[k] = dict(all=dict(median_m=float(np.median(d)), p90_m=float(np.percentile(d, 90))),
                  overlap=dict(n=int(m.sum()), median_m=float(np.median(d[m])), p90_m=float(np.percentile(d[m], 90))),
                  best_planar_offset=dict(along_m=float(best[1]), lateral_m=float(best[2]), median_m=float(best[0])))
    print(k, json.dumps(out[k]))
json.dump(out, open(M / 'headland_candidates.json', 'w'), indent=1)
