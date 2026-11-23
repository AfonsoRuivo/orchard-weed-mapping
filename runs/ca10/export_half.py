#!/usr/bin/env python3
"""Exports of one LIO-SAM half (after liosam_export.py has written keyframes_final.csv).

  export_half.py <run_dir> <t_from_bag_s> <t_to_bag_s> <out_dir>

Writes to out_dir (all times bag-unix seconds, poses = velodyne frame in the half's own map frame):
  keyframes.tum / keyframes.csv          final (loop-corrected) keyframe trajectory, cut to [t_from, t_to]
  odometry_native.tum                    /lio_sam/mapping/odometry as published (live), cut
  dense_lidar_rate.tum / .csv            pose at every /velodyne_points header stamp of the half inside the
                                         odometry span: live odometry moved onto the final keyframes
                                         (ca10lib.corrected_odometry), then interpolated (linear / slerp)
  gnss_fit_keyframes.json, gnss_fit_dense.json   SE2 fit to GNSS (gnss_fit.py; CB-11 method)
  export_summary.json                    counts, what the cut removed, loop corrections, last stamps
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ca10lib as L      # noqa: E402
import gnss_fit as G     # noqa: E402

run, t_from, t_to, out = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
lo, hi = L.T0 + t_from, L.T0 + t_to

kf_all = L.read_csv_traj(run / 'keyframes_final.csv')
od_all = L.read_csv_traj(run / 'odometry.csv')
live_all = L.read_csv_traj(run / 'keyframes_live.csv')
cut = lambda a: a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
kf, od, live = cut(kf_all), cut(od_all), cut(live_all)
summ = dict(run=str(run), window_bag_s=[t_from, t_to],
            keyframes=dict(n_all=len(kf_all), n_kept=len(kf), first_bag_s=float(kf_all[0, 0] - L.T0),
                           last_bag_s_all=float(kf_all[-1, 0] - L.T0), removed_bag_s=[float(t - L.T0) for t in
                                                                                      kf_all[kf_all[:, 0] > hi, 0]]),
            odometry=dict(n_all=len(od_all), n_kept=len(od), last_bag_s_all=float(od_all[-1, 0] - L.T0),
                          removed=int((od_all[:, 0] > hi).sum()),
                          median_dt_s=float(np.median(np.diff(od[:, 0])))))

dense, C = L.corrected_odometry(od, live, kf)
corr = np.linalg.norm(C[:, :3, 3], axis=1)
summ['loop_correction'] = dict(max_translation_m=float(corr.max()), median_m=float(np.median(corr)))
S = np.load('/home/afonso/ca10_gap/stamps.npz')
lid = np.sort(S['velodyne_points__hdr']) / 1e9
lid = lid[(lid >= max(lo, dense[0, 0])) & (lid <= min(hi, dense[-1, 0]))]
dl = L.interp_traj(dense, lid)
summ['dense_lidar_rate'] = dict(n=len(dl), first_bag_s=float(dl[0, 0] - L.T0), last_bag_s=float(dl[-1, 0] - L.T0))

hdr = [f'CA-10 {run.name}: t = header stamp (unix s, bag start {L.T0}), velodyne pose in the half map frame',
       f'cut to bag [{t_from}, {t_to}] s']
L.write_tum(out / 'keyframes.tum', kf, hdr + ['final LIO-SAM keyframes (transformations.pcd)'])
L.write_csv_traj(out / 'keyframes.csv', kf)
L.write_tum(out / 'odometry_native.tum', od, hdr + ['/lio_sam/mapping/odometry as published (live)'])
L.write_tum(out / 'dense_lidar_rate.tum', dl, hdr + ['pose at every LiDAR header stamp: live odometry moved onto '
                                                     'the final keyframes, interpolated'])
L.write_csv_traj(out / 'dense_lidar_rate.csv', dl)

for name, a in (('keyframes', kf), ('dense', dl)):
    a2 = a.copy(); a2[:, 0] -= L.T0
    r = G.fit(a2, t_from, t_to)
    json.dump(dict(traj=name, window=[t_from, t_to], skip_s=10.0, exclude=G.EXCLUDE, **G.public(r)),
              open(out / f'gnss_fit_{name}.json', 'w'), indent=1)
    summ[f'gnss_{name}'] = dict(ate_m=r['ate_m'], max_m=r['max_m'], rpe50_pct=r['rpe50_pct'], n=r['n'],
                                n_fixes=r['n_fixes'], yaw_deg=r['yaw_deg'])
    np.savetxt(out / f'gnss_err_{name}.csv', np.column_stack([r['_t'], r['_A'], r['_G'], r['_err']]),
               delimiter=',', header='t_bag_s,east_slam,north_slam,east_gnss,north_gnss,err_m', comments='', fmt='%.4f')
json.dump(summ, open(out / 'export_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
