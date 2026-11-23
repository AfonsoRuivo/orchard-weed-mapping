#!/usr/bin/env python3
"""
build_dense_traj.py - dense map from the SAME raw scans (default: the 612 keyframe scans of the July
LIO-SAM run, keyframe_scans.npz) placed with ANY estimator's trajectory.

  python3 build_dense_traj.py --traj liosam_rep1_keyframes.csv --label rep1
  python3 build_dense_traj.py --traj kiss_tuned_rep1.csv --label kiss [--dt 0.0]

Trajectory: CSV t,x,y,z,qx,qy,qz,qw of the velodyne frame (true for LIO-SAM, KISS-ICP and DLIO here);
rows with t < 1e9 (DLIO's initial zeros) are dropped. Each point is placed with the pose interpolated
at (header stamp + per-point time + dt) - linear translation, slerp rotation - so a sparse keyframe
trajectory and a 10-100 Hz odometry are handled the same way, and the sweep is deskewed.

Pre-alignment: each estimator has its own map frame (KISS-ICP starts at the tilted LiDAR attitude, so
its z is not vertical). The whole map is moved by ONE rigid SE(3) to the July run's frame (the frame of
the hand-picked trunks): rotation = chordal mean of R_july(t) R_est(t)^T, translation = mean residual,
both over the first --align-s seconds (before any drift). This is a rigid motion of the whole map, so the
map's internal geometry (what the tapes measure) is untouched; the scorer refines it in 2D (--align).
Output: dense_map/dense_<label>.npz (xyz float32) + dense_<label>_height.npy (local-ground height).
"""
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from common import DENSE, local_ground

TRAJ_DIR = Path.home() / 'Downloads/for_thesis_20261007/trajectories'
JULY = TRAJ_DIR / 'gmap_run_20260706_keyframes.csv'


def load_traj(path):
    a = np.array([[float(r[k]) for k in ('t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] for r in csv.DictReader(open(path))])
    a = a[np.isfinite(a).all(1) & (a[:, 0] > 1e9)]
    a = a[np.argsort(a[:, 0], kind='stable')]
    return a[np.r_[True, np.diff(a[:, 0]) > 0]]


class Traj:
    def __init__(self, a):
        self.t, self.p = a[:, 0], a[:, 1:4]; self.slerp = Slerp(self.t, Rotation.from_quat(a[:, 4:8]))

    def __call__(self, t):
        t = np.clip(t, self.t[0], self.t[-1])
        j = np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2)
        w = ((t - self.t[j]) / (self.t[j + 1] - self.t[j]))[:, None]
        return self.slerp(t), self.p[j] * (1 - w) + self.p[j + 1] * w


def prealign(est, ref, t_lo, t_hi):
    t = ref.t[(ref.t >= max(t_lo, est.t[0])) & (ref.t <= min(t_hi, est.t[-1]))]
    Rr, pr = ref(t); Re, pe = est(t)
    M = (Rr * Re.inv()).as_matrix().sum(0); U, _, Vt = np.linalg.svd(M)
    Ra = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    ta = (pr - pe @ Ra.T).mean(0)
    res = np.linalg.norm(pr - (pe @ Ra.T + ta), axis=1)
    return Ra, ta, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', required=True); ap.add_argument('--label', required=True)
    ap.add_argument('--scans', default=str(DENSE / 'keyframe_scans.npz'))
    ap.add_argument('--dt', type=float, default=0.0, help='added to every point time (pose-time convention test)')
    ap.add_argument('--align-s', type=float, default=120.0); ap.add_argument('--min-range', type=float, default=1.0)
    a = ap.parse_args()
    path = Path(a.traj); path = path if path.exists() else TRAJ_DIR / a.traj
    est = Traj(load_traj(path)); ref = Traj(load_traj(JULY))
    Ra, ta, res = prealign(est, ref, ref.t[0], ref.t[0] + a.align_s)
    print(f'{path.name}: {len(est.t)} poses, pre-align rot {np.degrees(Rotation.from_matrix(Ra).magnitude()):.2f} deg, '
          f'residual over first {a.align_s:.0f} s: RMS {np.sqrt(np.mean(res**2))*100:.1f} cm')
    d = np.load(a.scans); st, lens, pts = d['stamps'], d['lens'], d['pts']; off = np.r_[0, np.cumsum(lens)]
    out, n_out = [], 0
    for k in range(len(st)):
        s = pts[off[k]:off[k + 1]]; s = s[np.linalg.norm(s[:, :3], axis=1) >= a.min_range]
        t = st[k] + s[:, 5].astype(float) + a.dt
        if t.max() < est.t[0] or t.min() > est.t[-1]: n_out += 1; continue
        R, p = est(t)
        out.append((R.apply(s[:, :3].astype(float)) + p) @ Ra.T + ta)
    xyz = np.vstack(out).astype(np.float32)
    tag = a.label + (f'_dt{a.dt:+.3f}' if a.dt else '')
    np.savez(DENSE / f'dense_{tag}.npz', xyz=xyz)
    np.save(DENSE / f'dense_{tag}_height.npy', local_ground(xyz.astype(float)).astype(np.float32))
    print(f'{len(st) - n_out}/{len(st)} scans ({n_out} outside the trajectory), {len(xyz)} points -> dense_{tag}.npz')


if __name__ == '__main__':
    main()
