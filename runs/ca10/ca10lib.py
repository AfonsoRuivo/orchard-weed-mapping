"""Shared helpers for the CA-10 split run (trajectories, PCD I/O, rigid transforms, keyframe-scan maps)."""
import csv
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

T0 = 1782577258.896306          # bag start, unix s; "bag s" = unix - T0
A_END = 805.5                   # every A output is cut to t <= A_END (user)
B_START = 839.5
GAP = (A_END, B_START)
sys.path.insert(0, '/home/afonso/ca10_gap/scripts')
from icp_np import voxel, prep, icp as _icp, evaluate   # noqa: E402,F401  (gap-analysis numpy GICP)


# -- trajectories: arrays (N, 8) t_unix x y z qx qy qz qw ------------------------------------------------
def read_csv_traj(path):
    rows = list(csv.DictReader(open(path)))
    tk = 't' if 't' in rows[0] else 'timestamp'
    a = np.array([[float(r[tk])] + [float(r[k]) for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] for r in rows])
    a = a[np.isfinite(a).all(1)]
    a = a[np.argsort(a[:, 0], kind='stable')]
    return a[np.r_[True, np.diff(a[:, 0]) > 0]]


def write_tum(path, a, header=None):
    with open(path, 'w') as f:
        if header:
            f.write(''.join(f'# {h}\n' for h in header))
        for r in a:
            f.write(f'{r[0]:.9f} ' + ' '.join(f'{v:.9g}' for v in r[1:8]) + '\n')


def write_csv_traj(path, a):
    with open(path, 'w') as f:
        f.write('t,x,y,z,qx,qy,qz,qw\n')
        for r in a:
            f.write(f'{r[0]:.9f},' + ','.join(f'{v:.9g}' for v in r[1:8]) + '\n')


def to_T(a):
    """(N, 8) -> (N, 4, 4)"""
    T = np.repeat(np.eye(4)[None], len(a), 0)
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix(); T[:, :3, 3] = a[:, 1:4]
    return T


def from_T(t, T):
    return np.column_stack([t, T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()])


def transform_traj(a, M):
    """Left-multiply every pose by the rigid 4x4 M (moves the trajectory into another frame)."""
    return from_T(a[:, 0], M[None] @ to_T(a))


def interp_traj(a, t):
    """Pose at times t: linear translation, slerp rotation (as build_dense_traj.py). t must be inside a."""
    s = Slerp(a[:, 0], Rotation.from_quat(a[:, 4:8]))
    p = np.stack([np.interp(t, a[:, 0], a[:, k]) for k in (1, 2, 3)], 1)
    return np.column_stack([t, p, s(t).as_quat()])


def corrected_odometry(odom, kf_live, kf_final):
    """Live LIO-SAM odometry moved onto the final (loop-corrected) keyframe trajectory: each odometry pose gets
    the correction T_final(k) T_live(k)^-1 of the keyframe k at or before it (live and final keyframes are
    matched by stamp). Without loop corrections in the half this is the identity."""
    kt = kf_final[:, 0]
    live = {round(r[0], 4): r for r in kf_live}
    C = []
    for r in kf_final:
        l = live.get(round(r[0], 4))
        C.append(np.eye(4) if l is None else to_T(r[None])[0] @ np.linalg.inv(to_T(l[None])[0]))
    C = np.array(C)
    k = np.clip(np.searchsorted(kt, odom[:, 0], side='right') - 1, 0, len(kt) - 1)
    return from_T(odom[:, 0], C[k] @ to_T(odom)), C


def se3(R=np.eye(3), t=(0, 0, 0)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def se2_to_se3(R2, t2, dz=0.0):
    M = np.eye(4); M[:2, :2] = R2; M[:2, 3] = t2; M[2, 3] = dz; return M


def xyyaw(M):
    return float(M[0, 3]), float(M[1, 3]), float(M[2, 3]), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def rpy_deg(M):
    return Rotation.from_matrix(M[:3, :3]).as_euler('xyz', degrees=True).tolist()


# -- PCD (LIO-SAM binary x y z intensity float32) ----------------------------------------------------------
def read_pcd(path):
    f = open(path, 'rb').read()
    i = f.index(b'DATA binary\n') + 12
    head = f[:i].decode()
    fields = next(l for l in head.splitlines() if l.startswith('FIELDS')).split()[1:]
    a = np.frombuffer(f[i:], np.float32).reshape(-1, len(fields))
    return a.copy(), fields


def write_pcd(path, a, fields=('x', 'y', 'z', 'intensity')):
    a = np.ascontiguousarray(a, dtype=np.float32)
    n, k = a.shape
    head = (f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS {' '.join(fields)}\n"
            f"SIZE {' '.join(['4'] * k)}\nTYPE {' '.join(['F'] * k)}\nCOUNT {' '.join(['1'] * k)}\n"
            f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, 'wb') as f:
        f.write(head.encode()); f.write(a.tobytes())


def voxel_first(a, v):
    """One point per voxel (the first), keeping all columns."""
    k = np.floor(a[:, :3] / v).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return a[np.sort(i)]


# -- keyframe scans ----------------------------------------------------------------------------------------
class Scans:
    def __init__(self, path):
        d = np.load(path)
        self.stamps, self.lens, self.pts = d['stamps'], d['lens'], d['pts']
        self.off = np.r_[0, np.cumsum(self.lens)]

    def scan(self, k):
        return self.pts[self.off[k]:self.off[k + 1]]

    def index_of(self, t):
        j = np.argmin(np.abs(self.stamps - t))
        return int(j) if abs(self.stamps[j] - t) < 1e-3 else None


def part_map(scans, kf, t_from, t_to, zmin=-0.45, rmin=1.5, rmax=40.0, vox=0.15, ground=True):
    """Map part built from the keyframe scans with bag time in [t_from, t_to], placed with their keyframe poses.
    Per scan in the sensor frame: z > zmin (ground removed, the cut of the gap ICP and of the tuned KISS),
    rmin < r < rmax. Then voxel `vox` (first point per voxel). Returns (N, 3) float64 in the map frame."""
    out = []
    for r in kf:
        tb = r[0] - T0
        if not (t_from <= tb <= t_to):
            continue
        k = scans.index_of(r[0])
        if k is None:
            continue
        P = scans.scan(k)[:, :3].astype(float)
        rr = np.linalg.norm(P[:, :2], axis=1)
        m = (rr > rmin) & (rr < rmax)
        if ground:
            m &= P[:, 2] > zmin
        M = to_T(r[None])[0]
        out.append(P[m] @ M[:3, :3].T + M[:3, 3])
    P = np.vstack(out)
    return voxel(P, vox) if vox else P
