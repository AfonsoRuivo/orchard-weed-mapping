"""Shared helpers for the v2 trunk detector: maps, poses, ground, GT, scoring."""
import csv, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "orchard-slam/data/campinas_b/lio_sam/run_20260706"
GLOBALMAP = RUN / "GlobalMap.pcd"
DENSE = REPO / "data_set/campinas_b/dense_map"
GT_DIR = Path(__file__).resolve().parents[1] / "gt"


def read_pcd_xyz(path):
    f = open(path, 'rb').read(); i = f.index(b'DATA binary\n') + 12
    hdr = f[:i].decode()
    fields = hdr.split('FIELDS')[1].split('\n')[0].split()
    a = np.frombuffer(f[i:], np.float32).reshape(-1, len(fields))
    return a[:, :3].astype(np.float64)


def keyframe_poses():
    f = open(RUN / "transformations.pcd", 'rb').read(); i = f.index(b'DATA binary\n') + 12
    dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('i', 'f4'), ('roll', 'f4'), ('pitch', 'f4'),
                   ('yaw', 'f4'), ('t', 'f8')])
    return np.frombuffer(f[i:], dt).copy()


def rpy_to_R(r, p, y):
    """pcl::getTransformation convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def local_ground(P, cell=0.5, pctl=5, ref=None):
    """Height above local ground: ground = pctl-th percentile of z in cell x cell bins.
    ref: optional cloud used to compute the ground (defaults to P)."""
    Q = P if ref is None else ref
    x0, y0 = Q[:, 0].min() - 1, Q[:, 1].min() - 1
    def key(A):
        return (np.floor((A[:, 0] - x0) / cell).astype(np.int64) * 100000
                + np.floor((A[:, 1] - y0) / cell).astype(np.int64))
    kq = key(Q); order = np.argsort(kq, kind='stable'); ks = kq[order]; zs = Q[order, 2]
    uk, start = np.unique(ks, return_index=True); end = np.r_[start[1:], len(ks)]
    g = np.array([np.percentile(zs[s:e], pctl) for s, e in zip(start, end)])
    kp = key(P); pos = np.searchsorted(uk, kp); pos = np.clip(pos, 0, len(uk) - 1)
    ok = uk[pos] == kp
    h = np.full(len(P), np.nan); h[ok] = P[ok, 2] - g[pos[ok]]
    return h


def load_gt():
    G = list(csv.DictReader(open(GT_DIR / "gt_trunks_map.csv")))
    xy = np.array([[float(g['x']), float(g['y'])] for g in G])
    return G, xy


def write_det(path, dets, extra_cols=()):
    cols = ['x', 'y', 'r'] + list(extra_cols)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for d in dets: w.writerow(d)


def score(det_csv, ladder=None):
    """Run the provided scorer; return its SUMMARY line and parsed json."""
    import json, tempfile
    js = tempfile.mktemp(suffix='.json')
    cmd = [sys.executable, '-I', str(GT_DIR / 'eval_trunks.py'), '--det', str(det_csv), '--json', js]
    if ladder: cmd += ['--ladder', str(ladder)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res = json.load(open(js)); summ = [l for l in out.splitlines() if l.startswith('SUMMARY')][0]
    return summ, res, out
