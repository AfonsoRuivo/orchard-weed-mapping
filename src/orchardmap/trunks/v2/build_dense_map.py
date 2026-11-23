#!/usr/bin/env python3
"""
build_dense_map.py - dense map from the raw keyframe scans placed on the LIO-SAM keyframe poses.

The GlobalMap only keeps LIO-SAM's downsampled edge/planar features (0.2 / 0.4 m leaf), so trunk
slices are sparse. Here every raw point of the 612 keyframe scans is put in the map frame:
  - pose of keyframe k = (x, y, z, roll, pitch, yaw) from transformations.pcd (pcl convention),
    stamped at the END of the scan (Velodyne point times run from -0.10 s to 0);
  - deskew: each point is moved with the pose interpolated at stamp + t_point, using the
    neighbouring keyframes (linear translation, slerp rotation). ~2 s between keyframes, so this
    is a constant-velocity model over the 0.1 s sweep.
Output: dense_map/dense_kf.npz with xyz (float32), keyframe index, ring, range.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from common import DENSE, keyframe_poses, rpy_to_R


def main(deskew=True, min_range=1.0):
    K = keyframe_poses(); d = np.load(DENSE / 'keyframe_scans.npz'); pts, lens = d['pts'], d['lens']
    off = np.r_[0, np.cumsum(lens)]
    Rs = Rotation.from_matrix(np.stack([rpy_to_R(r, p, y) for r, p, y in zip(K['roll'], K['pitch'], K['yaw'])]))
    T = np.column_stack([K['x'], K['y'], K['z']]).astype(float); tk = K['t']
    slerp = Slerp(tk, Rs)
    out, kid, ring, rng = [], [], [], []
    for k in range(len(K)):
        s = pts[off[k]:off[k + 1]]
        r = np.linalg.norm(s[:, :3], axis=1); s = s[r >= min_range]; r = r[r >= min_range]
        p = s[:, :3].astype(float)
        if deskew:
            t = np.clip(tk[k] + s[:, 5].astype(float), tk[0], tk[-1])
            Ri = slerp(t)
            j = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2); a = ((t - tk[j]) / (tk[j + 1] - tk[j]))[:, None]
            Ti = T[j] * (1 - a) + T[j + 1] * a     # equals pose k at t = tk[k]
            w = Ri.apply(p) + Ti
        else:
            w = p @ rpy_to_R(K['roll'][k], K['pitch'][k], K['yaw'][k]).T + T[k]
        out.append(w.astype(np.float32)); kid.append(np.full(len(w), k, np.int16))
        ring.append(s[:, 4].astype(np.int8)); rng.append(r.astype(np.float16))
    xyz = np.vstack(out)
    name = 'dense_kf.npz' if deskew else 'dense_kf_nodeskew.npz'
    np.savez(DENSE / name, xyz=xyz, kf=np.concatenate(kid), ring=np.concatenate(ring), rng=np.concatenate(rng))
    print(f'{len(xyz)} points -> {DENSE / name}')


if __name__ == '__main__':
    import sys
    main(deskew='--no-deskew' not in sys.argv)
