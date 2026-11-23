#!/usr/bin/env python3
"""
extract_scans.py - pull raw /velodyne_points scans out of the CB-11 bag.

  python3 extract_scans.py --keyframes           # the 612 scans LIO-SAM used as keyframes
  python3 extract_scans.py --all --stride 2      # every 2nd scan (for the interpolated dense map)
  python3 extract_scans.py --times traj.csv --name rep1_keyframe_scans.npz   # scans at another run's keyframe stamps

Reads the sqlite .db3 directly (no ros2 bag play), deserialises PointCloud2 with rclpy and
stores, per scan: header stamp, and an (N, 6) float32 array x, y, z, intensity, ring, t_rel.
Output: one .npz per run in data_set/campinas_b/dense_map/.
"""
import argparse, sqlite3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

REPO = Path(__file__).resolve().parents[3]
BAG = Path.home() / "bags" / "session_20260706_140610(best)" / "session_20260706_140610_0-001.db3"
KF = REPO / "orchard-slam/data/campinas_b/lio_sam/run_20260706/transformations.pcd"
OUT = REPO / "data_set/campinas_b/dense_map"


def load_keyframes(path=KF):
    f = open(path, 'rb').read(); i = f.index(b'DATA binary\n') + 12
    dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('i', 'f4'), ('roll', 'f4'), ('pitch', 'f4'),
                   ('yaw', 'f4'), ('t', 'f8')])
    return np.frombuffer(f[i:], dt).copy()


def cloud_to_array(msg):
    names = {f.name: f for f in msg.fields}
    nfmt = {7: 'f4', 8: 'f8', 4: 'u2', 2: 'u1', 6: 'u4', 5: 'i4'}
    dt = np.dtype({'names': list(names), 'formats': [nfmt[names[n].datatype] for n in names],
                   'offsets': [names[n].offset for n in names], 'itemsize': msg.point_step})
    a = np.frombuffer(bytes(msg.data), dt)
    t = a['time'] if 'time' in names else np.zeros(len(a))
    out = np.column_stack([a['x'], a['y'], a['z'], a['intensity'], a['ring'], t]).astype(np.float32)
    return out[np.isfinite(out[:, :3]).all(1)]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--keyframes', action='store_true'); g.add_argument('--all', action='store_true')
    g.add_argument('--times', help='CSV with a t column (e.g. a LIO-SAM keyframes CSV): extract the scans with these stamps')
    ap.add_argument('--name', default=None, help='output file name in dense_map/')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-range', type=float, default=25.0, help='drop points beyond this range (m)')
    a = ap.parse_args()

    db = sqlite3.connect(f"file:{BAG}?mode=ro", uri=True)
    tid = db.execute("select id from topics where name='/velodyne_points'").fetchone()[0]
    rows = db.execute("select timestamp, data from messages where topic_id=? order by timestamp", (tid,))
    want = None
    if a.keyframes or a.times:
        kf_t = load_keyframes()['t'] if a.keyframes else np.array([float(r.split(',')[0]) for r in open(a.times).read().split('\n')[1:] if r])
        want = set()
    stamps, clouds, k = [], [], 0
    for ts, blob in rows:
        if a.all and k % a.stride:
            k += 1; continue
        k += 1
        msg = deserialize_message(blob, PointCloud2)
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if want is not None:
            j = np.argmin(np.abs(kf_t - st))
            if abs(kf_t[j] - st) > 1e-3: continue
            want.add(int(j))
        c = cloud_to_array(msg)
        c = c[np.linalg.norm(c[:, :3], axis=1) <= a.max_range]
        stamps.append(st); clouds.append(c)
        if len(clouds) % 500 == 0: print(len(clouds), 'scans')
    name = a.name or ('keyframe_scans.npz' if a.keyframes else f'all_scans_s{a.stride}.npz')
    lens = np.array([len(c) for c in clouds])
    np.savez(OUT / name, stamps=np.array(stamps), lens=lens, pts=np.vstack(clouds))
    print(f'{len(clouds)} scans, {lens.sum()} points -> {OUT / name}')
    if want is not None: print(f'matched {len(want)}/{len(kf_t)} keyframes')


if __name__ == '__main__':
    main()
