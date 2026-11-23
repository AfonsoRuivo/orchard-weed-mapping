#!/usr/bin/env python3
"""Raw /velodyne_points scans of CA-10 at given header stamps (LIO-SAM keyframe times), read-only.

Same output format and point decoding as orchard-weed-detection/trunk/v2/extract_scans.py (the CB-11
keyframe-scan extractor, `--times`): per scan the header stamp and an (N, 6) float32 array
x, y, z, intensity, ring, t_rel; points beyond --max-range dropped (CB-11: 25 m; here 40 m is stored so the
merge ICP can use the 1.5-40 m cut of the gap ICP, and the trunk map applies 25 m itself).
Difference: the db3 is queried by receive time around each stamp (timestamp index) instead of a full scan,
because only a few hundred of 13,002 scans are needed. The match must be exact (|dt| < 1 ms), as there.

Usage: extract_kf_scans.py <times.csv (t column, unix s)> <out.npz> [--max-range 40]
"""
import sys
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, '/home/afonso/thesis/orchard-weed-detection/trunk/v2')
from extract_scans import cloud_to_array   # noqa: E402  (unchanged decoder)

DB = '/home/afonso/bags/session_20260627_172049(half_good)/session_20260627_172049_0.db3'
times, out = sys.argv[1], sys.argv[2]
max_range = float(sys.argv[sys.argv.index('--max-range') + 1]) if '--max-range' in sys.argv else 40.0
kf_t = np.array([float(r.split(',')[0]) for r in open(times).read().split('\n')[1:] if r])
db = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
tid = db.execute("select id from topics where name='/velodyne_points'").fetchone()[0]
stamps, clouds, missing = [], [], []
for t in kf_t:
    lo, hi = int((t - 0.6) * 1e9), int((t + 0.6) * 1e9)
    best = None
    for ts, blob in db.execute("select timestamp, data from messages where topic_id=? and timestamp between ? and ?",
                               (tid, lo, hi)):
        msg = deserialize_message(blob, PointCloud2)
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if abs(st - t) < 1e-3:
            best = (st, msg); break
    if best is None:
        missing.append(float(t)); continue
    c = cloud_to_array(best[1])
    c = c[np.linalg.norm(c[:, :3], axis=1) <= max_range]
    stamps.append(best[0]); clouds.append(c)
lens = np.array([len(c) for c in clouds])
np.savez(out, stamps=np.array(stamps), lens=lens, pts=np.vstack(clouds))
print(f'{len(clouds)}/{len(kf_t)} keyframe scans, {lens.sum()} points -> {out}; missing {missing[:5]}')
