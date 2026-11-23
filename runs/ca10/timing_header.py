#!/usr/bin/env python3
"""Camera-timing test (a): frozen-stride frames re-paired with image time = HEADER stamp (frozen: receive time).

  timing_header.py <ca10_run dir>   ->  <ca10_run>/timing_header/frames/{images -> symlinks, lidar/}

- Image header stamp: the camera message index in the stem (ordinal in /camera/camera/color/image_raw/compressed,
  rosbag2 read order = receive time) is looked up in ~/ca10_gap/stamps.npz (rx/hdr of every message, this db3).
  Checked: rx - t0_cam reproduces the stem's rel_t to 10 ms rounding for every frame.
- LiDAR: the scan whose HEADER stamp is nearest the image header stamp; manifest target_ts = image hdr,
  matched_ts = that scan's hdr (the pose is then looked up at it, so pose, image and scan share the header clock, the
  trajectory's clock); dt = target - matched. Scans not already extracted are read read-only from the db3 and decoded
  as extract_dense_frames.parse_lidar (0.3-20 m, xyz).
Only the frozen subset (index % 50 == 0) is written. Images are the same files (symlinks).
"""
import csv, re, sqlite3, sys
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

R = Path(sys.argv[1]); O = R / 'timing_header/frames'; (O / 'images').mkdir(parents=True, exist_ok=True); (O / 'lidar').mkdir(exist_ok=True)
S = np.load('/home/afonso/ca10_gap/stamps.npz')
crx, chdr = S['camera_camera_color_image_raw_compressed__rx'], S['camera_camera_color_image_raw_compressed__hdr']
o = np.argsort(crx, kind='stable'); crx, chdr = crx[o], chdr[o]
lrx, lhdr = S['velodyne_points__rx'], S['velodyne_points__hdr']
ol = np.argsort(lhdr, kind='stable'); lrx, lhdr = lrx[ol], lhdr[ol]
t0c = crx[0]
DB = '/home/afonso/bags/session_20260627_172049(half_good)/session_20260627_172049_0.db3'
db = sqlite3.connect(f'file:{DB}?mode=ro&immutable=1', uri=True)
tid = db.execute("select id from topics where name='/velodyne_points'").fetchone()[0]


def parse_lidar(msg):                       # = extract_dense_frames.parse_lidar
    f = {x.name: x for x in msg.fields}; n = msg.width * msg.height
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([a[:, f[k].offset:f[k].offset + 4].copy().view('<f4')[:, 0] for k in 'xyz'], 1)
    v = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.3) & (np.linalg.norm(xyz, axis=1) < 20)
    return xyz[v]


man_old = {r['stem']: r for r in csv.DictReader(open(R / 'frames_stride10/lidar/manifest.csv'))}
rows, maxdev = [], 0.0
for stem in sorted(man_old):
    idx = int(re.match(r'frame_(\d+)_t([\d.]+)s', stem).group(1)); rel = float(re.match(r'frame_\d+_t([\d.]+)s', stem).group(1))
    if idx % 50: continue
    maxdev = max(maxdev, abs((crx[idx] - t0c) / 1e9 - rel))
    th = chdr[idx]; j = int(np.argmin(np.abs(lhdr - th))); lh, lr = lhdr[j], lrx[j]
    out = O / 'lidar' / f'{stem}.npz'
    blob = db.execute('select data from messages where topic_id=? and timestamp=?', (tid, int(lr))).fetchone()
    msg = deserialize_message(blob[0], PointCloud2)
    assert msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec == lh
    np.savez_compressed(out, xyz=parse_lidar(msg), timestamp=lh / 1e9)
    src = R / 'frames_stride10/images' / f'{stem}.jpg'; dst = O / 'images' / f'{stem}.jpg'
    if not dst.exists(): dst.symlink_to(src)
    rows.append((stem, th / 1e9, lh / 1e9, (th - lh) / 1e9, float(man_old[stem]['target_ts']), float(man_old[stem]['matched_ts'])))
with open(O / 'lidar' / 'manifest.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['stem', 'target_ts', 'matched_ts', 'dt', 'rx_target_ts_frozen', 'rx_matched_ts_frozen']); w.writerows(rows)
a = np.array([r[1:] for r in rows])
print(f'{len(rows)} frames; stem rel_t check max |dev| {maxdev * 1000:.1f} ms; image hdr - rx: median {np.median(a[:, 0] - a[:, 3]):+.3f} s '
      f'[p5 {np.percentile(a[:, 0] - a[:, 3], 5):+.3f}, p95 {np.percentile(a[:, 0] - a[:, 3], 95):+.3f}]; '
      f'pose time shift (new matched - frozen matched): median {np.median(a[:, 1] - a[:, 4]):+.3f} s, '
      f'p5 {np.percentile(a[:, 1] - a[:, 4], 5):+.3f}, p95 {np.percentile(a[:, 1] - a[:, 4], 95):+.3f}')
