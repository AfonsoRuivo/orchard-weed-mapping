#!/usr/bin/env python3
"""Supplementary dense frame extraction for the sigma_loc test (Ch7 Maps, 2026-10-08).

The frozen map run uses every 50th camera message (3.45 s, 1.7 m of travel), so a weed is seen in ~2 frames and
only 10 YOLO detections have a partner in the next frame. This re-extracts every STRIDE-th camera message (default
10 = 0.69 s, 0.34 m) with the nearest LiDAR scan, in the same layout as bag_frames_20260627 / lidar (stems
frame_<index>_t<rel_t>s, manifest.csv) so weed_map_cached.py runs on it unchanged.
Camera message index = ordinal in /camera/camera/color/image_raw/compressed, as in the existing stems (checked
against frame_000050: pixels equal up to JPEG re-encoding after the 180 deg rotation of calib rotate_180). Needs rosbag2_py (ROS 2 Humble).
Usage: python3 extract_dense_frames.py --stride 10 --out DIR [--limit N]"""
import argparse, csv, struct
from collections import deque
from pathlib import Path
import cv2, numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG = "/home/afonso/bags/session_20260706_140610(best)"
CAM, LID = '/camera/camera/color/image_raw/compressed', '/velodyne_points'

def parse_lidar(msg):
    f = {x.name: x for x in msg.fields}; n = msg.width * msg.height
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([a[:, f[k].offset:f[k].offset + 4].copy().view('<f4')[:, 0] for k in 'xyz'], 1)
    v = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.3) & (np.linalg.norm(xyz, axis=1) < 20)
    return xyz[v]

ap = argparse.ArgumentParser(); ap.add_argument('--stride', type=int, default=10)
ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args(); out = Path(a.out); (out / 'images').mkdir(parents=True, exist_ok=True); (out / 'lidar').mkdir(exist_ok=True)
rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=BAG, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
rd.set_filter(rosbag2_py.StorageFilter(topics=[CAM, LID]))
lt = get_message('sensor_msgs/msg/PointCloud2'); ct = get_message('sensor_msgs/msg/CompressedImage')
idx, t0, pend, prev, man, nsaved = -1, None, deque(), None, [], 0

def resolve(ts, raw):
    global prev, nsaved
    while pend and pend[0][1] <= ts:
        stem, tts = pend.popleft()
        if prev is not None and (tts - prev[0]) <= (ts - tts): cts, craw = prev
        else: cts, craw = ts, raw
        xyz = parse_lidar(deserialize_message(craw, lt))
        np.savez_compressed(out / 'lidar' / f'{stem}.npz', xyz=xyz, timestamp=cts)
        man.append((stem, tts, cts, tts - cts)); nsaved += 1
    prev = (ts, raw)

while rd.has_next():
    topic, raw, ns = rd.read_next(); ts = ns * 1e-9
    if topic == CAM:
        idx += 1
        if t0 is None: t0 = ts
        if idx % a.stride == 0:
            stem = f'frame_{idx:06d}_t{ts - t0:.2f}s'
            img = cv2.rotate(cv2.imdecode(np.frombuffer(bytes(deserialize_message(raw, ct).data), np.uint8), cv2.IMREAD_COLOR), cv2.ROTATE_180)  # calib rotate_180
            cv2.imwrite(str(out / 'images' / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])  # existing frames: q90
            pend.append((stem, t0 + float(f'{ts - t0:.2f}')))   # same 10 ms rounding as the existing manifest
    else:
        resolve(ts, raw)
        if a.limit and nsaved >= a.limit: break
with open(out / 'lidar' / 'manifest.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['stem', 'target_ts', 'matched_ts', 'dt']); w.writerows(man)
print(f'{nsaved} frames, {sum(abs(m[3]) > 0.15 for m in man)} outside 0.15 s')
