#!/usr/bin/env python3
"""Write a slim CA-10 half-bag: /velodyne_points, /camera/camera/imu, /gnss/fix, /gnss/vel, /tf_static.

  write_slim_bag.py <out_dir> <t_from_bag_s> <t_to_bag_s>

Source: the original db3, opened read-only (sqlite mode=ro&immutable=1). Messages with receive time in
[t_from, t_to] (bag s from the first message, T0 = 1782577258.896305676) are copied byte for byte (serialized
CDR, unchanged) with their original receive timestamps, in receive order (the timestamp index; equal stamps keep rowid order), into a new
rosbag2 sqlite3 bag written by Humble's rosbag2_py (so its metadata.yaml is v8-compatible; no v9->v8 fix needed).
/tf_static (one latched message at bag 0.029 s): written unchanged when it is inside the window; for a window
that starts later it is written with the receive timestamp of the window's first message (content unchanged),
so the half still carries the static camera transforms without making playback wait from bag 0.
"""
import sqlite3
import sys

import rosbag2_py

SRC = '/home/afonso/bags/session_20260627_172049(half_good)/session_20260627_172049_0.db3'
T0 = 1782577258896305676
TOPICS = ['/velodyne_points', '/camera/camera/imu', '/gnss/fix', '/gnss/vel', '/tf_static']
out, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
lo, hi = T0 + int(round(a * 1e9)), T0 + int(round(b * 1e9))
c = sqlite3.connect(f'file:{SRC}?mode=ro&immutable=1', uri=True)
tops = {name: (tid, typ, fmt) for tid, name, typ, fmt in c.execute('select id, name, type, serialization_format from topics')}
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=out, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
for name in TOPICS:
    _, typ, fmt = tops[name]
    w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format=fmt))
ids = {tops[n][0]: n for n in TOPICS}
q = ','.join(str(i) for i in ids)
first = c.execute(f'select min(timestamp) from messages where topic_id in ({q}) and timestamp between ? and ?',
                  (lo, hi)).fetchone()[0]
tf_id = tops['/tf_static'][0]
tf_ts, tf_data = c.execute('select timestamp, data from messages where topic_id=?', (tf_id,)).fetchone()
n = {k: 0 for k in TOPICS}
if not (lo <= tf_ts <= hi):
    w.write('/tf_static', bytes(tf_data), first); n['/tf_static'] += 1
    print(f'/tf_static re-stamped from bag {(tf_ts - T0) / 1e9:.3f} s to {(first - T0) / 1e9:.3f} s (content unchanged)')
for tid, ts, data in c.execute(f'select topic_id, timestamp, data from messages where topic_id in ({q}) and '
                               'timestamp between ? and ? order by timestamp', (lo, hi)):
    w.write(ids[tid], bytes(data), ts); n[ids[tid]] += 1
del w
print('written', n, '->', out)
