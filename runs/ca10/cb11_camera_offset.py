#!/usr/bin/env python3
"""Timing test (b): colour header-minus-receive offset of CB-11 over its 374 frozen frames (read-only).

  python3 cb11_camera_offset.py "<path to session_20260706_140610(best) .db3>" [out.json]

The CB-11 bag is not on this machine (searched 2026-10-27); run this once it is downloaded. Reads only the first
12 bytes (CDR header stamp) of each /camera/camera/color/image_raw/compressed message, ordered by receive time
(the rosbag2 read order that gave the stem index), and the stem index of the 374 frames used in the frozen map
(viz_bundle_20261014/weed_map/tables/frames.csv). Prints median, p5/p95, min/max of hdr - rx over those frames and over
the whole bag, and checks the index mapping against the stems' rel_t (10 ms rounding).
"""
import json, re, sqlite3, struct, sys
import numpy as np, pandas as pd

db, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'cb11_camera_offset.json')
c = sqlite3.connect(f'file:{db}?mode=ro&immutable=1', uri=True)
tid = c.execute("select id from topics where name='/camera/camera/color/image_raw/compressed'").fetchone()[0]
rows = c.execute('select timestamp, substr(data, 1, 12) from messages where topic_id=? order by timestamp', (tid,)).fetchall()
rx = np.array([r[0] for r in rows], np.int64)
hs = np.array([struct.unpack_from('<iI', bytes(r[1]), 4) for r in rows], np.int64); hdr = hs[:, 0] * 1_000_000_000 + hs[:, 1]
fr = pd.read_csv('/home/afonso/viz_bundle_20261014/weed_map/tables/frames.csv')
idx = fr.stem.str.extract(r'frame_(\d+)_')[0].astype(int).values; rel = fr.stem.str.extract(r'_t([\d.]+)s')[0].astype(float).values
dev = np.abs((rx[idx] - rx[0]) / 1e9 - rel)
d, dall = (hdr[idx] - rx[idx]) / 1e9, (hdr - rx) / 1e9
st = lambda v: dict(n=int(len(v)), median_s=float(np.median(v)), p5_s=float(np.percentile(v, 5)), p95_s=float(np.percentile(v, 95)),
                    min_s=float(v.min()), max_s=float(v.max()), iqr_s=float(np.subtract(*np.percentile(v, [75, 25]))))
res = dict(db=db, frozen_frames=st(d), whole_bag=st(dall), stem_index_check_max_dev_ms=float(dev.max() * 1000),
           ca10_reference=dict(median_s=-0.291, p5_s=-0.348, p95_s=-0.115, n=378))
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))
