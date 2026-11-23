#!/usr/bin/env python3
"""Stamp check of the messages that a playback window would publish (read-only on the source bag).

For every topic, takes the messages with receive time (bag timestamp) in [t_from, t_to] (bag seconds from
the first message), in receive order as `ros2 bag play` publishes them, and reads the header stamp from the
first bytes of the CDR blob (4-byte encapsulation header, then builtin_interfaces/Time sec int32, nanosec
uint32; every stamped message starts with std_msgs/Header). /tf_static has no top-level header and is skipped.

Usage: check_window_stamps.py <stamps.npz> <t_from> <t_to> <first30_s> <out.json>
The stamps come from ~/ca10_gap/stamps.npz (ca10_gap/scripts/extract_stamps.py: the same first-12-bytes read of
every message of this db3, read-only); reading the db3 per window was too slow (random reads over 23 GB).
Reports per stream: count, backwards header steps, duplicate stamps, hdr-rx median/min/max over the whole
window and over its first `first30_s` seconds, and the same statistics for the reference part of the bag
(rx 20-800 s), so "normal range" is measured, not assumed.
"""
import json
import sys

import numpy as np

npz, t_from, t_to, first, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
S = np.load(npz)
# reference statistics (rx 20-800 s) from another npz, e.g. the original bag when checking a written half
SR = np.load(sys.argv[sys.argv.index('--ref') + 1]) if '--ref' in sys.argv else S
T0 = 1782577258896305676          # bag start, ns (ros2 bag info)
names = sorted({k.rsplit('__', 1)[0] for k in S.files} - {'tf_static'})


def stamps(key, a, b, src=None):
    src = S if src is None else src
    rx, hdr = src[key + '__rx'], src[key + '__hdr']
    o = np.argsort(rx, kind='stable')            # playback order = receive time
    rx, hdr = rx[o], hdr[o]
    m = (rx >= T0 + int(a * 1e9)) & (rx <= T0 + int(b * 1e9))
    return rx[m], hdr[m]

def stats(rx, hdr):
    if len(rx) < 2:
        return dict(n=int(len(rx)))
    d = (hdr - rx) / 1e9
    st = np.diff(hdr)
    return dict(n=int(len(rx)), backwards=int((st < 0).sum()), duplicates=int((st == 0).sum()),
                largest_back_s=float(-st.min() / 1e9) if (st < 0).any() else 0.0,
                hdr_minus_rx_med=float(np.median(d)), hdr_minus_rx_min=float(d.min()), hdr_minus_rx_max=float(d.max()),
                hdr_minus_rx_p1=float(np.percentile(d, 1)), hdr_minus_rx_p99=float(np.percentile(d, 99)),
                rx_first_s=float((rx[0] - T0) / 1e9), rx_last_s=float((rx[-1] - T0) / 1e9),
                hdr_first_s=float((hdr[0] - T0) / 1e9), hdr_last_s=float((hdr[-1] - T0) / 1e9))


res = dict(npz=npz, t0_unix_ns=int(T0), window=[t_from, t_to], first_s=first, topics={})
for name in names:
    tid = name
    w = stats(*stamps(tid, t_from, t_to))
    f = stats(*stamps(tid, t_from, t_from + first))
    ref = stats(*stamps(tid, 20.0, 800.0, SR))
    res['topics'][name] = dict(window=w, first=f, reference_20_800=ref)
    ok = w.get('backwards', 0) == 0 and (f['hdr_minus_rx_min'] >= ref['hdr_minus_rx_min'] - 1e-3) \
        and (f['hdr_minus_rx_max'] <= ref['hdr_minus_rx_max'] + 1e-3)
    res['topics'][name]['ok'] = bool(ok)
    print(f"{name:55s} n={w['n']:7d} back={w.get('backwards')} dup={w.get('duplicates')} | first {first:.0f}s "
          f"hdr-rx med {f['hdr_minus_rx_med']:+.4f} [{f['hdr_minus_rx_min']:+.4f},{f['hdr_minus_rx_max']:+.4f}] | "
          f"ref [{ref['hdr_minus_rx_min']:+.4f},{ref['hdr_minus_rx_max']:+.4f}] p1/p99 "
          f"[{ref['hdr_minus_rx_p1']:+.4f},{ref['hdr_minus_rx_p99']:+.4f}]  {'OK' if ok else 'CHECK'}")
json.dump(res, open(out, 'w'), indent=1)
