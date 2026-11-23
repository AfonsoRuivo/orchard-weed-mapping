#!/usr/bin/env python3
"""Compare the frozen CA-10 weed map (image time = receive time) with the header-stamp re-run (timing_header.py +
run_weed_ca10.py --frames timing_header/frames --outname weed_timing_header). Nothing frozen is changed.

  timing_compare.py <ca10_run dir>  ->  <ca10_run>/timing_header/compare.json, entity_moves_<det>.csv

Detections are matched by (stem, u, v, area_px): the same image and the same detector give the same box / contact
pixel unless the height gate (from the newly paired scan) changes it. Movement of a matched detection = ENU
distance. Entity movement (entities of the area-limit-ON map points, eps 0.4 m, as the frozen tables): for each
frozen entity, the mean ENU position of its member detections that are also in the re-run, before and after
(same members, so only the timing moves it); plus the frozen-entity -> nearest re-run entity distance.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/afonso/viz_bundle_20261014/style')
from entities import entities
from priority_lib import priority, topk, jaccard

R = Path(sys.argv[1]); O = R / 'timing_header'
T_tr = pd.read_csv(R / 'trunks/trunks_final_ca10.csv')[['east_m', 'north_m']].values
st = lambda v: dict(n=int(len(v)), median_m=float(np.median(v)) if len(v) else None, p90_m=float(np.percentile(v, 90)) if len(v) else None,
                    max_m=float(np.max(v)) if len(v) else None)
key = lambda d: list(zip(d.stem, d.u.round(2), d.v.round(2), d.area_px.round(1)))
res = {}
for det in ('yolo', 'colour'):
    r = {}
    for kind, f in (('map_points', f'weed_map_{det}_cached.csv'), ('all_detections', f'weed_map_{det}_patches.csv')):
        a, b = pd.read_csv(R / 'weed/frozen' / f), pd.read_csv(R / 'weed_timing_header/frozen' / f)
        ka, kb = key(a), key(b); ib = {k: i for i, k in enumerate(kb)}
        m = [(i, ib[k]) for i, k in enumerate(ka) if k in ib]
        ia, jb = np.array([x[0] for x in m]), np.array([x[1] for x in m])
        d = np.hypot(a.x_enu.values[ia] - b.x_enu.values[jb], a.y_enu.values[ia] - b.y_enu.values[jb]) if len(m) else np.array([])
        r[kind] = dict(frozen=len(a), header=len(b), matched=len(m), only_frozen=len(a) - len(m), only_header=len(b) - len(m),
                       detection_move=st(d))
        if kind == 'all_detections':
            r[kind]['patch_frozen'] = int(a.patch.sum()); r[kind]['patch_header'] = int(b.patch.sum())
    a, b = pd.read_csv(R / 'weed/frozen' / f'weed_map_{det}_cached.csv'), pd.read_csv(R / 'weed_timing_header/frozen' / f'weed_map_{det}_cached.csv')
    ea, la = entities(a, 0.4); eb, lb = entities(b, 0.4)
    ib = {k: i for i, k in enumerate(key(b))}
    rows = []
    for eid, g in pd.Series(np.arange(len(a))).groupby(la):
        mem = [(i, ib[k]) for i, k in zip(g.values, [key(a)[i] for i in g.values]) if k in ib]
        if not mem: rows.append(dict(eid=eid, n_members=len(g), n_matched=0, move_m=np.nan)); continue
        i0, i1 = np.array([x[0] for x in mem]), np.array([x[1] for x in mem])
        p0 = a[['x_enu', 'y_enu']].values[i0].mean(0); p1 = b[['x_enu', 'y_enu']].values[i1].mean(0)
        rows.append(dict(eid=eid, n_members=len(g), n_matched=len(mem), move_m=float(np.hypot(*(p1 - p0)))))
    E = pd.DataFrame(rows).merge(ea[['eid', 'x', 'y', 'area_med']], on='eid')
    E['nn_to_header_entity_m'] = cKDTree(eb[['x', 'y']].values).query(E[['x', 'y']].values)[0]
    E.to_csv(O / f'entity_moves_{det}.csv', index=False)
    pa = priority(ea[['x', 'y']].values, ea.area_med.values, T_tr)['pri']; pb = priority(eb[['x', 'y']].values, eb.area_med.values, T_tr)['pri']
    ta, tb = ea[['x', 'y']].values[list(topk(pa))], eb[['x', 'y']].values[list(topk(pb))]
    top_nn = cKDTree(tb).query(ta)[0]
    r['entities'] = dict(frozen=len(ea), header=len(eb), entities_with_matched_members=int((E.n_matched > 0).sum()),
                         entity_move_same_members=st(E.move_m.dropna().values), frozen_to_nearest_header_entity=st(E.nn_to_header_entity_m.values),
                         total_area_frozen_m2=float(ea.area_med.sum()), total_area_header_m2=float(eb.area_med.sum()),
                         top10_frozen=len(ta), top10_header=len(tb), top10_frozen_with_header_top10_within_0p5m=int((top_nn <= 0.5).sum()))
    res[det] = r
fa = json.load(open(R / 'weed/frozen/frame_accounting.json')); fb = json.load(open(R / 'weed_timing_header/frozen/frame_accounting.json'))
res['frames'] = dict(frozen=fa['frames_used'], header=fb['frames_used'], implausible_plane_frozen=fa['implausible_plane'],
                     implausible_plane_header=fb['implausible_plane'], sync_dropped_frozen=fa['sync_dropped_dt_gt_0.15s'],
                     sync_dropped_header=fb['sync_dropped_dt_gt_0.15s'])
m = pd.read_csv(O / 'frames/lidar/manifest.csv')
res['offsets'] = dict(image_hdr_minus_rx_s=dict(median=float(np.median(m.target_ts - m.rx_target_ts_frozen)),
                                                p5=float(np.percentile(m.target_ts - m.rx_target_ts_frozen, 5)),
                                                p95=float(np.percentile(m.target_ts - m.rx_target_ts_frozen, 95)),
                                                min=float((m.target_ts - m.rx_target_ts_frozen).min()), max=float((m.target_ts - m.rx_target_ts_frozen).max())),
                      pose_time_shift_s=dict(median=float(np.median(m.matched_ts - m.rx_matched_ts_frozen)),
                                             p5=float(np.percentile(m.matched_ts - m.rx_matched_ts_frozen, 5)),
                                             p95=float(np.percentile(m.matched_ts - m.rx_matched_ts_frozen, 95))),
                      n_frames=len(m))
json.dump(res, open(O / 'compare.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
