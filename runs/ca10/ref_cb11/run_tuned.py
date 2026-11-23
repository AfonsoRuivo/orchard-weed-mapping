#!/usr/bin/env python3
"""Tuned colour map + area-limit-off maps for CB-11 (LIO-SAM rep1 frozen 20261008).

Same frame loop / calibration / trajectory / align_rep1 / ground plane / gate / COS_PHI_MIN / MAX_WEED_RANGE as
results/s1_freeze.py and weed_map_cached.py. Area limit (A > AREA_M2_MAX) does not drop: detections are flagged patch=1.
Per frame set, one pass computes: colour ORIGINAL (reproduction), colour TUNED, YOLO v2 @0.30.
Run with python3 -B (no __pycache__ in the repo). Writes only to this script's directory.

    python3 -B run_tuned.py --set frozen
    python3 -B run_tuned.py --set dense10
"""
import argparse, csv, json, sys, time
from pathlib import Path
import cv2, numpy as np

REPO = Path('/home/afonso/thesis')
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map/results'))
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map'))
import weed_map_cached as wm
import colour_detector as cd
from common import TRAJ, FROZEN, TAB, MAPDIR

OUT = Path(__file__).resolve().parent
SETS = {'frozen': dict(images=wm.IMAGES_DIR, lidar=REPO / 'data_set/campinas_b/lidar', frames=TAB / 'frames.csv',
                       ref=FROZEN, suffix=''),
        'dense10': dict(images=REPO / 'data_set/campinas_b/dense_frames_stride10_20261008/images',
                        lidar=REPO / 'data_set/campinas_b/dense_frames_stride10_20261008/lidar',
                        frames=TAB / 'dense10/frames.csv', ref=MAPDIR / 'dense10_20261008', suffix='_dense10')}
COLS = ['x_enu', 'y_enu', 'x_slam', 'y_slam', 'timestamp', 'area_px', 'area_m2', 'confidence', 'detector', 'stem',
        'u', 'v', 'range_m', 'cos_phi', 'patch', 'x1', 'y1', 'x2', 'y2']
STAGES = ['raw', 'above_horizon_or_gate', 'small_blob', 'no_ground_hit', 'range_gt_5m', 'area_gt_1m2_kept_as_patch',
          'kept']

ap = argparse.ArgumentParser(); ap.add_argument('--set', choices=list(SETS), required=True)
ap.add_argument('--no-yolo', action='store_true'); args = ap.parse_args()
S = SETS[args.set]
wm.load_calibration(); wm.load_trajectory(str(TRAJ)); wm.load_alignment(str(FROZEN / 'align_rep1.json'))

# -- frames: exactly the s1_freeze / weed_map_cached.main loop ------------------------------------------------
man = [r for r in csv.DictReader(open(S['lidar'] / 'manifest.csv')) if abs(float(r['dt'])) <= wm.SYNC_TOLERANCE]
t0 = time.time(); frames = []
for r in man:
    stem, lt = r['stem'], float(r['matched_ts'])
    ip, lp = S['images'] / f'{stem}.jpg', S['lidar'] / f'{stem}.npz'
    if not ip.exists() or not lp.exists(): continue
    if lt < wm.traj_t[0] or lt > wm.traj_t[-1]: continue
    try: a, b, c, d, xr, xt = wm.fit_ground(np.load(lp)['xyz'])
    except Exception: continue
    if not (wm.Z_MIN < -d / c < wm.Z_MAX): continue
    frames.append((stem, ip, lt, (a, b, c, d), xr, xt))
ref_stems = [r['stem'] for r in csv.DictReader(open(S['frames']))]
assert [f[0] for f in frames] == ref_stems, 'frame set differs from tables/frames.csv'
print(f'{args.set}: {len(frames)} frames (= {S["frames"].name}), ground fit {time.time() - t0:.0f} s', flush=True)


def colour(img, hz, p):
    """colour_detector.detect + contact pixel as weed_map_cached.get_colour_boxes (bbox bottom-centre, v = y2 - 1)."""
    dets, m = cd.detect(img, hz, p, return_mask=True)
    n = len(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    ok = [(x1 + (x2 - x1) / 2, y2 - 1, a, float('nan'), x1, y1, x2, y2) for x1, y1, x2, y2, a in dets]
    return ok, n


RUNS = {'colour_original': cd.ORIGINAL, 'colour_tuned': cd.TUNED}
if not args.no_yolo: RUNS['yolo'] = None
cnt = {k: dict.fromkeys(STAGES, 0) for k in RUNS}
rows = {k: [] for k in RUNS}; tdet = dict.fromkeys(RUNS, 0.0); hzs = {}
t1 = time.time()
for i, (stem, ip, lt, pl, xr, xt) in enumerate(frames):
    img = cv2.imread(str(ip)); T = wm.get_pose(lt); hz = wm.get_horizon(img.shape, xr, xt, *pl)
    if args.set == 'frozen': hzs[stem] = hz
    for k, p in RUNS.items():
        ts = time.time(); C = cnt[k]
        if p is None:
            ok, rej = wm.get_weed_boxes(img, hz, return_debug=True)
            C['raw'] += len(ok) + len(rej); C['above_horizon_or_gate'] += len(rej)
        else:
            ok, n = colour(img, hz, p)
            C['raw'] += n; C['small_blob'] += n - len(ok)
        for u, v, area_px, conf, x1, y1, x2, y2 in ok:
            pw, s, geo = wm.pixel_to_ground(u, v, *pl, T, return_geometry=True)
            if pw is None: C['no_ground_hit'] += 1; continue
            if np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE: C['range_gt_5m'] += 1; continue
            area_m2 = area_px * s; patch = int(area_m2 > wm.AREA_M2_MAX)
            C['area_gt_1m2_kept_as_patch' if patch else 'kept'] += 1
            xe, ye = wm.slam_to_enu(pw[0], pw[1])
            rows[k].append((xe, ye, pw[0], pw[1], lt, area_px, area_m2, conf, 'yolo' if p is None else 'colour', stem,
                            u, v, geo['range_m'], geo['cos_phi'], patch, x1, y1, x2, y2))
        tdet[k] += time.time() - ts
    if i % 100 == 0: print(f'  {i}/{len(frames)} {time.time() - t1:.0f} s', flush=True)

# -- outputs -----------------------------------------------------------------------------------------------------
names = {'colour_original': f'weed_map_colour_original_patches{S["suffix"]}.csv',
         'colour_tuned': f'weed_map_colour_tuned{S["suffix"]}.csv', 'yolo': f'weed_map_yolo_patches{S["suffix"]}.csv'}
table, checks = [], {}
for k in RUNS:
    with open(OUT / names[k], 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows[k])
    C = dict(cnt[k]); C['csv_rows'] = len(rows[k]); C['matches_csv'] = C['kept'] + C['area_gt_1m2_kept_as_patch'] == len(rows[k])
    C['detector'] = k; C['det_time_s'] = round(tdet[k], 1); table.append(C); print(C, flush=True)
    # reproduction: patch=0 rows must equal the frozen area-limit-on CSV of that detector (ORIGINAL colour / YOLO)
    if k in ('colour_original', 'yolo'):
        det = 'colour' if k.startswith('colour') else 'yolo'
        ref = list(csv.DictReader(open(S['ref'] / f'weed_map_{det}_cached.csv')))
        key = lambda st, u, v, a: (st, round(float(u), 3), round(float(v), 3), round(float(a), 3))
        mine = sorted(key(r[9], r[10], r[11], r[5]) for r in rows[k] if not r[14])
        theirs = sorted(key(r['stem'], r['u'], r['v'], r['area_px']) for r in ref)
        xe_mine = {key(r[9], r[10], r[11], r[5]): (r[0], r[1]) for r in rows[k] if not r[14]}
        dxe = max([np.hypot(xe_mine[key(r['stem'], r['u'], r['v'], r['area_px'])][0] - float(r['x_enu']),
                            xe_mine[key(r['stem'], r['u'], r['v'], r['area_px'])][1] - float(r['y_enu']))
                   for r in ref if key(r['stem'], r['u'], r['v'], r['area_px']) in xe_mine] or [np.nan])
        checks[k] = dict(kept_area_on=C['kept'], ref_rows=len(ref), rows_identical=mine == theirs,
                         max_enu_diff_m=float(dxe))
        print(k, 'reproduction vs', S['ref'].name, checks[k], flush=True)
import pandas as pd
pd.DataFrame(table).to_csv(OUT / f'filter_table_all{S["suffix"]}.csv', index=False)
pd.DataFrame([t for t in table if t['detector'] == 'colour_tuned']).to_csv(OUT / f'filter_table_tuned{S["suffix"]}.csv',
                                                                            index=False)
json.dump(dict(set=args.set, frames=len(frames), checks=checks, time_total_s=round(time.time() - t0, 1),
               det_time_s=tdet, TUNED=cd.TUNED, ORIGINAL=cd.ORIGINAL), open(OUT / f'run_{args.set}.json', 'w'),
          indent=1, default=str)
if args.set == 'frozen':
    np.savez_compressed(OUT / '_horizons_frozen.npz', **hzs)
print(f'done {time.time() - t0:.0f} s')
