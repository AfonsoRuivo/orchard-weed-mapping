#!/usr/bin/env python3
"""Section 1 - freeze the map run: frame accounting + per-filter table for YOLO v2 (0.30) and colour,
plus per-frame table (plane, pose) reused by sigma_loc. Uses the same loop as weed_map_cached.py.
Outputs: tables/frame_accounting.csv, tables/filter_table.csv, tables/frames.csv"""
import csv, subprocess
from pathlib import Path
import cv2, numpy as np
from common import *
import weed_map_cached as wm

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--images', default=str(wm.IMAGES_DIR)); ap.add_argument('--lidar', default=str(LIDAR))
ap.add_argument('--tables', default=str(TAB)); ap.add_argument('--frames-only', action='store_true',
                help='only the per-frame ground-plane table (used for the dense sigma_loc run)')
args = ap.parse_args(); IMAGES, LIDAR, TAB = Path(args.images), Path(args.lidar), Path(args.tables); TAB.mkdir(exist_ok=True, parents=True)
wm.load_calibration(); wm.load_trajectory(str(TRAJ)); wm.load_alignment(str(FROZEN / 'align_rep1.json'))
man_all = list(csv.DictReader(open(LIDAR / 'manifest.csv')))
acc = dict(manifest_rows=len(man_all), image_files=len(list(IMAGES.glob('*.jpg'))))
man = [r for r in man_all if abs(float(r['dt'])) <= wm.SYNC_TOLERANCE]
acc['sync_dropped_dt_gt_0.15s'] = len(man_all) - len(man)
acc['sync_dropped_stems'] = ';'.join(r['stem'] for r in man_all if abs(float(r['dt'])) > wm.SYNC_TOLERANCE)
frames, fr_rows = [], []
cnt = dict(out_of_traj=0, ground_failed=0, implausible_plane=0, no_files=0)
for r in man:
    stem, lt = r['stem'], float(r['matched_ts'])
    ip, lp = IMAGES / f'{stem}.jpg', LIDAR / f'{stem}.npz'
    if not ip.exists() or not lp.exists(): cnt['no_files'] += 1; continue
    if lt < wm.traj_t[0] or lt > wm.traj_t[-1]: cnt['out_of_traj'] += 1; continue
    try: a, b, c, d, xr, xt = wm.fit_ground(np.load(lp)['xyz'])
    except Exception: cnt['ground_failed'] += 1; continue
    if not (wm.Z_MIN < -d / c < wm.Z_MAX): cnt['implausible_plane'] += 1; continue
    T = wm.get_pose(lt); n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    fr_rows.append(dict(stem=stem, t=lt, a=a, b=b, c=c, d=d, tilt_deg=float(np.degrees(np.arccos(n[2]))),
                        height=float(-d / c), x_slam=T[0, 3], y_slam=T[1, 3],
                        yaw_slam=float(np.arctan2(T[1, 0], T[0, 0]))))
    frames.append((ip, lt, (a, b, c, d), xr, xt))
acc.update(cnt); acc['frames_used'] = len(frames)
acc['closes'] = acc['manifest_rows'] - acc['sync_dropped_dt_gt_0.15s'] - sum(cnt.values()) == len(frames)
pd.DataFrame(fr_rows).to_csv(TAB / 'frames.csv', index=False)
if args.frames_only:
    json.dump(acc, open(TAB / 'frame_accounting.json', 'w'), indent=1); print(acc); raise SystemExit

STAGES = ['raw', 'above_horizon_or_gate', 'small_blob', 'no_ground_hit', 'range_gt_5m', 'area_gt_1m2', 'kept']
out = []
for det in ('yolo', 'colour'):
    Cn = dict.fromkeys(STAGES, 0); Cn['ungated_blobs'] = 0
    for ip, lt, pl, xr, xt in frames:
        img = cv2.imread(str(ip)); T = wm.get_pose(lt); hz = wm.get_horizon(img.shape, xr, xt, *pl)
        if det == 'yolo':
            ok, rej = wm.get_weed_boxes(img, hz, return_debug=True)
            Cn['raw'] += len(ok) + len(rej); Cn['above_horizon_or_gate'] += len(rej)
            cand = [k[:3] for k in ok]
        else:
            ok, dbg = wm.get_colour_boxes(img, hz, return_debug=True)
            # raw = contours of the ungated mask; gate removes pixels (not whole blobs) so count blobs
            un, _ = cv2.findContours(dbg['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            Cn['ungated_blobs'] += sum(cv2.contourArea(c_) > wm.MIN_BLOB for c_ in un)
            Cn['raw'] += dbg['n_contours']; Cn['small_blob'] += dbg['n_contours'] - len(ok)
            cand = [k[:3] for k in ok]
        for u, v, area_px in cand:
            pw, s = wm.pixel_to_ground(u, v, *pl, T)
            if pw is None: Cn['no_ground_hit'] += 1
            elif np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE: Cn['range_gt_5m'] += 1
            elif area_px * s > wm.AREA_M2_MAX: Cn['area_gt_1m2'] += 1
            else: Cn['kept'] += 1
    nrows = len(load_weeds(det)); Cn['csv_rows'] = nrows; Cn['matches_csv'] = Cn['kept'] == nrows
    Cn['detector'] = det; out.append(Cn); print(Cn, flush=True)
pd.DataFrame(out).to_csv(TAB / 'filter_table.csv', index=False)
acc['git_commit'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD']).decode().strip()
acc['params'] = dict(MAX_PLANE_TILT_DEG=wm.MAX_PLANE_TILT_DEG, HORIZON_MODEL=wm.HORIZON_MODEL, CONF_THRESH=wm.CONF_THRESH,
                     YOLO_IMGSZ=wm.YOLO_IMGSZ, MIN_BLOB=wm.MIN_BLOB, MAX_WEED_RANGE=wm.MAX_WEED_RANGE, AREA_M2_MAX=wm.AREA_M2_MAX,
                     PERCENTILE=wm.PERCENTILE, H_WEED=wm.H_WEED, SYNC_TOLERANCE=wm.SYNC_TOLERANCE, weights=str(wm.MODEL_WEIGHTS.relative_to(REPO)))
json.dump(acc, open(TAB / 'frame_accounting.json', 'w'), indent=1); print(acc)
