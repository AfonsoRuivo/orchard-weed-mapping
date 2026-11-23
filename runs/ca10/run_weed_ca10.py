#!/usr/bin/env python3
"""CA-10 weed maps: CB-11 colour_tuned/run_tuned.py adapted (same frame loop as weed_map_cached.main / s1_freeze.py,
same calibration, ground plane, 3 deg tilt limit, line gate, COS_PHI_MIN 0.10, MAX_WEED_RANGE 5 m, YOLO v2 @0.30
960 px, colour TUNED = colour_detector.TUNED; detections with area_m2 > AREA_M2_MAX (1 m^2) kept and flagged
patch=1, as run_tuned.py). All functions and constants come from the unchanged weed_map_cached.py / colour_detector.py.

Differences from run_tuned.py (CA-10 inputs, nothing else):
  * trajectory = traj/merged_final/keyframes.csv (three-piece merge, A's frame); alignment = merge_final/align_A_to_enu.json
  * frames: stride-10 extraction of this bag (extract_frames_ca10.py); set 'frozen' = the stride-50 subset
    (camera index % 50 == 0, the CB-11 frozen stride), 'dense10' = all;
  * frames whose image time or matched LiDAR time is inside a window without poses (805.5-839.5 s, 966-975 s) are
    skipped and counted (the trajectory interpolation would otherwise bridge the windows);
  * no reproduction asserts against CB-11 tables; colour ORIGINAL not run.
Outputs in weed/<set>/: weed_map_{yolo,colour}_cached.csv (area limit ON = patch=0 rows, the 14 frozen columns),
weed_map_{yolo,colour}_patches.csv (all rows + patch,x1,y1,x2,y2), frames.csv, frame_accounting.json, filter_table.csv.

    python3 -B run_weed_ca10.py <ca10_run dir> --set frozen|dense10
"""
import argparse, csv, json, re, sys, time, hashlib
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path('/home/afonso/thesis')
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map/results'))
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map'))
import weed_map_cached as wm
import colour_detector as cd

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--set', choices=['frozen', 'dense10'], required=True)
ap.add_argument('--frames', default='frames_stride10', help='frame folder (images/, lidar/manifest.csv) under the run dir')
ap.add_argument('--outname', default='weed', help='output folder under the run dir (default weed/<set>)')
args = ap.parse_args(); R = Path(args.run)
T0 = 1782577258.896306
NOPOSE = [(805.5, 839.5), (966.0, 975.0)]                      # the brief's windows without poses
# the pose lookup interpolates the merged KEYFRAMES, whose spans across piece boundaries are wider than NOPOSE:
# A's last keyframe -> B-pre's first, B-pre's last -> B-post's first (frames there would be posed across a piece
# boundary, the second one across the 5.2 m alley jump). Skipped too and counted separately (advisor review).
KFSPAN = [(802.226, 845.403), (964.517, 975.511)]
IMG, LID = R / args.frames / 'images', R / args.frames / 'lidar'
OUT = R / args.outname / args.set; OUT.mkdir(parents=True, exist_ok=True)
TRAJ = R / 'traj/merged_final/keyframes.csv'; ALIGN = R / 'merge_final/align_A_to_enu.json'
wm.load_calibration(); wm.load_trajectory(str(TRAJ)); wm.load_alignment(str(ALIGN))
COLS = ['x_enu', 'y_enu', 'x_slam', 'y_slam', 'timestamp', 'area_px', 'area_m2', 'confidence', 'detector', 'stem',
        'u', 'v', 'range_m', 'cos_phi', 'patch', 'x1', 'y1', 'x2', 'y2']
STAGES = ['raw', 'above_horizon_or_gate', 'small_blob', 'no_ground_hit', 'range_gt_5m', 'area_gt_1m2_kept_as_patch', 'kept']

man_all = list(csv.DictReader(open(LID / 'manifest.csv')))
if args.set == 'frozen':
    man_all = [r for r in man_all if int(re.match(r'frame_(\d+)_', r['stem']).group(1)) % 50 == 0]
acc = dict(set=args.set, manifest_rows=len(man_all))
inwin = lambda tb: any(a <= tb <= b for a, b in NOPOSE)
skipped = {f'{a}-{b}': [] for a, b in NOPOSE}
man = []
for r in man_all:
    tb = [float(r['target_ts']) - T0, float(r['matched_ts']) - T0]
    w = next((f'{a}-{b}' for a, b in NOPOSE if any(a <= x <= b for x in tb)), None)
    if w: skipped[w].append(r['stem']); continue
    w = next((f'kfspan_{a}-{b}' for a, b in KFSPAN if any(a < x < b for x in tb)), None)
    if w: skipped.setdefault(w, []).append(r['stem']); continue
    man.append(r)
acc['skipped_no_pose'] = {k: len(v) for k, v in skipped.items()}
acc['skipped_note'] = ('805.5-839.5 / 966.0-975.0 = brief windows; kfspan_* = extra frames between the bracketing '
                       'keyframes of two different pieces (pose would be interpolated across a piece boundary)')
acc['skipped_no_pose_stems'] = {k: ';'.join(v) for k, v in skipped.items()}
m2 = [r for r in man if abs(float(r['dt'])) <= wm.SYNC_TOLERANCE]
acc['sync_dropped_dt_gt_0.15s'] = len(man) - len(m2)
acc['sync_dropped_stems'] = ';'.join(r['stem'] for r in man if abs(float(r['dt'])) > wm.SYNC_TOLERANCE)
cnt0 = dict(out_of_traj=0, ground_failed=0, implausible_plane=0, no_files=0)
t0 = time.time(); frames, fr_rows = [], []
for r in m2:
    stem, lt = r['stem'], float(r['matched_ts'])
    ip, lp = IMG / f'{stem}.jpg', LID / f'{stem}.npz'
    if not ip.exists() or not lp.exists(): cnt0['no_files'] += 1; continue
    if lt < wm.traj_t[0] or lt > wm.traj_t[-1]: cnt0['out_of_traj'] += 1; continue
    try: a, b, c, d, xr, xt = wm.fit_ground(np.load(lp)['xyz'])
    except Exception: cnt0['ground_failed'] += 1; continue
    if not (wm.Z_MIN < -d / c < wm.Z_MAX): cnt0['implausible_plane'] += 1; continue
    frames.append((stem, ip, lt, (a, b, c, d), xr, xt))
    T = wm.get_pose(lt); n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    fr_rows.append(dict(stem=stem, t=lt, a=a, b=b, c=c, d=d, tilt_deg=float(np.degrees(np.arccos(n[2]))),
                        height=float(-d / c), x_slam=T[0, 3], y_slam=T[1, 3], yaw_slam=float(np.arctan2(T[1, 0], T[0, 0]))))
acc.update(cnt0); acc['frames_used'] = len(frames)
acc['closes'] = acc['manifest_rows'] - sum(v for k, v in acc['skipped_no_pose'].items()) - acc['sync_dropped_dt_gt_0.15s'] - sum(cnt0.values()) == len(frames)
pd.DataFrame(fr_rows).to_csv(OUT / 'frames.csv', index=False)
print(args.set, json.dumps({k: v for k, v in acc.items() if 'stems' not in k}), f'ground fit {time.time() - t0:.0f} s', flush=True)


def colour(img, hz, p):
    dets, m = cd.detect(img, hz, p, return_mask=True)
    n = len(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    return [(x1 + (x2 - x1) / 2, y2 - 1, a, float('nan'), x1, y1, x2, y2) for x1, y1, x2, y2, a in dets], n


RUNS = {'colour': cd.TUNED, 'yolo': None}
cnt = {k: dict.fromkeys(STAGES, 0) for k in RUNS}; rows = {k: [] for k in RUNS}; tdet = dict.fromkeys(RUNS, 0.0)
t1 = time.time()
for i, (stem, ip, lt, pl, xr, xt) in enumerate(frames):
    img = cv2.imread(str(ip)); T = wm.get_pose(lt); hz = wm.get_horizon(img.shape, xr, xt, *pl)
    for k, p in RUNS.items():
        ts = time.time(); C = cnt[k]
        if p is None:
            ok, rej = wm.get_weed_boxes(img, hz, return_debug=True)
            C['raw'] += len(ok) + len(rej); C['above_horizon_or_gate'] += len(rej)
        else:
            ok, n = colour(img, hz, p); C['raw'] += n; C['small_blob'] += n - len(ok)
        for u, v, area_px, conf, x1, y1, x2, y2 in ok:
            pw, s, geo = wm.pixel_to_ground(u, v, *pl, T, return_geometry=True)
            if pw is None: C['no_ground_hit'] += 1; continue
            if np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE: C['range_gt_5m'] += 1; continue
            area_m2 = area_px * s; patch = int(area_m2 > wm.AREA_M2_MAX)
            C['area_gt_1m2_kept_as_patch' if patch else 'kept'] += 1
            xe, ye = wm.slam_to_enu(pw[0], pw[1])
            rows[k].append((xe, ye, pw[0], pw[1], lt, area_px, area_m2, conf, k, stem, u, v, geo['range_m'], geo['cos_phi'],
                            patch, x1, y1, x2, y2))
        tdet[k] += time.time() - ts
    if i % 200 == 0: print(f'  {i}/{len(frames)} {time.time() - t1:.0f} s', flush=True)
table = []
for k in RUNS:
    with open(OUT / f'weed_map_{k}_patches.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows[k])
    with open(OUT / f'weed_map_{k}_cached.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS[:14]); w.writerows([r[:14] for r in rows[k] if not r[14]])
    C = dict(cnt[k]); C['csv_rows'] = len(rows[k]); C['detector'] = k if k == 'yolo' else 'colour_tuned'
    C['det_time_s'] = round(tdet[k], 1); table.append(C); print(C, flush=True)
pd.DataFrame(table).to_csv(OUT / 'filter_table.csv', index=False)
wpath = wm.MODEL_WEIGHTS
acc['params'] = dict(MAX_PLANE_TILT_DEG=wm.MAX_PLANE_TILT_DEG, HORIZON_MODEL=wm.HORIZON_MODEL, CONF_THRESH=wm.CONF_THRESH,
                     YOLO_IMGSZ=wm.YOLO_IMGSZ, MAX_WEED_RANGE=wm.MAX_WEED_RANGE, AREA_M2_MAX=wm.AREA_M2_MAX,
                     PERCENTILE=wm.PERCENTILE, H_WEED=wm.H_WEED, SYNC_TOLERANCE=wm.SYNC_TOLERANCE, COS_PHI_MIN=wm.COS_PHI_MIN,
                     colour=cd.TUNED, weights=str(wpath), weights_sha256=hashlib.sha256(open(wpath, 'rb').read()).hexdigest(),
                     calib=str(wm.CALIB), T_lidar_camera=wm.T_lidar_cam.tolist(), K=wm.K.tolist(), dist=wm.dist.tolist(),
                     camera_time_offset_s=0.0, image_time='bag receive time (t0 + rel_t from the stem, 10 ms rounding)',
                     pose_time='matched LiDAR scan receive time (manifest matched_ts), as weed_map_cached.main',
                     traj=str(TRAJ), align=str(ALIGN))
acc['time_total_s'] = round(time.time() - t0, 1)
json.dump(acc, open(OUT / 'frame_accounting.json', 'w'), indent=1, default=str)
print(f'done {time.time() - t0:.0f} s')
