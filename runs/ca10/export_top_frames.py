#!/usr/bin/env python3
"""Camera frames behind the top-10 % priority entities (CA-10, frozen run), per detector. No judgement of content.

  export_top_frames.py <ca10_run dir>  ->  <ca10_run>/top_priority_frames/

Entities: connected components (eps 0.4 m, entities.entities) over ALL detections with the area limit off
(weed/frozen/weed_map_<det>_patches.csv: patch = 0 and 1), as in the CB-11 colour_tuned patch analysis. Priority:
priority_lib.priority with BASE (tau 1.0, R_d 1.5), trunks = trunks/trunks_final_ca10.csv; top 10 % =
rank < 0.1 n (s5_priority.py). For every frame with a detection in a top entity: the stored upright JPEG
(frames_stride10/images, rotated 180 deg at extraction, as CB-11) with each such detection drawn:
  YOLO   the xyxy box (red if patch, blue if not), conf, entity id
  colour the outline of its gated TUNED-mask component (colour_detector.detect with the frame's height gate; the
         component whose boundingRect equals the stored box), its box, entity id
  both   magenta dot = contact pixel (u, v); yellow line = height-gate boundary
Outputs: <det>/frames/<stem>.jpg, top_frames_<det>.csv (eid, detector, priority, rank, patch, stem, x1..y2,
area_px, area_m2, range_m, cos_phi, u, v, conf, x_enu, y_enu), top_entities_<det>.csv, contact_sheet_<det>_<k>.jpg
(4 x 4 tiles of 480 x 270 px, title = eid / stem).
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

R = Path(sys.argv[1]); O = R / 'top_priority_frames'; O.mkdir(exist_ok=True)
sys.path.insert(0, '/home/afonso/viz_bundle_20261014/style')
sys.path.insert(0, '/home/afonso/thesis/orchard-weed-detection/map')
sys.path.insert(0, '/home/afonso/thesis/orchard-weed-detection/map/results')
from entities import entities
from priority_lib import priority, BASE
import weed_map_cached as wm
import colour_detector as cd
wm.load_calibration(); wm.load_trajectory(str(R / 'traj/merged_final/keyframes.csv'))
wm.load_alignment(str(R / 'merge_final/align_A_to_enu.json'))
IMG, LID = R / 'frames_stride10/images', R / 'frames_stride10/lidar'
T_tr = pd.read_csv(R / 'trunks/trunks_final_ca10.csv')[['east_m', 'north_m']].values
summary = {}
for det in ('yolo', 'colour'):
    D = O / det / 'frames'; D.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(R / f'weed/frozen/weed_map_{det}_patches.csv')
    e, lab = entities(w, 0.4); w['eid'] = lab
    p = priority(e[['x', 'y']].values, e.area_med.values, T_tr)
    e['priority'] = p['pri']; e['rank'] = (-p['pri']).argsort().argsort(); e['top10'] = e['rank'] < len(e) * 0.1
    e['n_patch_det'] = [int(w.patch[w.eid == k].sum()) for k in e.eid]
    e.to_csv(O / f'top_entities_{det}.csv', index=False)
    top = e[e.top10].sort_values('rank')
    rows = w[w.eid.isin(top.eid)].merge(top[['eid', 'priority', 'rank']], on='eid')
    out = rows[['eid', 'detector', 'priority', 'rank', 'patch', 'stem', 'x1', 'y1', 'x2', 'y2', 'area_px', 'area_m2', 'range_m',
                'cos_phi', 'u', 'v', 'confidence', 'x_enu', 'y_enu']].rename(columns={'confidence': 'conf'})
    out['detector'] = det if det == 'yolo' else 'colour_tuned'
    out.sort_values(['rank', 'stem']).to_csv(O / f'top_frames_{det}.csv', index=False)
    tiles, unmatched = [], []
    for stem, g in rows.groupby('stem'):
        img = cv2.imread(str(IMG / f'{stem}.jpg'))
        a, b, c, d, xr, xt = wm.fit_ground(np.load(LID / f'{stem}.npz')['xyz'])
        hz = wm.get_horizon(img.shape, xr, xt, a, b, c, d)
        if det == 'colour':                      # on the clean image, before anything is drawn
            _, m = cd.detect(img.copy(), hz, cd.TUNED, return_mask=True)
            conts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            byrect = {}
            for c_ in conts:
                x, y, ww, hh = cv2.boundingRect(c_); byrect[(x, y, x + ww, y + hh)] = c_
        cv2.polylines(img, [np.c_[np.arange(img.shape[1]), hz].astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1)
        for _, r in g.iterrows():
            col = (0, 0, 255) if r.patch else (255, 128, 0)
            x1, y1, x2, y2 = (int(round(v)) for v in (r.x1, r.y1, r.x2, r.y2))
            if det == 'colour':
                c_ = byrect.get((x1, y1, x2, y2))
                if c_ is not None: cv2.drawContours(img, [c_], -1, (0, 255, 0), 2)
                else: unmatched.append(stem)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            lbl = f"e{r.eid} P{r.priority:.1f}" + (f" c{r.confidence:.2f}" if det == 'yolo' else '') + (' patch' if r.patch else '')
            cv2.putText(img, lbl, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.circle(img, (int(r.u), int(r.v)), 5, (255, 0, 255), -1)
        cv2.imwrite(str(D / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t = cv2.resize(img, (480, 270)); bar = np.full((22, 480, 3), 255, np.uint8)
        cv2.putText(bar, f"e{','.join(str(x) for x in sorted(g.eid.unique()))} | {stem}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append((int(g['rank'].min()), np.vstack([bar, t])))
    tiles.sort(key=lambda x: x[0]); tiles = [t for _, t in tiles]
    for k in range(0, len(tiles), 16):
        chunk = tiles[k:k + 16] + [np.full_like(tiles[0], 255)] * (16 - len(tiles[k:k + 16]))
        sheet = np.vstack([np.hstack(chunk[i:i + 4]) for i in range(0, 16, 4)])
        cv2.imwrite(str(O / f'contact_sheet_{det}_{k // 16 + 1}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary[det] = dict(entities=len(e), top10_entities=int(e.top10.sum()), top10_patch_entities=int((top.n_patch_det > 0).sum()),
                        detections=len(rows), frames=int(rows.stem.nunique()), sheets=(len(tiles) + 15) // 16,
                        colour_mask_unmatched=len(unmatched))
    print(det, summary[det], flush=True)
json.dump(dict(summary=summary, eps=0.4, priority=BASE, entities_over='all detections, area limit off (patch 0 and 1)'),
          open(O / 'summary.json', 'w'), indent=1)
