#!/usr/bin/env python3
"""CA-10 weed-map tables in the CB-11 format (maps-results s0_coverage / s4_map_summary / s5_priority), both frame sets.

  post_weed.py <ca10_run dir>

Per set (frozen = stride 50, dense10 = stride 10), from weed/<set>/ (run_weed_ca10.py):
  coverage_<set>.npz   s0_coverage.py: every used frame, a 12 px pixel grid below the gate line projected to the
                       ground plane (pipeline plane / horizon / cos_phi / 5 m range test), count of frames per 0.5 m
                       ENU cell. Same grid and test; the projection is vectorised (identical maths to
                       wm.pixel_to_ground). Grid origin/extent fitted to CA-10 (CB-11 used a fixed CB-11 box).
  entities_<det>.csv   entities.entities (connected components eps 0.4 m == DBSCAN min_samples 1) of the
                       area-limit-ON map points (patch = 0), + d_row/row/off_row (geom.dist_to_rows on the CA-10
                       trunk rows) + views (coverage), as s4_map_summary.py
  detection_to_entity_<det>_eps0.4.csv
  patch_entities_<det>.csv  same components on the patch = 1 detections (area > 1 m^2; CA-10 definition, CB-11 has
                       no patch-entity table): n, centroid, median / max area
  priority_<det>.csv, priority_zones_<det>.csv   priority_lib.priority / zones, BASE = tau 1.0, R_d 1.5 (+ zone
                       defaults), trunks = trunks/trunks_final_ca10.csv east_m/north_m, as s5_priority.py
  map_summary.csv, counts.csv
Rows: one PCA line per trunk row (geom.fit_rows) on rows with >= 3 trunks (row 7 has 1 trunk).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(sys.argv[1])
sys.path.insert(0, '/home/afonso/viz_bundle_20261014/style')          # entities.py, priority_lib.py (pure functions)
sys.path.insert(0, '/home/afonso/thesis/orchard-weed-detection/map')
from entities import entities
from priority_lib import priority, zones, BASE
import weed_map_cached as wm

EPS = 0.4; CELL, STEP = 0.5, 12
wm.load_calibration(); wm.load_trajectory(str(R / 'traj/merged_final/keyframes.csv'))
wm.load_alignment(str(R / 'merge_final/align_A_to_enu.json'))
t = pd.read_csv(R / 'trunks/trunks_final_ca10.csv').rename(columns={'east_m': 'xe', 'north_m': 'ye'})
t = t[t.row_id.notna() & (t.row_id.astype(str) != '')]; t['row_id'] = t.row_id.astype(int)
rows = {}
for r, g in t.groupby('row_id'):                                       # geom.fit_rows
    if len(g) < 3: continue
    p = g[['xe', 'ye']].values; c = p.mean(0); vt = np.linalg.svd(p - c)[2]
    ax = vt[0] if vt[0][1] > 0 else -vt[0]; s = (p - c) @ ax
    rows[r] = dict(c=c, u=ax, n=np.array([-ax[1], ax[0]]), s0=s.min(), s1=s.max(), n_trunks=len(g))
T_tr = pd.read_csv(R / 'trunks/trunks_final_ca10.csv')[['east_m', 'north_m']].values


def dist_to_rows(xy, extend=2.0):                                      # geom.dist_to_rows
    best = np.full(len(xy), np.inf); rid = np.full(len(xy), -1); sg = np.zeros(len(xy))
    for r, Rw in rows.items():
        q = xy - Rw['c']; s = q @ Rw['u']; off = q @ Rw['n']
        ds = np.maximum(np.maximum(Rw['s0'] - extend - s, s - Rw['s1'] - extend), 0)
        d = np.hypot(off, ds); m = d < best; best[m], rid[m], sg[m] = d[m], r, off[m]
    return best, rid, sg


kf = pd.read_csv(R / 'traj/merged_final/keyframes.csv'); E_tr = kf[['x', 'y']].values @ wm.R_align.T + wm.t_align
x0, y0 = np.floor(E_tr.min(0) - 10); nx, ny = (np.ceil((E_tr.max(0) + 10 - [x0, y0]) / CELL)).astype(int)


def coverage(fr, lidar):
    cnt = np.zeros((ny, nx), np.int32)
    uu, vv = np.meshgrid(np.arange(0, 1280, STEP), np.arange(0, 720, STEP)); uu, vv = uu.ravel().astype(float), vv.ravel().astype(float)
    ray = wm.K_inv @ np.stack([uu, vv, np.ones_like(uu)]); ray /= np.linalg.norm(ray, axis=0)
    rd = wm.R_tilt @ (wm.T_lidar_cam[:3, :3] @ ray); ro = wm.R_tilt @ wm.T_lidar_cam[:3, 3]
    for _, r in fr.iterrows():
        a, b, c, d, xr, xt = wm.fit_ground(np.load(lidar / f'{r.stem}.npz')['xyz'])
        T = wm.get_pose(r.t); hz = wm.get_horizon((720, 1280, 3), xr, xt, a, b, c, d)
        denom = a * rd[0] + b * rd[1] + c * rd[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            tr = -(a * ro[0] + b * ro[1] + c * ro[2] + d) / denom
        ok = (vv >= hz[uu.astype(int)]) & (np.abs(denom) >= 1e-6) & (tr >= 0.1) & (np.abs(denom) / np.sqrt(a * a + b * b + c * c) >= wm.COS_PHI_MIN)
        pl = wm.R_tilt.T @ (ro[:, None] + tr * rd); pw = T[:3, :3] @ pl + T[:3, 3:4]
        ok &= np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) <= wm.MAX_WEED_RANGE
        e = pw[:2, ok].T @ wm.R_align.T + wm.t_align
        i, j = ((e[:, 0] - x0) / CELL).astype(int), ((e[:, 1] - y0) / CELL).astype(int)
        m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        for jj, ii in set(zip(j[m], i[m])): cnt[jj, ii] += 1
    return cnt


def cov_at(cnt, xy):
    i = ((xy[:, 0] - x0) / CELL).astype(int); j = ((xy[:, 1] - y0) / CELL).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny); out = np.zeros(len(xy), int); out[ok] = cnt[j[ok], i[ok]]; return out


allcounts = []
for S in ('frozen', 'dense10'):
    D = R / 'weed' / S; fr = pd.read_csv(D / 'frames.csv'); acc = json.load(open(D / 'frame_accounting.json'))
    cnt = coverage(fr, R / 'frames_stride10/lidar')
    np.savez_compressed(D / f'coverage_{S}.npz', cnt=cnt, x0=x0, y0=y0, cell=CELL)
    obs = int((cnt > 0).sum()) * CELL ** 2
    print(S, 'coverage', obs, 'm2', flush=True)
    summ = []
    for det in ('yolo', 'colour'):
        w = pd.read_csv(D / f'weed_map_{det}_cached.csv'); wp = pd.read_csv(D / f'weed_map_{det}_patches.csv')
        e, lab = entities(w, EPS); e['det'] = det
        d, rid, sg = dist_to_rows(e[['x', 'y']].values); e['d_row'], e['row'], e['off_row'] = d, rid, sg
        e['views'] = cov_at(cnt, e[['x', 'y']].values); e.to_csv(D / f'entities_{det}.csv', index=False)
        pd.DataFrame(dict(row=np.arange(len(w)), stem=w.stem, eid=lab)).to_csv(D / f'detection_to_entity_{det}_eps0.4.csv', index=False)
        pw_ = wp[wp.patch == 1].reset_index(drop=True)
        if len(pw_):
            pe, plab = entities(pw_, EPS); pe['det'] = det; pe.to_csv(D / f'patch_entities_{det}.csv', index=False)
        else:
            pe = pd.DataFrame(columns=['eid', 'x', 'y', 'n_det', 'area_med', 'area_max']); pe.to_csv(D / f'patch_entities_{det}.csv', index=False)
        xy, size = e[['x', 'y']].values, e.area_med.values
        base = priority(xy, size, T_tr); Z, hot = zones(xy, base['pri'])
        out = e.copy()
        for k in ('pri', 'd_trunk', 'prox', 'dens', 'dens_n', 'size_n'): out[k if k != 'pri' else 'priority'] = base[k]
        out['rank'] = (-base['pri']).argsort().argsort(); out['in_zone'] = 0
        for zi, m in enumerate(Z, 1): out.loc[m, 'in_zone'] = zi
        out['top10'] = out['rank'] < len(out) * 0.1; out.to_csv(D / f'priority_{det}.csv', index=False)
        pd.DataFrame([dict(zone=zi, n=len(m), cx=xy[m, 0].mean(), cy=xy[m, 1].mean(), priority_sum=base['pri'][m].sum(),
                           priority_mean=base['pri'][m].mean(), area_m2=size[m].sum()) for zi, m in enumerate(Z, 1)],
                     columns=['zone', 'n', 'cx', 'cy', 'priority_sum', 'priority_mean', 'area_m2']).to_csv(D / f'priority_zones_{det}.csv', index=False)
        summ.append(dict(detector=det, frames=acc['frames_used'], map_points=len(w), entities=len(e),
                         pct_seen_once_frame=100 * (e.n_frames == 1).mean(), pct_one_detection=100 * (e.n_det == 1).mean(),
                         total_area_m2=e.area_med.sum(), total_area_max_m2=e.area_max.sum(), median_area_m2=e.area_med.median(),
                         trunks=len(T_tr), rows=len(rows), observed_ground_m2=obs, entities_per_100m2=100 * len(e) / obs,
                         cover_pct_area=100 * e.area_med.sum() / obs))
        top = out[out.top10]
        allcounts.append(dict(set=S, detector=det if det == 'yolo' else 'colour_tuned', frames=acc['frames_used'],
                              map_points=len(w), patch_detections=len(pw_), entities=len(e), patch_entities=len(pe),
                              total_area_m2=float(e.area_med.sum()), patch_entities_area_m2=float(pe.area_med.sum()) if len(pe) else 0.0,
                              top10_priority_entities=int(out.top10.sum()), top10_area_m2=float(top.area_med.sum()),
                              zones=len(Z), observed_ground_m2=obs))
        print(allcounts[-1], flush=True)
    pd.DataFrame(summ).to_csv(D / 'map_summary.csv', index=False)
pd.DataFrame(allcounts).to_csv(R / 'weed/counts.csv', index=False)
json.dump(dict(eps=EPS, priority=BASE, rows_used=sorted(int(k) for k in rows), coverage_grid=dict(x0=float(x0), y0=float(y0), nx=int(nx), ny=int(ny), cell=CELL, step_px=STEP)),
          open(R / 'weed/post_params.json', 'w'), indent=1)
