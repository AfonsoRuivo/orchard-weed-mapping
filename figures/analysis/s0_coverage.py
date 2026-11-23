#!/usr/bin/env python3
"""Ground coverage of the map run: for every used frame, project a pixel grid below the height-gate boundary to the
ground plane (same plane / horizon / range limit as the weed pipeline) and count, per 0.5 m ENU cell, how many frames
saw it.  -> tables/coverage_<run>.npz (grid origin, cell, counts).  Used for weeds per m^2 / per 10 m of row and
for greying the unobserved part of the map."""
import sys, csv
import cv2, numpy as np
from common import *
import weed_map_cached as wm
run = sys.argv[1] if len(sys.argv) > 1 else 'frozen'
from sigma_lib import RUNS
fr = pd.read_csv(RUNS[run][1]); imgdir = Path(sys.argv[2]) if len(sys.argv) > 2 else wm.IMAGES_DIR
lidar = Path(sys.argv[3]) if len(sys.argv) > 3 else LIDAR
wm.load_calibration(); wm.load_trajectory(str(TRAJ)); wm.load_alignment(str(FROZEN / 'align_rep1.json'))
CELL, STEP = 0.5, 12
x0, y0, nx, ny = -110.0, -130.0, 220, 260
cnt = np.zeros((ny, nx), np.int32)
uu, vv = np.meshgrid(np.arange(0, 1280, STEP), np.arange(0, 720, STEP)); uu, vv = uu.ravel(), vv.ravel()
for _, r in fr.iterrows():
    xyz = np.load(lidar / f'{r.stem}.npz')['xyz']
    a, b, c, d, xr, xt = wm.fit_ground(xyz)
    T = wm.get_pose(r.t); hz = wm.get_horizon((720, 1280, 3), xr, xt, a, b, c, d)
    seen = set()
    for u, v in zip(uu, vv):
        if v < hz[u]: continue
        pw, _ = wm.pixel_to_ground(u, v, a, b, c, d, T)
        if pw is None or np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE: continue
        e = wm.slam_to_enu(pw[0], pw[1]); i, j = int((e[0] - x0) / CELL), int((e[1] - y0) / CELL)
        if 0 <= i < nx and 0 <= j < ny: seen.add((j, i))
    for j, i in seen: cnt[j, i] += 1
np.savez_compressed(TAB / f'coverage_{run}.npz', cnt=cnt, x0=x0, y0=y0, cell=CELL)
print(run, 'covered cells', int((cnt > 0).sum()), 'area m2', (cnt > 0).sum() * CELL ** 2, 'mean views', cnt[cnt > 0].mean())
