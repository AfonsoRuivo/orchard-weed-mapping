#!/usr/bin/env python3
"""
colour_detector.py - parametrised colour weed detector + cached frame sets for tuning and scoring.

detect(img, horizon, p) is the colour branch of weed_map_cached.get_colour_boxes with
its constants in a dict p, so variants can be compared on the same frames:

    p['index']  'hsv_exg'  [H in range, S >= s_min, V >= v_min] AND [2G-R-B > tau]   (Ch5, original)
                'exg'      2G-R-B > tau only
                'exgr'     ExG - ExR > tau on chromatic coords (Meyer & Neto 2008), x 100
                'cive'     -(CIVE) > tau  (Kataoka et al. 2003)
                'lab_a'    -(a* - 128) > tau  (OpenCV 8-bit Lab, green is a* < 128)
              any index can be combined with the HSV condition by p['hsv']=True
    p['open'], p['dilate']  square kernel sizes (0 = skip)
    p['a_min']  minimum component area (px), measured after the gate
    p['gate']   True: mask pixels above the LiDAR boundary are removed before components

load_set(name) returns a list of frames with image, labels (weed + weed_cluster),
and the gate boundary for ground percentiles 5..25 (None if the plane is implausible),
all computed with weed_map_cached.py and cached as .npz next to the set.
"""
import csv, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weed_map_cached as wm

REPO = HERE.parents[1]
SETS = REPO / 'data_set' / 'weed_labelling_20261008'
LIDAR = REPO / 'data_set' / 'campinas_b' / 'lidar'
PCTS = [5, 10, 15, 20, 25]
WEED_IDS = {0, 2}

ORIGINAL = dict(index='hsv_exg', h_lo=wm.HSV_H_LOW, h_hi=wm.HSV_H_HIGH, s_min=wm.HSV_S_MIN, v_min=wm.HSV_V_MIN,
                tau=wm.EXG_THRESH, open=3, dilate=3, a_min=wm.MIN_BLOB, gate=True)
# Tuned on tune_T (colour_tune.py): only two constants changed. H_lo 14 -> 36 (28 deg -> 72 deg) drops the
# yellow of dry grass and straw; A_min 300 -> 2000 px drops the speckle of shadowed leaf litter.
TUNED = dict(ORIGINAL, h_lo=36, a_min=2000)


def index_mask(img, p):
    f = img.astype(np.float32); B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    k = p['index']
    if k in ('hsv_exg', 'exg'):
        m = (2 * G - R - B) > p['tau']
    elif k == 'exgr':
        s = R + G + B + 1e-6; r, g, b = R / s, G / s, B / s
        m = 100 * ((2 * g - r - b) - (1.4 * r - g)) > p['tau']
    elif k == 'cive':
        m = -(0.441 * R - 0.811 * G + 0.385 * B + 18.78745) > p['tau']
    elif k == 'lab_a':
        a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
        m = (128 - a) > p['tau']
    else:
        raise ValueError(k)
    if k == 'hsv_exg' or p.get('hsv'):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m &= cv2.inRange(hsv, np.array([p['h_lo'], p['s_min'], p['v_min']]), np.array([p['h_hi'], 255, 255])) > 0
    return m.astype(np.uint8) * 255


def mask_of(img, p):
    m = index_mask(img, p)
    if p.get('open'):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((p['open'], p['open']), np.uint8))
    if p.get('dilate'):
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, np.ones((p['dilate'], p['dilate']), np.uint8))
    return m


def detect(img, horizon, p, return_mask=False):
    """Components as (x1, y1, x2, y2, area), largest first."""
    m = mask_of(img, p)
    if p.get('gate', True) and horizon is not None:
        m[np.arange(m.shape[0])[:, None] < horizon[None, :]] = 0
    conts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in conts:
        a = float(cv2.contourArea(c))
        if a <= p['a_min']:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((x, y, x + w, y + h, a))
    out.sort(key=lambda d: -d[4])
    return (out, m) if return_mask else out


def load_labels(path, w, h):
    out = []
    if path.exists():
        for line in path.read_text().split('\n'):
            q = line.split()
            if len(q) >= 5 and int(q[0]) in WEED_IDS:
                cx, cy, bw, bh = map(float, q[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def load_set(name, traj=None):
    """Frames of a labelling set with their gate boundary at every percentile (cached)."""
    d = SETS / name
    cache = d / f'_gate_cache_tilt{wm.MAX_PLANE_TILT_DEG}_{wm.HORIZON_MODEL}.npz'
    stems = (d / 'frames.txt').read_text().split()
    if cache.exists():
        z = np.load(cache, allow_pickle=True); hz = z['hz'].item(); planes = z['planes'].item()
    else:
        hz, planes = {}, {}
        for st in stems:
            img = cv2.imread(str(d / 'images' / f'{st}.jpg'))
            a, b, c, dd, xr, xt, dbg = wm.fit_ground(np.load(LIDAR / f'{st}.npz')['xyz'], return_debug=True)
            a0, b0, c0, d0 = dbg['plane_before_offset']; nl = np.linalg.norm([a0, b0, c0])
            for P in PCTS:
                dP = d0 - np.percentile(dbg['candidate_heights'], P) * nl
                ok = wm.Z_MIN < -dP / c0 < wm.Z_MAX
                hz[(st, P)] = wm.get_horizon(img.shape, xr, xt, a0, b0, c0, dP) if ok else None
                planes[(st, P)] = (a0, b0, c0, dP) if ok else None
        np.savez(cache, hz=np.array(hz, dtype=object), planes=np.array(planes, dtype=object))
    frames = []
    for st in stems:
        img = cv2.imread(str(d / 'images' / f'{st}.jpg')); h, w = img.shape[:2]
        frames.append(dict(stem=st, img=img, gts=load_labels(d / 'labels' / f'{st}.txt', w, h),
                           hz={P: hz[(st, P)] for P in PCTS}, plane={P: planes[(st, P)] for P in PCTS}))
    return frames
