#!/usr/bin/env python3
"""CA-10 copy of CB-11 patch_export/export_patches.py: weed PATCHES (A > AREA_M2_MAX) and the camera ground footprint.
Header and geometry helpers copied verbatim; only inputs, the colour detector (TUNED, colour_detector.detect) and the
frame list (CA-10 frozen set, weed/frozen/frames.csv) differ; no CB-11 table checks, no contact sheets.

Original docstring:
Export the frozen run's area_limit detections (A > AREA_M2_MAX) as weed PATCHES.

Read-only with respect to the thesis repo: imports weed_map_cached (wm) for every
function and parameter, re-runs the s1_freeze.py frame loop (copied, not imported:
s1_freeze writes tables/frames.csv at import), writes only to ~/patch_export.

Run:  PYTHONDONTWRITEBYTECODE=1 python3 export_patches.py
"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path('/home/afonso/thesis')
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map'))
import weed_map_cached as wm   # noqa: E402

OUT = Path.home() / 'ca10_run/weed/frozen/patches'
OUT.mkdir(parents=True, exist_ok=True)
TAB = Path.home() / 'ca10_run/weed/frozen'
TRAJ = Path.home() / 'ca10_run/traj/merged_final/keyframes.csv'
LIDAR = Path.home() / 'ca10_run/frames_stride10/lidar'
IMAGES = Path.home() / 'ca10_run/frames_stride10/images'
DENSIFY_PX = 10.0
APPROX_EPS = 2.0
ERODE_PX = 2          # valid mask eroded so simplified/densified vertices still project

# exactly the frozen run's setup (s1_freeze.py): calibration (with manual_tweaks), rep1 keyframes, align_rep1
wm.load_calibration(wm.CALIB)
wm.load_trajectory(str(TRAJ))
wm.load_alignment(str(Path.home() / 'ca10_run/merge_final/align_A_to_enu.json'))
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map/results'))
import colour_detector as cd   # noqa: E402
YAW_ALIGN = float(np.arctan2(wm.R_align[1, 0], wm.R_align[0, 0]))


# ---------------------------------------------------------------- geometry helpers
def valid_mask(shape, horizon, plane, T):
    """Per-pixel copy of the pipeline's acceptance region: below the gate (v >= horizon[u]),
    ray hits the plane (|denom| >= 1e-6, t_r >= 0.1, cos_phi >= COS_PHI_MIN) and the ground
    point is <= MAX_WEED_RANGE from the robot (horizontal, map frame), as in main()/s1_freeze.
    Vectorised copy of wm.pixel_to_ground; spot-checked against it below."""
    h, w = shape[:2]
    a, b, c, d = plane
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)])
    ray = wm.K_inv @ pix
    ray /= np.linalg.norm(ray, axis=0)
    rd = wm.R_tilt @ (wm.T_lidar_cam[:3, :3] @ ray)
    ro = wm.R_tilt @ wm.T_lidar_cam[:3, 3]
    denom = a * rd[0] + b * rd[1] + c * rd[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t_r = -(a * ro[0] + b * ro[1] + c * ro[2] + d) / denom
    nl = np.sqrt(a * a + b * b + c * c)
    cos_phi = np.abs(denom) / nl
    ok = (np.abs(denom) >= 1e-6) & (t_r >= 0.1) & (cos_phi >= wm.COS_PHI_MIN)
    p_l = wm.R_tilt.T @ (ro[:, None] + t_r * rd)
    pw = T[:3, :3] @ p_l + T[:3, 3:4]
    rng = np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3])
    ok &= rng <= wm.MAX_WEED_RANGE
    ok = ok.reshape(h, w)
    ok &= vv >= horizon[None, :]
    return ok


def spot_check(m, plane, T, horizon, rng, n=300):
    h, w = m.shape
    us, vs = rng.integers(0, w, n), rng.integers(0, h, n)
    bad = 0
    for u, v in zip(us, vs):
        pw, _ = wm.pixel_to_ground(float(u), float(v), *plane, T)
        ref = (pw is not None and np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) <= wm.MAX_WEED_RANGE
               and v >= horizon[u])
        bad += ref != bool(m[v, u])
    return bad


def densify(poly, step=DENSIFY_PX):
    """Closed polygon (N,2) -> vertices with spacing <= step px (original vertices kept)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        L = float(np.hypot(*(q - p)))
        k = max(1, int(np.ceil(L / step)))
        for j in range(k):
            out.append(p + (q - p) * j / k)
    return np.array(out)


def shoelace(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def project(poly_px, plane, T):
    """Project vertices with wm.pixel_to_ground. Returns list of rows and the count of vertices
    that did not project (should be 0)."""
    rows, miss = [], 0
    for u, v in poly_px:
        pw, s, geo = wm.pixel_to_ground(float(u), float(v), *plane, T, return_geometry=True)
        if pw is None:
            miss += 1; continue
        xe, ye = wm.slam_to_enu(pw[0], pw[1])
        rows.append((float(u), float(v), pw[0], pw[1], xe, ye, geo['range_m'], geo['cos_phi']))
    return rows, miss


def clip_outline(outline_px, vmask_er, shape):
    """Raster clip: fill the outline, AND with the (eroded) valid mask, outer contours,
    approxPolyDP(eps 2). Returns list of parts (each (N,2) float), largest first."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(outline_px).astype(np.int32).reshape(-1, 1, 2)], 255)
    m &= vmask_er
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c_ in cs:
        ap = cv2.approxPolyDP(c_, APPROX_EPS, True).reshape(-1, 2).astype(float)
        if len(ap) < 3:
            # degenerate (1-2 px wide sliver): keep the raw contour if it has an area
            ap = c_.reshape(-1, 2).astype(float)
            if len(ap) < 3:
                continue
        parts.append(ap)
    parts.sort(key=lambda p: -cv2.contourArea(p.astype(np.float32)))
    return parts


# ---------------------------------------------------------------- frame list: the CA-10 frozen run (run_weed_ca10.py)
fr_ref = pd.read_csv(TAB / 'frames.csv')
frames, plane_dev = [], 0.0
for _, r in fr_ref.iterrows():
    a, b, c, d, xr, xt = wm.fit_ground(np.load(LIDAR / f'{r.stem}.npz')['xyz'])
    plane_dev = max(plane_dev, float(np.max(np.abs(np.array([a, b, c, d]) - r[['a', 'b', 'c', 'd']].values.astype(float)))))
    frames.append((r.stem, IMAGES / f'{r.stem}.jpg', float(r.t), (a, b, c, d), xr, xt))
check = dict(frames_used=len(frames), plane_max_abs_diff_vs_frames_csv=plane_dev, colour=cd.TUNED)
print(check, flush=True)

patches, verts, foot = [], [], []
counts = {det: dict(raw=0, gate=0, small_blob=0, no_ground_hit=0, range_gt_5m=0, area_gt_1m2=0, kept=0) for det in ('yolo', 'colour')}
per_frame_area, spot_bad, miss_total, pid = [], 0, 0, 0
srng = np.random.default_rng(1); t0 = time.time()
for i, (stem, ip, lt, pl, xr, xt) in enumerate(frames):
    img = cv2.imread(str(ip)); T = wm.get_pose(lt)
    hz = wm.get_horizon(img.shape, xr, xt, *pl)
    vm = valid_mask(img.shape, hz, pl, T)
    spot_bad += spot_check(vm, pl, T, hz, srng)
    vm_er = cv2.erode(vm.astype(np.uint8) * 255, np.ones((2 * ERODE_PX + 1,) * 2, np.uint8))
    xe_r, ye_r = wm.slam_to_enu(T[0, 3], T[1, 3])
    yaw_enu = float(np.arctan2(T[1, 0], T[0, 0])) + YAW_ALIGN
    yaw_enu = float(np.arctan2(np.sin(yaw_enu), np.cos(yaw_enu)))
    cs, _ = cv2.findContours(vm_er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        big = max(cs, key=cv2.contourArea)
        fp = densify(cv2.approxPolyDP(big, APPROX_EPS, True).reshape(-1, 2).astype(float))
        rows, miss = project(fp, pl, T); miss_total += miss
        for k, rw in enumerate(rows):
            foot.append((stem, k, rw[4], rw[5]))
    dets = []
    ok, rej = wm.get_weed_boxes(img, hz, return_debug=True)
    counts['yolo']['raw'] += len(ok) + len(rej); counts['yolo']['gate'] += len(rej)
    for (u, v, apx, conf, x1, y1, x2, y2) in ok:
        dets.append(('yolo', u, v, apx, conf, (x1, y1, x2, y2), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)))
    cdet, m = cd.detect(img, hz, cd.TUNED, return_mask=True)          # TUNED colour, gated mask
    conts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counts['colour']['raw'] += len(conts)
    n_all = len(conts); conts = [c_ for c_ in conts if float(cv2.contourArea(c_)) > cd.TUNED['a_min']]
    counts['colour']['small_blob'] += n_all - len(conts)
    assert len(conts) == len(cdet)
    for c_ in conts:
        x, y, w, h = cv2.boundingRect(c_); a_ = float(cv2.contourArea(c_))
        assert (x, y, x + w, y + h, a_) in cdet
        dets.append(('colour', x + w / 2, y + h - 1, a_, float('nan'), (x, y, x + w, y + h),
                     cv2.approxPolyDP(c_, APPROX_EPS, True).reshape(-1, 2).astype(float)))
    n_area = dict(yolo=0, colour=0)
    for det, u, v, apx, conf, box, outline in dets:
        C = counts[det]
        pw, s = wm.pixel_to_ground(u, v, *pl, T)
        if pw is None: C['no_ground_hit'] += 1; continue
        if np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE: C['range_gt_5m'] += 1; continue
        A_pipe = apx * s
        if A_pipe <= wm.AREA_M2_MAX: C['kept'] += 1; continue
        C['area_gt_1m2'] += 1; n_area[det] += 1
        parts = clip_outline(outline, vm_er, img.shape)
        rec = dict(pid=pid, stem=stem, det=det, conf=conf, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                   u_contact=u, v_contact=v, area_px=apx, A_pipeline_m2=A_pipe, A_polygon_m2=np.nan,
                   n_vertices=0, n_parts=len(parts), A_other_parts_m2=0.0,
                   clip_status='empty_after_clip' if not parts else 'ok',
                   x_enu=xe_r, y_enu=ye_r, yaw=yaw_enu, timestamp=lt)
        for j, part in enumerate(parts):
            rows, miss = project(densify(part), pl, T); miss_total += miss
            if len(rows) < 3: continue
            A = shoelace(np.array([[r_[4], r_[5]] for r_ in rows]))
            if j == 0:
                rec['A_polygon_m2'] = A; rec['n_vertices'] = len(rows)
                for k, rw in enumerate(rows): verts.append((pid,) + (k,) + rw)
            else:
                rec['A_other_parts_m2'] += A
        patches.append(rec); pid += 1
    per_frame_area.append(dict(stem=stem, yolo_area_gt_1m2=n_area['yolo'], colour_area_gt_1m2=n_area['colour']))
    if i % 50 == 0: print(f'{i}/{len(frames)} {time.time() - t0:.0f}s patches={pid}', flush=True)
P = pd.DataFrame(patches); P.to_csv(OUT / 'patches.csv', index=False)
pd.DataFrame(verts, columns=['pid', 'k', 'u', 'v', 'x_map', 'y_map', 'x_enu', 'y_enu', 'range_m', 'cos_phi']).to_csv(OUT / 'patch_vertices.csv', index=False)
pd.DataFrame(foot, columns=['stem', 'k', 'x_enu', 'y_enu']).to_csv(OUT / 'footprint_vertices.csv', index=False)
pd.DataFrame(per_frame_area).to_csv(OUT / 'per_frame_area_limit_counts.csv', index=False)
ft = pd.read_csv(TAB / 'filter_table.csv').set_index('detector')
for det, ref in (('yolo', 'yolo'), ('colour', 'colour_tuned')):
    C = counts[det]
    check[det] = dict(counts=C, run_weed_ca10=dict(no_ground_hit=int(ft.loc[ref, 'no_ground_hit']), range_gt_5m=int(ft.loc[ref, 'range_gt_5m']),
                      area_gt_1m2=int(ft.loc[ref, 'area_gt_1m2_kept_as_patch']), kept=int(ft.loc[ref, 'kept'])))
    check[det]['matches_run_weed_ca10'] = all(C[k] == v for k, v in check[det]['run_weed_ca10'].items())
check.update(vertices_not_projected=miss_total, valid_mask_spot_check_mismatches=int(spot_bad),
             valid_mask_spot_check_pixels=300 * len(frames), n_patches=len(P),
             clip_status=P['clip_status'].value_counts().to_dict() if len(P) else {}, multi_part=int((P['n_parts'] > 1).sum()) if len(P) else 0,
             frames_with_footprint=int(pd.Series([f[0] for f in foot]).nunique()), time_s=round(time.time() - t0, 1))
json.dump(check, open(OUT / 'checks.json', 'w'), indent=1, default=str)
print(json.dumps(check, indent=1, default=str), flush=True)
