#!/usr/bin/env python3
"""
Weed Map Builder - cached frames, one pipeline for both detectors.

Replaces weed_map_yolo_cached.py (YOLO) and the colour branch of weed_map.py
(live bag) with one script, so that both detectors are evaluated on the SAME
frames, calibration, trajectory, ground plane, height gate and projection.
Only the detector differs:

    --detector yolo    YOLO11n boxes; gate on the box bottom-centre;
                       contact pixel = box bottom-centre; area = 0.7 * box area
    --detector colour  HSV + ExG mask, morphology (open 3x3, dilate 3x3);
                       gate applied to the mask pixel by pixel (Eq. horizon-gate);
                       contours > MIN_BLOB px; contact pixel = bottom-centre of
                       the contour's bounding box (was the centroid in
                       weed_map.py, which projects too far); area = contour area

Changes with respect to the two old scripts (all deliberate):
  * single calibration / trajectory / alignment for both detectors (CLI args);
  * colour contact pixel = bounding-box bottom-centre instead of centroid;
  * quaternion sign continuity enforced before interpolation (q and -q are
    the same rotation; linear interpolation across a sign flip is wrong);
  * ground RANSAC written in NumPy with a seeded generator: Open3D's
    segment_plane gave different planes on repeated runs even after
    o3d.utility.random.seed (tested with Open3D 0.20), so frames were not
    reproducible. Same algorithm: 3-point samples, 500 iterations, 0.06 m
    inlier distance, best consensus, least-squares refit on its inliers;
  * YOLO model loaded lazily, so the colour run does not need it;
  * ground plane limited to MAX_PLANE_TILT_DEG (3 deg) from the LiDAR's level, and the gate
    boundary is a straight line along the plane's vanishing direction (HORIZON_MODEL 'line')
    instead of a quadratic: before, 22 % of frames got planes rolled onto the ridge sides and
    the boundary crossed the image diagonally (2026-10-08, weed_detection_C1 report Sec. 1);
  * output CSV keeps the old columns, in the same order, plus extra columns
    (detector, stem, u, v, range_m, cos_phi) needed for sigma_loc and the
    map audit in the Results chapter.

Module-level functions keep their old names and signatures (fit_ground,
get_horizon, get_weed_boxes, pixel_to_ground), so the figure scripts that
import weed_map_yolo_cached can import this module instead.

Usage:
    python3 weed_map_cached.py --detector yolo
    python3 weed_map_cached.py --detector colour
    python3 weed_map_cached.py --detector yolo --traj path/to/final_trajectory.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

SEED = 0

# -- Defaults (CB-11); override from the command line ------------------------
REPO       = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "data_set" / "campinas_b" / "images" / "bag_frames_20260627"  # CB-11 frames despite the name
LIDAR_DIR  = REPO / "data_set" / "campinas_b" / "lidar"
CALIB      = REPO / "orchard-calibration" / "calibration" / "calib_campinas_b.json"
TRAJ_CSV   = REPO / "orchard-slam" / "data" / "campinas_b" / "lio_sam" / "trajectory_20260706.csv"
OUT_DIR    = REPO / "data_set" / "campinas_b" / "maps"
MODEL_WEIGHTS = REPO / "data_set" / "weed_dataset" / "runs" / "train" / "weeds_v2_n960" / "weights" / "best.pt"

# ENU alignment of CB-11 (planar Procrustes fit of SLAM to GNSS). If the
# trajectory changes, this must be recomputed for that trajectory: pass
# --align file.json with {"R": [[..],[..]], "t": [..]}.
R_ALIGN_DEFAULT = [[-0.14508231170482752, -0.9894195888653019],
                   [ 0.9894195888653021, -0.14508231170482772]]
T_ALIGN_DEFAULT = [-0.14110743455980668, -2.798850687960666]

# -- Parameters (Table of parameters, Appendix) -------------------------------
SYNC_TOLERANCE = 0.15
RADIUS, PERCENTILE = 8.0, 15
CANDIDATE_FRACTION = 0.20
RANSAC_DIST, RANSAC_N, RANSAC_ITERS = 0.06, 3, 500
H_GROUND, H_WEED, H_SHRUB = 0.08, 0.40, 1.0      # gate band = H_WEED +- 0.15 = (0.25, 0.55)
Z_MIN, Z_MAX = -1.1, -0.3
N_STRIPS = 40
# Ground-plane tilt limit: RANSAC accepts only planes within this angle of the LiDAR's level
# (None = no limit, the original fit). Without it, 22 % of CB-11 frames got planes rolled 5-15 deg
# onto the side of the tree-row ridges, uncorrelated with the robot's own roll (r = -0.01, LIO-SAM);
# the robot stands on the floor it sees, so the local ground is close to the LiDAR's level.
# 3 deg keeps the usual 0.5-1.3 deg tilt with margin (gate_tilt report, 2026-10-08).
MAX_PLANE_TILT_DEG = 3.0
# Gate boundary model: 'quad' = quadratic through the strip maxima (original);
# 'line' = straight line parallel to the ground plane's vanishing line, at the median strip maximum.
HORIZON_MODEL = 'line'
MAX_WEED_RANGE = 5.0
COS_PHI_MIN = 0.10
AREA_M2_MAX = 1.0
# YOLO
CONF_THRESH, BOX_SHRINK, YOLO_DEVICE = 0.30, 0.7, "cpu"   # v2: F1 peaks at 0.30 (weed_yolo_v2 report)
YOLO_IMGSZ = 960                                          # v2 was trained at 960 px
# colour
HSV_H_LOW, HSV_H_HIGH, HSV_S_MIN, HSV_V_MIN = 14, 110, 31, 36
EXG_THRESH = 10
MORPH_KERNEL = np.ones((3, 3), np.uint8)
MIN_BLOB = 300                                    # px, contour area

R_tilt = np.eye(3)  # no tilt correction for this session

# -- Globals filled by load_calibration / load_trajectory --------------------
T_lidar_cam = T_cam_lid = K = K_inv = dist = None
FX_FY = None
R_align = np.array(R_ALIGN_DEFAULT); t_align = np.array(T_ALIGN_DEFAULT)
_interp = None; traj_t = None
_model = None


def load_calibration(path=CALIB):
    global T_lidar_cam, T_cam_lid, K, K_inv, dist, FX_FY
    calib = json.load(open(path))
    T = np.array(calib['T_lidar_camera'])
    for tw in calib.get('manual_tweaks', []):
        dR = Rotation.from_euler('xyz', [tw.get('droll_deg', 0), tw.get('dpitch_deg', 0),
                                         tw.get('dyaw_deg', 0)], degrees=True).as_matrix()
        dT = np.eye(4); dT[:3, :3] = dR
        dT[:3, 3] = [tw.get('dtx_m', 0), tw.get('dty_m', 0), tw.get('dtz_m', 0)]
        T = dT @ T
    T_lidar_cam = T
    T_cam_lid = np.linalg.inv(T)
    K = np.array(calib['camera_K']); K_inv = np.linalg.inv(K)
    dist = np.array(calib['camera_dist'])
    FX_FY = float(K[0, 0]) * float(K[1, 1])


def load_trajectory(path=TRAJ_CSV):
    """Linear position, normalised linear quaternion blend (Eq. pose-interp),
    with quaternion signs made continuous first."""
    global _interp, traj_t
    rows = list(csv.DictReader(open(path)))
    tk = 'timestamp' if 'timestamp' in rows[0] else 't'   # LIO-SAM trajectory.csv / keyframe exports
    traj_t = np.array([float(r[tk]) for r in rows])
    q = np.array([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in rows])
    for i in range(1, len(q)):                      # sign continuity
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    cols = {'x': np.array([float(r['x']) for r in rows]),
            'y': np.array([float(r['y']) for r in rows]),
            'z': np.array([float(r['z']) for r in rows]),
            'qx': q[:, 0], 'qy': q[:, 1], 'qz': q[:, 2], 'qw': q[:, 3]}
    _interp = {k: interp1d(traj_t, v, bounds_error=False, fill_value='extrapolate')
               for k, v in cols.items()}


def load_alignment(path=None):
    global R_align, t_align
    if path:
        a = json.load(open(path))
        R_align, t_align = np.array(a['R']), np.array(a['t'])


def get_pose(ts):
    tx, ty, tz = [float(_interp[k](ts)) for k in ('x', 'y', 'z')]
    q = np.array([float(_interp[k](ts)) for k in ('qx', 'qy', 'qz', 'qw')])
    q /= np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def slam_to_enu(x, y):
    p = R_align @ np.array([x, y]) + t_align
    return p[0], p[1]


# -- Ground (Sec. local ground reference) --------------------------------------
def ransac_plane(pts, dist=RANSAC_DIST, iters=RANSAC_ITERS, seed=SEED, max_tilt_deg=None):
    """Seeded RANSAC plane (a, b, c, d), |(a, b, c)| = 1. A new generator per
    call, so the plane of a frame does not depend on which frames ran before.
    max_tilt_deg: samples whose normal is further than this from the z axis are
    skipped; if the least-squares refit leaves the limit, the sampled plane is kept."""
    rng = np.random.default_rng(seed)
    p = pts.astype(np.float64)
    cos_max = np.cos(np.radians(max_tilt_deg)) if max_tilt_deg is not None else -1.0
    best_n, best_mask, best_plane = -1, None, None
    for _ in range(iters):
        s = p[rng.choice(len(p), RANSAC_N, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) < cos_max:
            continue
        mask = np.abs(p @ n - n @ s[0]) <= dist
        cnt = int(mask.sum())
        if cnt > best_n:
            best_n, best_mask, best_plane = cnt, mask, (n[0], n[1], n[2], -n @ s[0])
    if best_mask is None:
        raise RuntimeError("RANSAC found no plane")
    q = p[best_mask]                                  # least-squares refit on inliers
    cen = q.mean(axis=0)
    n = np.linalg.svd(q - cen, full_matrices=False)[2][-1]
    if abs(n[2]) < cos_max:
        return best_plane
    return n[0], n[1], n[2], -n @ cen


def fit_ground(xyz_raw, return_debug=False):
    xt = (R_tilt @ xyz_raw.T).T
    d2 = np.sqrt(xt[:, 0] ** 2 + xt[:, 1] ** 2)
    mr = d2 < RADIUS
    xr, xt = xyz_raw[mr], xt[mr]
    z_thr = np.percentile(xt[:, 2], 100 * CANDIDATE_FRACTION)
    gc = xt[xt[:, 2] <= z_thr]
    a, b, c, d = ransac_plane(gc, max_tilt_deg=MAX_PLANE_TILT_DEG)
    if c < 0:
        a, b, c, d = -a, -b, -c, -d
    nl = np.linalg.norm([a, b, c])
    gh = (a * gc[:, 0] + b * gc[:, 1] + c * gc[:, 2] + d) / nl
    offset = np.percentile(gh, PERCENTILE)     # negative: plane moves down
    d -= offset * nl
    if return_debug:
        return a, b, c, d, xr, xt, {'plane_before_offset': (a, b, c, d + offset * nl),
                                    'offset_m': float(offset), 'candidate_pts': gc,
                                    'candidate_heights': gh}
    return a, b, c, d, xr, xt


# -- Height gate (Sec. LiDAR height gate) ---------------------------------------
def get_horizon(img_shape, xr, xt, a, b, c, d, return_debug=False):
    h_img, w_img = img_shape[:2]
    heights = (a * xt[:, 0] + b * xt[:, 1] + c * xt[:, 2] + d) / np.linalg.norm([a, b, c])
    bpts = xr[(heights > H_WEED - 0.15) & (heights < H_WEED + 0.15)]
    horizon = np.full(w_img, h_img // 2, dtype=int)
    strip_maxima, cf = [], None
    if len(bpts) >= 5:
        pc = (T_cam_lid[:3, :3] @ bpts.T).T + T_cam_lid[:3, 3]
        pc = pc[pc[:, 2] > 0.1]
        if len(pc) >= 5:
            ip, _ = cv2.projectPoints(pc.astype(np.float32), np.zeros(3), np.zeros(3), K, dist)
            ip = ip.reshape(-1, 2); bw = w_img / N_STRIPS; cp = {}
            for u, v in ip:
                if not (0 <= u < w_img and 0 <= v < h_img):
                    continue
                s = int(u / bw)
                if s not in cp or v < cp[s][1]:
                    cp[s] = (int(u), int(v))
            if len(cp) >= 3:
                strip_maxima = sorted(cp.values(), key=lambda p: p[0])
                xs = np.array([p[0] for p in strip_maxima]); ys = np.array([p[1] for p in strip_maxima])
                if HORIZON_MODEL == 'line':
                    # slope of the ground plane's vanishing line l = K^-T R_CL n (lines at a fixed
                    # height above the plane converge to it); offset = median strip maximum
                    l = K_inv.T @ (T_cam_lid[:3, :3] @ np.array([a, b, c]))
                    slope = -l[0] / l[1]
                    cf = np.array([slope, np.median(ys - slope * xs)])
                else:
                    cf = np.polyfit(xs, ys, deg=2)
                horizon = np.clip(np.polyval(cf, np.arange(w_img)).astype(int), 0, h_img - 1)
    if return_debug:
        return horizon, {'band_pts_lidar': bpts, 'n_strips_nonempty': len(strip_maxima),
                         'strip_maxima_px': strip_maxima, 'fallback_used': cf is None,
                         'polyfit_coeffs': cf}
    return horizon


# -- Detectors: both return (u_contact, v_contact, area_px, conf, x1, y1, x2, y2) --
def _model_lazy():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_WEIGHTS)
    return _model


def get_weed_boxes(img, horizon, return_debug=False):
    """YOLO. Gate on the bottom-centre of each box."""
    h_img, w_img = img.shape[:2]
    res = _model_lazy().predict(img, conf=CONF_THRESH, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE, verbose=False)[0]
    kept, rejected = [], []
    for bx in res.boxes:
        x1, y1, x2, y2 = bx.xyxy[0].tolist()
        conf = float(bx.conf[0])
        u_c, v_b = (x1 + x2) / 2, y2
        if v_b < horizon[min(max(int(u_c), 0), w_img - 1)]:
            rejected.append((x1, y1, x2, y2, conf))
            continue
        kept.append((u_c, v_b, (x2 - x1) * (y2 - y1) * BOX_SHRINK, conf, x1, y1, x2, y2))
    if return_debug:
        return kept, rejected
    return [k[:4] for k in kept]


def colour_mask(img):
    """HSV + ExG + morphology, before the height gate."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m_hsv = cv2.inRange(hsv, np.array([HSV_H_LOW, HSV_S_MIN, HSV_V_MIN]),
                        np.array([HSV_H_HIGH, 255, 255]))
    f = img.astype(np.float32)
    m_exg = ((2 * f[:, :, 1] - f[:, :, 2] - f[:, :, 0]) > EXG_THRESH).astype(np.uint8) * 255
    mask = cv2.bitwise_and(m_hsv, m_exg)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL)
    return mask


def get_colour_boxes(img, horizon, return_debug=False):
    """Colour detector. Gate applied to the mask pixel by pixel, then contours
    larger than MIN_BLOB; contact pixel = bottom-centre of the bounding box."""
    h_img, w_img = img.shape[:2]
    mask = colour_mask(img)
    gated = mask.copy()
    rows = np.arange(h_img)[:, None]
    gated[rows < horizon[None, :]] = 0
    conts, _ = cv2.findContours(gated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for cnt in conts:
        area = float(cv2.contourArea(cnt))
        if area <= MIN_BLOB:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        kept.append((x + w / 2, y + h - 1, area, float('nan'), x, y, x + w, y + h))
    if return_debug:
        return kept, {'mask': mask, 'mask_gated': gated, 'n_contours': len(conts)}
    return [k[:4] for k in kept]


# -- Ray-plane projection and footprint (Secs. ray-plane, metric footprint) -----
def pixel_to_ground(u, v, a, b, c, d, T_world, return_geometry=False):
    ray_cam = K_inv @ np.array([u, v, 1.0]); ray_cam /= np.linalg.norm(ray_cam)
    rd = R_tilt @ (T_lidar_cam[:3, :3] @ ray_cam)
    ro = R_tilt @ T_lidar_cam[:3, 3]
    denom = a * rd[0] + b * rd[1] + c * rd[2]
    if abs(denom) < 1e-6:
        return (None, None, None) if return_geometry else (None, None)
    t_r = -(a * ro[0] + b * ro[1] + c * ro[2] + d) / denom
    if t_r < 0.1:
        return (None, None, None) if return_geometry else (None, None)
    p_lidar = R_tilt.T @ (ro + t_r * rd)
    pw = T_world[:3, :3] @ p_lidar + T_world[:3, 3]
    nl = np.sqrt(a * a + b * b + c * c)
    cos_th, cos_phi = ray_cam[2], abs(denom) / nl
    if cos_phi < COS_PHI_MIN:
        return (None, None, None) if return_geometry else (None, None)
    px_scale = (t_r ** 2) * (cos_th ** 3) / (FX_FY * cos_phi)
    if return_geometry:
        return pw, px_scale, {'range_m': float(t_r), 'cos_phi': float(cos_phi)}
    return pw, px_scale


# Calibration and trajectory are loaded at import with the defaults, so that
# the figure scripts can use the functions directly; main() reloads them if
# other paths are given on the command line.
for _load, _path in ((load_calibration, CALIB), (load_trajectory, TRAJ_CSV)):
    try:
        _load(_path)
    except FileNotFoundError:
        pass   # figure scripts that do not need it (e.g. the ground fit) still import


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detector', choices=['yolo', 'colour'], required=True)
    ap.add_argument('--images', default=str(IMAGES_DIR))
    ap.add_argument('--lidar', default=str(LIDAR_DIR))
    ap.add_argument('--calib', default=str(CALIB))
    ap.add_argument('--traj', default=str(TRAJ_CSV))
    ap.add_argument('--align', default=None, help='JSON with R (2x2) and t (2) of the SLAM->ENU alignment')
    ap.add_argument('--out', default=str(OUT_DIR))
    ap.add_argument('--weights', default=None, help='YOLO weights (default: weeds_v2_n960)')
    ap.add_argument('--conf', type=float, default=None, help='YOLO confidence threshold (default 0.30)')
    ap.add_argument('--imgsz', type=int, default=None, help='YOLO inference size (default 960)')
    args = ap.parse_args()
    global MODEL_WEIGHTS, CONF_THRESH, YOLO_IMGSZ
    if args.weights: MODEL_WEIGHTS = Path(args.weights)
    if args.conf is not None: CONF_THRESH = args.conf
    if args.imgsz: YOLO_IMGSZ = args.imgsz

    load_calibration(args.calib); load_trajectory(args.traj); load_alignment(args.align)
    images, lidar, out = Path(args.images), Path(args.lidar), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detect = get_weed_boxes if args.detector == 'yolo' else get_colour_boxes

    manifest = [r for r in csv.DictReader(open(lidar / 'manifest.csv'))
                if abs(float(r['dt'])) <= SYNC_TOLERANCE]
    counts = dict(frames=0, no_files=0, out_of_traj=0, ground_failed=0, implausible_plane=0)
    rows = []
    for r in manifest:
        stem, lt = r['stem'], float(r['matched_ts'])
        ip, lp = images / f'{stem}.jpg', lidar / f'{stem}.npz'
        if not ip.exists() or not lp.exists():
            counts['no_files'] += 1; continue
        if lt < traj_t[0] or lt > traj_t[-1]:
            counts['out_of_traj'] += 1; continue
        img = cv2.imread(str(ip)); xyz_raw = np.load(lp)['xyz']
        try:
            a, b, c, d, xr, xt = fit_ground(xyz_raw)
        except Exception:
            counts['ground_failed'] += 1; continue
        if not (Z_MIN < -d / c < Z_MAX):
            counts['implausible_plane'] += 1; continue
        counts['frames'] += 1
        T_world = get_pose(lt); x_s, y_s = T_world[0, 3], T_world[1, 3]
        horizon = get_horizon(img.shape, xr, xt, a, b, c, d)
        for u, v, area_px, conf in detect(img, horizon):
            pw, px_scale, geo = pixel_to_ground(u, v, a, b, c, d, T_world, return_geometry=True)
            if pw is None or np.hypot(pw[0] - x_s, pw[1] - y_s) > MAX_WEED_RANGE:
                continue
            area_m2 = area_px * px_scale
            if area_m2 > AREA_M2_MAX:
                continue
            xe, ye = slam_to_enu(pw[0], pw[1])
            rows.append((xe, ye, pw[0], pw[1], lt, area_px, area_m2, conf,
                         args.detector, stem, u, v, geo['range_m'], geo['cos_phi']))

    csv_path = out / f'weed_map_{args.detector}_cached.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x_enu', 'y_enu', 'x_slam', 'y_slam', 'timestamp', 'area_px', 'area_m2',
                    'confidence', 'detector', 'stem', 'u', 'v', 'range_m', 'cos_phi'])
        w.writerows(rows)
    print(f"{args.detector}: {counts['frames']} frames used, {len(rows)} detections -> {csv_path}")
    print(f"skipped: {counts}")


if __name__ == '__main__':
    main()
