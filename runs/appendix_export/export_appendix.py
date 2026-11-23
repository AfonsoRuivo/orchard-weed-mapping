#!/usr/bin/env python3
"""Read-only export of appendix raw material (Tasks A, B1, B2). Writes only under ~/appendix_export."""
import csv, json, shutil, sys
from pathlib import Path
import cv2, numpy as np

REPO = Path('/home/afonso/thesis')
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map'))
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map/results'))
import weed_map_cached as wm
import colour_detector as cd
from common import TRAJ, FROZEN, LIDAR, TAB

OUT = Path.home() / 'appendix_export'
IMAGES = wm.IMAGES_DIR
wm.load_calibration(); wm.load_trajectory(str(TRAJ)); wm.load_alignment(str(FROZEN / 'align_rep1.json'))

man = [r for r in csv.DictReader(open(LIDAR / 'manifest.csv')) if abs(float(r['dt'])) <= wm.SYNC_TOLERANCE]
by_num = {int(r['stem'].split('_')[1]): r for r in man}
frames_csv = {r['stem']: r for r in csv.DictReader(open(TAB / 'frames.csv'))}


def fit(stem):
    """Plane exactly as the map loop; returns (plane, xr, xt, status)."""
    a, b, c, d, xr, xt = wm.fit_ground(np.load(LIDAR / f'{stem}.npz')['xyz'])
    ok = wm.Z_MIN < -d / c < wm.Z_MAX
    return (a, b, c, d), xr, xt, ('ok' if ok else 'implausible_plane')


def lidar_uv(stem, plane, shape):
    """All scan points, pipeline extrinsics + cv2.projectPoints with K, dist (as get_horizon)."""
    xyz = np.load(LIDAR / f'{stem}.npz')['xyz'].astype(np.float64)
    a, b, c, d = plane
    h = (xyz @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])   # R_tilt = I
    pc = xyz @ wm.T_cam_lid[:3, :3].T + wm.T_cam_lid[:3, 3]
    keep = pc[:, 2] > 0.1
    ip, _ = cv2.projectPoints(pc[keep].astype(np.float32), np.zeros(3), np.zeros(3), wm.K, wm.dist)
    ip = ip.reshape(-1, 2); hh = h[keep]
    H, W = shape[:2]
    inn = (ip[:, 0] >= 0) & (ip[:, 0] < W) & (ip[:, 1] >= 0) & (ip[:, 1] < H)
    return ip[inn], hh[inn]


def ground_status(u, v, area_px, plane, T):
    """Filters after the gate, in the order of the map loop; grazing ray split out of no-hit."""
    a, b, c, d = plane
    pw, s, geo = wm.pixel_to_ground(u, v, a, b, c, d, T, return_geometry=True)
    if pw is None:
        ray = wm.K_inv @ np.array([u, v, 1.0]); ray /= np.linalg.norm(ray)
        rd = wm.T_lidar_cam[:3, :3] @ ray; ro = wm.T_lidar_cam[:3, 3]
        den = a * rd[0] + b * rd[1] + c * rd[2]
        if abs(den) < 1e-6:
            return 'no_ground_hit(parallel)', None, None
        tr = -(a * ro[0] + b * ro[1] + c * ro[2] + d) / den
        if tr < 0.1:
            return 'no_ground_hit(behind_camera)', None, None
        return 'grazing_ray(cos_phi<0.10)', None, dict(range_m=tr, cos_phi=abs(den) / np.linalg.norm([a, b, c]))
    if np.hypot(pw[0] - T[0, 3], pw[1] - T[1, 3]) > wm.MAX_WEED_RANGE:
        return 'out_of_range(>5m)', pw, geo
    if area_px * s > wm.AREA_M2_MAX:
        return 'area_limit(>1m2)', pw, geo
    return 'kept', pw, geo


def write_uv(path, uv, h):
    np.savetxt(path, np.c_[uv, h], fmt='%.2f,%.2f,%.4f', header='u,v,height_above_plane_m', comments='')


def export_frame(stem, dest, with_detectors=True):
    img = cv2.imread(str(IMAGES / f'{stem}.jpg'))
    assert max(img.shape[:2]) <= 1600
    cv2.imwrite(str(dest / f'{stem}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    plane, xr, xt, st = fit(stem)
    uv, h = lidar_uv(stem, plane, img.shape)
    write_uv(dest / f'{stem}_lidar_uv.csv', uv, h)
    info = dict(stem=stem, plane=plane, status=st, plane_height=-plane[3] / plane[2], n_uv=len(uv))
    if not with_detectors or st != 'ok':
        return info
    lt = float(by_num[int(stem.split('_')[1])]['matched_ts'])
    T = wm.get_pose(lt)
    hz = wm.get_horizon(img.shape, xr, xt, *plane)
    np.savetxt(dest / f'{stem}_gate_line.csv', np.c_[np.arange(len(hz)), hz], fmt='%d', delimiter=',',
               header='u,v_gate (pixels above v_gate are removed by the height gate)', comments='')
    # YOLO
    ok, rej = wm.get_weed_boxes(img, hz, return_debug=True)
    rows = [(x1, y1, x2, y2, cf, 'above_gate', None, None) for x1, y1, x2, y2, cf in rej]
    for u_c, v_b, area, cf, x1, y1, x2, y2 in ok:
        s, pw, geo = ground_status(u_c, v_b, area, plane, T)
        rows.append((x1, y1, x2, y2, cf, s, pw, geo))
    with open(dest / f'{stem}_yolo.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x1', 'y1', 'x2', 'y2', 'conf', 'class', 'status', 'x_map', 'y_map',
                    'x_enu', 'y_enu', 'range_m', 'cos_phi'])
        for x1, y1, x2, y2, cf, s, pw, geo in rows:
            g = ['', '', '', '']
            if s == 'kept':
                g = [f'{pw[0]:.3f}', f'{pw[1]:.3f}', *(f'{q:.3f}' for q in wm.slam_to_enu(pw[0], pw[1]))]
            w.writerow([f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}', f'{cf:.4f}', 'weed', s, *g,
                        f"{geo['range_m']:.3f}" if geo else '', f"{geo['cos_phi']:.4f}" if geo else ''])
    # colour, tuned
    blobs, mask = cd.detect(img, hz, cd.TUNED, return_mask=True)
    cv2.imwrite(str(dest / f'{stem}_colour_mask.png'), mask)
    with open(dest / f'{stem}_colour.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x1', 'y1', 'x2', 'y2', 'area_px', 'status', 'x_map', 'y_map', 'x_enu', 'y_enu', 'range_m', 'cos_phi'])
        for x1, y1, x2, y2, ar in blobs:
            s, pw, geo = ground_status(x1 + (x2 - x1) / 2, y2 - 1, ar, plane, T)
            g = ['', '', '', '']
            if s == 'kept':
                g = [f'{pw[0]:.3f}', f'{pw[1]:.3f}', *(f'{q:.3f}' for q in wm.slam_to_enu(pw[0], pw[1]))]
            w.writerow([x1, y1, x2, y2, f'{ar:.0f}', s, *g,
                        f"{geo['range_m']:.3f}" if geo else '', f"{geo['cos_phi']:.4f}" if geo else ''])
    info.update(t=lt, pose=(T[0, 3], T[1, 3], float(np.arctan2(T[1, 0], T[0, 0]))),
                n_yolo=len(rows), n_yolo_kept=sum(r[5] == 'kept' for r in rows), n_colour=len(blobs))
    return info


def stem_of(n):
    return by_num[n]['stem']


def main():
    OUT.mkdir(exist_ok=True)
    # pipeline's own implausible-plane list (all synced frames inside the trajectory)
    implaus = []
    for r in man:
        lt = float(r['matched_ts'])
        if lt < wm.traj_t[0] or lt > wm.traj_t[-1]:
            continue
        if fit(r['stem'])[3] != 'ok':
            implaus.append(r['stem'])
    json.dump(implaus, open(OUT / '_implausible_from_pipeline.json', 'w'), indent=1)
    print('implausible:', implaus, flush=True)

    a_rows = []
    for run, nums in (('A_run1', range(5400, 5851, 50)), ('A_run2', range(15950, 16351, 50))):
        dest = OUT / run; dest.mkdir(exist_ok=True)
        for n in nums:
            info = export_frame(stem_of(n), dest)
            assert info['status'] == 'ok', info
            fr = frames_csv.get(info['stem'])
            a, b, c, d = info['plane']
            a_rows.append(dict(run=run, stem=info['stem'], timestamp=f"{info['t']:.6f}", a=a, b=b, c=c, d=d,
                               x_slam=info['pose'][0], y_slam=info['pose'][1], yaw_slam=info['pose'][2],
                               matches_frames_csv=bool(fr) and np.allclose(
                                   [float(fr[k]) for k in ('a', 'b', 'c', 'd', 'x_slam', 'y_slam', 'yaw_slam')],
                                   [a, b, c, d, *info['pose']], atol=1e-6)))
            print(run, info['stem'], info['n_yolo'], info['n_yolo_kept'], info['n_colour'], flush=True)
    with open(OUT / 'A_frames.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(a_rows[0])); w.writeheader(); w.writerows(a_rows)

    dest = OUT / 'B_failures'; dest.mkdir(exist_ok=True)
    b_rows = []
    for stem in implaus + [stem_of(13250), stem_of(13450)]:
        info = export_frame(stem, dest, with_detectors=False)
        a, b, c, d = info['plane']
        b_rows.append(dict(stem=stem, status=info['status'], a=a, b=b, c=c, d=d, plane_height_m=info['plane_height'],
                           tilt_deg=float(np.degrees(np.arccos(abs(c) / np.linalg.norm([a, b, c])))), n_points_in_image=info['n_uv']))
        print('B', b_rows[-1], flush=True)
    with open(dest / 'B_planes.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(b_rows[0])); w.writeheader(); w.writerows(b_rows)


if __name__ == '__main__':
    main()
