#!/usr/bin/env python3
"""CA-10 figure-data bundle in the structure and file formats of ~/viz_bundle_20261014 (CB-11).

  build_bundle.py <ca10_run dir>  ->  <ca10_run>/viz_bundle_ca10/

Relative paths and file names are kept identical to the CB-11 bundle so the figure scripts run with only their root
constant changed; MANIFEST.md maps every name to its CA-10 content ("rep1 frame" = A's LIO-SAM map frame, cb11/ =
the CA-10 session, frozen_374frames/ = the CA-10 frozen-stride run with 361 frames, ...).
"""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation, Slerp

R = Path(sys.argv[1]); B = R / 'viz_bundle_ca10'
if B.exists(): shutil.rmtree(B)
for d in ('alignments', 'trajectories/cb11', 'trunks/final_rep1', 'weed_map/frozen_374frames', 'weed_map/dense_stride10',
          'weed_map/tables/dense10', 'weed_map/patches', 'dense', 'perception', 'reports'):
    (B / d).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / 'scripts')); sys.path.insert(0, '/home/afonso/viz_bundle_20261014/style')
import ca10lib as L
from entities import entities
T0 = L.T0
al = json.load(open(R / 'merge_final/align_A_to_enu.json')); RA, tA = np.array(al['R']), np.array(al['t'])
enu = lambda xy: np.asarray(xy) @ RA.T + tA
cp = lambda s, d: shutil.copy2(R / s, B / d)

# -- GNSS ------------------------------------------------------------------------------------------------------
g = pd.read_csv('/home/afonso/ca10_gap/gnss_enu.csv'); fx = pd.read_csv('/home/afonso/ca10_gap/gnss_fix.csv')
t0_unix = float(g.timestamp.iloc[0])
pd.DataFrame(dict(t_rel_s=g.timestamp - t0_unix, east_m=g.e, north_m=g.n)).to_csv(B / 'trajectories/cb11/gnss_enu.csv', index=False, float_format='%.4f')
src = pd.DataFrame(dict(timestamp=g.timestamp, x_east=g.e, y_north=g.n, z_up=g.altitude - g.altitude.iloc[0]))
src = src.merge(fx[['timestamp', 'latitude', 'longitude', 'altitude']], on='timestamp', how='left')
src.to_csv(B / 'trajectories/cb11/gnss_raw_source.csv', index=False, float_format='%.8f')

# -- trajectories ----------------------------------------------------------------------------------------------
runs, fits = {}, {}
TRJ = {'liosam_rep1_keyframes': 'traj/merged_final/keyframes.csv', 'liosam_merged_dense': 'traj/merged_final/dense_lidar_rate.csv',
       'liosam_A_keyframes': 'traj/A/keyframes.csv', 'liosam_B_own_frame_keyframes': 'traj/B/keyframes.csv'}
import gnss_fit as G
for name, p in TRJ.items():
    a = L.read_csv_traj(R / p)
    with open(B / f'trajectories/cb11/{name}_raw.csv', 'w') as f:
        f.write('t,x,y,z,qx,qy,qz,qw\n')
        for r in a: f.write(f'{r[0]:.9f},' + ','.join(f'{v:.9f}' for v in r[1:]) + '\n')
    if name == 'liosam_B_own_frame_keyframes':
        continue                                    # B's own frame is not in A's frame: no ENU file
    e = enu(a[:, 1:3])
    pd.DataFrame(dict(t_rel_s=a[:, 0] - t0_unix, east_m=e[:, 0], north_m=e[:, 1], z_slam_m=a[:, 3])).to_csv(
        B / f'trajectories/cb11/{name}_enu.csv', index=False, float_format='%.3f')
    r = G.fit(np.column_stack([a[:, 0] - T0, a[:, 1:]]), 0, 1400, R=RA, t=tA)
    fits[name] = dict(source=p, R_slam_to_enu=RA.tolist(), t_slam_to_enu=tA.tolist(), ate_m=round(r['ate_m'], 3),
                      rpe50_pct=round(r['rpe50_pct'], 2), n=len(a), t_end_s=int(a[-1, 0] - t0_unix),
                      note='one SE2 for the merged run (align_rep1.json); A-only keyframes evaluated with it' if name != 'liosam_rep1_keyframes' else 'the merged-run fit')
json.dump({'cb11': dict(session='CA-10 = session_20260627_172049(half_good) (names kept from the CB-11 bundle)',
                        t0_unix=t0_unix, gnss_enu='gnss_enu.csv: t_rel_s = timestamp - t0_unix; E/N metres, origin = first CA-10 fix',
                        runs=fits, no_pose_windows_bag_s=[[805.5, 839.5], [966.0, 975.0]], bag_t0_unix=T0),
           '_readme': 'CA-10 bundle: see MANIFEST.md'}, open(B / 'alignments/trajectory_to_enu_fits.json', 'w'), indent=1)
json.dump(dict(R=al['R'], t=al['t'], traj='trajectories/cb11/liosam_rep1_keyframes_raw.csv', gps='trajectories/cb11/gnss_raw_source.csv',
               method=al['method'], n=al['n'], ate_m=al['ate_m'], _ca10='A frame (merged LIO-SAM) -> CA-10 bag-ENU; called align_rep1 for script compatibility'),
          open(B / 'alignments/align_rep1.json', 'w'), indent=1)
cp('merge_final/T_final.json', 'alignments/merge_T_final.json')

# -- trunks ----------------------------------------------------------------------------------------------------
t = pd.read_csv(R / 'trunks/trunks_final_ca10.csv'); ta = pd.read_csv(R / 'trunks/trunks_final_ca10_all.csv')
def trunk_table(d):
    return pd.DataFrame(dict(x_rep1=d.x_A, y_rep1=d.y_A, east_m=d.east_m, north_m=d.north_m, r=d.r, x_july='', y_july='',
                             x_enu_OLD_DO_NOT_USE='', y_enu_OLD_DO_NOT_USE='', support=d.support, n_up=d.n_up, vocc=d.vocc,
                             kept_by_row_prior=d.kept_by_row_prior, row_id=d.row_id, source=d.source, gt_id='', survey_row=''))
trunk_table(t).to_csv(B / 'trunks/trunks_final_rep1_enu.csv', index=False)
pd.DataFrame(dict(x=t.x_A, y=t.y_A, r=t.r, support=t.support, n_up=t.n_up, vocc=t.vocc, kept_by_row_prior=t.kept_by_row_prior,
                  row_id=t.row_id, source=t.source)).to_csv(B / 'trunks/trunks_final_rep1.csv', index=False)
pd.DataFrame(dict(x=ta.x_A, y=ta.y_A, r=ta.r, support=ta.support, n_up=ta.n_up, vocc=ta.vocc, kept_by_row_prior=ta.kept_by_row_prior,
                  row_id=ta.row_id, source=ta.source)).to_csv(B / 'trunks/trunks_final_rep1_all.csv', index=False)
rl = pd.read_csv(R / 'trunks/row_lines.csv'); rl = rl[rl.n_trunks >= 3]
pd.DataFrame(dict(row_id=rl.row_id, survey_row='', n_trunks=rl.n_trunks, c_x_rep1=rl.c_x_A, c_y_rep1=rl.c_y_A, u_x_rep1=rl.u_x_A,
                  u_y_rep1=rl.u_y_A, start_x_rep1=rl.start_x_A, start_y_rep1=rl.start_y_A, end_x_rep1=rl.end_x_A, end_y_rep1=rl.end_y_A,
                  start_east=rl.start_east, start_north=rl.start_north, end_east=rl.end_east, end_north=rl.end_north,
                  length_m=rl.length_m, mean_spacing_m=rl.mean_spacing_m, perp_rms_cm=rl.perp_rms_cm)).to_csv(B / 'trunks/row_lines.csv', index=False)
for f in ('row_coverage.csv', 'row_gaps.csv', 'summary.json', 'whole_map_ca10.png', 'merge_check_trunks_Aonly.csv',
          'merge_check_trunks_Bonly.csv', 'merge_check_shift_curve_Bpre.csv', 'merge_check_shift_curve_Bpost.csv'):
    if (R / 'trunks' / f).exists(): cp(f'trunks/{f}', f'trunks/final_rep1/{f}')
cp('trunks/row_lines.csv', 'trunks/final_rep1/row_lines_all_columns.csv')

# -- weed maps -------------------------------------------------------------------------------------------------
for S, dst, sfx in (('frozen', 'frozen_374frames', ''), ('dense10', 'dense_stride10', '_dense')):
    D = R / 'weed' / S
    for det in ('yolo', 'colour'):
        w = pd.read_csv(D / f'weed_map_{det}_cached.csv'); w.to_csv(B / f'weed_map/{dst}/weed_map_{det}_cached.csv', index=False)
        shutil.copy2(D / f'weed_map_{det}_patches.csv', B / f'weed_map/{dst}/weed_map_{det}_patches.csv')
        _, lab = entities(w, 0.4)
        pd.DataFrame(dict(raw_row=np.arange(len(w)), stem=w.stem, timestamp=w.timestamp, x_enu=w.x_enu, y_enu=w.y_enu, eid=lab)).to_csv(
            B / f'weed_map/{dst}/detection_to_entity_{det}_eps0.4.csv', index=False)
        tag = det if sfx == '' else det
        shutil.copy2(D / f'priority_{det}.csv', B / f'weed_map/tables/priority_{det}{sfx}.csv')
        shutil.copy2(D / f'priority_zones_{det}.csv', B / f'weed_map/tables/priority_zones_{det}{sfx}.csv')
        shutil.copy2(D / f'patch_entities_{det}.csv', B / f'weed_map/tables/patch_entities_{det}{sfx}.csv')
        if S == 'frozen':
            shutil.copy2(D / f'entities_{det}.csv', B / f'weed_map/tables/entities_{det}.csv')
        else:
            shutil.copy2(D / f'entities_{det}.csv', B / f'weed_map/tables/dense10/entities_{det}.csv')
    shutil.copy2(R / 'merge_final/align_A_to_enu.json', B / f'weed_map/{dst}/align_rep1.json')
    for f in ('frame_accounting.json', 'filter_table.csv'):
        shutil.copy2(D / f, B / f'weed_map/{dst}/{f}')
    if S == 'frozen':
        for f in ('frames.csv', 'frame_accounting.json', 'filter_table.csv', 'map_summary.csv'):
            shutil.copy2(D / f, B / f'weed_map/tables/{f}')
        shutil.copy2(D / 'coverage_frozen.npz', B / 'weed_map/tables/coverage_frozen.npz')
    else:
        for f in ('frames.csv', 'frame_accounting.json', 'map_summary.csv'):
            shutil.copy2(D / f, B / f'weed_map/tables/dense10/{f}')
        shutil.copy2(D / 'coverage_dense10.npz', B / 'weed_map/tables/coverage_dense.npz')
for f in ('patches.csv', 'patch_vertices.csv', 'footprint_vertices.csv', 'per_frame_area_limit_counts.csv', 'checks.json'):
    cp(f'weed/frozen/patches/{f}', f'weed_map/patches/{f}')
cp('weed/counts.csv', 'weed_map/tables/counts_ca10.csv'); cp('weed/post_params.json', 'weed_map/tables/post_params.json')
shutil.copy2('/home/afonso/thesis/orchard-calibration/calibration/calib_campinas_b.json', B / 'perception/calibration_campinas_b.json')

# -- dense map (F1/F2 format, 0.1 m voxels for the zip) + ground DEM --------------------------------------------
import trunks as Tr
from common import local_ground
F = json.load(open(R / 'merge_final/T_final.json')); T1, T2 = np.array(F['T1_Bpre']), np.array(F['T2'])
kfA = L.read_csv_traj(R / 'traj/A/keyframes.csv'); kfB = L.read_csv_traj(R / 'traj/B/keyframes.csv')
SA, SB = L.Scans(R / 'scans/A_kf_scans.npz'), L.Scans(R / 'scans/B_kf_scans.npz')
xyz, inten = [], []
for S, kf, a, b, T in ((SA, kfA, 0.0, L.A_END, np.eye(4)), (SB, kfB, L.B_START, 966.0, T1), (SB, kfB, 975.0, 1400.0, T2)):
    m = (kf[:, 0] - T0 >= a - 1) & (kf[:, 0] - T0 <= b + 1); k = kf[m]; sl = Slerp(k[:, 0], Rotation.from_quat(k[:, 4:8]))
    for r in k:
        if not (a <= r[0] - T0 <= b): continue
        s = S.scan(S.index_of(r[0])); rr = np.linalg.norm(s[:, :3], axis=1); s = s[(rr >= 1.0) & (rr <= 25.0)]
        tt = np.clip(r[0] + s[:, 5].astype(float), k[0, 0], k[-1, 0])
        p = np.stack([np.interp(tt, k[:, 0], k[:, c]) for c in (1, 2, 3)], 1)
        xyz.append((sl(tt).apply(s[:, :3].astype(float)) + p) @ T[:3, :3].T + T[:3, 3]); inten.append(s[:, 3])
P = np.vstack(xyz); I = np.concatenate(inten); h = local_ground(P)
key = np.floor(P / 0.1).astype(np.int64); _, idx = np.unique(key, axis=0, return_index=True); idx = np.sort(idx)
np.savez_compressed(B / 'dense/dense_liosam_rep1.npz', xyz=P[idx].astype(np.float32), intensity=I[idx].astype(np.float16),
                    h_above_ground=h[idx].astype(np.float16))
cell = 0.5; x0, y0 = np.floor(P[:, :2].min(0)); ix = ((P[:, 0] - x0) / cell).astype(int); iy = ((P[:, 1] - y0) / cell).astype(int)
nx, ny = ix.max() + 1, iy.max() + 1; z = np.full((ny, nx), np.nan, np.float32)
o = np.lexsort((P[:, 2], iy * nx + ix)); kk = (iy * nx + ix)[o]; zz = P[o, 2]
u_, st = np.unique(kk, return_index=True); en = np.r_[st[1:], len(kk)]
for k_, s_, e_ in zip(u_, st, en): z[k_ // nx, k_ % nx] = np.percentile(zz[s_:e_], 5)
np.savez_compressed(B / 'dense/ground_dem_liosam_rep1.npz', z=z, x0=x0, y0=y0, cell=cell)
print('dense', len(P), '->', len(idx), 'points at 0.1 m')
for f in ('RUNLOG.md',):
    cp(f, f'reports/{f}')
print('bundle written', B)

# -- format parity with the CB-11 bundle (reports/format_check.txt) -----------------------------------------------
import subprocess
a = json.load(open(B / 'alignments/align_rep1.json'))
a.update(_vm_traj='/home/afonso/ca10_run/traj/merged_final/keyframes.csv', _vm_gps='/home/afonso/ca10_gap/gnss_enu.csv')
for p in ('alignments/align_rep1.json', 'weed_map/frozen_374frames/align_rep1.json', 'weed_map/dense_stride10/align_rep1.json'):
    json.dump(a, open(B / p, 'w'), indent=1)
d = dict(np.load(B / 'dense/ground_dem_liosam_rep1.npz')); d['x0'] = np.float32(d['x0']); d['y0'] = np.float32(d['y0'])
np.savez_compressed(B / 'dense/ground_dem_liosam_rep1.npz', **d)
for f, src in (('trunks/trunks_final_rep1.csv', t), ('trunks/trunks_final_rep1_all.csv', ta)):
    pd.DataFrame(dict(x=src.x_A, y=src.y_A, r=src.r, x_july='', y_july='', x_enu=src.east_m, y_enu=src.north_m, support=src.support,
                      n_up=src.n_up, vocc=src.vocc, kept_by_row_prior=src.kept_by_row_prior, row_id=src.row_id, source=src.source,
                      gt_id='')).to_csv(B / f, index=False)
for S, dst in (('frozen', 'weed_map/tables/filter_table.csv'), ('dense10', 'weed_map/dense_stride10/filter_table.csv')):
    ft = pd.read_csv(R / 'weed' / S / 'filter_table.csv')
    pd.DataFrame(dict(raw=ft.raw, above_horizon_or_gate=ft.above_horizon_or_gate, small_blob=ft.small_blob, no_ground_hit=ft.no_ground_hit,
                      range_gt_5m=ft.range_gt_5m, area_gt_1m2=ft.area_gt_1m2_kept_as_patch, kept=ft.kept, ungated_blobs='',
                      csv_rows=ft.kept, matches_csv=True, detector=ft.detector.replace({'colour_tuned': 'colour'}))).to_csv(B / dst, index=False)
shutil.copy2(B / 'weed_map/tables/filter_table.csv', B / 'weed_map/frozen_374frames/filter_table.csv')
head = subprocess.run(['git', '-C', '/home/afonso/thesis', 'log', '-1', '--format=%h', '--', 'orchard-weed-detection/map/weed_map_cached.py'],
                      capture_output=True, text=True).stdout.strip()
for p in ('weed_map/tables/frame_accounting.json', 'weed_map/tables/dense10/frame_accounting.json',
          'weed_map/frozen_374frames/frame_accounting.json', 'weed_map/dense_stride10/frame_accounting.json'):
    fa = json.load(open(B / p)); n_img = len(list((R / 'frames_stride10/images').glob('*.jpg')))
    fa['image_files'] = fa['manifest_rows'] if 'frozen' in p or p.endswith('tables/frame_accounting.json') else n_img
    fa['git_commit'] = head + ' (last commit of weed_map_cached.py; file unchanged)'
    json.dump(fa, open(B / p, 'w'), indent=1)
for det in ('yolo', 'colour'):
    p = B / f'weed_map/tables/priority_{det}_dense.csv'
    pd.read_csv(p).drop(columns=['det', 'd_row', 'row', 'off_row', 'views']).to_csv(p, index=False)
print('format fixes applied')
subprocess.run([sys.executable, str(R / 'scripts/finalize_bundle.py'), str(R)], check=True)
