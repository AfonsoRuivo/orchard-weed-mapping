#!/usr/bin/env python3
"""J8 of the 2026-10-07 thesis job list: build data/thesis_jobs_20261007/for_thesis_20261007/
(+ .zip) and jobs_summary.md from what the queue (tools/thesis_jobs_20261007.sh) produced.

    python3 thesis_pack_20261007.py

Every trajectory is rewritten as t,x,y,z,qx,qy,qz,qw (t = ROS time, s). Missing inputs are
listed in the summary instead of failing the pack. J9 answers are read from
data/thesis_jobs_20261007/j9_answers.md (written by hand) and appended to the summary.
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'analysis'))
from ate_vs_gps import GPS, evaluate, load_gps, load_traj  # noqa: E402

JD = ROOT / 'data' / 'thesis_jobs_20261007'
PACK = JD / 'for_thesis_20261007'
D = ROOT / 'data'
SW = D / 'lio_sam' / 'sweeps'
COLS = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']


def jobs():
    out = {}
    tsv = JD / 'jobs.tsv'
    if tsv.exists():
        for r in csv.DictReader(open(tsv), delimiter='\t'):
            out[r['job']] = r  # last entry per job wins
    return out


def norm_csv(src, dst):
    """Copy a trajectory CSV with columns t|timestamp,x,y,z,qx,qy,qz,qw to the standard header."""
    rows = list(csv.DictReader(open(src)))
    tk = 't' if 't' in rows[0] else 'timestamp'
    with open(dst, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[tk]] + [r[c] for c in COLS[1:]])
    return len(rows)


def rtab_poses(db, dst):
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(db, td)
        subprocess.run(['rtabmap-export', '--poses', '--poses_format', '10', '--output_dir', td,
                        str(Path(td) / Path(db).name)], capture_output=True, check=True)
        txt = next(Path(td).glob('*_poses.txt'))
        with open(dst, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for line in open(txt):
                if not line.startswith('#') and line.strip():
                    w.writerow(line.split())


def main():
    J = jobs()
    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / 'trajectories').mkdir(parents=True)
    missing, notes = [], []
    shutil.copy(GPS, PACK / GPS.name)
    q = GPS.with_name(GPS.stem + '_quality.csv')
    if q.exists():
        shutil.copy(q, PACK / q.name)

    def jd(j):
        r = J.get(j)
        # 'diverged' runs (bag played to the end, trajectory blew up) are still packed and summarised
        return Path(r['run_dir']) if r and r['status'] in ('ok', 'diverged') and r['run_dir'] else None

    traj = {}  # pack name -> (source, kind)
    exp = JD / 'exports'
    for name, src in [
        ('liosam_baseline_r08_keyframes', exp / 'liosam_baseline' / 'keyframes_final.csv'),
        ('liosam_rep1_keyframes', exp / 'liosam_rep1' / 'keyframes_final.csv'),
        ('liosam_rep2_keyframes', exp / 'liosam_rep2' / 'keyframes_final.csv'),
        ('kiss_tuned_ng_p20_nodeskew', D / 'kiss_icp/runs/session_20260706_140610_best_ng_p20_nodeskew_20261004_192110/kiss_odom.csv'),
        ('kiss_tuned_rep1', D / 'kiss_icp/runs/session_20260706_140610_best_tuned_rep1_20261005_093554/kiss_odom.csv'),
        ('kiss_tuned_rep2', D / 'kiss_icp/runs/session_20260706_140610_best_tuned_rep2_20261005_095843/kiss_odom.csv'),
        ('kiss_in_rtabmap_loop_run', D / 'rtabmap/runs/session_20260706_140610_best_kisstuned_rtabloop_r1.0_20261004_195621/kiss_odom.csv'),
        ('dlio_base_full', D / 'dlio/runs/session_20260706_140610_best_base_full_20261006_172420/dlio_odom.csv'),
        ('gmap_run_20260706_keyframes', D / 'campinas_b/lio_sam/trajectory_20260706.csv'),
    ]:
        traj[name] = src
    for j, v in [('J1', 'baseline_loops'), ('J2', 'loop_closure_off'), ('J3', 'imu_rot_perm_tilt'), ('J4', 'imu_rot_camlidar'),
                 ('J3b', 'imu_rot_perm_tilt_repeat'), ('J3c', 'imu_rot_perm_tilt_repeat2'), ('J4b', 'imu_rot_camlidar_repeat'),
                 ('Jperm', 'imu_rot_permutation_r05')]:
        d = jd(j)
        if d is None and j in ('J3b', 'J3c', 'J4b', 'Jperm'):
            continue  # optional extra job
        if d:
            traj[f'liosam_{j}_{v}_odometry'] = d / 'odometry.csv'
            traj[f'liosam_{j}_{v}_keyframes'] = d / 'keyframes_final.csv'
        else:
            missing.append(f'{j} ({v}) trajectories: job not ok')
    d = jd('J5')
    if d:
        traj['dlio_J5_rep3'] = d / 'dlio_odom.csv'
    elif J.get('J5') and J['J5']['run_dir'] and Path(J['J5']['run_dir'], 'dlio_odom.csv').exists():
        traj['dlio_J5_rep3_crashed_at_1155s'] = Path(J['J5']['run_dir'], 'dlio_odom.csv')
    else:
        missing.append('J5 DLIO repeat: job not ok')
    d = jd('J5b')
    if d:
        traj['dlio_J5b_rep4'] = d / 'dlio_odom.csv'

    for name, src in traj.items():
        if Path(src).exists():
            norm_csv(src, PACK / 'trajectories' / f'{name}.csv')
        else:
            missing.append(f'{name}: {src} not found')
    try:
        rtab_poses(D / 'rtabmap/runs/session_20260706_140610_best_kisstuned_rtabloop_r1.0_20261004_195621/rtabmap_kissicp_loop.db',
                   PACK / 'trajectories' / 'rtabmap_loop_on_kiss_tuned.csv')
    except Exception as e:  # noqa: BLE001
        missing.append(f'RTAB-Map poses export failed: {e}')

    d = jd('J1')
    for f in ('loops.csv', 'kf_corrections.csv'):
        if d and (d / f).exists():
            shutil.copy(d / f, PACK / f)
        else:
            missing.append(f'{f} (J1)')
    for sub, pats in (('entropy', ['mme_*.csv']), ('entropy_15_600s', ['mme_*.csv']), ('trunks', ['dense_trunks_*.npz', 'trunks_pred_*.csv'])):
        files = [p for pat in pats for p in (JD / sub).glob(pat)]
        if files:
            (PACK / sub).mkdir(exist_ok=True)
            for p in files:
                shutil.copy(p, PACK / sub / p.name)
        else:
            missing.append(f'{sub}/ outputs')

    # ---------------- jobs_summary.md ----------------
    g = load_gps()
    lines = ['# Thesis job queue — summary (2026-10-07)', '',
             'Bag `~/bags/session_20260706_140610(best)`. ATE = `analysis/ate_vs_gps.py` (SE(2) fit, whole run). '
             'CPU = CPU-seconds of the SLAM processes (`tools/proc_sampler.py`), peak RAM = their summed RSS. '
             'LIO-SAM ATE is on the final optimized keyframes (as in the report).', '',
             '| Job | What | Status | ATE [m] | max [m] | end t [s] | CPU [s] | peak RAM [MB] | starved | Notes |',
             '|---|---|---|---|---|---|---|---|---|---|']
    what = {'J1': 'LIO-SAM baseline + loop recording', 'J2': 'LIO-SAM loop closure off',
            'J3': 'LIO-SAM IMU R = perm + tilt', 'J4': 'LIO-SAM IMU R = camera-LiDAR implied',
            'J5': 'DLIO full repeat', 'J6': 'map entropy', 'J7': 'trunk transfer + dense export',
            'J3b': 'LIO-SAM IMU R = perm + tilt, repeat (extra)',
            'J3c': 'LIO-SAM IMU R = perm + tilt, 2nd repeat (extra)',
            'J4b': 'LIO-SAM IMU R = camera-LiDAR implied, repeat (extra)',
            'Jperm': 'LIO-SAM IMU R = plain permutation, rate 0.5 (extra)',
            'J6b': 'map entropy 15-600 s, incl. J3/J4 before divergence (extra)',
            'J5b': 'DLIO full repeat, race-fixed build (extra)'}
    for j in ['J1', 'J2', 'J3', 'J3b', 'J3c', 'J4', 'J4b', 'Jperm', 'J5', 'J5b', 'J6', 'J6b', 'J7']:
        if j in ('J3b', 'J3c', 'J4b', 'Jperm', 'J6b', 'J5b') and j not in J:
            continue
        r = J.get(j)
        if not r:
            lines.append(f'| {j} | {what[j]} | not run | | | | | | | |')
            continue
        d = Path(r['run_dir']) if r['run_dir'] else None
        ate = mx = tend = cpu = ram = starv = ''
        note = ''
        tr = None
        if d and j in ('J1', 'J2', 'J3', 'J3b', 'J3c', 'J4', 'J4b', 'Jperm'):
            tr = d / 'keyframes_final.csv'
            e = json.loads((d / 'run_entry.json').read_text()) if (d / 'run_entry.json').exists() else {}
            res = e.get('resources') or {}
            cpu, ram, starv = res.get('cpu_seconds', ''), res.get('peak_rss_mb', ''), e.get('starvation_warnings', '')
            note = e.get('export', '')
        elif d and j in ('J5', 'J5b'):
            tr = d / 'dlio_odom.csv'
            e = json.loads((d / 'run_entry.json').read_text()) if (d / 'run_entry.json').exists() else {}
            res = e.get('resources') or {}
            cpu, ram, starv = res.get('cpu_seconds', ''), res.get('peak_rss_mb', ''), e.get('starvation_warnings', '')
        elif d and j in ('J6', 'J6b') and (d / 'mme_summary.csv').exists():
            note = '; '.join(f"{x['label']} {x['mme']}" for x in csv.DictReader(open(d / 'mme_summary.csv')))
            res = json.loads((d / 'resources.json').read_text()) if (d / 'resources.json').exists() else {}
            cpu, ram = res.get('cpu_seconds', ''), res.get('peak_rss_mb', '')
        elif d and j == 'J7':
            note = ', '.join(f"{p.name} {p.stat().st_size / 2**20:.0f} MB" for p in sorted(d.glob('dense_trunks_*.npz')))
        if tr is not None and tr.exists():
            try:
                ev = evaluate(load_traj(tr), g)
                ate, mx, tend = f"{ev['ate']:.2f}", f"{ev['max']:.1f}", f"{ev['t_end']:.0f}"
            except Exception as ex:  # noqa: BLE001
                note += f' (ATE failed: {ex})'
        lines.append(f"| {j} | {what[j]} | {r['status']} | {ate} | {mx} | {tend} | {cpu} | {ram} | {starv} | {note} |")
    lines += ['', 'Status: ok = the bag played to the end and the run tracked; diverged = the bag played to the end '
              'but the trajectory blew up (set by hand after checking it against J1); see the ATE/max column.', '']
    if (JD / 'notes.md').exists():
        lines += [(JD / 'notes.md').read_text(), '']
    lines += ['## Start/end closure (J9.4)', '',
              'Distance between the first and last pose of each trajectory (GPS: first vs last fix).', '',
              '| Trajectory | first-last XY [m] | first-last Z [m] |', '|---|---|---|',
              f"| GPS | {np.linalg.norm(g[-1, 1:3] - g[0, 1:3]):.2f} | |"]
    for p in sorted((PACK / 'trajectories').glob('*.csv')):
        a = load_traj(p)
        lines.append(f'| {p.stem} | {np.linalg.norm(a[-1, 1:3] - a[0, 1:3]):.2f} | {a[-1, 3] - a[0, 3]:.2f} |')
    lines.append('')
    if missing:
        lines += ['## Missing from the pack', ''] + [f'- {m}' for m in missing] + ['']
    lines += ['## Pack contents', ''] + [f'- `{p.relative_to(PACK)}`' for p in sorted(PACK.rglob('*')) if p.is_file()] + ['']
    ans = JD / 'j9_answers.md'
    lines += [ans.read_text()] if ans.exists() else ['## J9', '', '(j9_answers.md not written yet)']
    (JD / 'jobs_summary.md').write_text('\n'.join(lines) + '\n')
    shutil.copy(JD / 'jobs_summary.md', PACK / 'jobs_summary.md')
    zp = shutil.make_archive(str(JD / 'for_thesis_20261007'), 'zip', JD, 'for_thesis_20261007')
    print(f'pack: {PACK} ({len(list(PACK.rglob("*")))} entries), zip {Path(zp).stat().st_size / 2**20:.1f} MB; '
          f'missing: {len(missing)}')


if __name__ == '__main__':
    main()
