#!/usr/bin/env python3
"""Post-process one LIO-SAM run directory into the thesis exports.

    python3 liosam_export.py <run_dir> [out_dir]

out_dir defaults to run_dir; give one to export an older run without adding files to it.

Reads <run_dir>/map/transformations.pcd (final optimized keyframes: x y z intensity roll
pitch yaw time; intensity = keyframe index) and, when present, the liosam_run_recorder.py
outputs. Writes:
  keyframes_final.csv  t,x,y,z,qx,qy,qz,qw   (final optimized keyframe trajectory, with times)
  kf_corrections.csv   kf,t,x_live,y_live,z_live,x_final,y_final,z_final,dxy,dz,d3
                       (live = pose when the keyframe was created; needs keyframes_live.csv)
  loops.csv            t,kf_from,kf_to,x_from,y_from,x_to,y_to,match_err_m,dt_kf_s,dist_xy_m,path_dist_m,revisit
                       (one row per accepted loop; positions are final; needs loops_raw.csv
                        and loops_final_edges.csv; match_err_m = distance of the edge endpoints
                        to the keyframes they were matched to, ~0 if the match is exact;
                        dt_kf_s = time between the two keyframes, dist_xy_m = straight-line distance,
                        path_dist_m = distance driven between them, revisit = 1 if path_dist >
                        3*dist_xy + 10 m, i.e. the robot came back rather than just moving on)
Prints a one-line summary (counts, max correction).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read_pcd(path):
    """Minimal PCD reader (ascii or binary, no compression) -> structured numpy array."""
    with open(path, 'rb') as f:
        hdr = {}
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if not line or line.startswith('#'):
                continue
            k, *v = line.split()
            hdr[k] = v
            if k == 'DATA':
                break
        names = hdr['FIELDS']
        types = {('F', '4'): 'f4', ('F', '8'): 'f8', ('U', '4'): 'u4', ('I', '4'): 'i4', ('U', '1'): 'u1',
                 ('U', '2'): 'u2', ('I', '2'): 'i2', ('I', '1'): 'i1'}
        dt = np.dtype([(n, types[(t, s)]) for n, t, s in zip(names, hdr['TYPE'], hdr['SIZE'])])
        n = int(hdr['POINTS'][0])
        if hdr['DATA'][0] == 'binary':
            return np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        if hdr['DATA'][0] == 'ascii':
            return np.loadtxt(f, dtype=dt, max_rows=n, ndmin=1)
        raise ValueError(f'unsupported PCD DATA {hdr["DATA"]}')


def final_keyframes(run_dir):
    k = read_pcd(Path(run_dir) / 'map' / 'transformations.pcd')
    k = k[np.argsort(k['intensity'])]
    # LIO-SAM: Rot3::RzRyRx(roll, pitch, yaw) == extrinsic xyz Euler (roll about x first)
    q = Rotation.from_euler('xyz', np.stack([k['roll'], k['pitch'], k['yaw']], 1)).as_quat()
    return np.column_stack([k['time'], k['x'], k['y'], k['z'], q]), k['intensity'].astype(int)


def write_csv(path, header, rows, fmt='{:.9f}'):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([x if isinstance(x, (int, np.integer)) else fmt.format(x) for x in r])


def main():
    rd = Path(sys.argv[1])
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else rd
    od.mkdir(parents=True, exist_ok=True)
    kf, idx = final_keyframes(rd)
    write_csv(od / 'keyframes_final.csv', ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], kf)
    msg = [f'{len(kf)} final keyframes']

    live_p = rd / 'keyframes_live.csv'
    if live_p.exists():
        live = {}
        for r in csv.DictReader(open(live_p)):
            live.setdefault(round(float(r['t']), 4), [float(r[c]) for c in 'xyz'])
        rows = []
        for i, k in zip(idx, kf):
            L = live.get(round(k[0], 4))
            if L is None:
                continue
            d = np.array(k[1:4]) - L
            rows.append([int(i), k[0], *L, *k[1:4], np.linalg.norm(d[:2]), d[2], np.linalg.norm(d)])
        write_csv(od / 'kf_corrections.csv',
                  ['kf', 't', 'x_live', 'y_live', 'z_live', 'x_final', 'y_final', 'z_final', 'dxy', 'dz', 'd3'], rows)
        d3 = np.array([r[-1] for r in rows]) if rows else np.zeros(1)
        msg.append(f'kf_corrections: {len(rows)} matched, max 3D {d3.max():.2f} m, >10 cm: {(d3 > 0.1).sum()}')

    raw_p, fin_p = rd / 'loops_raw.csv', rd / 'loops_final_edges.csv'
    if raw_p.exists() and fin_p.exists():
        first_t = {int(r['edge_idx']): float(r['t']) for r in csv.DictReader(open(raw_p))}
        tree = cKDTree(kf[:, 1:4])
        # cumulative path length along the final keyframes, to tell a revisit (long way round)
        # from a constraint to an older keyframe on the same stretch (path ~= straight line)
        s_path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kf[:, 1:3], axis=0), axis=1))])
        rows = []
        for r in csv.DictReader(open(fin_p)):
            a = np.array([float(r[c]) for c in ('x_from', 'y_from', 'z_from')])
            b = np.array([float(r[c]) for c in ('x_to', 'y_to', 'z_to')])
            (da, ia), (db, ib) = tree.query(a), tree.query(b)
            dt = kf[ia, 0] - kf[ib, 0]
            de = np.linalg.norm(a[:2] - b[:2])
            dp = abs(s_path[ia] - s_path[ib])
            rows.append([first_t.get(int(r['edge_idx']), float('nan')), int(idx[ia]), int(idx[ib]),
                         a[0], a[1], b[0], b[1], max(da, db), dt, de, dp, int(dp > 3 * de + 10)])
        rows.sort(key=lambda r: r[0])
        write_csv(od / 'loops.csv', ['t', 'kf_from', 'kf_to', 'x_from', 'y_from', 'x_to', 'y_to', 'match_err_m',
                                     'dt_kf_s', 'dist_xy_m', 'path_dist_m', 'revisit'], rows)
        nrev = sum(r[-1] for r in rows)
        msg.append(f'loops: {len(rows)} accepted, {nrev} true revisits (path > 3x straight-line + 10 m)' +
                   (f', max endpoint match error {max(r[7] for r in rows):.3f} m' if rows else ''))
    print(f'{rd.name}: ' + '; '.join(msg))


if __name__ == '__main__':
    main()
