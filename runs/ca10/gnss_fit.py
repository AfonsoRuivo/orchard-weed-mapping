#!/usr/bin/env python3
"""SE2 fit of a SLAM trajectory (or a part of it) to the CA-10 GNSS track, as in the CB-11 evaluation
(orchard-slam/analysis/ate_vs_gps.py: GNSS east/north interpolated at the SLAM sample times, rigid 2D Umeyama,
no scale, ATE = RMSE of the residuals, max, RPE10/RPE50 = drift per 10/50 m of GNSS travel, %).

Differences, all from the brief or forced by the gap:
  * only GNSS fixes inside [t_from, t_to] (bag s) are used, and fixes in the EXCLUDE window 822.8-838.8 s
    (receiver re-acquiring, REPORT section 3) are dropped;
  * SLAM samples are kept only where GNSS can be interpolated without crossing a GNSS gap > 2.5 s
    (the 811.6-822.8 s outage), and, as in CB-11, the first 10 s of the half are skipped (initialisation);
GNSS = ~/ca10_gap/gnss_enu.csv (bag-ENU, origin = first CA-10 fix; columns tb = bag s, e, n).

Library use: fit(traj_array, t_from, t_to) -> dict(R, t, ...); apply(R, t, xy).
CLI: gnss_fit.py traj.csv t_from t_to out.json [--skip 10]
"""
import csv
import json
import sys

import numpy as np

T0 = 1782577258.896306
GNSS = '/home/afonso/ca10_gap/gnss_enu.csv'
EXCLUDE = (822.8, 838.8)


def load_gnss(path=GNSS):
    rows = list(csv.DictReader(open(path)))
    g = np.array([[float(r['tb']), float(r['e']), float(r['n'])] for r in rows])
    g = g[np.isfinite(g).all(1)]
    return g[np.argsort(g[:, 0])]


def load_traj(path):
    """t (bag s), x, y, z, qx, qy, qz, qw from a CSV with t/timestamp (unix) or a TUM file."""
    if str(path).endswith('.tum') or str(path).endswith('.txt'):
        a = np.loadtxt(path)
        a[:, 0] -= T0
        return a
    rows = list(csv.DictReader(open(path)))
    tk = 't' if 't' in rows[0] else 'timestamp'
    a = np.array([[float(r[tk]) - T0] + [float(r[k]) for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] for r in rows])
    return a[np.argsort(a[:, 0], kind='stable')]


def se2(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mq - R @ mp


def gnss_at(t, g, max_gap=2.5):
    """Interpolated GNSS at t, NaN where the bracketing fixes are > max_gap apart or t is outside."""
    i = np.searchsorted(g[:, 0], t)
    ok = (i > 0) & (i < len(g))
    i = np.clip(i, 1, len(g) - 1)
    ok &= (g[i, 0] - g[i - 1, 0]) <= max_gap
    out = np.stack([np.interp(t, g[:, 0], g[:, k]) for k in (1, 2)], 1)
    out[~ok] = np.nan
    return out


def rpe(A, G, L):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
    errs = []
    for i in range(len(A)):
        j = np.searchsorted(d, d[i] + L)
        if j >= len(A):
            break
        errs.append(np.linalg.norm((A[j] - A[i]) - (G[j] - G[i])) / L * 100)
    return float(np.mean(errs)) if errs else float('nan')


def fit(a, t_from, t_to, skip=10.0, g=None, R=None, t=None):
    """Fit (or, with R,t given, only evaluate) trajectory a against GNSS in [t_from, t_to]."""
    g = load_gnss() if g is None else g
    gm = (g[:, 0] >= t_from) & (g[:, 0] <= t_to) & ~((g[:, 0] >= EXCLUDE[0]) & (g[:, 0] <= EXCLUDE[1]))
    gh = g[gm]
    m = (a[:, 0] >= t_from + skip) & (a[:, 0] <= t_to)
    s = a[m]
    G = gnss_at(s[:, 0], gh)
    ok = np.isfinite(G).all(1)
    s, G = s[ok], G[ok]
    if R is None:
        R, t = se2(s[:, 1:3], G)
    A = s[:, 1:3] @ np.asarray(R).T + t
    e = np.linalg.norm(A - G, axis=1)
    return dict(R=np.asarray(R).tolist(), t=np.asarray(t).tolist(), yaw_deg=float(np.degrees(np.arctan2(R[1][0], R[0][0]))),
                n=int(len(s)), n_fixes=int(len(gh)), t_first=float(s[0, 0]), t_last=float(s[-1, 0]),
                ate_m=float(np.sqrt((e ** 2).mean())), max_m=float(e.max()), median_m=float(np.median(e)),
                rpe10_pct=rpe(A, G, 10), rpe50_pct=rpe(A, G, 50),
                _t=s[:, 0], _err=e, _A=A, _G=G)


def public(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}


if __name__ == '__main__':
    path, t_from, t_to, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    skip = float(sys.argv[sys.argv.index('--skip') + 1]) if '--skip' in sys.argv else 10.0
    r = fit(load_traj(path), t_from, t_to, skip)
    r = dict(traj=path, window=[t_from, t_to], skip_s=skip, exclude=EXCLUDE, method='SE2 Umeyama as ate_vs_gps.py',
             **public(r))
    json.dump(r, open(out, 'w'), indent=1)
    print(f"{path}: n={r['n']} ATE {r['ate_m']:.3f} m, max {r['max_m']:.2f} m, RPE50 {r['rpe50_pct']:.2f} %, "
          f"yaw {r['yaw_deg']:.2f} deg")
