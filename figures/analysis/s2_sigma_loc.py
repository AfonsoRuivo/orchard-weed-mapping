#!/usr/bin/env python3
"""Section 2 - position consistency sigma_loc from re-sightings of the same weed in two frames dn camera frames apart.
usage: s2_sigma_loc.py RUN DET [dn ...]      RUN = frozen | dense
Pairing: mutual nearest neighbours (ENU/SLAM distance) between the detections of frame n and n+dn, radius R.
d = p(n+dn) - p(n) split along / across the (smoothed) travel direction; sigma = RMS(d)/sqrt2 and 1.4826 MAD/sqrt2.
Outputs (tables/sigma_<run>_<det>_dn<dn>_*.csv): sweep (R), null, pairs, bins, fit."""
import sys, json
from scipy.optimize import nnls
from common import *
from sigma_lib import *
import weed_map_cached as wm
from sigma_model import jacobian

run, det = sys.argv[1], sys.argv[2]; dns = [int(x) for x in sys.argv[3:]] or [50]
Rs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]; rng = np.random.default_rng(0)
w, f = prep(det, run)
for dn in dns:
    tag = f'sigma_{run}_{det}_dn{dn}'; rows = []; nulls = []
    for R in Rs:
        P = build_pairs(w, f, R, dn=dn)
        nl = [len(build_pairs(w, f, R, shift=s, rng=rng, dn=dn)) for s in (2.0, 3.0, 4.0) for _ in range(10)]
        row = dict(R=R, n_pairs=len(P), null_pairs=float(np.mean(nl)), false_frac=float(np.mean(nl)) / max(len(P), 1))
        if len(P) > 3:
            for ax in ('along', 'across'):
                x = P[ax].values
                row.update({f'rms_{ax}': rms_sigma(x), f'mad_{ax}': mad_sigma(x), f'mean_{ax}': x.mean()})
            lo, hi = np.percentile(P.dist, [50, 95]); row.update(dist_med=lo, dist_p95=hi)
        rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(TAB / f'{tag}_sweep.csv', index=False); print(tag); print(S.round(3).to_string())
