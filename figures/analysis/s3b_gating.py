#!/usr/bin/env python3
"""Section 3b - gating interval  sigma_loc < eps < d_min  for trunks, isolated weeds and weeds in dense patches.
usage: s3b_gating.py [DN] [R]    (re-sightings from the dense run, lag DN camera frames, matching radius R)
 * eps_min = 95th percentile of the position difference of true re-sightings (a radius that links 95 % of them)
   (for a Gaussian difference of two views eps_min = 2.45 sqrt2 sigma); sigma_loc reported as RMS/sqrt2 and MAD/sqrt2.
 * d_min   = 5th percentile of the nearest-neighbour distance between detections of the SAME frame (always distinct
   objects, no sigma_loc contamination); trunks: nearest-neighbour distance among surveyed trunks.
 * class: local density = colour-entity area (eps 0.4 m) within 1.5 m of the detection, from the frozen colour map
   (not from the detector under test for the weeds' d_min); isolated = lowest third, dense = highest third.
   interval exists  <=>  eps_min < d_min."""
import sys, json
from scipy.spatial import cKDTree
from common import *
from sigma_lib import *
from entities import entities
dn = int(sys.argv[1]) if len(sys.argv) > 1 else 50; R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
ec = pd.read_csv(TAB / 'entities_colour.csv'); tree = cKDTree(ec[['x', 'y']].values); area = ec.area_med.values
def density(xy):
    return np.array([area[i].sum() for i in tree.query_ball_point(xy, 1.5)])
rows = []
tr = load_trunks(); d = cKDTree(tr[['xe', 'ye']].values).query(tr[['xe', 'ye']].values, k=2)[0][:, 1]
m = pd.read_csv(REPO / 'orchard-weed-detection/trunk/v2_out/per_trunk_matches.csv'); m['err_cm'] = pd.to_numeric(m.err_cm, errors='coerce'); m = m.dropna(subset=['err_cm'])
rows.append(dict(det='trunk', cls='all', n_pairs=len(m), sigma_rms=0.080, sigma_mad=np.nan, eps_min=float(np.percentile(m.err_cm, 95) / 100 * np.sqrt(2)),
                 d_min=float(np.percentile(d, 5)), n_dmin=len(d), note='sigma = tape RMS (8.0 cm); eps_min = p95 of error x sqrt2 (two detections)'))
# thresholds from the colour density at all YOLO+colour detections of the frozen run
allxy = np.vstack([load_weeds(x)[['x_enu', 'y_enu']].values for x in ('yolo', 'colour')])
# density is computed in ENU entity coordinates (ec.x, ec.y are ENU)
q1, q2 = np.percentile(density(allxy), [100 / 3, 200 / 3]); print('density terciles (m2 colour area within 1.5 m):', q1, q2)
for det in ('yolo', 'colour'):
    w, f = prep(det, 'dense'); w['dens'] = density(w[['x_enu', 'y_enu']].values)
    P = build_pairs(w, f, R, dn=dn); P['dens'] = w.dens.loc[P.ia].values
    # d_min: same-frame NN among detections, per class
    dm = {}
    for cls, sel in (('isolated', w.dens <= q1), ('dense', w.dens >= q2), ('all', w.dens >= 0)):
        nn = []
        for st, g in w[sel].groupby('stem'):
            if len(g) > 1:
                dd = cKDTree(g[['x_enu', 'y_enu']].values).query(g[['x_enu', 'y_enu']].values, k=2)[0][:, 1]; nn += dd.tolist()
        dm[cls] = nn
    for cls, sel in (('isolated', P.dens <= q1), ('dense', P.dens >= q2), ('all', P.dens >= 0)):
        Q = P[sel]; x = np.r_[Q.along, Q.across]
        rows.append(dict(det=det, cls=cls, n_pairs=len(Q), sigma_rms=rms_sigma(x) if len(Q) > 3 else np.nan, sigma_mad=mad_sigma(x) if len(Q) > 3 else np.nan,
                         eps_min=float(np.percentile(Q.dist, 95)) if len(Q) > 5 else np.nan, eps_min_gauss=(2.447 * np.sqrt(2) * np.sqrt((mad_sigma(Q.along) ** 2 + mad_sigma(Q.across) ** 2) / 2)) if len(Q) > 5 else np.nan, d_min=float(np.percentile(dm[cls], 5)) if len(dm[cls]) > 5 else np.nan,
                         n_dmin=len(dm[cls]), note=f'R={R}, lag {dn} frames; false-pair contamination possible'))
G = pd.DataFrame(rows); G['interval_exists'] = G.eps_min_gauss.fillna(G.eps_min) < G.d_min; G['ratio_dmin_over_epsmin'] = G.d_min / G.eps_min_gauss.fillna(G.eps_min)
G.to_csv(TAB / f'gating_dn{dn}_R{R}.csv', index=False); print(G.round(3).to_string())
