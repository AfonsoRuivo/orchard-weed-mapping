#!/usr/bin/env python3
"""
eval_trunks.py - score a trunk detector against the two ground truths.

  python3 eval_trunks.py --det my_detections.csv
  python3 eval_trunks.py --det my_detections.csv --ladder 2 --json score.json

--det: CSV with the detected trunks, in the GlobalMap frame. Columns x, y (also accepted:
       cx, cy) and optionally r (or radius). One row per trunk. Extra columns are ignored.

Two references (both in this folder):
  gt_trunks_map.csv   the 106 trunk centres picked by hand in the GlobalMap
                      (street-view tool, checked from above). Used to match detections.
  gt_rows_all.csv     the tape survey: 206 distances between those same trunks
                      (S along the row, X across, D diagonal). Independent of the map.

What it reports, only inside the surveyed rows (a corridor around each row, so trees and
bushes outside the survey are not counted as false positives):
  - detection:  TP / FP / FN, precision, recall, F1  (Hungarian matching, gate --gate m)
  - position:   distance from each matched detection to the hand-picked centre (median, p90)
  - tape:       using the DETECTED centres, |d_map - d_tape| for every tape whose two trunks
                were both detected (RMS per type). This is the number that does not depend
                on how the trunks were picked by hand. The two tapes known to be misread
                (L1_SB3, L1_SB7) are left out.
  - radius:     mean / spread of the detected radii (no circumferences measured yet)
The last line is a one-line summary, handy to compare runs.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BAD_TAPES = {'L1_SB3', 'L1_SB7'}


def load_det(p):
    rows = list(csv.DictReader(open(p)))
    if not rows: sys.exit('no detections in ' + p)
    k = {c.strip().lower(): c for c in rows[0]}
    kx, ky = k.get('x') or k.get('cx'), k.get('y') or k.get('cy')
    kr = k.get('r') or k.get('radius') or k.get('r_m')
    if not kx or not ky: sys.exit(f'need columns x,y (found {list(rows[0])})')
    xy = np.array([[float(r[kx]), float(r[ky])] for r in rows])
    rr = np.array([float(r[kr]) if kr and r[kr] not in ('', 'nan') else np.nan for r in rows])
    return xy, rr


def corridor_mask(pts, gt_xy, gt_rows, half_width, end_ext):
    """True for points within half_width of a surveyed row segment (extended by end_ext)."""
    keep = np.zeros(len(pts), bool)
    for r in np.unique(gt_rows):
        g = gt_xy[gt_rows == r]
        c = g.mean(0); u = np.linalg.svd(g - c)[2][0]; n = np.array([-u[1], u[0]])
        s = (g - c) @ u; s0, s1 = s.min() - end_ext, s.max() + end_ext
        ps, pn = (pts - c) @ u, (pts - c) @ n
        keep |= (ps >= s0) & (ps <= s1) & (np.abs(pn) <= half_width)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--det', required=True)
    ap.add_argument('--gt-map', default=str(HERE / 'gt_trunks_map.csv'))
    ap.add_argument('--gt-tape', default=str(HERE / 'gt_rows_all.csv'))
    ap.add_argument('--gate', type=float, default=0.5, help='max distance for a match (m)')
    ap.add_argument('--corridor', type=float, default=1.2, help='half-width of the row corridor (m)')
    ap.add_argument('--ladder', type=int, choices=[1, 2], default=None, help='score one ladder only')
    ap.add_argument('--json', default=None); ap.add_argument('--matches', default=None, help='write per-trunk CSV')
    a = ap.parse_args()

    G = list(csv.DictReader(open(a.gt_map)))
    if a.ladder: G = [g for g in G if (int(g['row']) <= 2) == (a.ladder == 1)]
    gid = [g['id'] for g in G]; gxy = np.array([[float(g['x']), float(g['y'])] for g in G]); grow = np.array([int(g['row']) for g in G])
    dxy, dr = load_det(a.det)
    inside = corridor_mask(dxy, gxy, grow, a.corridor, 1.5)
    D, R = dxy[inside], dr[inside]

    # Hungarian matching with a gate
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(gxy[:, None] - D[None], axis=2) if len(D) else np.zeros((len(gxy), 0))
    big = 1e6; Cg = np.where(C <= a.gate, C, big)
    match = {}
    if len(D):
        gi, di = linear_sum_assignment(Cg)
        match = {int(i): int(j) for i, j in zip(gi, di) if Cg[i, j] < big}
    tp = len(match); fn = len(G) - tp; fp = len(D) - tp
    P = tp / (tp + fp) if tp + fp else 0.0; Rc = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
    perr = np.array([C[i, j] for i, j in match.items()]) * 100

    # tape comparison with the detected centres
    det_of = {gid[i]: D[j] for i, j in match.items()}
    tape = {}
    for m in csv.DictReader(open(a.gt_tape)):
        if not m['value_m'].strip() or m['measurement_id'] in BAD_TAPES: continue
        f, t = m['from_trunk'], m['to_trunk']
        if f in det_of and t in det_of:
            tape.setdefault(m['type'], []).append(np.linalg.norm(det_of[f] - det_of[t]) - float(m['value_m']))
    n_tape_total = sum(1 for m in csv.DictReader(open(a.gt_tape)) if m['value_m'].strip()
                       and m['measurement_id'] not in BAD_TAPES and m['from_trunk'] in gid and m['to_trunk'] in gid)
    T = {k: dict(n=len(v), bias_cm=float(np.mean(v) * 100), rms_cm=float(np.sqrt(np.mean(np.square(v))) * 100)) for k, v in tape.items()}
    allv = [x for v in tape.values() for x in v]
    rr = np.array([R[j] for j in match.values()]); rr = rr[np.isfinite(rr)]

    res = dict(n_gt=len(G), n_det_in_rows=int(len(D)), n_det_outside_rows=int((~inside).sum()), tp=tp, fp=fp, fn=fn,
               precision=P, recall=Rc, f1=F1,
               pos_err_cm=dict(median=float(np.median(perr)) if tp else None, p90=float(np.percentile(perr, 90)) if tp else None),
               tape=T, tape_all_rms_cm=float(np.sqrt(np.mean(np.square(allv))) * 100) if allv else None,
               tapes_used=len(allv), tapes_possible=n_tape_total,
               radius_cm=dict(n=int(len(rr)), mean=float(rr.mean() * 100) if len(rr) else None,
                              std=float(rr.std() * 100) if len(rr) else None))
    print(f"ground truth: {len(G)} trunks{' (ladder %d)' % a.ladder if a.ladder else ''}; detections in the surveyed rows: {len(D)} "
          f"(+{(~inside).sum()} elsewhere, not scored)")
    print(f"detection : TP {tp}  FP {fp}  FN {fn}   P {P:.2f}  R {Rc:.2f}  F1 {F1:.2f}")
    if tp: print(f"position  : vs hand-picked centre  median {np.median(perr):.1f} cm   p90 {np.percentile(perr, 90):.1f} cm")
    for k in 'SXD':
        if k in T: print(f"tape {k}    : n {T[k]['n']:3d}  bias {T[k]['bias_cm']:+5.1f} cm  RMS {T[k]['rms_cm']:5.1f} cm")
    if len(rr): print(f"radius    : mean {rr.mean()*100:.1f} cm  std {rr.std()*100:.1f} cm  (n {len(rr)})")
    print(f"SUMMARY  F1 {F1:.3f} | pos median {np.median(perr) if tp else float('nan'):.1f} cm | tape RMS "
          f"{res['tape_all_rms_cm'] or float('nan'):.1f} cm on {len(allv)}/{n_tape_total} tapes")
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)
    if a.matches:
        with open(a.matches, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['id', 'gt_x', 'gt_y', 'det_x', 'det_y', 'err_cm', 'r_det'])
            for i, g in enumerate(G):
                if i in match:
                    j = match[i]; w.writerow([gid[i], *gxy[i], *D[j], f'{C[i, j]*100:.1f}', R[j]])
                else:
                    w.writerow([gid[i], *gxy[i], '', '', 'MISSED', ''])


if __name__ == '__main__':
    main()
