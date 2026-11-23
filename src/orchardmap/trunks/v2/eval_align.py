#!/usr/bin/env python3
"""
eval_align.py - score detections from a map whose frame differs from the July GlobalMap frame.

  python3 eval_align.py --det trunks.csv --out score_dir/ [--no-align]

1. --align: fit ONE rigid SE(2) (rotation + translation, NO scale: map scale is part of what is measured)
   detections -> hand-picked trunks, iterated (match with Hungarian, gate 1.5 -> 0.5 m; pairs > 0.30 m are
   left out of the last fits so one bad trunk cannot pull the frame). One transform for both ladders, so
   drift between the ladders is not hidden.
2. Run the unchanged ../gt/eval_trunks.py on the aligned detections (all, L1, L2).
3. Extras:
   - S scale per ladder: least-squares d_map = (1 + s) d_tape over the S tapes (s in %; S bias in cm too).
   - F1 strict: matches farther than --strict m from the hand pick counted as a miss + a false positive
     (R2-T2 sits on a stake 46 cm away; at 0.30 m it is a miss).
   - hand-pick-free: per ladder, the tape-only layout (gt_trunks_tape_fit.csv) is fitted to the detections
     with a similarity (rotation, translation, uniform scale: the map is ~1-2 % short, ~0.5-1 m over a row,
     so a rigid fit cannot match the far end), hand picks only give the initial guess; detections are
     matched to it (gate 0.5 m) and we report the tape RMS on that matching, the layout RMS (detection vs
     fitted tape-fit position) and the fitted scale. None of these uses where a person clicked a trunk.
   - similarity: per ladder, detections -> hand picks with a uniform scale too: the hand picks inherit the July
     map's scale (~2 % short), so a map with another scale shows along-row errors growing to the row ends; this
     separates that from centre noise (pos RMS, strict F1 recomputed after the fit).
Writes <out>/score_{all,L1,L2}.{json,txt}, matches.csv, aligned detections and summary.json.
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import GT_DIR, load_gt

BAD = {'L1_SB3', 'L1_SB7'}


def rigid2d(A, B, scale=False):
    """R, t minimising |R A + t - B| (with scale=True, R includes a uniform scale: Umeyama)."""
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H)
    Dg = np.diag([1, np.linalg.det(Vt.T @ U.T)]); R = Vt.T @ Dg @ U.T
    if scale: R = R * (S * np.diag(Dg)).sum() / ((A - ca) ** 2).sum()
    return R, cb - R @ ca


def hmatch(A, B, gate):
    C = np.linalg.norm(A[:, None] - B[None], axis=2); gi, di = linear_sum_assignment(np.where(C < gate, C, 1e6))
    k = C[gi, di] < gate
    return gi[k], di[k], C[gi[k], di[k]]


def align(D, G, iters=10):
    R, t = np.eye(2), np.zeros(2)
    for it in range(iters):
        gate = max(0.5, 1.5 - 0.25 * it); gi, di, e = hmatch(G, D @ R.T + t, gate)
        k = e < (0.30 if it >= 4 else gate)
        R, t = rigid2d(D[di[k]], G[gi[k]])
    gi, di, e = hmatch(G, D @ R.T + t, 0.5)
    return R, t, dict(theta_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), tx=float(t[0]), ty=float(t[1]),
                      n_pairs=int((e < 0.30).sum()), rms_cm=float(np.sqrt(np.mean(e[e < 0.30] ** 2)) * 100))


def tapes():
    return [m for m in csv.DictReader(open(GT_DIR / 'gt_rows_all.csv')) if m['value_m'].strip() and m['measurement_id'] not in BAD]


def tape_res(pos):
    out = []
    for m in tapes():
        f, t = m['from_trunk'], m['to_trunk']
        if f in pos and t in pos:
            out.append((m['type'], int(f[1]) <= 2, np.linalg.norm(pos[f] - pos[t]), float(m['value_m'])))
    return out


def s_scale(res, ladder1):
    v = [(dm, dt) for ty, l1, dm, dt in res if ty == 'S' and l1 == ladder1]
    if not v: return None, None
    dm, dt = np.array(v).T
    return float((dm @ dt / (dt @ dt) - 1) * 100), float(np.mean(dm - dt) * 100)


def handpick_free(D, G, gxy):
    TF = list(csv.DictReader(open(GT_DIR / 'gt_trunks_tape_fit.csv'))); hp = {g['id']: xy for g, xy in zip(G, gxy)}
    pos, lay, scl = {}, [], {}
    for L in (1, 2):
        ids = [r['id'] for r in TF if int(r['ladder']) == L]
        T = np.array([[float(r['x_ladder']), float(r['y_ladder'])] for r in TF if int(r['ladder']) == L])
        H = np.array([hp[i] for i in ids]); cands = []
        for flip in (1, -1):    # the tape-fit frame has arbitrary handedness
            Tf = T * [1, flip]; R, t = rigid2d(Tf, H, True); cands.append((np.linalg.norm(Tf @ R.T + t - H), Tf, R, t))
        _, T, R, t = min(cands, key=lambda c: c[0])     # hand picks give the initial guess only
        for it in range(8):
            gi, di, e = hmatch(T @ R.T + t, D, 0.5 if it else 1.0)
            k = e < (0.30 if it >= 3 else 1.0); R, t = rigid2d(T[gi[k]], D[di[k]], True)
        gi, di, e = hmatch(T @ R.T + t, D, 0.5)
        for i, j, ee in zip(gi, di, e): pos[ids[i]] = D[j]; lay.append((L, ee))
        scl[L] = float((np.sqrt(abs(np.linalg.det(R))) - 1) * 100)
    res = tape_res(pos); e = np.array([dm - dt for _, _, dm, dt in res]) * 100
    lay = np.array(lay)
    out = dict(n_matched=len(pos), tape_rms_cm=float(np.sqrt(np.mean(e ** 2))), tapes_used=len(e),
               layout_rms_cm=float(np.sqrt(np.mean(lay[:, 1] ** 2)) * 100))
    for L in (1, 2):
        el = np.array([dm - dt for _, l1, dm, dt in res if l1 == (L == 1)]) * 100
        out[f'L{L}'] = dict(tape_rms_cm=float(np.sqrt(np.mean(el ** 2))), tapes_used=len(el),
                            layout_rms_cm=float(np.sqrt(np.mean(lay[lay[:, 0] == L, 1] ** 2)) * 100), layout_scale_pct=scl[L])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--det', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--no-align', action='store_true'); ap.add_argument('--strict', type=float, default=0.30)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.det))); D = np.array([[float(r['x']), float(r['y'])] for r in rows])
    G, gxy = load_gt()
    R, t, info = (np.eye(2), np.zeros(2), dict(theta_deg=0, tx=0, ty=0)) if a.no_align else align(D, gxy)
    Da = D @ R.T + t
    with open(out / 'det_aligned.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r, p in zip(rows, Da): w.writerow(dict(r, x=f'{p[0]:.4f}', y=f'{p[1]:.4f}'))
    S = dict(align=info)
    ev = [sys.executable, '-I', str(GT_DIR / 'eval_trunks.py'), '--det', str(out / 'det_aligned.csv')]
    for L, tag in [(None, 'all'), (1, 'L1'), (2, 'L2')]:
        cmd = ev + ['--json', str(out / f'score_{tag}.json')] + (['--ladder', str(L)] if L else [])
        if tag == 'all': cmd += ['--matches', str(out / 'matches.csv')]
        txt = subprocess.run(cmd, capture_output=True, text=True).stdout; open(out / f'score_{tag}.txt', 'w').write(txt)
        S[tag] = json.load(open(out / f'score_{tag}.json'))
    M = list(csv.DictReader(open(out / 'matches.csv')))
    pos = {m['id']: np.array([float(m['det_x']), float(m['det_y'])]) for m in M if m['err_cm'] != 'MISSED'}
    res = tape_res(pos)
    for L in (1, 2):
        sc, sb = s_scale(res, L == 1); S[f'L{L}']['S_scale_pct'], S[f'L{L}']['S_bias_cm'] = sc, sb
    far = sum(1 for m in M if m['err_cm'] != 'MISSED' and float(m['err_cm']) > a.strict * 100)
    tp, fp, fn = S['all']['tp'] - far, S['all']['fp'] + far, S['all']['fn'] + far
    S['strict'] = dict(gate_m=a.strict, n_far=far, f1=2 * tp / (2 * tp + fp + fn),
                       far_ids=[m['id'] for m in M if m['err_cm'] != 'MISSED' and float(m['err_cm']) > a.strict * 100])
    S['handpick_free'] = handpick_free(Da, G, gxy)
    # the hand picks carry the July map's own scale error (map ~2 % short); a map with a different scale gets
    # along-row position errors that grow towards the row ends. Same numbers after a per-ladder similarity fit:
    sim = {}; far_s = 0
    for L, rr in ((1, '12'), (2, '34')):
        mm = [m for m in M if m['err_cm'] != 'MISSED' and m['id'][1] in rr]
        Gm = np.array([[float(m['gt_x']), float(m['gt_y'])] for m in mm]); Dm = np.array([[float(m['det_x']), float(m['det_y'])] for m in mm])
        Rs, ts = rigid2d(Dm, Gm, True); e = np.linalg.norm(Dm @ Rs.T + ts - Gm, axis=1)
        sim[f'L{L}'] = dict(scale_vs_handpicks_pct=float((np.sqrt(np.linalg.det(Rs)) - 1) * 100), pos_rms_cm=float(np.sqrt(np.mean(e ** 2)) * 100),
                            pos_median_cm=float(np.median(e) * 100))
        far_s += int((e > a.strict).sum())
    tp, fp, fn = S['all']['tp'] - far_s, S['all']['fp'] + far_s, S['all']['fn'] + far_s
    sim['strict_f1'] = 2 * tp / (2 * tp + fp + fn); sim['n_far'] = far_s; S['similarity'] = sim
    json.dump(S, open(out / 'summary.json', 'w'), indent=1)
    A, hf = S['all'], S['handpick_free']; T = A['tape']
    print(f"align {info['theta_deg']:+.3f} deg ({info['tx']:+.3f}, {info['ty']:+.3f}) | F1 {A['f1']:.3f} (strict {S['strict']['f1']:.3f}) "
          f"| tape RMS {A['tape_all_rms_cm']:.1f} (S {T['S']['rms_cm']:.1f} X {T['X']['rms_cm']:.1f} D {T['D']['rms_cm']:.1f}) "
          f"on {A['tapes_used']} | S scale L1 {S['L1']['S_scale_pct']:+.2f}% L2 {S['L2']['S_scale_pct']:+.2f}% "
          f"| hand-pick-free tape {hf['tape_rms_cm']:.1f} on {hf['tapes_used']}, layout {hf['layout_rms_cm']:.1f}")


if __name__ == '__main__':
    main()
