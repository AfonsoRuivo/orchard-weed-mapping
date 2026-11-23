#!/usr/bin/env python3
"""Section 2c - sigma_loc vs range (binned, with bootstrap CI over frame pairs), predicted curve from plane noise,
and the signed along-track range bias through the origin: d_along = b2 (rho2^2 - rho1^2)  [plane/pitch-like] or
b1 (rho2 - rho1) [range-scale-like].  Reads tables/sigma_<run>_<det>_dn<dn>_pairs_R<R>.csv from s2b."""
import json
from common import *
CASES = [('yolo', 10, 0.5), ('colour', 10, 0.4), ('yolo', 50, 0.75), ('colour', 50, 0.5), ('yolo', 20, 0.5), ('colour', 20, 0.4)]
EDGES = [1.0, 2.0, 3.0, 4.0, 5.0]; rng = np.random.default_rng(0); out, bias = [], []
for det, dn, R in CASES:
    f = TAB / f'sigma_dense_{det}_dn{dn}_pairs_R{R}.csv'
    if not f.exists(): continue
    M = pd.read_csv(f); M['rbar'] = 0.5 * (M.rho1 + M.rho2); keys = M.n.unique()
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        B = M[(M.rbar >= lo) & (M.rbar < hi)]
        if len(B) < 5: continue
        row = dict(det=det, dn=dn, R=R, lo=lo, hi=hi, n=len(B), rbar=B.rbar.mean())
        for ax, v in (('along', 'a'), ('across', 'c')):
            x = B[ax].values; row[f'rms_{ax}'] = np.sqrt(np.mean(x ** 2) / 2); row[f'mad_{ax}'] = 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)
            row[f'mean_{ax}'] = x.mean(); row[f'pred_{ax}'] = np.sqrt(np.mean(B[f'Vp_{v}']) / 2)
            bs = []
            for _ in range(300):
                kk = rng.choice(keys, len(keys)); Bb = pd.concat([B[B.n == k] for k in kk]) if len(keys) < 400 else B.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
                bs.append(np.sqrt(np.mean(Bb[ax].values ** 2) / 2))
            row[f'rms_{ax}_lo'], row[f'rms_{ax}_hi'] = np.percentile(bs, [5, 95])
        out.append(row)
    # bias through origin
    for name, x in (('quad', M.rho2 ** 2 - M.rho1 ** 2), ('lin', M.rho2 - M.rho1)):
        x = x.values; y = M.along.values; b = (x @ y) / (x @ x); bs = []
        for _ in range(300):
            kk = rng.choice(keys, len(keys)); Mb = pd.concat([M[M.n == k] for k in kk]); xb = (Mb.rho2 ** 2 - Mb.rho1 ** 2 if name == 'quad' else Mb.rho2 - Mb.rho1).values
            bs.append((xb @ Mb.along.values) / (xb @ xb))
        bias.append(dict(det=det, dn=dn, R=R, form=name, b=b, lo=np.percentile(bs, 5), hi=np.percentile(bs, 95), n=len(M), mean_drho=float((M.rho2 - M.rho1).mean())))
pd.DataFrame(out).to_csv(TAB / 'sigma_range_bins.csv', index=False); pd.DataFrame(bias).to_csv(TAB / 'sigma_along_bias.csv', index=False)
print(pd.DataFrame(out)[['det', 'dn', 'lo', 'n', 'rms_along', 'rms_along_lo', 'rms_along_hi', 'pred_along', 'rms_across', 'pred_across']].round(3).to_string())
print(pd.DataFrame(bias).round(3).to_string())
