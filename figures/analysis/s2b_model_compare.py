#!/usr/bin/env python3
"""Section 2b - compare the measured pair scatter with the geometric prediction.
usage: s2b_model_compare.py RUN DET DN R
 * plane noise from the frames themselves: world-frame plane normal (tilt along / across travel) and plane height at
   the pair midpoint, difference between the two frames of each pair, /sqrt2  (upper bound: includes real terrain change)
 * per pair: Jacobian of the ground projection at each detection -> predicted along/across variance, plane part and pixel part
 * measured: d_axis^2 of true re-sightings (pairs within R; false pairs reduced by the dist<R cut, see s2)
 * fit: d^2 = k_plane * V_plane + sv^2 * S_v + su^2 * S_u + c  (non-negative LS, per axis), CI by bootstrap over frame pairs"""
import sys, json
from scipy.optimize import nnls
from common import *
from sigma_lib import *
from sigma_model import jacobian
import weed_map_cached as wm
run, det, dn, R = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
wm.load_calibration(); wm.load_trajectory(str(TRAJ))
w, f = prep(det, run); fi = f.reset_index().set_index('n')
P = build_pairs(w, f, R, dn=dn)
# plane noise
def plane_world(r):
    T = wm.get_pose(r.t); n = np.array([r.a, r.b, r.c]); nl = np.linalg.norm(n); n, d = n / nl, r.d / nl
    nw = T[:3, :3] @ n; dw = d - nw @ T[:3, 3]; return nw, dw
cp = consecutive_pairs(f, dn); tilt_a, tilt_c, dz = [], [], []
for n, m, ux, uy in cp:
    a, b = fi.loc[n], fi.loc[m]; (na, da), (nb, db) = plane_world(a), plane_world(b)
    u = np.array([ux, uy]); v = np.array([-uy, ux])
    tilt_a.append((nb[:2] - na[:2]) @ u / nb[2]); tilt_c.append((nb[:2] - na[:2]) @ v / nb[2])
    mid = 0.5 * (np.array([a.x_slam, a.y_slam]) + np.array([b.x_slam, b.y_slam]))
    za = -(da + na[:2] @ mid) / na[2]; zb = -(db + nb[:2] @ mid) / nb[2]; dz.append(zb - za)
rm = lambda x: float(np.sqrt(np.mean(np.square(x)) / 2)); mad = lambda x: float(1.4826 * np.median(np.abs(np.array(x) - np.median(x))) / np.sqrt(2))
noise = dict(sig_pitch=rm(tilt_a), sig_roll=rm(tilt_c), sig_off=rm(dz), sig_pitch_mad=mad(tilt_a), sig_roll_mad=mad(tilt_c), sig_off_mad=mad(dz), n_framepairs=len(cp))
print('plane noise (rad, rad, m):', {k: round(v, 4) for k, v in noise.items()})
# per-pair Jacobians
sig = dict(pitch=noise['sig_pitch_mad'], roll=noise['sig_roll_mad'], off=noise['sig_off_mad'], v=0, u=0)
rows = []
for _, p in P.iterrows():
    n, m = int(p.n), int(p.m); u_ = np.array([fi.loc[n].hx + fi.loc[m].hx, fi.loc[n].hy + fi.loc[m].hy]); u_ /= np.linalg.norm(u_)
    out = dict(Vp_a=0, Vp_c=0, Sv_a=0, Sv_c=0, Su_a=0, Su_c=0, ok=True)
    for side, ia in (('1', p.ia), ('2', p.ib)):
        r = w.loc[ia]; fr = fi.loc[n if side == '1' else m]
        J = jacobian(r.u, r.v, (fr.a, fr.b, fr.c, fr.d), wm.get_pose(fr.t))
        if J is None: out['ok'] = False; break
        e1, e2 = u_, np.array([-u_[1], u_[0]])
        for tag, e in (('a', e1), ('c', e2)):
            out[f'Vp_{tag}'] += (e @ J[:, 0]) ** 2 * sig['pitch'] ** 2 + (e @ J[:, 1]) ** 2 * sig['roll'] ** 2 + (e @ J[:, 2]) ** 2 * sig['off'] ** 2
            out[f'Sv_{tag}'] += (e @ J[:, 3]) ** 2; out[f'Su_{tag}'] += (e @ J[:, 4]) ** 2
    rows.append(out)
M = pd.concat([P.reset_index(drop=True), pd.DataFrame(rows)], axis=1); M = M[M.ok].copy()
M.to_csv(TAB / f'sigma_{run}_{det}_dn{dn}_pairs_R{R}.csv', index=False)
def fit_axis(M, ax):
    y = (M[ax] - M[ax].median()) ** 2          # remove the constant bias (reported separately)
    A = np.c_[M[f'Vp_{ax[0]}'], M[f'Sv_{ax[0]}'], M[f'Su_{ax[0]}'], np.ones(len(M))]
    # trim the 3 % largest d^2 (false pairs) for stability
    k = y <= np.percentile(y, 97); x, _ = nnls(A[k], y[k]); return x
res = {}
for ax in ('along', 'across'):
    x = fit_axis(M, ax); bs = []
    keys = M.n.unique(); rng = np.random.default_rng(0)
    for _ in range(300):
        pick = rng.choice(keys, len(keys)); Mb = pd.concat([M[M.n == k] for k in pick]); bs.append(fit_axis(Mb, ax))
    bs = np.array(bs)
    res[ax] = dict(k_plane=x[0], sig_v_px=float(np.sqrt(x[1])), sig_u_px=float(np.sqrt(x[2])), c0_m=float(np.sqrt(x[3])),
                   k_plane_ci=np.percentile(bs[:, 0], [5, 95]).tolist(), sig_v_ci=np.sqrt(np.percentile(bs[:, 1], [5, 95])).tolist(),
                   c0_ci=np.sqrt(np.percentile(bs[:, 3], [5, 95])).tolist())
# bias regression: d_along = b0 + b1 (rho2^2 - rho1^2)
X = np.c_[np.ones(len(M)), M.rho2 ** 2 - M.rho1 ** 2]; bb = np.linalg.lstsq(X, M.along.values, rcond=None)[0]
res['bias_along'] = dict(b0=float(bb[0]), b1_per_m2=float(bb[1]), mean=float(M.along.mean()))
X = np.c_[np.ones(len(M)), M.rho2 - M.rho1]; bc = np.linalg.lstsq(X, M.across.values, rcond=None)[0]
res['bias_across'] = dict(b0=float(bc[0]), b1_per_m=float(bc[1]), mean=float(M.across.mean()))
res['plane_noise'] = noise; res['n_pairs'] = len(M); res['R'] = R
json.dump(res, open(TAB / f'sigma_{run}_{det}_dn{dn}_model_R{R}.json', 'w'), indent=1); print(json.dumps(res, indent=1))
