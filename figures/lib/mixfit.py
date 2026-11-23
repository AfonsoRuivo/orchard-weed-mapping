"""Mixture model for re-sighting pairs: a fraction pi are true re-sightings, whose difference d = p2 - p1 is Gaussian
(per-axis variance = sum of the two single-view variances, centred on a small bias mu), the rest are false pairs of
different objects, uniform over the matching disc of radius R.  Fitted by maximum likelihood; the truncation of the
Gaussian to the disc is accounted for (isotropic approximation of the mass).  Per-pair single-view variance  s_ax(rho)^2 = s0^2 + (k rho^p)^2
(p = 2 along track, 1 across: plane-tilt lever arm rho^2/h resp. yaw/pixel lever arm rho) or constant (k = 0)."""
import numpy as np
from scipy.optimize import minimize

def _sig(x, lo, hi): return lo + (hi - lo) / (1 + np.exp(-x))

def _sk(theta, hetero):
    return _sig(theta[3:5], 0.01, 0.6), (_sig(theta[5:7], 0.0, 0.2) if hetero else np.zeros(2))

def _nll(theta, d, r1, r2, R, hetero, p):
    pi = 1 / (1 + np.exp(-theta[0])); mu = theta[1:3]
    s0, k = _sk(theta, hetero)
    V = np.stack([s0[a] ** 2 * 2 + (k[a] * 1.0) ** 2 * (r1 ** p[a] * r1 ** p[a] + r2 ** p[a] * r2 ** p[a]) for a in (0, 1)], 1)   # (n, 2)
    sd = np.sqrt(V)
    x = (d - mu) / sd
    g = np.exp(-0.5 * (x ** 2).sum(1)) / (2 * np.pi * sd[:, 0] * sd[:, 1])
    # mass of the Gaussian inside the matching disc: isotropic approximation with the mean variance
    mass = np.maximum(1 - np.exp(-R ** 2 / (V.sum(1))), 1e-3)
    f = pi * g / mass + (1 - pi) / (np.pi * R ** 2)
    return -np.log(f + 1e-300).sum()

def fit(P, R, hetero=False, p=(2, 1), x0=None):
    """P: DataFrame with along, across, rho1, rho2.  Returns dict(pi, mu_along, mu_across, s_along, s_across[, k_*])."""
    d = P[['along', 'across']].values; r1, r2 = P.rho1.values, P.rho2.values
    th0 = np.array([1.0, 0, 0, -1.5, -1.5] + ([-2.0, -2.0] if hetero else []))
    best = None
    for s in (0.5, 1.0, 2.0):
        t0 = th0.copy(); t0[3:5] += np.log(s)*1.2
        r = minimize(_nll, t0, args=(d, r1, r2, R, hetero, p), method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-3, fatol=1e-3))
        if best is None or r.fun < best.fun: best = r
    th = best.x; out = dict(pi=1 / (1 + np.exp(-th[0])), mu_along=th[1], mu_across=th[2], nll=best.fun, n=len(P))
    s0, k = _sk(th, hetero); out.update(s_along=s0[0], s_across=s0[1])     # single-view sigma (pair variance = 2 s^2) for k=0
    if hetero: out.update(k_along=k[0], k_across=k[1])
    return out

def sigma_at(fitres, rho, axis):
    s0 = fitres[f's_{axis}']; k = fitres.get(f'k_{axis}', 0.0); p = 2 if axis == 'along' else 1
    return np.sqrt(s0 ** 2 + (k * rho ** p) ** 2)
