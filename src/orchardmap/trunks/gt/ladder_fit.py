import numpy as np
from scipy.optimize import least_squares
def fit(nA, nB, meas):
    """meas: list of (id,type,('A',i),('B',j),value). Straight parallel rows: A on y=0, B on y=W+t*x."""
    def unpack(p):
        a, b, W, t = p[:nA], p[nA:nA + nB], p[-2], p[-1]
        P = {('A', k): np.array([a[k - 1], 0.]) for k in range(1, nA + 1)}
        P.update({('B', k): np.array([b[k - 1], W + t * b[k - 1]]) for k in range(1, nB + 1)})
        return P
    def f(p):
        P = unpack(p)
        return [np.linalg.norm(P[m[2]] - P[m[3]]) - m[4] for m in meas] + [p[0] * 10]
    best = None
    for off in (-1.5, 0, 1.5):
        p0 = np.r_[np.arange(nA) * 3.0, np.arange(nB) * 3.0 + off, 4.8, 0]
        s = least_squares(f, p0, loss='soft_l1', f_scale=0.05)
        if best is None or s.cost < best.cost: best = s
    r = np.array(best.fun[:-1])
    return unpack(best.x), r, best.x[-2], best.x[-1]
def robust_sigma(r): return 1.4826 * np.median(np.abs(r - np.median(r)))
