"""Candidate generation + per-candidate features for the v2 trunk detector."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def density_peaks(P, h, zlo, zhi, cell=0.03, sigma=3, nms=0.6, thr=0.3):
    m = (h > zlo) & (h < zhi); Q = P[m]
    x0, y0 = Q[:, 0].min(), Q[:, 1].min()
    H, _, _ = np.histogram2d(Q[:, 0], Q[:, 1], bins=[np.arange(x0, Q[:, 0].max() + cell, cell),
                                                     np.arange(y0, Q[:, 1].max() + cell, cell)])
    S = ndimage.gaussian_filter(H, sigma); mx = ndimage.maximum_filter(S, size=int(nms / cell))
    pk = np.argwhere((S == mx) & (S > thr))
    return np.c_[x0 + (pk[:, 0] + .5) * cell, y0 + (pk[:, 1] + .5) * cell], S[pk[:, 0], pk[:, 1]]


def candidate_features(cand, P, h, tree2d, band=(0.10, 0.40)):
    """Features around each candidate (x, y). tree2d: cKDTree on P[:, :2]."""
    F = []
    for x, y in cand:
        idx = np.array(tree2d.query_ball_point([x, y], 0.8), int)
        Q, hh = P[idx], h[idx]; d = np.hypot(Q[:, 0] - x, Q[:, 1] - y)
        inb = (hh > band[0]) & (hh < band[1])
        n_core = (inb & (d < 0.20)).sum()
        n_ring = (inb & (d > 0.35) & (d < 0.8)).sum()
        area_core, area_ring = np.pi * 0.2 ** 2, np.pi * (0.8 ** 2 - 0.35 ** 2)
        # vertical support: points in the column right above the band (branches/forks)
        n_up = ((hh > 0.4) & (hh < 1.2) & (d < 0.35)).sum()
        n_low = ((hh > 0.0) & (hh < 0.08) & (d < 0.2)).sum()
        # occupancy of height bins in the core column (0.05 m bins, 0.05-1.2 m)
        hb = hh[d < 0.25]; occ = np.histogram(hb, bins=np.arange(0.05, 1.21, 0.05))[0]
        F.append(dict(n_core=n_core, contrast=(n_core / area_core + 1) / (n_ring / area_ring + 1),
                      n_up=n_up, n_low=n_low, vocc=(occ >= 3).mean()))
    return F
