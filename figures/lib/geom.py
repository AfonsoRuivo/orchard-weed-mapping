"""Rows, coverage and distance helpers."""
import numpy as np, pandas as pd
from common import *

def fit_rows(t):
    """One straight line per row_id (PCA): returns dict row -> (centre, dir, normal, s_min, s_max)."""
    rows = {}
    for r, g in t.groupby('row_id'):
        p = g[['xe', 'ye']].values; c = p.mean(0); vt = np.linalg.svd(p - c)[2]
        ax = vt[0] if vt[0][1] > 0 else -vt[0]; s = (p - c) @ ax
        rows[r] = dict(c=c, u=ax, n=np.array([-ax[1], ax[0]]), s0=s.min(), s1=s.max(), n_trunks=len(g))
    return rows

def dist_to_rows(xy, rows, extend=2.0):
    """Perpendicular distance (m) to the nearest row line (finite segment extended by `extend` m at the ends);
    returns (dist, row_id, signed offset)."""
    best = np.full(len(xy), np.inf); rid = np.full(len(xy), -1); sg = np.zeros(len(xy))
    for r, R in rows.items():
        q = xy - R['c']; s = q @ R['u']; off = q @ R['n']
        ds = np.maximum(np.maximum(R['s0'] - extend - s, s - R['s1'] - extend), 0)
        d = np.hypot(off, ds); m = d < best
        best[m], rid[m], sg[m] = d[m], r, off[m]
    return best, rid, sg

class Coverage:
    def __init__(self, run='frozen'):
        z = np.load(TAB / f'coverage_{run}.npz'); self.cnt, self.x0, self.y0, self.cell = z['cnt'], float(z['x0']), float(z['y0']), float(z['cell'])
    def at(self, xy):
        i = ((xy[:, 0] - self.x0) / self.cell).astype(int); j = ((xy[:, 1] - self.y0) / self.cell).astype(int)
        ok = (i >= 0) & (i < self.cnt.shape[1]) & (j >= 0) & (j < self.cnt.shape[0])
        out = np.zeros(len(xy), int); out[ok] = self.cnt[j[ok], i[ok]]; return out
    def cells(self):
        j, i = np.nonzero(self.cnt > 0)
        return np.c_[self.x0 + (i + .5) * self.cell, self.y0 + (j + .5) * self.cell], self.cnt[j, i]
