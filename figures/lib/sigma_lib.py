"""Pairing of detections in consecutive frames -> position differences along / across travel."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import *

RUNS = {'frozen': (FROZEN, TAB / 'frames.csv'),
        'dense': (MAPDIR / 'dense10_20261008', TAB / 'dense10' / 'frames.csv')}

def load_frames(run='frozen'):
    f = pd.read_csv(RUNS[run][1])
    f['n'] = f.stem.str.slice(6, 12).astype(int)
    return add_heading(f)

def add_heading(f, half=50):
    """Smoothed direction of travel: pose(n+half) - pose(n-half), nearest available frames (SLAM frame)."""
    n = f.n.values; P = f[['x_slam', 'y_slam']].values; hx, hy, ok = [], [], []
    for k in range(len(f)):
        a = np.argmin(np.abs(n - (n[k] - half))); b = np.argmin(np.abs(n - (n[k] + half)))
        d = P[b] - P[a]; L = np.hypot(*d)
        good = abs(n[a] - (n[k] - half)) <= 20 and abs(n[b] - (n[k] + half)) <= 20 and L > 0.5
        hx.append(d[0] / L if good else np.nan); hy.append(d[1] / L if good else np.nan)
    f['hx'], f['hy'] = hx, hy
    return f

def load_weeds_run(det, run='frozen'):
    return pd.read_csv(RUNS[run][0] / f'weed_map_{det}_cached.csv')

def prep(det, run='frozen'):
    """Weeds with robot-relative horizontal range rho (m) and frame number."""
    w = load_weeds_run(det, run); f = load_frames(run).set_index('stem')
    w['n'] = w.stem.str.slice(6, 12).astype(int)
    w['xr'], w['yr'] = f.loc[w.stem, 'x_slam'].values, f.loc[w.stem, 'y_slam'].values
    w['rho'] = np.hypot(w.x_slam - w.xr, w.y_slam - w.yr)
    return w, f

def consecutive_pairs(f, dn=50, turn_deg=20):
    """Frame-number pairs (n, n+dn), robot moving (0.018-0.054 m per camera frame) and heading (smoothed) not
    turning. Returns (n, m, ux, uy) with u = smoothed travel direction at the pair's midpoint."""
    fi = f.reset_index(drop=True).set_index('n'); out = []
    for n in fi.index:
        m = n + dn
        if m not in fi.index: continue
        a, b = fi.loc[n], fi.loc[m]
        if np.isnan(a.hx) or np.isnan(b.hx): continue
        d = np.hypot(b.x_slam - a.x_slam, b.y_slam - a.y_slam)
        if not (0.018 * dn <= d <= 0.054 * dn): continue
        if np.degrees(np.arccos(np.clip(a.hx * b.hx + a.hy * b.hy, -1, 1))) > turn_deg: continue
        ux, uy = a.hx + b.hx, a.hy + b.hy; L = np.hypot(ux, uy)
        out.append((n, m, ux / L, uy / L))
    return out

def mutual_nn(A, B, R):
    """index pairs (i, j) that are each other's nearest neighbour, dist <= R."""
    if len(A) == 0 or len(B) == 0: return np.empty((0, 2), int), np.empty(0)
    dA, jA = cKDTree(B).query(A); dB, iB = cKDTree(A).query(B)
    ok = [(i, j) for i, (j, d) in enumerate(zip(jA, dA)) if d <= R and iB[j] == i]
    ok = np.array(ok, int).reshape(-1, 2)
    return ok, (np.hypot(*(A[ok[:, 0]] - B[ok[:, 1]]).T) if len(ok) else np.empty(0))

def build_pairs(w, f, R, shift=None, rng=None, dn=50):
    """Pairs of detections in consecutive frames. d = later - earlier, along/across = components on the
    robot's displacement direction u (travel) and its left normal. shift: random translation (m) applied
    to the later frame's detections (null model for false pairs)."""
    g = {n: s for n, s in w.groupby('n')}; rows = []
    for n, m, ux, uy in consecutive_pairs(f, dn):
        if n not in g or m not in g: continue
        A, B = g[n], g[m]
        pa, pb = A[['x_slam', 'y_slam']].values, B[['x_slam', 'y_slam']].values
        if shift is not None:
            ang = rng.uniform(0, 2 * np.pi); pb = pb + shift * np.array([np.cos(ang), np.sin(ang)])
        ij, _ = mutual_nn(pa, pb, R)
        for i, j in ij:
            d = pb[j] - pa[i]
            rows.append(dict(n=n, m=m, dx=d[0], dy=d[1], along=d[0] * ux + d[1] * uy, across=-d[0] * uy + d[1] * ux,
                             dist=np.hypot(*d), rho1=A.rho.iloc[i], rho2=B.rho.iloc[j], ia=A.index[i], ib=B.index[j],
                             area1=A.area_m2.iloc[i], area2=B.area_m2.iloc[j]))
    return pd.DataFrame(rows)

def rms_sigma(x): return np.sqrt(np.mean(np.square(x)) / 2)
def mad_sigma(x): return 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)
