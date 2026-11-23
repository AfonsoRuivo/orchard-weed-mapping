"""Priority score and intervention zones - same formulae and defaults as weed_actionable.py (equal weights, TAU 1.0,
R_DENS 1.5, ZONE_PCTL 85, ZONE_EPS 2.0, ZONE_MIN 4), as functions so the variants can be run and compared."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from entities import cc_labels

BASE = dict(tau=1.0, r_dens=1.5, w_prox=1.0, w_dens=1.0, w_size=1.0, zone_pctl=85, zone_eps=2.0, zone_min=4)

def norm95(v):
    return np.clip(v / max(np.percentile(v, 95), 1e-9), 0, 1)

def priority(xy, size, trunks_xy, **kw):
    p = {**BASE, **kw}
    d_tr = cKDTree(trunks_xy).query(xy)[0]
    prox = np.exp(-d_tr / p['tau'])
    neigh = cKDTree(xy).query_ball_point(xy, p['r_dens'])
    dens = np.array([size[i].sum() for i in neigh])
    dn, sn = norm95(dens), norm95(size)
    pri = 10 * (p['w_prox'] * prox + p['w_dens'] * dn + p['w_size'] * sn) / (p['w_prox'] + p['w_dens'] + p['w_size'])
    return dict(pri=pri, d_trunk=d_tr, prox=prox, dens=dens, dens_n=dn, size_n=sn)

def zones(xy, pri, **kw):
    p = {**BASE, **kw}
    hot = np.where(pri >= np.percentile(pri, p['zone_pctl']))[0]; Z = []
    if len(hot):
        lab = cc_labels(xy[hot], p['zone_eps'])
        for c in np.unique(lab):
            m = hot[lab == c]
            if len(m) >= p['zone_min']: Z.append(m)
    Z.sort(key=lambda m: -pri[m].sum())
    return Z, hot

def topk(pri, frac=0.10):
    k = max(1, int(np.ceil(frac * len(pri)))); return set(np.argsort(-pri)[:k])

def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)

def compare(base_pri, var_pri, base_Z, var_Z, frac=0.10):
    rho = spearmanr(base_pri, var_pri)[0]
    j = jaccard(topk(base_pri, frac), topk(var_pri, frac))
    bm = set(np.concatenate(base_Z)) if base_Z else set(); vm = set(np.concatenate(var_Z)) if var_Z else set()
    # zones matched one-to-one by best member overlap (Jaccard >= 0.5)
    matched = 0
    for z in base_Z:
        if any(jaccard(set(z), set(y)) >= 0.5 for y in var_Z): matched += 1
    return dict(jaccard_top10=j, spearman=rho, n_zones=len(var_Z), zones_matched=matched, zone_members_jaccard=jaccard(bm, vm))

def random_null_jaccard(n, frac=0.10, reps=5000, seed=0):
    rng = np.random.default_rng(seed); k = max(1, int(np.ceil(frac * n))); a = set(range(k)); out = []
    for _ in range(reps): out.append(jaccard(a, set(rng.choice(n, k, replace=False))))
    return float(np.mean(out)), float(np.percentile(out, 95))
