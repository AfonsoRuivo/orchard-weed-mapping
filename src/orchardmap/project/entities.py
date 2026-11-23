"""Connected-components de-duplication (== DBSCAN min_samples=1, as weed_cluster_yolo_cached.py) with extra stats."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist

def cc_labels(xy, eps):
    n = len(xy)
    pairs = cKDTree(xy).query_pairs(eps, output_type='ndarray')
    if len(pairs) == 0: return np.arange(n)
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]

def entities(w, eps, xcol='x_enu', ycol='y_enu'):
    """One row per entity: centroid, n detections, n distinct frames, span (max pairwise distance), area stats,
    n_samefr_merge = detections beyond one per frame (distinct objects forced together)."""
    xy = w[[xcol, ycol]].values; lab = cc_labels(xy, eps); rows = []
    for k, idx in pd.Series(np.arange(len(w))).groupby(lab):
        i = idx.values; p = xy[i]
        rows.append(dict(eid=k, x=p[:, 0].mean(), y=p[:, 1].mean(), n_det=len(i), n_frames=w.stem.iloc[i].nunique(),
                         span=float(pdist(p).max()) if len(i) > 1 else 0.0,
                         area_med=float(np.median(w.area_m2.iloc[i])), area_max=float(w.area_m2.iloc[i].max()),
                         area_sum_frame=float(w.assign(_i=np.arange(len(w))).iloc[i].groupby('stem').area_m2.sum().max()),
                         conf=float(w.confidence.iloc[i].mean()) if 'confidence' in w else np.nan,
                         samefr=len(i) - w.stem.iloc[i].nunique()))
    return pd.DataFrame(rows), lab
