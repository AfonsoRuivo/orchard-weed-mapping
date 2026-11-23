#!/usr/bin/env python3
"""
Weed de-duplication / clustering — YOLO cached-pipeline variant.

Same clustering logic as weed_cluster.py (DBSCAN via connected components,
min_samples=1), pointed at weed_map_yolo_cached.csv instead. Detector-agnostic
— only the input/output paths differ from the original.

Outputs:
    weed_clusters_yolo.csv   one row per weed entity
    weed_clusters_yolo.png   clusters over faded raw points + detections/cluster hist

Usage:
    python3 weed_cluster_yolo_cached.py
"""
import numpy as np, csv
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parents[2]
WEED_CSV    = REPO / "data_set" / "campinas_b" / "maps" / "weed_map_yolo_cached.csv"
OUT_DIR     = REPO / "data_set" / "campinas_b" / "maps"
EPS         = 0.4                       # merge radius (m) — rerun to retune
MIN_SAMPLES = 1                         # keep single-frame weeds as their own entity
EPS_SWEEP   = [0.2, 0.3, 0.4, 0.5, 0.6] # printed so you can pick the elbow
N_DEBUG     = 15                        # how many clusters to dump to console
N_LABEL     = 5                         # how many largest clusters to label on the map

# Sequential, single-hue, colorblind-safe (viridis) — magnitude encodes area_m2.
# The original weed_cluster.py used 'RdYlGn_r', a diverging map, for a pure
# magnitude quantity; swapped here per the dataviz skill's sequential-encoding rule.
SEQ_CMAP = 'viridis'

# ── Connected-components clustering (== DBSCAN, min_samples=1) ────────────────
def cluster_eps(xy, eps):
    """Label each point by connected component of the within-eps graph."""
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    pairs = cKDTree(xy).query_pairs(eps, output_type='ndarray')  # (M,2), i<j, dist<=eps
    if len(pairs) == 0:
        return np.arange(n)                                      # all singletons
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels

# ── Load raw detections ──────────────────────────────────────────────────────
rows = list(csv.DictReader(open(WEED_CSV)))
area_key = 'area_m2' if (rows and 'area_m2' in rows[0]) else 'area_px'
if area_key != 'area_m2':
    print("WARNING: area_m2 not found in CSV — falling back to area_px.")

xy   = np.array([[float(r['x_enu']),  float(r['y_enu'])]  for r in rows])
xs   = np.array([float(r['x_slam'])   for r in rows])
ys   = np.array([float(r['y_slam'])   for r in rows])
ts   = np.array([float(r['timestamp']) for r in rows])
area = np.array([float(r[area_key])   for r in rows])
conf = np.array([float(r['confidence']) for r in rows]) if rows and 'confidence' in rows[0] else None
print(f"Loaded {len(rows)} raw detections  (size key: {area_key})")

# ── eps sweep ────────────────────────────────────────────────────────────────
print("\neps sweep (cluster counts):")
for e in EPS_SWEEP:
    nc = len(set(cluster_eps(xy, e)))
    print(f"  eps={e:.2f} m -> {nc:>4} clusters  "
          f"(avg {len(rows)/max(nc,1):.1f} detections/cluster)")

# ── Cluster at chosen EPS ────────────────────────────────────────────────────
labels = cluster_eps(xy, EPS)
uniq   = sorted(set(labels))
print(f"\nChosen eps={EPS} m -> {len(uniq)} weed entities (from {len(rows)} detections)")

clusters = []
for cid in uniq:
    m  = labels == cid
    a  = area[m]
    cx, cy = xy[m, 0].mean(), xy[m, 1].mean()        # centroid
    span = float(np.sqrt(((xy[m] - [cx, cy])**2).sum(axis=1)).max())  # extent from centroid
    clusters.append({
        'cluster_id'    : int(cid),
        'x_enu'         : float(cx),
        'y_enu'         : float(cy),
        'x_slam'        : float(xs[m].mean()),
        'y_slam'        : float(ys[m].mean()),
        'n_detections'  : int(m.sum()),
        'area_m2_median': float(np.median(a)),
        'area_m2_max'   : float(a.max()),
        'span_m'        : span,
        't_first'       : float(ts[m].min()),
        't_last'        : float(ts[m].max()),
        'conf_mean'     : float(conf[m].mean()) if conf is not None else None,
        '_areas'        : np.sort(a),                # debug only
    })

# ── Debug dump: per-cluster area_m2 lists (top N by detection count) ─────────
print(f"\n--- per-cluster {area_key} lists (top {N_DEBUG} by detection count) ---")
print("   id   n   median    max    span_m  CV%   [sorted measurements]")
for c in sorted(clusters, key=lambda c: -c['n_detections'])[:N_DEBUG]:
    a  = c['_areas']
    cv = (a.std() / a.mean() * 100) if a.mean() > 0 else 0.0
    lst = ", ".join(f"{v:.3f}" for v in a)
    print(f"  {c['cluster_id']:>4} {c['n_detections']:>3}  "
          f"{c['area_m2_median']:.3f}  {c['area_m2_max']:.3f}  "
          f"{c['span_m']:5.2f}  {cv:4.0f}  [{lst}]")
print("  (high CV% = noisy size measurements; large span_m = possible chaining -> lower EPS)")

# ── Save CSV ─────────────────────────────────────────────────────────────────
cols = ['cluster_id', 'x_enu', 'y_enu', 'x_slam', 'y_slam', 'n_detections',
        'area_m2_median', 'area_m2_max', 'span_m', 't_first', 't_last', 'conf_mean']
out_csv = OUT_DIR / "weed_clusters_yolo.csv"
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for c in clusters:
        w.writerow({k: c[k] for k in cols})
print(f"\nSaved -> {out_csv}")

# ── Diagnostic plot ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0a0f1a')

# Left: raw (faded) + cluster entities (color/size by median area)
ax1.set_facecolor('#0f1e2e')
ax1.scatter(xy[:, 0], xy[:, 1], s=6, c='#475569', alpha=0.35, zorder=1,
            label=f'raw detections ({len(rows)})')
cx = np.array([c['x_enu']          for c in clusters])
cy = np.array([c['y_enu']          for c in clusters])
cm = np.array([c['area_m2_median'] for c in clusters])
amax = np.percentile(cm, 95)
cn   = np.clip(cm / max(amax, 1e-9), 0, 1)
sc = ax1.scatter(cx, cy, c=cn, cmap=SEQ_CMAP, s=20 + cn * 90, alpha=0.9, zorder=3,
                 edgecolors='#0a0f1a', linewidths=0.5,
                 label=f'weed entities ({len(clusters)})')
cbar = fig.colorbar(sc, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label(f'median area m² (clip @95pct={amax:.3f})', color='#94a3b8', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='#64748b')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#64748b')

# Selective direct labels: only the N_LABEL largest entities, not every point
top = sorted(clusters, key=lambda c: -c['area_m2_median'])[:N_LABEL]
for c in top:
    ax1.annotate(f"{c['area_m2_median']:.2f} m²", (c['x_enu'], c['y_enu']),
                 xytext=(6, 6), textcoords='offset points',
                 fontsize=8, color='#e8f0f7',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#1a2a3a', ec='#334155', alpha=0.85))

ax1.set_xlabel('East (m)',  color='#94a3b8', fontsize=11)
ax1.set_ylabel('North (m)', color='#94a3b8', fontsize=11)
ax1.set_title(f'De-duplicated weed map  (eps={EPS} m)', color='#e8f0f7', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#64748b')
ax1.grid(True, alpha=0.15, color='#334155')
ax1.set_aspect('equal')
ax1.legend(fontsize=9, facecolor='#1a2a3a', labelcolor='#e8f0f7', edgecolor='#334155')

# Right: detections-per-cluster histogram
ax2.set_facecolor('#0f1e2e')
ns = np.array([c['n_detections'] for c in clusters])
ax2.hist(ns, bins=range(1, ns.max() + 2), color='#38bdf8', alpha=0.85, align='left')
ax2.set_xlabel('detections per cluster', color='#94a3b8', fontsize=11)
ax2.set_ylabel('# clusters', color='#94a3b8', fontsize=11)
ax2.set_title(f'{len(rows)} detections → {len(clusters)} entities '
              f'({len(rows)/max(len(clusters),1):.1f}× dedup)',
              color='#e8f0f7', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#64748b')
ax2.grid(True, alpha=0.15, color='#334155', axis='y')

plt.tight_layout()
out_png = OUT_DIR / "weed_clusters_yolo.png"
plt.savefig(str(out_png), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved -> {out_png}")
print("Done!")
