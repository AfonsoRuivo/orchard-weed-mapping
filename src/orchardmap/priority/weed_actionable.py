#!/usr/bin/env python3
"""
Actionable weed-intervention map.

Combines the weed entities (weed_clusters.csv) and trunks (trunk_map.csv) into a
0-10 priority per weed from three equally-weighted, normalized factors:

  proximity  = exp(-d_trunk / TAU)         closeness to nearest trunk (closer = higher)
  density    = sum of weed area within R   local infestation (95th-pct normalized)
  size       = area_m2 of the weed         (95th-pct normalized)

  priority = 10 * (w_p*prox + w_d*dens_n + w_s*size_n) / (w_p + w_d + w_s)

Tune W_* to reweight. Outputs a ranked CSV and a priority map with intervention zones.

Usage:
    python3 ~/ros2_ws/weed_actionable.py
"""
import numpy as np, csv
from pathlib import Path
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# -- Config -------------------------------------------------------------------
WEEDS_CSV  = "/home/afonso/ros2_ws/maps/weed_clusters.csv"
TRUNKS_CSV = "/home/afonso/ros2_ws/maps/trunk_map.csv"
TRAJ_CSV   = "/home/afonso/ros2_ws/debug_recordings/liosam_trajectory.csv"
OUT_DIR    = Path("/home/afonso/ros2_ws/maps")

TAU      = 1.0     # trunk-proximity falloff length (m)
R_DENS   = 1.5     # local-density radius (m)
W_PROX, W_DENS, W_SIZE = 1.0, 1.0, 1.0   # factor weights (equal for now)

ZONE_PCTL = 85     # weeds at/above this priority percentile seed intervention zones
ZONE_EPS  = 2.0    # group hot weeds within this distance into one zone (m)
ZONE_MIN  = 4      # min hot weeds to call something a zone

R_align = np.array([[-0.57747287, -0.81640987],
                     [ 0.81640987, -0.57747287]])
t_align = np.array([-1.68209879, 0.59453907])
def slam_to_enu(x, y):
    p = R_align @ np.array([x, y]) + t_align
    return p[0], p[1]

def norm95(v):
    hi = np.percentile(v, 95)
    return np.clip(v / max(hi, 1e-9), 0, 1)

def cluster_xy(xy, eps):
    n = len(xy)
    if n == 0: return np.array([], int)
    pairs = cKDTree(xy).query_pairs(eps, output_type='ndarray')
    if len(pairs) == 0: return np.arange(n)
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab

# -- Load ---------------------------------------------------------------------
wr = list(csv.DictReader(open(WEEDS_CSV)))
W_xy = np.array([[float(r['x_enu']), float(r['y_enu'])] for r in wr])
W_sl = np.array([[float(r['x_slam']), float(r['y_slam'])] for r in wr])
size = np.array([float(r['area_m2_median']) for r in wr])
n = len(wr)
print(f"{n} weed entities")

try:
    tr = list(csv.DictReader(open(TRUNKS_CSV)))
    T_xy = np.array([[float(r['x_enu']), float(r['y_enu'])] for r in tr])
    print(f"{len(tr)} trunks")
except Exception:
    T_xy = np.empty((0, 2)); print("no trunks found")

# -- Factors ------------------------------------------------------------------
# trunk proximity
if len(T_xy):
    d_trunk, _ = cKDTree(T_xy).query(W_xy)
else:
    d_trunk = np.full(n, np.inf)
prox = np.exp(-d_trunk / TAU)                      # in (0,1], 1 at a trunk

# local density (area-weighted) within R_DENS
neigh = cKDTree(W_xy).query_ball_point(W_xy, R_DENS)
dens = np.array([size[idx].sum() for idx in neigh])

dens_n, size_n = norm95(dens), norm95(size)

# priority 0-10
priority = 10.0 * (W_PROX * prox + W_DENS * dens_n + W_SIZE * size_n) / (W_PROX + W_DENS + W_SIZE)
rank = np.argsort(-priority)                       # indices, highest first

print(f"\npriority: mean {priority.mean():.2f}  max {priority.max():.2f}  "
      f"(>{np.percentile(priority, ZONE_PCTL):.2f} = top {100-ZONE_PCTL}%)")
print("top 5 weeds:")
for k in rank[:5]:
    print(f"  pri {priority[k]:4.1f}  d_trunk {d_trunk[k]:4.2f}m  "
          f"area {size[k]:.3f}  dens {dens[k]:.3f}")

# -- Save ranked CSV ----------------------------------------------------------
cols = ['rank', 'x_enu', 'y_enu', 'x_slam', 'y_slam', 'area_m2', 'dist_trunk',
        'dens_local', 'prox', 'dens_norm', 'size_norm', 'priority']
out_csv = OUT_DIR / "weed_actionable.csv"
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols)
    for new_rank, k in enumerate(rank):
        w.writerow([new_rank, W_xy[k, 0], W_xy[k, 1], W_sl[k, 0], W_sl[k, 1],
                    size[k], d_trunk[k], dens[k], prox[k], dens_n[k], size_n[k], priority[k]])
print(f"Saved -> {out_csv}")

# -- Intervention zones (cluster the high-priority weeds) ---------------------
hot = np.where(priority >= np.percentile(priority, ZONE_PCTL))[0]
zones = []
if len(hot):
    lab = cluster_xy(W_xy[hot], ZONE_EPS)
    for c in set(lab):
        members = hot[lab == c]
        if len(members) >= ZONE_MIN:
            zones.append(members)
    zones.sort(key=lambda m: -priority[m].sum())   # rank zones by total priority
print(f"{len(zones)} intervention zones")

# -- Plot ---------------------------------------------------------------------
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#333', 'axes.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(13, 10), facecolor='white'); ax.set_facecolor('#fafafa')

try:
    traj = list(csv.DictReader(open(TRAJ_CSV)))
    txy = np.array([slam_to_enu(float(r['x']), float(r['y'])) for r in traj])
    ax.plot(txy[:, 0], txy[:, 1], color='#cccccc', lw=0.8, ls='--', alpha=0.8, zorder=1,
            label='Robot path')
except Exception:
    pass

# trunks
for t in T_xy:
    ax.add_patch(Circle((t[0], t[1]), 0.12, facecolor='#8d5a2b', edgecolor='#3e2723',
                        lw=0.6, alpha=0.85, zorder=2))
if len(T_xy):
    ax.scatter([], [], marker='o', s=70, facecolor='#8d5a2b', edgecolor='#3e2723',
               label=f'Trunks ({len(T_xy)})')

# weeds coloured by priority (low = pale, high = deep red), sized by area
wsize = 10 + norm95(size) * 45
sc = ax.scatter(W_xy[:, 0], W_xy[:, 1], c=priority, cmap='YlOrRd', vmin=0, vmax=10,
                s=wsize, alpha=0.9, edgecolors='#88000022', linewidths=0.3, zorder=3)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Intervention priority (0–10)', fontsize=10)

# intervention zones
for zi, members in enumerate(zones, 1):
    pts = W_xy[members]
    if len(pts) >= 3:
        try:
            h = ConvexHull(pts)
            poly = pts[h.vertices]
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor='#111111',
                                 lw=1.4, ls='-', alpha=0.8, zorder=5))
            cen = pts.mean(0)
            ax.text(cen[0], cen[1], f'{zi}', fontsize=11, fontweight='bold',
                    ha='center', va='center', color='#111111',
                    bbox=dict(boxstyle='circle', fc='white', ec='#111111', alpha=0.9), zorder=6)
        except Exception:
            pass
if zones:
    ax.plot([], [], color='#111111', lw=1.4, label=f'Intervention zones ({len(zones)})')

ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_aspect('equal')
ax.grid(True, alpha=0.18, color='#cccccc')
ax.set_title('Actionable weed map — priority = proximity × density × size  '
             f'(equal weights)\nCampinas orchard · {n} weeds · {len(zones)} zones',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
plt.tight_layout()
out_png = OUT_DIR / "weed_actionable.png"
plt.savefig(str(out_png), dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved -> {out_png}")
print("Done!")
