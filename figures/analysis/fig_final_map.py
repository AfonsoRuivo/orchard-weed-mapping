#!/usr/bin/env python3
"""final_map.pdf: trunks, row lines, path, weed entities (marker area ~ entity area) for YOLO v2 and colour, + zoom."""
from figlib import *
from geom import *
plt = style()
t = load_trunks(); rows = fit_rows(t); tr = load_traj()
ey = pd.read_csv(TAB / 'entities_yolo.csv'); ec = pd.read_csv(TAB / 'entities_colour.csv')
fig = plt.figure(figsize=(15 * CM, 9.2 * CM)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.08)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
def draw(ax, big):
    coverage_bg(ax)
    for r, R in rows.items():
        p = R['c'][None] + np.array([R['s0'] - 1.5, R['s1'] + 1.5])[:, None] * R['u']
        ax.plot(p[:, 0], p[:, 1], color=C['row'], lw=0.5, zorder=1)
    ax.plot(tr.xe, tr.ye, color=C['path'], lw=1.0 if big else 1.2, zorder=2)
    ax.scatter(t.xe, t.ye, s=5 if big else 22, color=C['trunk'], lw=0, zorder=3, rasterized=big)
    k = 1.0 if big else 2.2
    ax.scatter(ec.x, ec.y, s=np.clip(3 + ec.area_med * 250, 3, 40) * k, facecolor=C['colour'], edgecolor='none', alpha=0.55, zorder=4, rasterized=True)
    ax.scatter(ey.x, ey.y, s=np.clip(6 + ey.area_med * 90, 6, 60) * k, facecolor='none', edgecolor=C['yolo'], lw=0.7 if big else 1.0, zorder=5)
    ax.set_aspect('equal')
draw(a, True)
a.set_xlim(-84, 8); a.set_ylim(-106, 0); a.set_xlabel('East (m)'); a.set_ylabel('North (m)'); scalebar(a, 20, (0.05, 0.01)); north(a, (0.9, 0.02), 0.07)
# zoom: densest YOLO neighbourhood within the observed part
from scipy.spatial import cKDTree
k = cKDTree(ey[['x', 'y']].values).query_ball_point(ey[['x', 'y']].values, 6.0); i = int(np.argmax([len(q) for q in k]))
cx, cy = ey.x[i], ey.y[i]; hw, hh = 8.5, 6.0
draw(b, False); b.set_xlim(cx - hw, cx + hw); b.set_ylim(cy - hh, cy + hh); scalebar(b, 5, (0.05, 0.05)); b.set_xlabel('East (m)')
b.yaxis.tick_right(); b.yaxis.set_label_position('right'); b.set_ylabel('North (m)')
a.add_patch(mp.Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fill=False, ec='k', lw=0.6, zorder=10))
a.text(0.02, 0.98, '(a)', transform=a.transAxes, fontsize=9, fontweight='bold', va='top'); b.text(0.02, 0.98, '(b)', transform=b.transAxes, fontsize=9, fontweight='bold', va='top')
from matplotlib.ticker import MaxNLocator
b.xaxis.set_major_locator(MaxNLocator(5, integer=True)); b.yaxis.set_major_locator(MaxNLocator(5, integer=True))
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', mfc='none', mec=C['yolo'], ms=5, label=f'YOLO v2 ({len(ey)})'),
     Line2D([], [], marker='o', ls='', mfc=C['colour'], mec='none', alpha=0.7, ms=5, label=f'colour ({len(ec)})'),
     Line2D([], [], marker='o', ls='', color=C['trunk'], ms=3.5, label=f'trunks ({len(t)})'),
     Line2D([], [], color=C['path'], lw=1.2, label='robot path'), Line2D([], [], color=C['row'], lw=0.8, label='row lines')]
fig.legend(handles=h, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handletextpad=0.3)
fig.savefig(FIG / 'final_map.pdf', bbox_inches='tight'); fig.savefig(FIG / 'final_map.png', dpi=200, bbox_inches='tight')
