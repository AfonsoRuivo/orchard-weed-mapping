#!/usr/bin/env python3
"""priority_map.pdf: entities coloured by priority (single-hue ramp), marker size ~ area, intervention zones outlined."""
from figlib import *
from geom import *
from scipy.spatial import ConvexHull
from priority_lib import zones
plt = style()
t = load_trunks(); rows = fit_rows(t)
fig = plt.figure(figsize=(15 * CM, 9.6 * CM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.06)
axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]; cax = fig.add_subplot(gs[2])
for ax, det, lab, sz in zip(axs, ('yolo', 'colour'), ('YOLO v2', 'colour'), (lambda a: np.clip(6 + a * 90, 6, 60), lambda a: np.clip(3 + a * 250, 3, 40))):
    e = pd.read_csv(TAB / f'priority_{det}.csv'); coverage_bg(ax)
    ax.scatter(t.xe, t.ye, s=2.5, color=C['trunk'], lw=0, zorder=2, rasterized=True)
    o = np.argsort(e.priority.values)
    sc = ax.scatter(e.x.values[o], e.y.values[o], c=e.priority.values[o], cmap='viridis', vmin=0, vmax=10, s=sz(e.area_med.values[o]), edgecolor='k', linewidth=0.15, zorder=4, rasterized=True)
    Z, _ = zones(e[['x', 'y']].values, e.priority.values)
    for zi, m in enumerate(Z, 1):
        p = e[['x', 'y']].values[m]
        if len(p) >= 1:
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); pp = (p[:, None, :] + 0.8 * np.c_[np.cos(ang), np.sin(ang)][None]).reshape(-1, 2)
            q = pp[ConvexHull(pp).vertices]
            ax.add_patch(mp.Polygon(q, closed=True, fill=False, ec='k', lw=0.9, zorder=6))
        ax.text(*(p.mean(0) + [3.5, 3.5]), str(zi), fontsize=7, ha='center', va='center', zorder=7, bbox=dict(boxstyle='circle,pad=0.12', fc='w', ec='k', lw=0.4))
    ax.set_aspect('equal'); ax.set_xlim(-84, 8); ax.set_ylim(-106, 0); ax.set_xlabel('East (m)')
    ax.text(0.0, 1.01, f'({"ab"[det == "colour"]}) {lab}', transform=ax.transAxes, fontsize=8, va='bottom')
    scalebar(ax, 20, (0.05, 0.01))
axs[0].set_ylabel('North (m)'); axs[1].set_yticklabels([]); north(axs[1], (0.9, 0.02), 0.07)
cb = fig.colorbar(sc, cax=cax); cb.set_label('priority (0-10)'); cb.outline.set_linewidth(0.4)
fig.savefig(FIG / 'priority_map.pdf', bbox_inches='tight'); fig.savefig(FIG / 'priority_map.png', dpi=200, bbox_inches='tight')
