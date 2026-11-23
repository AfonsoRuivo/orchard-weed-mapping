#!/usr/bin/env python3
"""eps_sweep.pdf: merge-radius sweep. (a) entities per detection, (b) % entities seen from one frame only,
(c) % entities that hold two detections of ONE frame (distinct objects merged), (d) entity span at eps = 0.4 m."""
from figlib import *
from entities import entities
from sigma_lib import load_weeds_run
plt = style(); S = pd.read_csv(TAB / 'eps_sweep.csv')
fig, ax = plt.subplots(2, 2, figsize=(15 * CM, 9 * CM)); ax = ax.ravel()
T = pd.read_csv(TAB / 'tradeoff_dn50.csv'); T10 = pd.read_csv(TAB / 'tradeoff_dn10.csv'); D = pd.read_csv(TAB / 'eps_sweep_dense.csv')
for det, lab in (('yolo', 'YOLO v2'), ('colour', 'colour')):
    s = S[S.det == det]; col = C[det]
    ax[0].plot(s.eps, s.n_entities / s.n_det, color=col, label=lab)
    ax[1].plot(s.eps, s.pct_once, color=col)
    t = T[(T.det == det) & (T.cls == 'all')]; t10 = T10[(T10.det == det) & (T10.cls == 'all')]
    ax[2].plot(t.eps, t.link.clip(upper=1), color=col, lw=1.2); ax[2].plot(t10.eps, t10.link.clip(upper=1), color=col, lw=0.8, ls='--')
    ax[2].plot(t.eps, t.over, color=col, lw=1.2, ls=':')
ax[2].set_ylim(0, 1.05)
for a_ in ax[:3]: a_.axvline(0.4, color='k', lw=0.5, ls=':')
ax[0].set_ylabel('entities / detections'); ax[1].set_ylabel('entities seen in one frame (%)'); ax[2].set_ylabel('share')
for a_ in ax[:3]: a_.set_xlabel('merge radius $\\varepsilon$ (m)')
from matplotlib.lines import Line2D
ax[1].legend(handles=[Line2D([], [], color=C['yolo'], label='YOLO v2'), Line2D([], [], color=C['colour'], label='colour')], loc='lower left', fontsize=7, handlelength=1.6)
fig.legend(handles=[Line2D([], [], color='k', lw=1.2, label='(c) re-sightings linked, lag 3.5 s'), Line2D([], [], color='k', lw=0.8, ls='--', label='(c) re-sightings linked, lag 0.7 s'), Line2D([], [], color='k', lw=1.2, ls=':', label='(c) detections with a same-frame neighbour within $\\varepsilon$')], loc='lower center', ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.05), columnspacing=1.0, handlelength=1.8)
bins = np.linspace(0, 3, 31)
for det in ('yolo', 'colour'):
    e, _ = entities(load_weeds_run(det, 'frozen'), 0.4); m = e.n_det > 1
    ax[3].hist(e.span[m], bins=bins, histtype='step', color=C[det], lw=1.1)
ax[3].axvline(0.8, color='k', lw=0.5, ls=':'); ax[3].text(0.84, 40, '$2\\varepsilon$', fontsize=7)
ax[3].set_yscale('log'); ax[3].set_xlabel('span of multi-detection entities (m), $\\varepsilon$ = 0.4 m'); ax[3].set_ylabel('entities')
for a_, l in zip(ax, 'abcd'): a_.text(-0.02, 1.02, f'({l})', transform=a_.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
fig.tight_layout(); fig.savefig(FIG / 'eps_sweep.pdf', bbox_inches='tight'); fig.savefig(FIG / 'eps_sweep.png', dpi=200, bbox_inches='tight')
