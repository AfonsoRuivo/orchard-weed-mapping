#!/usr/bin/env python3
"""sigma_loc.pdf: (a, b) single-view position scatter along / across travel vs range, measured (YOLO, colour; lag 0.69 s
solid, 3.45 s dashed) against the plane-noise prediction; (c) sensitivity of the estimate to the matching radius."""
from figlib import *
plt = style(); B = pd.read_csv(TAB / 'sigma_range_bins.csv')
fig, ax = plt.subplots(1, 3, figsize=(15 * CM, 5.0 * CM), gridspec_kw=dict(wspace=0.38))
for a_, axn, ttl in ((ax[0], 'along', 'along track'), (ax[1], 'across', 'across track')):
    pr = B[(B.det == 'colour') & (B.dn == 10)]
    a_.plot(pr.rbar, pr[f'pred_{axn}'], color=C['null'], ls=':', lw=1.2, label='prediction')
    for det in ('yolo', 'colour'):
        for dn, ls in ((10, '-'), (50, '--')):
            g = B[(B.det == det) & (B.dn == dn) & (B.n >= 15)]
            if len(g) == 0: continue
            a_.errorbar(g.rbar, g[f'rms_{axn}'], yerr=[(g[f'rms_{axn}'] - g[f'rms_{axn}_lo']).fillna(0), (g[f'rms_{axn}_hi'] - g[f'rms_{axn}']).fillna(0)], color=C[det], ls=ls, marker='o', ms=2.5, lw=1.0, capsize=1.5, elinewidth=0.6)
    a_.set_xlabel('range from robot (m)'); a_.set_ylabel(f'$\\sigma_{{loc}}$ {ttl} (m)'); a_.set_ylim(0, 0.45); a_.set_xlim(1, 4.6)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=C['yolo'], marker='o', ms=2.5, label='YOLO v2'), Line2D([], [], color=C['colour'], marker='o', ms=2.5, label='colour'),
                      Line2D([], [], color='k', ls='-', label='lag 0.7 s'), Line2D([], [], color='k', ls='--', label='lag 3.5 s'),
                      Line2D([], [], color=C['null'], ls=':', label='prediction')], loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=6.5, ncol=5, handlelength=1.6, columnspacing=1.2)
for det in ('yolo', 'colour'):
    S = pd.read_csv(TAB / f'sigma_dense_{det}_dn10_sweep.csv'); S50 = pd.read_csv(TAB / f'sigma_dense_{det}_dn50_sweep.csv')
    for s_, ls in ((S, '-'), (S50, '--')):
        ax[2].plot(s_.R, np.sqrt(s_.mad_along ** 2 + s_.mad_across ** 2) / np.sqrt(2) * 1.0, color=C[det], ls=ls, marker='o', ms=2.2)
ax[2].set_xlabel('matching radius $R$ (m)'); ax[2].set_ylabel('$\\sigma_{loc}$ (m), both axes (MAD)'); ax[2].set_ylim(0, 0.4)
for a_, l in zip(ax, 'abc'): a_.text(-0.02, 1.03, f'({l})', transform=a_.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
fig.savefig(FIG / 'sigma_loc.pdf', bbox_inches='tight'); fig.savefig(FIG / 'sigma_loc.png', dpi=200, bbox_inches='tight')
