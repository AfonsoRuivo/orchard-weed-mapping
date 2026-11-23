#!/usr/bin/env python3
"""priority_stability.pdf: top-10 % Jaccard and Spearman rho of each variant against the baseline, for YOLO (frozen, 99 entities),
YOLO (dense run, 312) and colour (frozen, 925); bottom row = noise floor (baseline vs baseline with positions +-0.25 m, areas +-30 %)."""
from figlib import *
plt = style()
S = {'YOLO v2, 99 entities': pd.read_csv(TAB / 'priority_stability.csv').query("det=='yolo'"), 'YOLO v2, 312 (dense run)': pd.read_csv(TAB / 'priority_stability_dense.csv').query("det=='yolo'"),
     'colour, 925': pd.read_csv(TAB / 'priority_stability.csv').query("det=='colour'")}
cols = {'YOLO v2, 99 entities': ('#0072B2', 'o'), 'YOLO v2, 312 (dense run)': ('#56B4E9', 's'), 'colour, 925': ('#D55E00', '^')}
labs = ['$\\tau$ = 0.5 m', '$\\tau$ = 2 m', 'density weight ×2', 'proximity weight ×2', 'size weight ×2', 'noise floor']
keys = ['tau=0.5', 'tau=2', 'w_dens x2', 'w_prox x2', 'w_size x2', 'noise']
fig, ax = plt.subplots(1, 2, figsize=(15 * CM, 5.4 * CM), sharey=True, gridspec_kw=dict(wspace=0.08))
for a_, m, xl in ((ax[0], 'jaccard_top10', 'Jaccard index, top 10 % entities'), (ax[1], 'spearman', 'Spearman $\\rho$, full ranking')):
    for k, (nm, d) in enumerate(S.items()):
        col, mk = cols[nm]
        for i, key in enumerate(keys):
            r = d[d.variant.str.startswith(key)]
            if not len(r): continue
            r = r.iloc[0]; y = i + (k - 1) * 0.2
            a_.plot(r[m], y, mk, color=col, ms=4, mfc=col if key != 'noise' else 'none', label=nm if i == 0 else None)
            if key == 'noise':
                sd = r['jaccard_sd' if m == 'jaccard_top10' else 'spearman_sd']; a_.errorbar(r[m], y, xerr=sd, color=col, capsize=1.5, lw=0.8)
    a_.set_xlabel(xl); a_.set_yticks(range(len(labs))); a_.set_yticklabels(labs); a_.axhline(4.5, color='k', lw=0.4)
ax[0].invert_yaxis(); ax[0].axvspan(0, 0.18, color='0.9', zorder=0); ax[0].text(0.09, 2.5, 'random\nsets', fontsize=6, va='center', ha='center', rotation=90); ax[0].set_xlim(0, 1.02); ax[1].set_xlim(0.85, 1.0)
ax[1].legend(loc='center left', bbox_to_anchor=(0.0, 0.52), fontsize=6, handletextpad=0.2)
for a_, l in zip(ax, 'ab'): a_.text(-0.0, 1.02, f'({l})', transform=a_.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')
fig.savefig(FIG / 'priority_stability.pdf', bbox_inches='tight'); fig.savefig(FIG / 'priority_stability.png', dpi=200, bbox_inches='tight')
