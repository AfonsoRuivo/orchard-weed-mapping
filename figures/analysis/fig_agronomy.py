#!/usr/bin/env python3
"""agronomy.pdf: (a, b) weed entities per 100 m^2 of observed ground vs distance to the nearest row line (YOLO v2, colour),
(c) agreement of the two maps: share of entities of one map with an entity of the other within r, against a random-placement null."""
import json
from figlib import *
plt = style(); A = pd.read_csv(TAB / 'agronomy_distance_profile.csv'); G = json.load(open(TAB / 'agreement.json'))
fig, ax = plt.subplots(1, 3, figsize=(15 * CM, 5.0 * CM), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 1.05]))
for a_, det, lab in ((ax[0], 'yolo', 'YOLO v2'), (ax[1], 'colour', 'colour')):
    d = A[(A.det == det) & (A.observed_m2 > 100)]; x = 0.5 * (d.d_lo + d.d_hi)
    y = d.per_100m2; err = 100 * np.sqrt(d.n) / d.observed_m2
    a_.bar(x, y, width=0.22, color=C[det], alpha=0.85, lw=0); a_.errorbar(x, y, yerr=err, fmt='none', ecolor='k', lw=0.6, capsize=1.2)
    a_.set_xlabel('distance to nearest row line (m)'); a_.set_ylabel(f'{lab} entities / 100 m$^2$'); a_.set_xlim(0, 2.6); a_.set_ylim(bottom=0)
labels = ['0.5', '1.0', '1.5']; x = np.arange(3); w = 0.2
yv = [G[f'r{r}']['yolo_with_colour_within'] for r in labels]; cv = [G[f'r{r}']['colour_with_yolo_within'] for r in labels]
nv = [G[f'r{r}']['null_mean'] for r in labels]; n95 = [G[f'r{r}']['null_p95'] for r in labels]
ax[2].bar(x - w, yv, w, color=C['yolo'], lw=0, label='YOLO with colour near'); ax[2].bar(x, nv, w, color=C['null'], lw=0, label='random placement (mean, p95)')
ax[2].errorbar(x, nv, yerr=[np.zeros(3), np.array(n95) - np.array(nv)], fmt='none', ecolor='k', lw=0.6, capsize=1.2)
ax[2].bar(x + w, cv, w, color=C['colour'], lw=0, label='colour with YOLO near')
ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_xlabel('matching radius $r$ (m)'); ax[2].set_ylabel('share of entities'); ax[2].set_ylim(0, 1.45)
ax[2].legend(loc='upper left', fontsize=6, handlelength=1.0, bbox_to_anchor=(-0.05, 1.02))
for a_, l in zip(ax, 'abc'): a_.text(-0.02, 1.02, f'({l})', transform=a_.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')
fig.savefig(FIG / 'agronomy.pdf', bbox_inches='tight'); fig.savefig(FIG / 'agronomy.png', dpi=200, bbox_inches='tight')
