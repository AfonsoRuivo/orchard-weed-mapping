#!/usr/bin/env python3
"""Builds REPORT_maps_20261008.md from the CSV/JSON tables (every number in the report is read from tables/)."""
import json, sys
from common import *
def md(df, fmt=None, index=False):
    df = df.copy(); fmt = fmt or {}
    cols = list(df.columns); out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join((fmt.get(c, '{:.3g}').format(r[c]) if isinstance(r[c], (float, np.floating)) and not pd.isna(r[c]) else ('' if (isinstance(r[c], float) and pd.isna(r[c])) else str(r[c]))) for c in cols) + ' |')
    return '\n'.join(out)
J = lambda f: json.load(open(TAB / f)); R = lambda f: pd.read_csv(TAB / f)
parts = sys.argv[1:] or ['1', '2', '3']
o = []
acc = J('frame_accounting.json'); F = R('filter_table.csv')
o.append(f"""# Maps results for Ch7 §7.4 - report ({'interim, sections 1-3' if parts == ['1','2','3'] else 'final'})
Branch `maps-results`, 2026-10-08. All numbers below are read from `data_set/campinas_b/maps/results_20261008/tables/` by `make_report.py`;
the method is described in `LOG.md`; scripts are in `orchard-weed-detection/map/results/`.
""")
o.append(f"""## 1. Frozen run
- Commit `{acc['git_commit']}` (`weed_map_cached.py` unchanged since 9fa8611). Trajectory: LIO-SAM rep1 keyframes; alignment `align_rep1.json` (SE(2), ATE 1.82 m).
  Rerun into `maps/liosam_rep1_frozen_20261008/`; **bit-identical** to the earlier `maps/liosam_rep1/` CSVs (`cmp`).
- Parameters: {', '.join(f'{k}={v}' for k, v in acc['params'].items())}.
- Frame accounting (closes: {acc['closes']}): {acc['manifest_rows']} manifest rows (= {acc['image_files']} images) - {acc['sync_dropped_dt_gt_0.15s']} dropped for LiDAR sync (|dt|>0.15 s; frames 0 and 50, LiDAR not yet running)
  - {acc['out_of_traj']} outside the trajectory - {acc['implausible_plane']} implausible ground plane - {acc['ground_failed']} RANSAC failures = **{acc['frames_used']} frames used**.
- Per-filter table (boxes / blobs at each stage, in the order of the pipeline; kept == CSV rows for both detectors):

""")
Ft = F.set_index('detector').T.loc[['raw', 'above_horizon_or_gate', 'small_blob', 'no_ground_hit', 'range_gt_5m', 'area_gt_1m2', 'kept']].reset_index().rename(columns={'index': 'stage'})
o.append(md(Ft.assign(yolo=Ft.yolo.astype(int), colour=Ft.colour.astype(int)), {}))
o.append(f"""
Notes: for colour, `raw` = contours of the *gated* mask, `small_blob` = contours <= MIN_BLOB (300 px); the gate itself acts pixel-wise
(ungated mask would give {int(F.set_index('detector').loc['colour','ungated_blobs'])} blobs > 300 px against {int(F.set_index('detector').loc['colour','kept']+F.set_index('detector').loc['colour','no_ground_hit']+F.set_index('detector').loc['colour','range_gt_5m']+F.set_index('detector').loc['colour','area_gt_1m2'])} after the gate).
`no_ground_hit` (ray grazes the plane, cos(phi) < 0.10, or t < 0.1 m) is the largest YOLO filter (187 of 491 boxes): those are boxes whose
bottom edge is near the horizon, i.e. far weeds, which the 5 m range limit would have removed anyway.
""")
open(RES / 'REPORT_maps_20261008.md', 'w').write('\n'.join(o))
print('\n'.join(o)[:200])
