#!/usr/bin/env python3
"""Section 3a - merge-radius sweep 0.1..1.0 m for YOLO and colour (frozen run, and dense run if available).
Per eps: entities, % seen once (single frame), span quantiles, # entities with span > 2 eps, same-frame merges
(entities holding >1 detection of one frame = distinct objects forced together; independent of sigma_loc)."""
import sys
from common import *
from entities import *
from sigma_lib import load_weeds_run
run = sys.argv[1] if len(sys.argv) > 1 else 'frozen'; dets = sys.argv[2:] or ['yolo', 'colour']
EPS = np.round(np.arange(0.1, 1.0001, 0.05), 2); out = []
for det in dets:
    w = load_weeds_run(det, run)
    for e in EPS:
        E, lab = entities(w, e)
        out.append(dict(det=det, eps=e, n_det=len(w), n_entities=len(E), pct_once=100 * (E.n_frames == 1).mean(),
                        pct_once_det=100 * (E.n_det == 1).mean(), span_med=E.span.median(), span_p95=E.span.quantile(.95),
                        span_max=E.span.max(), n_span_gt_2eps=int((E.span > 2 * e).sum()), n_samefr_entities=int((E.samefr > 0).sum()),
                        pct_samefr_entities=100 * (E.samefr > 0).mean(), n_samefr_extra=int(E.samefr.sum()),
                        biggest=int(E.n_det.max())))
pd.DataFrame(out).to_csv(TAB / ('eps_sweep.csv' if run == 'frozen' else f'eps_sweep_{run}.csv'), index=False)
print(pd.DataFrame(out).round(2).to_string())
