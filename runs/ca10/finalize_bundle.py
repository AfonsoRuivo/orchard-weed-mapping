#!/usr/bin/env python3
"""Last step of build_bundle.py (also runnable alone): MANIFEST.md, REPORT.md, RUNLOG.md into the bundle, and the
format check against the CB-11 bundle (reports/format_check.txt). build_bundle.py recreates the folder, so
hand-written files must come from here."""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); B = R / 'viz_bundle_ca10'; A = Path('/home/afonso/viz_bundle_20261014')
shutil.copy2(R / 'scripts/bundle_MANIFEST.md', B / 'MANIFEST.md')
if (R / 'satellite').exists(): shutil.copytree(R / 'satellite', B / 'satellite', dirs_exist_ok=True)   # satellite_ca10.py output
for f in ('REPORT.md', 'RUNLOG.md'): shutil.copy2(R / f, B / 'reports' / f)
EXPECTED = {'alignments/trajectory_to_enu_fits.json': 'CB-09/CB-01 sessions not part of CA-10',
            'trunks/final_rep1/summary.json': 'no ground truth -> no GT score keys'}
def cols(p):
    try: return list(pd.read_csv(p, nrows=1).columns)
    except pd.errors.EmptyDataError: return 'EMPTY'
lines, same, diff, new = [], 0, 0, []
for f in sorted(B.rglob('*')):
    if not f.is_file() or f.suffix not in ('.csv', '.npz', '.json'): continue
    rel = f.relative_to(B); g = A / rel
    if not g.exists(): new.append(str(rel)); continue
    if f.suffix == '.csv':
        a, b = cols(g), cols(f); ok = a == b or a == 'EMPTY'
        info = f'CB-11 file is empty (0 zones); CA-10 columns {b}' if a == 'EMPTY' else sorted(set(a) ^ set(b))
    elif f.suffix == '.npz':
        x, y = np.load(g), np.load(f); a = {k: str(x[k].dtype) for k in x.files}; b = {k: str(y[k].dtype) for k in y.files}; ok = a == b; info = (a, b)
    else:
        a, b = json.load(open(g)), json.load(open(f)); ok = set(a) <= set(b); info = sorted(set(a) - set(b))
    tag = 'EXPECTED' if not ok and str(rel) in EXPECTED else ('OK  ' if ok else 'DIFF')
    if tag == 'EXPECTED': info = EXPECTED[str(rel)]
    same += tag != 'DIFF'; diff += tag == 'DIFF'
    lines.append(f"{tag} {rel} {'' if tag == 'OK  ' and not (f.suffix == '.csv' and a == 'EMPTY') else info}")
import zipfile
z = zipfile.ZipFile('/home/afonso/patch_export.zip')
for fn in ('patches.csv', 'patch_vertices.csv', 'footprint_vertices.csv', 'per_frame_area_limit_counts.csv'):
    a = z.open(f'patch_export/{fn}').readline().decode().strip(); b = open(B / 'weed_map/patches' / fn).readline().strip()
    lines.append(f"{'OK  ' if a == b else 'DIFF'} weed_map/patches/{fn} (vs CB-11 patch_export columns)"); diff += a != b
lines.append(f'\n{same} same format (incl. expected), {diff} different, {len(new)} CA-10-only files:')
lines += ['  ' + n for n in new]
(B / 'reports/format_check.txt').write_text('\n'.join(lines) + '\n'); print(lines[-len(new) - 1])
