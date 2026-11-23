"""Shared paths, loaders and plotting style for the Ch7 'Maps' analysis (2026-10-08)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
MAPDIR = REPO / 'data_set/campinas_b/maps'
FROZEN = MAPDIR / 'liosam_rep1_frozen_20261008'
RES = MAPDIR / 'results_20261008'
FIG, TAB = RES / 'figures', RES / 'tables'
TRAJ = REPO / 'orchard-slam/data/thesis_jobs_20261007/for_thesis_20261007/trajectories/liosam_rep1_keyframes.csv'
TRUNKS = REPO / 'orchard-weed-detection/trunk/v2_out/trunks_final_rep1.csv'
LIDAR = REPO / 'data_set/campinas_b/lidar'
sys.path.insert(0, str(REPO / 'orchard-weed-detection/map'))

_a = json.load(open(FROZEN / 'align_rep1.json'))
R_ALIGN, T_ALIGN = np.array(_a['R']), np.array(_a['t'])

def to_enu(xy):
    return np.asarray(xy) @ R_ALIGN.T + T_ALIGN

def load_weeds(det):
    return pd.read_csv(FROZEN / f'weed_map_{det}_cached.csv')

def load_trunks():
    """Trunks in ENU through align_rep1 (NOT the x_enu column, which uses the July alignment)."""
    t = pd.read_csv(TRUNKS)
    e = to_enu(t[['x', 'y']].values)
    t['xe'], t['ye'] = e[:, 0], e[:, 1]
    return t

def load_traj():
    d = pd.read_csv(TRAJ)
    tk = 'timestamp' if 'timestamp' in d else 't'
    d = d.rename(columns={tk: 't'})
    e = to_enu(d[['x', 'y']].values)
    d['xe'], d['ye'] = e[:, 0], e[:, 1]
    return d

# colour-blind-safe, fixed meaning across all figures (Okabe-Ito)
# Chapter-7 palette: keep identical to thesis-document/figures/scripts/thesis_style.py::COL
C = dict(trunk='#8B5A2B', yolo='#334155', colour='#00A6A6', weed='#5B8C2A',
         path='#D4A62A', row='#999999', zone='#222222', null='#7f7f7f')

def style():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.grid': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'legend.frameon': False,
        'lines.linewidth': 1.0,
    })
    import matplotlib.pyplot as plt
    return plt
CM = 1 / 2.54
