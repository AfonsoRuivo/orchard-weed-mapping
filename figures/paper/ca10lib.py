import sys, json
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import pandas as pd
CAB=S+'ca10run/ca10_run/viz_bundle_ca10/'; OUTD=S+'fig/ca10/'
T0=1782577258.896306
AL=json.load(open(CAB+'alignments/align_rep1.json')); RA=np.array(AL['R']); TA=np.array(AL['t'])
def enu(xy): return np.asarray(xy)@RA.T+TA
PIECE_COL={'A':'#D4A62A','B-pre':'#0072B2','B-post':'#CC79A7'}
GREENS=mpl.colors.LinearSegmentedColormap.from_list('g',['#C9D9A0','#8DB25A','#5B8C2A','#2F5E1E','#1B3D12'])
def traj(kind='keyframes'):
    d=pd.read_csv(CAB+('trajectories/cb11/liosam_rep1_keyframes_raw.csv' if kind=='keyframes' else 'trajectories/cb11/liosam_merged_dense_raw.csv')); d['tb']=d.t-T0
    d['piece']=np.where(d.tb<810,'A',np.where(d.tb<970,'B-pre','B-post')); return d
def trunks():
    t=pd.read_csv(CAB+'trunks/trunks_final_rep1_enu.csv'); t=t[(t.kept_by_row_prior==1)]
    return t[t.groupby('row_id').row_id.transform('size')>=3]
def dense():
    d=np.load(CAB+'dense/dense_liosam_rep1.npz'); return d['xyz'].astype(float), d['h_above_ground'].astype(float)
def gnss():
    g=pd.read_csv(CAB+'trajectories/cb11/gnss_raw_source.csv'); g['tb']=g.timestamp-T0; return g
LAB={'A':'segment 1','B-pre':'segment 2','B-post':'segment 3'}
