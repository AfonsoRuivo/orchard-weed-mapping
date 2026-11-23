import numpy as np,pandas as pd,json
from scipy.spatial.transform import Rotation as Rot, Slerp
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
C=json.load(open(S+'vb/perception/calibration_campinas_b.json'))
K=np.array(C['camera_K']);T_lc=np.array(C['T_lidar_camera'])
tr=pd.read_csv(S+'px/traj_rep1.csv'); tt=tr.timestamp.values; sl=Slerp(tt,Rot.from_quat(tr[['qx','qy','qz','qw']].values))
def ground_map(a,b,c,d,t,u,v):
    n=np.array([a,b,c]); ray=T_lc[:3,:3]@(np.linalg.inv(K)@np.array([u,v,1.0])); o=T_lc[:3,3]
    s=-(n@o+d)/(n@ray); pl=o+s*ray
    R=sl([t]).as_matrix()[0]; p=np.array([np.interp(t,tt,tr[k].values) for k in 'xyz'])
    return (R@pl+p)[:2]
