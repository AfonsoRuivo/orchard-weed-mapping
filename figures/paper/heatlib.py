import sys, json
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
B=S+'vb/'
G=json.load(open(B+'satellite/georeference.json'))['enu_png']
img=np.asarray(Image.open(B+'satellite/'+G['file']).convert('RGB'))/255.; img=0.4*img+0.6*img.mean(2,keepdims=True); img=0.5*img+0.5
e0,e1,n0,n1=G['extent_east_north']
cv=np.load(B+'done/tables_maps_results/coverage_frozen.npz'); cnt=cv['cnt']; x0,y0,c=float(cv['x0']),float(cv['y0']),float(cv['cell'])
veil=np.zeros(cnt.shape+(4,)); veil[...,:3]=1; veil[...,3]=np.where(cnt>0,0,0.55)
Tr=pd.read_csv(B+'trunks/trunks_final_rep1_enu.csv'); Tr=Tr[Tr.kept_by_row_prior==1]
cs=0.25; gx=np.arange(-86,10,cs)+cs/2; gy=np.arange(-108,2,cs)+cs/2; GX,GY=np.meshgrid(gx,gy)
ci=((GX-x0)/c).astype(int); cj=((GY-y0)/c).astype(int)
seen=(ci>=0)&(ci<cnt.shape[1])&(cj>=0)&(cj<cnt.shape[0]); seen[seen]=cnt[cj[seen],ci[seen]]>0
CM_H=LinearSegmentedColormap.from_list('h',['#FFF5D6','#FDBF6F','#F16913','#B10026','#4A0010'])
def heat(e,sig=1.0):
    H=np.zeros_like(GX)
    for x,y,p in zip(e.x,e.y,e.priority):
        m=(np.abs(GX-x)<4*sig)&(np.abs(GY-y)<4*sig); H[m]+=p*np.exp(-((GX[m]-x)**2+(GY[m]-y)**2)/(2*sig**2))
    return H
def base(a):
    a.imshow(img,extent=[e0,e1,n0,n1],origin='upper',interpolation='bilinear',rasterized=True,zorder=0)
    a.imshow(veil,extent=[x0,x0+cnt.shape[1]*c,y0,y0+cnt.shape[0]*c],origin='lower',interpolation='nearest',rasterized=True,zorder=1)
    a.set_xlim(-84,8); a.set_ylim(-106,0); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_linewidth(0.5)
def overlay(a,H,vmax,tsize=1.6):
    Hn=np.clip(H/vmax,0,1); rgba=CM_H(Hn); rgba[...,3]=np.clip(Hn*1.6,0,0.92); rgba[~seen,3]=0
    a.imshow(rgba,extent=[gx[0]-cs/2,gx[-1]+cs/2,gy[0]-cs/2,gy[-1]+cs/2],origin='lower',interpolation='bilinear',zorder=2,rasterized=True)
    a.scatter(Tr.east_m,Tr.north_m,s=tsize,color=COL['trunk'],lw=0,zorder=3,rasterized=True)
