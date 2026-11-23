import sys
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
CAB=S+'ca10run/ca10_run/viz_bundle_ca10/'
import json
from PIL import Image
G=json.load(open(S+'ca10run2/ca10_run/viz_bundle_ca10/satellite/georeference.json'))['enu_png']
img=np.asarray(Image.open(S+'ca10run2/ca10_run/viz_bundle_ca10/satellite/'+G['file']).convert('RGB'))/255.; img=0.4*img+0.6*img.mean(2,keepdims=True); img=0.5*img+0.5
e0,e1,n0,n1=G['extent_east_north']; OV=S+'px/out/ca10p/'
cv=np.load(CAB+'weed_map/tables/coverage_frozen.npz'); cnt=cv['cnt']; x0,y0,c=float(cv['x0']),float(cv['y0']),float(cv['cell'])
Tr=pd.read_csv(CAB+'trunks/trunks_final_rep1_enu.csv'); Tr=Tr[(Tr.kept_by_row_prior==1)&(Tr.groupby('row_id').row_id.transform('size')>=3)]
K=pd.read_csv(CAB+'trajectories/cb11/liosam_merged_dense_enu.csv') if False else pd.read_csv(CAB+'trajectories/cb11/liosam_rep1_keyframes_enu.csv')
cs=0.25; gx=np.arange(-30,30,cs)+cs/2; gy=np.arange(-12,102,cs)+cs/2; GX,GY=np.meshgrid(gx,gy)
ci=((GX-x0)/c).astype(int); cj=((GY-y0)/c).astype(int)
seen=(ci>=0)&(ci<cnt.shape[1])&(cj>=0)&(cj<cnt.shape[0]); seen[seen]=cnt[cj[seen],ci[seen]]>0
CM_H=LinearSegmentedColormap.from_list('h',['#FFF5D6','#FDBF6F','#F16913','#B10026','#4A0010'])
def heat(e,sig=1.0):
    H=np.zeros_like(GX)
    for x,y,p in zip(e.x,e.y,e.priority):
        m=(np.abs(GX-x)<4*sig)&(np.abs(GY-y)<4*sig); H[m]+=p*np.exp(-((GX[m]-x)**2+(GY[m]-y)**2)/(2*sig**2))
    return H
XL=(-24,20); YL=(-8,97)
def base(a):
    a.imshow(img,extent=[e0,e1,n0,n1],origin='upper',interpolation='bilinear',rasterized=True,zorder=0)
    veil=np.zeros(cnt.shape+(4,)); veil[...,:3]=1; veil[...,3]=np.where(cnt>0,0,0.55)
    a.imshow(veil,extent=[x0,x0+cnt.shape[1]*c,y0,y0+cnt.shape[0]*c],origin='lower',interpolation='nearest',rasterized=True,zorder=1)
    a.set_xlim(*XL); a.set_ylim(*YL); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_linewidth(0.5)
def overlay(a,H,vmax):
    Hn=np.clip(H/vmax,0,1); rgba=CM_H(Hn); rgba[...,3]=np.clip(Hn*1.6,0,0.92); rgba[~seen,3]=0
    a.imshow(rgba,extent=[gx[0]-cs/2,gx[-1]+cs/2,gy[0]-cs/2,gy[-1]+cs/2],origin='lower',interpolation='bilinear',zorder=2,rasterized=True)
    a.scatter(Tr.east_m,Tr.north_m,s=1.6,color=COL['trunk'],lw=0,zorder=3,rasterized=True)
Hs={d:heat(pd.read_csv(OV+f'priority_{d}.csv')) for d in ('yolo','colour')}; vmax=max(np.percentile(H[H>0.05],99) for H in Hs.values())
fig=plt.figure(figsize=(FIG_W*0.78,FIG_W*0.78*0.98))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.06],wspace=0.05); axs=[fig.add_subplot(gs[i]) for i in (0,1)]; cax=fig.add_subplot(gs[2])
for a,d,lab in zip(axs,('yolo','colour'),('(a) YOLO11n','(b) colour')):
    base(a); overlay(a,Hs[d],vmax); a.text(0,1.01,lab,transform=a.transAxes,fontsize=8,fontweight='bold',va='bottom')
axs[0].plot([-21,-11],[-5.5]*2,color='k',lw=1.5); axs[0].text(-16,-4.6,'10 m',ha='center',va='bottom',fontsize=7)
axs[1].annotate('',xy=(16,3),xytext=(16,-5),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='k')); axs[1].text(16,3.6,'N',ha='center',va='bottom',fontsize=7)
sm=plt.cm.ScalarMappable(cmap=CM_H,norm=plt.Normalize(0,1)); cb=fig.colorbar(sm,cax=cax); cb.set_label('priority density',fontsize=7); cb.set_ticks([0,1]); cb.set_ticklabels(['low','high']); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
fig.subplots_adjust(left=0.01,right=0.9,top=0.95,bottom=0.02)
fig.savefig(S+'fig/ca10/ca10_priority_heat.pdf',dpi=300); fig.savefig(S+'fig/ca10/ca10_priority_heat.png',dpi=150)
