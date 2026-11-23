import sys, json, os
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
import matplotlib.patches as mp
B=S+'vb/'; TAB=B+'done/tables_maps_results/'; OV=S+'px/out/patches/'
ey=pd.read_csv(OV+'entities_yolo.csv'); ec=pd.read_csv(OV+'entities_colour.csv')
T=pd.read_csv(B+'trunks/trunks_final_rep1_enu.csv'); T=T[T.kept_by_row_prior==1]
A=json.load(open(B+'alignments/align_rep1.json')); Rm=np.array(A['R']); tv=np.array(A['t'])
tr=pd.read_csv(B+'trajectories/cb11/liosam_rep1_keyframes_raw.csv'); P=tr[['x','y']].values@Rm.T+tv
G=json.load(open(B+'satellite/georeference.json'))['enu_png']
img=np.asarray(Image.open(B+'satellite/'+G['file']).convert('RGB'))/255.; img=0.45*img+0.55*img.mean(2,keepdims=True); img=0.65*img+0.35
e0,e1,n0,n1=G['extent_east_north']
cv=np.load(TAB+'coverage_frozen.npz'); cnt=cv['cnt']; x0,y0,cell=float(cv['x0']),float(cv['y0']),float(cv['cell'])
veil=np.zeros(cnt.shape+(4,)); veil[...,:3]=1.0; veil[...,3]=np.where(cnt>0,0.0,0.62)
vext=[x0,x0+cnt.shape[1]*cell,y0,y0+cnt.shape[0]*cell]
DARK='#111827'; TEAL_E='#065F5F'
def draw(ax,zoom):
    k=2.6 if zoom else 1.0
    ax.imshow(img,extent=[e0,e1,n0,n1],origin='upper',interpolation='bilinear',rasterized=True,zorder=0)
    ax.imshow(veil,extent=vext,origin='lower',interpolation='nearest',rasterized=True,zorder=1)
    ax.plot(P[:,0],P[:,1],color=COL['path'],lw=1.4 if zoom else 0.8,zorder=2)
    ax.scatter(T.east_m,T.north_m,s=18 if zoom else 3,color=COL['trunk'],lw=0,zorder=3,rasterized=not zoom)
    q=ec[ec.patch==0]; ax.scatter(q.x,q.y,s=np.clip(3+q.area_med*40,3,14)*k,color=COL['colour'],lw=0,alpha=0.6,zorder=4,rasterized=True)
    q=ec[ec.patch==1]; ax.scatter(q.x,q.y,s=np.clip(8+q.area_med*8,10,30)*k,marker='s',color=COL['colour'],edgecolor=TEAL_E,lw=0.5,alpha=0.85,zorder=5,rasterized=True)
    q=ey[ey.patch==0]; ax.scatter(q.x,q.y,s=np.clip(10+q.area_med*60,10,40)*k,facecolor='none',edgecolor=DARK,lw=1.6 if zoom else 0.9,zorder=6)
    q=ey[ey.patch==1]; ax.scatter(q.x,q.y,s=np.clip(12+q.area_med*6,16,40)*k,marker='s',facecolor='none',edgecolor=DARK,lw=1.6 if zoom else 0.9,zorder=7)
    ax.set_aspect('equal')
fig=plt.figure(figsize=(FIG_W,FIG_W*0.63))
gs=fig.add_gridspec(1,2,width_ratios=[1.0,1.0],wspace=0.06)
a=fig.add_subplot(gs[0]); b=fig.add_subplot(gs[1])
draw(a,False); a.set_xlim(-84,4); a.set_ylim(-104,4); a.set_xticks([]); a.set_yticks([])
cx,cy=-74.4,-66.7; hw,hh=7.0,9.5
draw(b,True); b.set_xlim(cx-hw,cx+hw); b.set_ylim(cy-hh,cy+hh); b.set_xticks([]); b.set_yticks([])
a.add_patch(mp.Rectangle((cx-hw,cy-hh),2*hw,2*hh,fill=False,ec='k',lw=0.8,zorder=10))
for ax,L,lab in [(a,20,'20 m'),(b,5,'5 m')]:
    xl=ax.get_xlim(); yl=ax.get_ylim(); dx=xl[1]-xl[0]; dy=yl[1]-yl[0]
    ax.plot([xl[0]+0.05*dx,xl[0]+0.05*dx+L],[yl[0]+0.04*dy]*2,color='k',lw=1.6,solid_capstyle='butt',zorder=12)
    ax.text(xl[0]+0.05*dx+L/2,yl[0]+0.055*dy,lab,ha='center',va='bottom',fontsize=7,zorder=12,bbox=dict(fc='white',ec='none',pad=0.4,alpha=0.85))
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
xl=a.get_xlim(); yl=a.get_ylim()
a.annotate('',xy=(xl[1]-5,yl[1]-4),xytext=(xl[1]-5,yl[1]-13),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='k'),zorder=12)
a.text(xl[1]-5,yl[1]-15,'N',ha='center',va='top',fontsize=7,zorder=12)
a.text(0,1.01,'(a)',transform=a.transAxes,fontsize=8,fontweight='bold',va='bottom')
b.text(0,1.01,'(b)',transform=b.transAxes,fontsize=8,fontweight='bold',va='bottom')
h=[Line2D([],[],marker='o',ls='',mfc='none',mec=DARK,mew=1.1,ms=6,label=f'YOLO11n plant ({int((ey.patch==0).sum())})'),
   Line2D([],[],marker='s',ls='',mfc='none',mec=DARK,mew=1.1,ms=6,label=f'YOLO11n patch ({int(ey.patch.sum())})'),
   Line2D([],[],marker='o',ls='',mfc=COL['colour'],mec='none',alpha=0.6,ms=4,label=f'colour plant ({int((ec.patch==0).sum())})'),
   Line2D([],[],marker='s',ls='',mfc=COL['colour'],mec=TEAL_E,mew=0.5,ms=5,label=f'colour patch ({int(ec.patch.sum())})'),
   Line2D([],[],marker='o',ls='',color=COL['trunk'],ms=3.5,label='trunk'),
   Line2D([],[],color=COL['path'],lw=1.4,label='robot path'),
   mp.Patch(fc='white',ec='0.6',lw=0.5,label='not seen by the camera')]
fig.legend(handles=h,loc='lower center',ncol=4,fontsize=7,frameon=False,bbox_to_anchor=(0.5,0.0),columnspacing=1.1,handletextpad=0.3)
fig.subplots_adjust(left=0.01,right=0.99,top=0.95,bottom=0.11)
fig.savefig(S+'ov7/Images/Results/weed_map.pdf',dpi=350); fig.savefig(S+'fig/weed_map.png',dpi=130)
