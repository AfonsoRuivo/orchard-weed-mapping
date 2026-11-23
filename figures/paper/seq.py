import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'proj.py')).read())
import sys; sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
A=S+'ax/appendix_export/A_run2/'
F=pd.read_csv(S+'ax/appendix_export/A_frames.csv').set_index('stem')
stems=[s for s in F.index if s[6:12] in ('016000','016050','016100','016150')]
FC=['#7B3294','#D6604D','#2166AC','#1B7837']; YC=COL['yolo']; CC=COL['colour']; PC='#A0522D'
R=1.16; fig=plt.figure(figsize=(FIG_W,FIG_W*R)); W=0.49; H=W*9/16/R
pos=[(0.0,1-H),(0.51,1-H),(0.0,1-2*H-0.012),(0.51,1-2*H-0.012)]
pts={}
for k,s in enumerate(stems):
    r=F.loc[s]; ax=fig.add_axes([pos[k][0],pos[k][1],W,H])
    im=mpimg.imread(A+s+'.jpg').astype(float)/255.
    m=mpimg.imread(A+s+'_colour_mask.png'); m=(m[...,0] if m.ndim==3 else m)>0.5
    tint=np.array([0,166,166])/255.; im[m]=0.55*im[m]+0.45*tint
    ax.imshow(im[::2,::2],extent=[0,1280,720,0]); ax.set_xlim(0,1280); ax.set_ylim(720,0); ax.axis('off')
    g=pd.read_csv(A+s+'_gate_line.csv'); ax.plot(g.iloc[:,0],g.iloc[:,1],color='#F2B701',lw=1.0,ls=(0,(4,2)))
    pts[s]=[]
    for f,dv in (('_colour.csv',1),('_yolo.csv',0)):
        d=pd.read_csv(A+s+f)
        for q in d.itertuples():
            patch=q.status.startswith('area'); kept=q.status=='kept' or patch
            ls='-' if kept else (0,(3,2))
            if f=='_yolo.csv':
                ax.add_patch(plt.Rectangle((q.x1,q.y1),q.x2-q.x1,q.y2-q.y1,fill=False,ec=YC,lw=2.6,ls=ls))
                ax.add_patch(plt.Rectangle((q.x1,q.y1),q.x2-q.x1,q.y2-q.y1,fill=False,ec='white',lw=1.0,ls=ls))
            else:
                ax.add_patch(plt.Rectangle((q.x1,q.y1),q.x2-q.x1,q.y2-q.y1,fill=False,ec=CC,lw=1.0,ls=ls))
            if kept:
                gm=ground_map(r.a,r.b,r.c,r.d,r.timestamp,(q.x1+q.x2)/2,q.y2-dv); pts[s].append((gm,f,patch))
            if patch or not kept:
                tag='patch' if patch else ('gate' if q.status.startswith('above') else {'g':'grazing','o':'range'}.get(q.status[0],''))
                ax.text(q.x1+4,q.y1-6,tag,fontsize=5.2,color='white',va='bottom',bbox=dict(fc=PC if patch else '0.15',ec='none',pad=0.6,alpha=0.85))
    ax.text(14,22,'abcd'[k],fontsize=8,fontweight='bold',color='white',va='top',ha='left',bbox=dict(fc=FC[k],ec='none',boxstyle='round,pad=0.25'))
    ax.text(1266,22,f't = {r.timestamp-F.loc[stems[0],"timestamp"]:.1f} s',fontsize=6.5,color='white',ha='right',va='top',bbox=dict(fc='0.1',ec='none',pad=1,alpha=0.6))
b=fig.add_axes([0.0,0.075,1.0,1-2*H-0.012-0.075-0.045])
d=np.load(S+'vb/../vb_dense_none.npz') if False else None
x0,x1,y0,y1=-58.6,-42.6,78.6,87.4
P=pd.read_csv(S+'vb/trunks/trunks_final_rep1_all.csv'); P=P[P.kept_by_row_prior==1]
Q=np.load(S+'liosam_raw.npy').astype(float)
m=(Q[:,0]>x0-4)&(Q[:,0]<x1+4)&(Q[:,1]>y0-4)&(Q[:,1]<y1+4); X=Q[m]; cs_=0.1
ix=((X[:,0]-x0+4)/cs_).astype(int); iy=((X[:,1]-y0+4)/cs_).astype(int); Wd=ix.max()+1; Hd=iy.max()+1
gmin=np.full((Hd,Wd),np.inf); np.minimum.at(gmin,(iy,ix),X[:,2]); gf=np.where(np.isfinite(gmin),gmin,np.nan)
gm_=ndi.gaussian_filter(ndi.grey_erosion(np.nan_to_num(gf,nan=np.nanmax(gf)),size=(21,21)),6)
hh=X[:,2]-gm_[iy,ix]; top=np.full((Hd,Wd),np.nan); np.fmax.at(top,(iy,ix),hh)
veg=LinearSegmentedColormap.from_list('v',['#efe9dc','#c8d6a2','#8fb36c','#5f8f4c','#3f6e3a'])
bg=veg(np.clip(np.nan_to_num(top,nan=0)/3.0,0,1)); bg[np.isnan(top)]=[1,1,1,1]; bg[...,:3]=0.25+0.75*bg[...,:3]
b.imshow(bg,extent=[x0-4,x0-4+Wd*cs_,y0-4,y0-4+Hd*cs_],origin='lower',interpolation='bilinear',rasterized=True)
tr2=pd.read_csv(S+'px/traj_rep1.csv'); b.plot(tr2.x,tr2.y,color=COL['path'],lw=1.2,zorder=2)
b.scatter(P.x,P.y,s=16,c=COL['trunk'],edgecolors='white',linewidths=0.4,zorder=3)
hf=np.arctan(640/924.)
for k,s in enumerate(stems):
    r=F.loc[s]; yw=r.yaw_slam; p=np.array([r.x_slam,r.y_slam]); ang=np.linspace(yw-hf,yw+hf,20)
    poly=np.r_[[p+1.0*np.array([np.cos(a_),np.sin(a_)]) for a_ in ang],[p+5.0*np.array([np.cos(a_),np.sin(a_)]) for a_ in ang[::-1]]]
    b.add_patch(Polygon(poly,closed=True,fc=FC[k],alpha=0.07,ec='none',zorder=3)); b.add_patch(Polygon(poly,closed=True,fill=False,ec=FC[k],lw=0.7,alpha=0.8,zorder=3))
    b.scatter(*p,s=26,c=FC[k],edgecolors='white',linewidths=0.6,zorder=6); b.text(p[0]+0.05,p[1]-0.45,'abcd'[k],fontsize=7,fontweight='bold',color=FC[k],ha='center',va='top',zorder=6)
    for gm,f,patch in pts[s]:
        if f=='_colour.csv': b.scatter(*gm,s=30 if patch else 14,marker='s' if patch else 'o',c=FC[k],edgecolors='white',linewidths=0.3,zorder=7)
        else: b.scatter(*gm,s=60 if patch else 46,marker='s',facecolors='none',edgecolors=FC[k],linewidths=1.3 if not patch else 1.8,zorder=8)
b.annotate('',xy=(-43.0,79.6),xytext=(-46.0,79.6),arrowprops=dict(arrowstyle='->',color='0.25',lw=1.0)); b.text(-44.5,79.35,'direction of travel',fontsize=6.5,ha='center',va='top',color='0.25')
b.set_xlim(x0,x1); b.set_ylim(y0,y1); b.set_aspect('equal'); b.set_xticks([]); b.set_yticks([])
for sp in b.spines.values(): sp.set_linewidth(0.5)
b.plot([x0+0.4,x0+2.4],[y0+0.5]*2,color='k',lw=1.5); b.text(x0+1.4,y0+0.7,'2 m',ha='center',va='bottom',fontsize=6.5)
b.text(0.0,1.01,'(e) Top view: what each frame sees (up to 5 m) and where its detections land',transform=b.transAxes,fontsize=8,fontweight='bold',ha='left',va='bottom')
h=[Line2D([],[],marker='s',ls='none',mfc='none',mec=YC,mew=1.4,ms=6,label='YOLO box / point'),
   Line2D([],[],marker='o',ls='none',mfc=CC,mec='white',ms=5,label='colour blob / point'),
   Line2D([],[],color='0.3',lw=1,ls=(0,(3,2)),label='dropped by a filter'),
   Line2D([],[],color='#F2B701',lw=1,ls=(0,(4,2)),label='height gate'),
   Line2D([],[],marker='o',ls='none',mfc=COL['trunk'],mec='white',ms=5,label='trunk'),
   Line2D([],[],color=COL['path'],lw=1.2,label='robot path')]
fig.legend(handles=h,loc='lower center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(0.5,0.0),columnspacing=1.4)
fig.savefig(S+'fig/seq_raw.pdf',dpi=220); fig.savefig(S+'fig/detection_sequence.png',dpi=110)
