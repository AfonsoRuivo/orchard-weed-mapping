import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
import sys,json; sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]
from thesis_style import *
import pandas as pd
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
D='/mnt/user-data/uploads/trunk/v2_out/fig_method/'
OUT=S + 'paper/figures/trunk_method_col'
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5})
M=json.load(open(D+'meta.json')); A=pd.read_csv(D+'a_side_points.csv'); B=np.load(D+'b_density.npz'); C=pd.read_csv(D+'b_candidates.csv'); R=pd.read_csv(D+'c_row.csv')
GRN='#2E8B3D'; WEED='#C0392B'; BAND='#F4A261'; COLM='#3A6EA5'
W=8.8/2.54; fig=plt.figure(figsize=(W,W*0.96))
a=fig.add_axes([0.12,0.56,0.36,0.38]); b=fig.add_axes([0.63,0.56,0.35,0.38]); c=fig.add_axes([0.12,0.19,0.86,0.24])
# (a) side view
a.axhspan(0.10,0.40,color=BAND,alpha=0.35,lw=0,zorder=0)
a.scatter(A.s,A.h,s=0.6,c='0.25',lw=0,rasterized=True,zorder=2)
a.add_patch(Rectangle((-0.35,0.4),0.7,0.8,fill=False,ec=COLM,lw=1.0,ls=(0,(3,2)),zorder=3))
a.text(0.38,1.25,'column',color=COLM,fontsize=6.5,va='bottom',ha='left'); a.text(0.55,0.25,'band',color='#B5651D',fontsize=6.5,va='center')
a.set_xlim(-1,1); a.set_ylim(0,1.6); a.set_xlabel('along the row (m)'); a.set_ylabel('height above ground (m)')
a.set_title('(a) side view of one tree',loc='left',fontsize=7,pad=3)
# (b) density
tx,ty=M['tree_xy']; ex,ey=B['x_edges']-tx,B['y_edges']-ty
b.imshow(B['S'].T,origin='lower',extent=[ex[0],ex[-1],ey[0],ey[-1]],cmap='Greys',vmax=np.percentile(B['S'],99.5),interpolation='bilinear',rasterized=True)
for r in C.itertuples():
    x,y=r.x-tx,r.y-ty
    if r.kind=='trunk': b.add_patch(Circle((x,y),0.2,fill=False,ec=GRN,lw=1.3)); b.text(x-0.9,y+0.28,f'trunk: band {r.n_core},\ncolumn {r.n_up}',color=GRN,fontsize=6,va='bottom',ha='left')
    elif r.kind=='weed': b.add_patch(Circle((x,y),0.2,fill=False,ec=WEED,lw=1.3)); b.text(x+0.1,y-0.3,f'weed: band {r.n_core},\ncolumn {r.n_up}',color=WEED,fontsize=6,va='top',ha='center')
    else: b.plot(x,y,'+',color='0.45',ms=4,mew=0.7)
b.set_aspect('equal'); b.set_xlim(-1.0,2.0); b.set_ylim(-2.7,0.9)
b.set_xlabel('east (m)'); b.set_ylabel('north (m)'); b.set_title('(b) band density',loc='left',fontsize=7,pad=3)
# (c) row model
m=(R.s>17.5)&(R.s<48.5)&(R.n.abs()<0.9); r=R[m]
c.axhspan(-0.45,0.45,color='0.93',lw=0,zorder=0)
lo=r[r.strength<0.4]; c.scatter(lo.s,lo.n,s=3,c='0.65',lw=0,zorder=2)
pool=r[(r.strength>=0.4)&(r.picked==0)]; c.scatter(pool.s,pool.n,s=14,facecolors='none',edgecolors='0.45',lw=0.7,zorder=3)
g=r[(r.picked==1)&(r.gate==1)]; c.scatter(g.s,g.n,s=18,c=GRN,lw=0,zorder=4)
w=r[(r.picked==1)&(r.gate==0)]; c.scatter(w.s,w.n,s=18,facecolors='white',edgecolors=GRN,lw=1.2,zorder=4)
for q in r[r.picked==1].itertuples():
    c.text(q.s,q.n-0.2,str(q.gt_id).split('-')[1] if isinstance(q.gt_id,str) else '',ha='center',va='top',fontsize=5.8,color='0.3')
    c.text(q.s,q.n+0.2,f'{q.strength:.2f}',ha='center',va='bottom',fontsize=5.8,color=GRN)
c.set_xlim(17.5,48.5); c.set_ylim(-0.9,0.9); c.set_yticks([-0.5,0,0.5]); c.set_xlabel('along the row (m)'); c.set_ylabel('across (m)')
c.set_title(f'(c) row model at a row end (spacing {M["spacing_m"]:.2f} m)',loc='left',fontsize=7,pad=3)
h=[Line2D([],[],marker='o',ls='',color='0.65',ms=2.5,label='strength < 0.4'),
   Line2D([],[],marker='o',ls='',mfc='none',mec='0.45',ms=4,label='in pool, not chosen'),
   Line2D([],[],marker='o',ls='',color=GRN,ms=4.5,label='chosen, passes gate'),
   Line2D([],[],marker='o',ls='',mfc='white',mec=GRN,mew=1.2,ms=4.5,label='chosen by row model')]
fig.legend(handles=h,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.55,0.0),columnspacing=1.2,handletextpad=0.3)
for ax in (a,b,c): ax.tick_params(length=2)
fig.savefig(OUT+'.pdf',dpi=400); fig.savefig(OUT+'.png',dpi=250)
