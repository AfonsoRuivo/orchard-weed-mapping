import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'heatlib.py')).read())
from matplotlib.lines import Line2D
import matplotlib.patches as mp
OV=S+'px/out/patches/'
ey=pd.read_csv(OV+'entities_yolo.csv'); ec=pd.read_csv(OV+'entities_colour.csv')
A=json.load(open(B+'alignments/align_rep1.json')); Rm=np.array(A['R']); tv=np.array(A['t'])
tr=pd.read_csv(B+'trajectories/cb11/liosam_rep1_keyframes_raw.csv'); P=tr[['x','y']].values@Rm.T+tv
DARK='#111827'; TE='#065F5F'
W=18.1/2.54
fig=plt.figure(figsize=(W,W*0.415))
axs=[fig.add_axes([0.002+i*0.315,0.085,0.305,0.85]) for i in range(3)]
cax=fig.add_axes([0.952,0.25,0.010,0.55])
a=axs[0]; base(a)
a.plot(P[:,0],P[:,1],color=COL['path'],lw=0.7,zorder=2)
a.scatter(Tr.east_m,Tr.north_m,s=1.6,color=COL['trunk'],lw=0,zorder=3,rasterized=True)
q=ec[ec.patch==0]; a.scatter(q.x,q.y,s=np.clip(2+q.area_med*30,2,9),color=COL['colour'],lw=0,alpha=0.7,zorder=4,rasterized=True)
q=ec[ec.patch==1]; a.scatter(q.x,q.y,s=np.clip(6+q.area_med*6,8,22),marker='s',color=COL['colour'],edgecolor=TE,lw=0.4,zorder=5,rasterized=True)
q=ey[ey.patch==0]; a.scatter(q.x,q.y,s=np.clip(7+q.area_med*40,7,26),facecolor='none',edgecolor=DARK,lw=0.7,zorder=6)
q=ey[ey.patch==1]; a.scatter(q.x,q.y,s=np.clip(9+q.area_med*5,12,28),marker='s',facecolor='none',edgecolor=DARK,lw=0.7,zorder=7)
Hs={d:heat(pd.read_csv(OV+f'priority_{d}.csv')) for d in ('yolo','colour')}; vmax=max(np.percentile(H[H>0.05],99) for H in Hs.values())
for ax,d in zip(axs[1:],('yolo','colour')): base(ax); overlay(ax,Hs[d],vmax,1.2)
for ax,t in zip(axs,['(a) weed entities','(b) priority density, YOLO11n','(c) priority density, colour']):
    ax.set_title(t,fontsize=7.5,loc='left',pad=2)
axs[0].plot([-80,-60],[-102]*2,color='k',lw=1.4,zorder=12); axs[0].text(-70,-100.6,'20 m',ha='center',va='bottom',fontsize=6.5,zorder=12,bbox=dict(fc='white',ec='none',pad=0.5,alpha=0.9))
axs[0].annotate('',xy=(3,-4),xytext=(3,-12),arrowprops=dict(arrowstyle='-|>',lw=0.7,color='k')); axs[0].text(3,-13,'N',ha='center',va='top',fontsize=6.5)
sm=plt.cm.ScalarMappable(cmap=CM_H,norm=plt.Normalize(0,1)); cb=fig.colorbar(sm,cax=cax); cb.set_ticks([0,1]); cb.set_ticklabels(['low','high']); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
h=[Line2D([],[],marker='o',ls='',mfc='none',mec=DARK,mew=0.9,ms=5,label=f'YOLO11n plant ({int((ey.patch==0).sum())})'),
   Line2D([],[],marker='s',ls='',mfc='none',mec=DARK,mew=0.9,ms=5,label=f'YOLO11n patch ({int(ey.patch.sum())})'),
   Line2D([],[],marker='o',ls='',mfc=COL['colour'],mec='none',ms=4,label=f'colour plant ({int((ec.patch==0).sum())})'),
   Line2D([],[],marker='s',ls='',mfc=COL['colour'],mec=TE,mew=0.4,ms=4.5,label=f'colour patch ({int(ec.patch.sum())})'),
   Line2D([],[],marker='o',ls='',color=COL['trunk'],ms=3,label='trunk'),
   Line2D([],[],color=COL['path'],lw=1.2,label='path'),
   mp.Patch(fc='white',ec='0.6',lw=0.5,label='not seen')]
fig.legend(handles=h,loc='lower center',ncol=7,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,0.0),columnspacing=0.9,handletextpad=0.3)
fig.savefig(S+'paper/figures/weed_map_priority.pdf',dpi=300); fig.savefig(S+'fig/paper_map.png',dpi=130)
