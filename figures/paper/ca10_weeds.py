import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'ca10lib.py')).read())
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, Patch
from scipy.spatial import cKDTree
OV=S+'px/out/ca10p/'
tr=trunks(); tj=traj(); P,H=dense()
toA=lambda e:(np.asarray(e)-TA)@RA
ang=[np.arctan(np.polyfit(q.x_rep1,q.y_rep1,1)[0]) for _,q in tr.groupby('row_id')]; th=float(np.mean(ang))
c,s=np.cos(-th),np.sin(-th); Rm=np.array([[c,-s],[s,c]]); c0=np.array([tr.x_rep1.mean(),tr.y_rep1.mean()])
F=lambda xy:(np.asarray(xy)-c0)@Rm.T
tuv=F(tr[['x_rep1','y_rep1']].values)
E={d:pd.read_csv(OV+f'priority_{d}.csv') for d in ('yolo','colour')}
for d,e in E.items(): q=F(toA(e[['x','y']].values)); e['u'],e['v']=q[:,0],q[:,1]
pv=pd.read_csv(CAB+'weed_map/patches/patch_vertices.csv'); pm=pd.read_csv(CAB+'weed_map/patches/patches.csv').set_index('pid')
q=F(pv[['x_map','y_map']].values); pv['u'],pv['v']=q[:,0],q[:,1]
cv=np.load(CAB+'weed_map/tables/coverage_frozen.npz'); cnt=cv['cnt']; x0,y0,cc=float(cv['x0']),float(cv['y0']),float(cv['cell'])
jj,ii=np.nonzero(cnt>0); ce=np.c_[x0+(ii+0.5)*cc,y0+(jj+0.5)*cc]; cuv=F(toA(ce))
DC={'yolo':COL['yolo'],'colour':COL['colour']}
fig=plt.figure(figsize=(FIG_W,FIG_W*0.84))
gs=fig.add_gridspec(2,1,height_ratios=[1,0.95],hspace=0.22,left=0.07,right=0.92,top=0.95,bottom=0.07)
ax=fig.add_subplot(gs[0]); ax.set_aspect('equal')
ax.scatter(cuv[:,0],cuv[:,1],s=4.2,marker='s',c=COL['ground'],lw=0,rasterized=True,zorder=0)
ax.scatter(tuv[:,0],tuv[:,1],s=2.2,c=COL['trunk'],lw=0,zorder=1)
for pid,g in pv.groupby('pid'):
    d=pm.loc[pid,'det']; ax.add_patch(Polygon(g[['u','v']].values,closed=True,fc=DC[d],ec=DC[d],alpha=0.28,lw=0.4,zorder=2))
for d,mk in (('yolo','o'),('colour','D')):
    e=E[d][E[d].patch==0]; ax.scatter(e.u,e.v,s=6+12*np.sqrt(e.area_med),marker=mk,fc='none',ec=DC[d],lw=0.6,zorder=3)
# hot spot = highest-priority YOLO entity
h=E['yolo'].sort_values('priority').iloc[-1]; U0,V0=h.u,h.v; UW,VW=7.0,3.6
ax.add_patch(Rectangle((U0-UW,V0-VW),2*UW,2*VW,fc='none',ec='k',lw=0.7,ls='--',zorder=5)); ax.text(U0+UW+0.6,V0,'(b)',fontsize=7,va='center')
ax.set_xlabel('along the rows (m)',labelpad=1); ax.set_ylabel('across (m)')
ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc='none',mec=COL['yolo'],ms=4,label='YOLO weed'),Line2D([],[],marker='D',ls='',mfc='none',mec=COL['colour'],ms=3.5,label='colour weed'),
   Patch(fc=COL['yolo'],alpha=0.35,label='YOLO patch'),Patch(fc=COL['colour'],alpha=0.35,label='colour patch'),Patch(fc=COL['ground'],label='ground seen by camera'),
   Line2D([],[],marker='o',ls='',ms=2.5,color=COL['trunk'],label='trunk')],ncol=6,loc='lower left',bbox_to_anchor=(0.0,1.0),fontsize=6.3,handlelength=1.2,columnspacing=0.9,borderaxespad=0.2)
panel_label(ax,'(a)',x=-0.07,y=1.06)
# (b) top-view close-up: LiDAR points low above the ground, with the detections
ax2=fig.add_subplot(gs[1]); ax2.set_aspect('equal')
uv=F(P[:,:2]); HC=0.8
m=(np.abs(uv[:,0]-U0)<UW)&(np.abs(uv[:,1]-V0)<VW)&(H>-0.3)&(H<HC); uvb,Hb=uv[m],H[m]
g=Hb<0.10; ax2.scatter(uvb[g,0],uvb[g,1],s=1.2,c=COL['ground'],marker='s',lw=0,rasterized=True,zorder=0)
o=np.argsort(Hb[~g]); ax2.scatter(uvb[~g,0][o],uvb[~g,1][o],s=1.6,c=GREENS(np.clip(Hb[~g][o]/HC,0,1)),marker='s',lw=0,rasterized=True,zorder=1)
for pid,gp in pv.groupby('pid'):
    if not ((np.abs(gp.u-U0)<UW+0.5)&(np.abs(gp.v-V0)<VW+0.5)).any(): continue
    d=pm.loc[pid,'det']; ax2.add_patch(Polygon(gp[['u','v']].values,closed=True,fc='none',ec=DC[d],lw=1.0,ls='-' if d=='yolo' else '--',zorder=4))
for d,mk in (('yolo','o'),('colour','D')):
    e=E[d][(E[d].patch==0)&(np.abs(E[d].u-U0)<UW)&(np.abs(E[d].v-V0)<VW)]
    ax2.scatter(e.u,e.v,s=18,marker=mk,fc='white',ec=DC[d],lw=1.0,zorder=5)
ts=(np.abs(tuv[:,0]-U0)<UW)&(np.abs(tuv[:,1]-V0)<VW)
ax2.scatter(tuv[ts,0],tuv[ts,1],s=22,c=COL['trunk'],edgecolors='white',lw=0.5,zorder=6)
ax2.set_xlim(U0-UW,U0+UW); ax2.set_ylim(V0-VW,V0+VW); ax2.set_xlabel('along the rows (m)',labelpad=1); ax2.set_ylabel('across (m)')
panel_label(ax2,'(b)',x=-0.07,y=1.02)
cax=ax2.inset_axes([1.02,0.1,0.018,0.8]); sm=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,HC),cmap=GREENS)
cb=fig.colorbar(sm,cax=cax); cb.set_ticks([0,0.4,0.8]); cb.ax.tick_params(labelsize=6,length=2,width=0.4); cb.outline.set_linewidth(0.4); cb.set_label('LiDAR height above ground (m)',fontsize=6.5)
ax2.legend(handles=[Line2D([],[],color=COL['yolo'],lw=1.0,label='YOLO patch outline'),Line2D([],[],color=COL['colour'],lw=1.0,ls='--',label='colour patch outline')],
           ncol=2,loc='lower right',bbox_to_anchor=(1.0,1.0),fontsize=6.3,borderaxespad=0.2)
fig.savefig(OUTD+'ca10_weed_map.pdf',dpi=400); fig.savefig(OUTD+'ca10_weed_map.png',dpi=200)
print('hot spot',U0,V0,h.priority,h.patch,h.area_med)
