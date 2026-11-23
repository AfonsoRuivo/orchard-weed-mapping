import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'ca10lib.py')).read())
from scipy.spatial import cKDTree
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
P,H=dense(); tr=trunks(); tj=traj()
# row frame: rows run along A-frame x; angle from the trunk row lines
ang=[]; 
for r,g in tr.groupby('row_id'): ang.append(np.arctan(np.polyfit(g.x_rep1,g.y_rep1,1)[0]))
th=float(np.mean(ang)); c,s=np.cos(-th),np.sin(-th); Rm=np.array([[c,-s],[s,c]]); c0=np.array([tr.x_rep1.mean(),tr.y_rep1.mean()])
def F(xy): return (np.asarray(xy)-c0)@Rm.T
uv=F(P[:,:2]); tuv=F(tr[['x_rep1','y_rep1']].values); puv=F(tj[['x','y']].values)
m=(uv[:,0]>tuv[:,0].min()-9)&(uv[:,0]<tuv[:,0].max()+9)&(uv[:,1]>tuv[:,1].min()-6)&(uv[:,1]<tuv[:,1].max()+6)&(H>-0.3)&(H<4.5)
uv,P,H=uv[m],P[m],H[m]
g=H<0.12; kd=cKDTree(uv[g]); zg=P[g,2]
def zground(q): _,ii=kd.query(q,k=8); return np.median(zg[ii],axis=1)
H_TOP=3.0
def colours(h,alpha_can=1.0):
    col=GREENS(np.clip(h/H_TOP,0,1)); gr=h<0.12; col[gr]=mpl.colors.to_rgba(COL['ground']); return col
fig=plt.figure(figsize=(FIG_W,FIG_W*0.6))
gs=fig.add_gridspec(2,1,height_ratios=[2.2,1],hspace=0.2,left=0.07,right=0.99,top=0.99,bottom=0.1)
# (a) oblique block view
ax=fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_aspect('equal')
cam=dict(az=0.0,el=50.0)
rng=np.random.default_rng(0); keep=(H<0.12)&(rng.random(len(H))<0.5) | (H>=0.12)
X,Y,D=project(np.c_[uv,P[:,2]][keep],**cam); o=np.argsort(-D); cc=colours(H[keep])[o]
ax.scatter(X[o],Y[o],s=0.25,c=cc,marker='s',linewidths=0,rasterized=True)
tz=zground(tuv); tX,tY,_=project(np.c_[tuv,tz+0.05],**cam)
ax.scatter(tX,tY,s=3.5,c=COL['trunk'],edgecolors='white',linewidths=0.3,zorder=10)
pz=zground(puv)+0.1
for p,gp in tj.groupby('piece',sort=False):
    idx=gp.index.values; pX,pY,_=project(np.c_[puv[idx],pz[idx]],**cam)
    ax.plot(pX,pY,color='white',lw=2.2,zorder=11,solid_capstyle='round'); ax.plot(pX,pY,color=PIECE_COL[p],lw=1.1,zorder=12,solid_capstyle='round')
ax.set_xlim(X.min()-1,X.max()+1); ax.set_ylim(Y.min()-1,Y.max()+1)
scale_bar(ax,10,'10 m',xy=(0.0,0.04))
nr=F(np.array([[0,0],[*(RA.T@[0,1])]]));nv=nr[1]-nr[0]; nX,nY,_=project(np.array([[0,0,0],[nv[0],nv[1],0]]),**cam)
north_arrow(ax,nX[1]-nX[0],nY[1]-nY[0],xy=(0.95,0.72),size=0.07)
panel_label(ax,'(a)',x=0.0,y=0.9)
cax=ax.inset_axes([0.74,-0.035,0.24,0.03]); sm=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,H_TOP),cmap=GREENS)
cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.set_ticks([0,1,2,3]); cb.ax.tick_params(labelsize=6,length=2,width=0.4); cb.outline.set_linewidth(0.4)
cax.set_title('height above ground (m)',fontsize=6.5,pad=1.5)
ax.legend(handles=[Line2D([],[],color=PIECE_COL[k],lw=1.3,label=f'path, {LAB[k]}') for k in PIECE_COL]+
          [Line2D([],[],marker='o',ls='',ms=3,mfc=COL['trunk'],mec='white',mew=0.3,label='trunk')],
          loc='upper left',bbox_to_anchor=(0.0,-0.005),ncol=4,fontsize=6.5,handlelength=1.4,labelspacing=0.25,borderaxespad=0)
# (b) 3 m slice across the rows
ax2=fig.add_subplot(gs[1]); U0=np.median(tuv[:,0])+15
sl=np.abs(uv[:,0]-U0)<1.5; v=uv[sl,1]; z=P[sl,2]-zground(uv[sl]) ; h=H[sl]
o=np.argsort(uv[sl,0]); ax2.scatter(v[o],z[o],s=0.6,c=colours(h)[o],marker='s',linewidths=0,rasterized=True)
ts=np.abs(tuv[:,0]-U0)<1.5
ax2.scatter(tuv[ts,1],np.zeros(ts.sum())+0.05,s=10,c=COL['trunk'],edgecolors='white',linewidths=0.4,zorder=5,marker='o')
ps=np.abs(puv[:,0]-U0)<0.6
for p in PIECE_COL:
    q=ps&(tj.piece.values==p)
    if not q.any(): continue
    vv=np.sort(puv[q,1]); groups=np.split(vv,np.where(np.diff(vv)>2)[0]+1)
    for gv in groups: ax2.plot(np.median(gv),1.0,'v',ms=5,color=PIECE_COL[p],mec='k',mew=0.4,zorder=6)
ax2.set_xlim(uv[:,1].min(),uv[:,1].max()); ax2.set_ylim(-0.3,4.2); ax2.set_ylabel('height (m)'); ax2.set_xlabel('across the rows (m)')
ax2.set_aspect('equal'); panel_label(ax2,'(b)',x=0.0,y=1.02)
fig.savefig(OUTD+'ca10_orchard_3d.pdf',dpi=400); fig.savefig(OUTD+'ca10_orchard_3d.png',dpi=200)
print('row angle',np.degrees(th),'slice u',U0)
