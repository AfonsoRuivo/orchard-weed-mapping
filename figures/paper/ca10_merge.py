import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'ca10lib.py')).read())
from matplotlib.lines import Line2D
tr=trunks(); tj=traj(); g=gnss()
T1=np.array(json.load(open(CAB+'alignments/merge_T_final.json'))['T1_Bpre'])
B=pd.read_csv(CAB+'trajectories/cb11/liosam_B_own_frame_keyframes_raw.csv'); B['tb']=B.t-T0
Bxy=(np.c_[B[['x','y','z']].values,np.ones(len(B))]@T1.T)[:,:2]
gA=(g[['x_east','y_north']].values-TA)@RA        # ENU -> A frame
gok=~((g.tb>822.8)&(g.tb<838.8))
ang=[np.arctan(np.polyfit(q.x_rep1,q.y_rep1,1)[0]) for _,q in tr.groupby('row_id')]; th=float(np.mean(ang))
c,s=np.cos(-th),np.sin(-th); Rm=np.array([[c,-s],[s,c]]); c0=np.array([tr.x_rep1.mean(),tr.y_rep1.mean()])
F=lambda xy:(np.asarray(xy)-c0)@Rm.T
tuv=F(tr[['x_rep1','y_rep1']].values); puv=F(tj[['x','y']].values); guv=F(gA); buv=F(Bxy)
fig=plt.figure(figsize=(FIG_W,FIG_W*0.78))
gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],width_ratios=[1.25,1],hspace=0.32,wspace=0.22,left=0.07,right=0.99,top=0.96,bottom=0.09)
ax=fig.add_subplot(gs[0,:]); ax.set_aspect('equal')
ax.scatter(tuv[:,0],tuv[:,1],s=2.5,c=COL['trunk'],lw=0,zorder=2)
ax.plot(guv[gok,0],guv[gok,1],'.',ms=1.6,color=COL['gnss'],zorder=1,alpha=0.6)
for p,gp in tj.groupby('piece',sort=False): ax.plot(puv[gp.index,0],puv[gp.index,1],color=PIECE_COL[p],lw=1.1,zorder=3)
def mark(tb,txt,dx,dy):
    i=np.argmin(np.abs(g.tb.values-tb)); x,y=guv[i]
    ax.annotate(txt,xy=(x,y),xytext=(x+dx,y+dy),fontsize=6.5,ha='center',arrowprops=dict(arrowstyle='-',lw=0.5,color='k'),zorder=6,
                bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.9))
mark(822,'sensor gap',-1,3.0); mark(970,'U-turn',-5,-5.5)
ax.set_xlabel('along the rows (m)'); ax.set_ylabel('across (m)')
ax.legend(handles=[Line2D([],[],color=PIECE_COL[k],lw=1.3,label=LAB[k]) for k in PIECE_COL]+[Line2D([],[],marker='.',ls='',color=COL['gnss'],label='GNSS'),Line2D([],[],marker='o',ls='',ms=2.5,color=COL['trunk'],label='trunk')],
          ncol=5,loc='lower left',bbox_to_anchor=(0.04,1.0),fontsize=6.5,handlelength=1.4,borderaxespad=0.2)
panel_label(ax,'(a)',x=-0.06,y=1.02)
# (b) U-turn
ax2=fig.add_subplot(gs[1,0]); ax2.set_aspect('equal')
w=(g.tb>930)&(g.tb<1030); ax2.plot(guv[w,0],guv[w,1],'.',ms=2.4,color=COL['gnss'],label='GNSS',zorder=1)
wb=(B.tb>930)&(B.tb<1030); ax2.plot(buv[wb,0],buv[wb,1],'--',color=PIECE_COL['B-pre'],lw=1.0,label='segments 2 and 3 as one SLAM run',zorder=3)
for p in ('B-pre','B-post'):
    q=(tj.piece==p)&(tj.tb>930)&(tj.tb<1030); ax2.plot(puv[q,0],puv[q,1],color=PIECE_COL[p],lw=1.1,zorder=2,label=LAB[p])
ax2.scatter(tuv[:,0],tuv[:,1],s=4,c=COL['trunk'],lw=0,zorder=0)
xs=np.r_[guv[w,0],buv[wb,0]]; ys=np.r_[guv[w,1],buv[wb,1]]
ax2.set_xlim(xs.min()-6,xs.max()+2); ax2.set_ylim(ys.min()-2.5,ys.max()+2.5)
ax2.set_xlabel('along the rows (m)'); ax2.set_ylabel('across (m)')
ax2.legend(fontsize=6,loc='lower left',bbox_to_anchor=(0.0,1.0),ncol=2,handlelength=1.6,labelspacing=0.2,columnspacing=0.8,borderaxespad=0.2)
panel_label(ax2,'(b)',x=-0.12,y=1.3)
# (c) trunk shift curves
ax3=fig.add_subplot(gs[1,1])
for p,f in (('B-pre','Bpre'),('B-post','Bpost')):
    d=pd.read_csv(CAB+f'trunks/final_rep1/merge_check_shift_curve_{f}.csv'); ax3.plot(d.shift_along_m,d.n_within_0p3m,color=PIECE_COL[p],lw=1.0,label=LAB[p])
ax3.axvline(0,color='k',lw=0.5,ls=':'); 
for x in (-3.06,3.06): ax3.axvline(x,color=COL['grey'],lw=0.4,ls=':')
ax3.text(3.06,ax3.get_ylim()[1]*0.98,' ±1 tree',fontsize=6,color=COL['grey'],va='top')
ax3.set_xlabel('shift along the rows (m)'); ax3.set_ylabel('trunk pairs within 0.3 m'); ax3.legend(fontsize=6.5,loc='upper right',bbox_to_anchor=(1.0,0.88))
panel_label(ax3,'(c)',x=-0.18,y=1.02)
fig.savefig(OUTD+'ca10_merge.pdf',dpi=400); fig.savefig(OUTD+'ca10_merge.png',dpi=200)
