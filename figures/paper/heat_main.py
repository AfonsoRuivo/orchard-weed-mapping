import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'heatlib.py')).read())
OV=S+'px/out/patches/'
Hs={d:heat(pd.read_csv(OV+f'priority_{d}.csv')) for d in ('yolo','colour')}; vmax=max(np.percentile(H[H>0.05],99) for H in Hs.values())
fig=plt.figure(figsize=(FIG_W,FIG_W*0.55))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.05],wspace=0.05); axs=[fig.add_subplot(gs[i]) for i in (0,1)]; cax=fig.add_subplot(gs[2])
for a,d,lab in zip(axs,('yolo','colour'),('(a) YOLO11n','(b) colour')):
    base(a); overlay(a,Hs[d],vmax); a.text(0,1.01,lab,transform=a.transAxes,fontsize=8,fontweight='bold',va='bottom')
axs[0].plot([-80,-60],[-102]*2,color='k',lw=1.5); axs[0].text(-70,-100.8,'20 m',ha='center',va='bottom',fontsize=7)
axs[1].annotate('',xy=(3,-4),xytext=(3,-12),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='k')); axs[1].text(3,-13,'N',ha='center',va='top',fontsize=7)
sm=plt.cm.ScalarMappable(cmap=CM_H,norm=plt.Normalize(0,1)); cb=fig.colorbar(sm,cax=cax); cb.set_label('priority density',fontsize=7); cb.set_ticks([0,1]); cb.set_ticklabels(['low','high']); cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
fig.subplots_adjust(left=0.01,right=0.93,top=0.95,bottom=0.02)
fig.savefig(S+'ov7/Images/Results/priority_heat.pdf',dpi=300); fig.savefig(S+'fig/priority_heat.png',dpi=130)
