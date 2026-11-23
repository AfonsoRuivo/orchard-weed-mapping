import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
exec(open(_os.path.join(_os.path.dirname(__file__), 'heatlib.py')).read())
from matplotlib.lines import Line2D
D=pd.read_csv(S+'sw/colour_sweep/sweep_scores.csv'); t=D[D.set.str.contains('test')]
V=[(14,300,'before tuning','orig'),(36,800,None,'a800'),(36,1400,None,None),(36,2000,'tuned, used in the map','patches')]
fig=plt.figure(figsize=(FIG_W,FIG_W*0.80))
gs=fig.add_gridspec(2,4,hspace=0.22,wspace=0.05,left=0.01,right=0.99,top=0.94,bottom=0.02)
L=iter('abcdefgh')
for k,(h,am,lab,var) in enumerate(V):
    a=fig.add_subplot(gs[0,k]); base(a)
    w=pd.read_csv(S+f'sw/colour_sweep/weed_map_colour_h{h}_a{am}.csv'); r=t[(t.h_lo==h)&(t.a_min==am)].iloc[0]
    p=w[w.patch==0]; q=w[w.patch==1]
    a.scatter(p.x_enu,p.y_enu,s=1.8,color=COL['colour'],lw=0,alpha=0.85,rasterized=True,zorder=3)
    a.scatter(q.x_enu,q.y_enu,s=8,marker='s',color=COL['colour'],edgecolor='#065F5F',lw=0.4,rasterized=True,zorder=4)
    a.set_title(f'({next(L)}) $H\\geq${2*h}°, $A_{{\\min}}$ {am} px'+(f'\n{lab}' if lab else '\n '),fontsize=6.8,loc='left',pad=2)
    a.text(0.5,-0.025,f'{len(w)} points, P {r.P:.2f}, R {r.R:.2f}',transform=a.transAxes,ha='center',va='top',fontsize=6.5)
Hs={v[3]:heat(pd.read_csv(S+f'px/out/{v[3]}/priority_colour.csv')) for v in V if v[3]}
vmax=max(np.percentile(H[H>0.05],99) for H in Hs.values())
for k,(h,am,lab,var) in enumerate([v for v in V if v[3]]):
    a=fig.add_subplot(gs[1,k]); base(a); overlay(a,Hs[var],vmax,0.8)
    a.set_title(f'({next(L)}) priority, $A_{{\\min}}$ {am} px'+(f'\n{lab}' if lab else '\n '),fontsize=6.8,loc='left',pad=2)
a=fig.add_subplot(gs[1,3]); pos=a.get_position(); a.set_position([pos.x0+0.045,pos.y0+0.05,pos.width-0.05,pos.height-0.09])
mk={300:'o',800:'s',1400:'D',2000:'^'}; cols={14:'#c6dbef',22:'#6baed6',30:'#2171b5',36:'#08306b'}
for f in (0.1,0.2,0.3,0.4):
    rr=np.linspace(0.3,1,200); pp=f*rr/(2*rr-f); m=(pp>0)&(pp<=1); a.plot(rr[m],pp[m],color='0.9',lw=0.5,zorder=0)
for _,r in t.iterrows(): a.scatter(r.R,r.P,marker=mk[r.a_min],s=16,color=cols[r.h_lo],edgecolor='white',lw=0.4,zorder=3)
a.set_xlim(0.48,0.78); a.set_ylim(0,0.3); a.set_xlabel('recall',fontsize=6.5,labelpad=1); a.set_ylabel('precision',fontsize=6.5,labelpad=1)
a.tick_params(labelsize=5.8,length=2); a.set_xticks([0.5,0.6,0.7]); a.set_yticks([0,0.1,0.2,0.3])
a.set_title(f'({next(L)}) test set, 16 settings',fontsize=6.8,loc='left',pad=2)
hl=[Line2D([],[],marker=mk[k],ls='',color='0.45',ms=3.5,label=f'{k}') for k in mk]+[Line2D([],[],marker='o',ls='',color=cols[k],ms=3.5,label=f'{2*k}°') for k in cols]
a.legend(handles=hl,ncol=2,fontsize=5.3,loc='upper right',frameon=False,handletextpad=0.1,columnspacing=0.5,borderaxespad=0.2,title='$A_{\\min}$    $H$',title_fontsize=5.5)
fig.savefig(S+'ov7/Images/Results/colour_sweep.pdf',dpi=300); fig.savefig(S+'fig/colour_sweep.png',dpi=130)
