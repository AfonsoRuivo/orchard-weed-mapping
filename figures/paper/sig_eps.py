import sys; sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]
from thesis_style import *
import pandas as pd
from matplotlib.lines import Line2D
import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
T=S+'px/out/sig/'; OUT=S+'ov7/Images/Results/'
C={'yolo':COL['yolo'],'colour':COL['colour']}
B=pd.read_csv(T+'sigma_range_bins.csv')
fig,ax=plt.subplots(1,2,figsize=(FIG_W,FIG_W*0.36),gridspec_kw=dict(wspace=0.32))
for a,axn,ttl in ((ax[0],'along','along track'),(ax[1],'across','across track')):
    pr=B[(B.det=='colour')&(B.dn==10)]
    a.plot(pr.rbar,pr[f'pred_{axn}'],color='0.55',ls=':',lw=1.2)
    for det in ('yolo','colour'):
        for dn,ls in ((50,'-'),(10,'--')):
            g=B[(B.det==det)&(B.dn==dn)&(B.n>=15)]
            if len(g)==0: continue
            a.errorbar(g.rbar,g[f'rms_{axn}'],yerr=[(g[f'rms_{axn}']-g[f'rms_{axn}_lo']).fillna(0),(g[f'rms_{axn}_hi']-g[f'rms_{axn}']).fillna(0)],
                       color=C[det],ls=ls,marker='o',ms=2.8,lw=1.0,capsize=1.5,elinewidth=0.6)
    a.set_xlabel('distance from the robot (m)'); a.set_ylabel(r'$\sigma_{\mathrm{loc}}$ '+ttl+' (m)'); a.set_ylim(0,0.45); a.set_xlim(1,4.6)
    a.grid(lw=0.3,color='0.9')
for a,l in zip(ax,'ab'): a.text(-0.02,1.03,f'({l})',transform=a.transAxes,fontsize=8,fontweight='bold',va='bottom',ha='right')
fig.legend(handles=[Line2D([],[],color=C['yolo'],marker='o',ms=2.8,label='YOLO11n'),Line2D([],[],color=C['colour'],marker='o',ms=2.8,label='colour'),
    Line2D([],[],color='k',ls='-',label='frames 3.45 s apart'),Line2D([],[],color='k',ls='--',label='0.69 s apart'),Line2D([],[],color='0.55',ls=':',label='prediction')],
    loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=5,handlelength=1.8,columnspacing=1.2)
fig.subplots_adjust(left=0.08,right=0.99,top=0.80,bottom=0.17)
fig.savefig(OUT+'sigma_loc.pdf'); fig.savefig(S+'fig/sigma_loc.png',dpi=150)
D=pd.read_csv(T+'tradeoff_dn50.csv')
fig,a=plt.subplots(figsize=(FIG_W,FIG_W*0.38))
for det in ('yolo','colour'):
    t=D[(D.det==det)&(D.cls=='all')]
    a.plot(t.eps,t.link.clip(upper=1),color=C[det],lw=1.4); a.plot(t.eps,t.over,color=C[det],lw=1.4,ls=':')
a.axvline(0.4,color='k',lw=0.6,ls='--'); a.text(0.41,0.02,r'$\varepsilon$ = 0.4 m',fontsize=7,va='bottom')
a.set_xlim(0.1,1.0); a.set_ylim(0,1.0); a.set_xlabel(r'merge radius $\varepsilon$ (m)'); a.set_ylabel('share'); a.grid(lw=0.3,color='0.9')
a.legend(handles=[Line2D([],[],color=C['yolo'],lw=1.4,label='YOLO11n'),Line2D([],[],color=C['colour'],lw=1.4,label='colour'),
    Line2D([],[],color='k',lw=1.4,label='L: re-sightings linked'),Line2D([],[],color='k',lw=1.4,ls=':',label='O: same-frame detections merged')],
    loc='upper center',bbox_to_anchor=(0.5,1.17),ncol=4,handlelength=1.8,columnspacing=1.2)
fig.subplots_adjust(left=0.08,right=0.99,top=0.86,bottom=0.17)
fig.savefig(OUT+'eps_sweep.pdf'); fig.savefig(S+'fig/eps_sweep.png',dpi=150)
