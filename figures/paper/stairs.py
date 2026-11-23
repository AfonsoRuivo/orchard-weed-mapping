import os as _os
# Root of the unpacked Zenodo data; see docs/data.md and config/paths.example.py
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
import sys; sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]
from thesis_style import *
OUT=S + ''
n=[491,447,260,222,174]; plants=[None,None,None,122,91]; patch=[None,None,None,100,83]
PC='#A0522D'
fig,a=plt.subplots(figsize=(FIG_W,FIG_W*0.40))
x=np.arange(len(n)); xs=np.repeat(np.arange(len(n)+1),2)[1:-1]-0.5; ys=np.repeat(n,2)
a.fill_between(xs[:6],ys[:6],0,color=COL['yolo'],alpha=0.10,lw=0)
for i in (3,4):
    a.fill_between([i-0.5,i+0.5],[plants[i]]*2,0,color=COL['yolo'],alpha=0.10,lw=0)
    a.fill_between([i-0.5,i+0.5],[n[i]]*2,[plants[i]]*2,color=PC,alpha=0.22,lw=0)
    a.plot([i-0.5,i+0.5],[plants[i]]*2,color=PC,lw=0.8,ls=(0,(3,2)))
    a.text(i,plants[i]/2,f'{plants[i]}\nplants',ha='center',va='center',fontsize=6.8,color=COL['yolo'])
    a.text(i,(plants[i]+n[i])/2,f'{patch[i]}\npatches',ha='center',va='center',fontsize=6.8,color=PC)
a.plot(xs,ys,color=COL['yolo'],lw=1.6)
for i,v in enumerate(n): a.text(i,v+12,f'{v}',ha='center',va='bottom',fontsize=7.5,fontweight='bold',color=COL['yolo'])
for i in range(len(n)-1):
    xm=i+0.5; a.annotate('',xy=(xm+0.1,n[i+1]+3),xytext=(xm+0.1,n[i]-3),arrowprops=dict(arrowstyle='-|>',color='#C0392B',lw=0.8,shrinkA=1,shrinkB=1))
    a.text(xm-0.05,(n[i]+n[i+1])/2,f'−{n[i]-n[i+1]}',fontsize=7,color='#C0392B',va='center',ha='right')
a.set_xticks(x); a.set_xticklabels(['YOLO\nboxes','above the\nheight gate','ray meets\nthe ground','within\n5 m','weed\nentities'])
a.set_xlim(-0.5,len(n)-0.5); a.set_ylim(0,560); a.set_ylabel('count')
a.grid(axis='y',lw=0.3,color='0.9'); a.tick_params(axis='x',length=0)
fig.subplots_adjust(left=0.09,right=0.99,top=0.97,bottom=0.18)
fig.savefig(OUT+'ov7/Images/Results/yolo_funnel.pdf'); fig.savefig(OUT+'fig/yolo_funnel.png',dpi=150)
