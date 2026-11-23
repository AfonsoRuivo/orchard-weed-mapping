import sys, json
import os as _os
S = _os.environ.get("ORCHARD_DATA", "./data").rstrip("/") + "/"
sys.path[:0] = [_os.path.join(_os.path.dirname(__file__), '..', 'lib'),
                _os.path.join(_os.path.dirname(__file__), '..', '..', 'src', 'orchardmap')]; from thesis_style import *
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def load_img(path, gfile):
    G=json.load(open(path+'georeference.json'))['enu_png']
    im=np.asarray(Image.open(path+G['file']).convert('RGB'))/255.
    im=0.55*im+0.45*im.mean(2,keepdims=True)          # desaturate
    im=np.clip(0.80*im+0.17,0,1)                      # lift, so dark overlays read
    e0,e1,n0,n1=G['extent_east_north']
    return im,(e0,e1,n0,n1)

# ---- Campinas B (CB-11) ----
B=S+'vb/'
imgB,extB=load_img(B+'satellite/','')
TB=pd.read_csv(B+'trunks/trunks_final_rep1_enu.csv'); TB=TB[TB.kept_by_row_prior==1]
AL=json.load(open(B+'alignments/align_rep1.json')); Rm=np.array(AL['R']); tv=np.array(AL['t'])
trB=pd.read_csv(B+'trajectories/cb11/liosam_rep1_keyframes_raw.csv')
PB=trB[['x','y']].values@Rm.T+tv
RL=pd.read_csv(B+'trunks/row_lines.csv')

# ---- Campinas A (CA-10) ----
CAB=S+'ca10run/ca10_run/viz_bundle_ca10/'
SATA=S+'ca10run2/ca10_run/satellite/'
imgA,extA=load_img(SATA,'')
ALA=json.load(open(CAB+'alignments/align_rep1.json')); RA=np.array(ALA['R']); TA=np.array(ALA['t'])
TA_=pd.read_csv(CAB+'trunks/trunks_final_rep1_enu.csv'); TA_=TA_[TA_.kept_by_row_prior==1]
TA_=TA_[TA_.groupby('row_id').row_id.transform('size')>=3]
T0=1782577258.896306
tjA=pd.read_csv(CAB+'trajectories/cb11/liosam_rep1_keyframes_raw.csv'); tjA['tb']=tjA.t-T0
tjA['piece']=np.where(tjA.tb<810,'A',np.where(tjA.tb<970,'B-pre','B-post'))
PA=tjA[['x','y']].values@RA.T+TA
PIECE={'A':'#D4A62A','B-pre':'#0072B2','B-post':'#CC79A7'}
print('CB-11 trunks',len(TB),'rows',TB.row_id.nunique(),'| CA-10 trunks',len(TA_),'rows',TA_.row_id.nunique())

def panel(ax,img,ext,xlim,ylim,trunks,path=None,pieces=None,ts=2.0,lw=0.9):
    ax.imshow(img,extent=[ext[0],ext[1],ext[2],ext[3]],origin='upper',interpolation='bilinear',rasterized=True,zorder=0)
    if pieces is not None:
        for p,gp in pieces[0].groupby('piece',sort=False):
            ax.plot(pieces[1][gp.index,0],pieces[1][gp.index,1],color=PIECE[p],lw=lw,zorder=2)
    elif path is not None:
        ax.plot(path[:,0],path[:,1],color=COL['path'],lw=lw,zorder=2)
    ax.scatter(trunks[:,0],trunks[:,1],s=ts,color=COL['trunk'],lw=0.25,edgecolor='white',zorder=3)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color('0.35')

def scalebar(ax,x,y,L,txt,c='white'):
    ax.plot([x,x+L],[y,y],color=c,lw=1.8,solid_capstyle='butt',zorder=12)
    ax.text(x+L/2,y+1.2,txt,ha='center',va='bottom',fontsize=6.5,color=c,zorder=12)
def north(ax,x,y,L=6,c='white'):
    ax.annotate('',xy=(x,y+L),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',lw=0.8,color=c),zorder=12)
    ax.text(x,y+L+0.6,'N',ha='center',va='bottom',fontsize=6.5,color=c,zorder=12)

fig=plt.figure(figsize=(FIG_W,FIG_W*0.495))
gs=fig.add_gridspec(1,4,width_ratios=[0.98,0.50,0.76,0.76],wspace=0.05,left=0.005,right=0.995,top=0.905,bottom=0.115)
tB=TB[['east_m','north_m']].values; tA=TA_[['east_m','north_m']].values

ax=fig.add_subplot(gs[0,0]); panel(ax,imgB,extB,(-84,6),(-104,2),tB,path=PB,ts=1.8)
zb=(-30,-5,-52,-27)
ax.add_patch(Rectangle((zb[0],zb[2]),zb[1]-zb[0],zb[3]-zb[2],fill=False,ec='white',lw=0.8,zorder=11))
scalebar(ax,-80,-100,20,'20 m'); north(ax,2,-100)
ax.set_title('(a) Campinas B, 325 trunks on 13 rows',fontsize=7,loc='left',pad=2)

ax=fig.add_subplot(gs[0,1]); panel(ax,imgA,extA,(-24,24),(-5,92),tA,pieces=(tjA,PA),ts=1.8)
scalebar(ax,-21,-1,20,'20 m'); north(ax,20,-1)
ax.set_title('(b) Campinas A, 168 trunks on 7 rows',fontsize=7,loc='left',pad=2)

ax=fig.add_subplot(gs[0,2]); panel(ax,imgB,extB,(zb[0],zb[1]),(zb[2],zb[3]),tB,path=PB,ts=7,lw=1.1)
scalebar(ax,-28.5,-50.5,10,'10 m')
ax.set_title('(c) detail of (a)',fontsize=7,loc='left',pad=2)

ax=fig.add_subplot(gs[0,3]); panel(ax,imgA,extA,(-14,11),(28,53),tA,pieces=(tjA,PA),ts=7,lw=1.1)
scalebar(ax,-13,29.5,10,'10 m')
ax.set_title('(d) detail of (b)',fontsize=7,loc='left',pad=2)

h=[Line2D([],[],marker='o',ls='',ms=3,mfc=COL['trunk'],mec='white',mew=0.3,label='detected trunk'),
   Line2D([],[],color=COL['path'],lw=1.3,label='path (Campinas A: segment 1)'),
   Line2D([],[],color=PIECE['B-pre'],lw=1.3,label='Campinas A, segment 2'),
   Line2D([],[],color=PIECE['B-post'],lw=1.3,label='segment 3')]
fig.legend(handles=h,loc='lower center',ncol=4,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,-0.012),columnspacing=1.1,handletextpad=0.35,handlelength=1.5)
fig.savefig(S+'ras/figures/trunk_map_satellite.pdf',dpi=400)
fig.savefig(S+'fig/sat_maps.png',dpi=150)
print('written')
