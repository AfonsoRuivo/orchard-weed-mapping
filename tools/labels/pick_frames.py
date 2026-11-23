import csv, numpy as np, shutil, sys
from pathlib import Path
R=Path(sys.argv[1]); L=R/'data_set/campinas_b/lidar'; I=R/'data_set/campinas_b/images/bag_frames_20260627'
OUT=R/'data_set/weed_labelling_20261008'
traj=np.array([float(r['timestamp']) for r in csv.DictReader(open(R/'orchard-slam/data/campinas_b/lio_sam/trajectory_20260706.csv'))])
RADIUS,PCT,ZMIN,ZMAX=8.0,15,-1.1,-0.3
def zg(xyz,seed):
    d2=np.hypot(xyz[:,0],xyz[:,1]); x=xyz[d2<RADIUS]; g=x[x[:,2]<=np.percentile(x[:,2],20)]
    rng=np.random.default_rng(seed); best=(-1,None)
    for _ in range(500):
        p=g[rng.choice(len(g),3,replace=False)]; n=np.cross(p[1]-p[0],p[2]-p[0]); nn=np.linalg.norm(n)
        if nn<1e-9: continue
        n/=nn; d=-n@p[0]; k=(np.abs(g@n+d)<0.06).sum()
        if k>best[0]: best=(k,(n,d))
    n,d=best[1]
    if n[2]<0: n,d=-n,-d
    h=g@n+d; d-=np.percentile(h,PCT); return -d/n[2]
rows=[]
for r in csv.DictReader(open(L/'manifest.csv')):
    t=float(r['matched_ts']); ok=abs(float(r['dt']))<=0.15 and traj[0]+15<t<traj[-1]-15
    if ok:
        xyz=np.load(L/f"{r['stem']}.npz")['xyz']; z=[zg(xyz,s) for s in range(5)]
        ok=all(ZMIN+0.05<v<ZMAX-0.05 for v in z)
    rows.append((r['stem'],t,ok))
good=[i for i,(s,t,ok) in enumerate(rows) if ok]
print(f'{len(rows)} frames, {len(good)} pass sync/trajectory/plane checks (with margin)')
# C1: 30 evenly spaced in time over the passing frames
ts=np.array([rows[i][1] for i in good]); tgt=np.linspace(ts[0],ts[-1],30)
C1=sorted({good[int(np.argmin(np.abs(ts-x)))] for x in tgt})
# T: 25 frames, between C1 frames, at least 3 frames (~10 s) from any C1 frame
mids=[(C1[k]+C1[k+1])//2 for k in range(len(C1)-1)]
T=[]
for m in mids:
    cand=sorted([g for g in good if min(abs(g-c) for c in C1)>=3], key=lambda g:abs(g-m))
    if cand and cand[0] not in T: T.append(cand[0])
T=sorted(T)
for name,S in (('test_C1',C1),('tune_T',T)):
    d=OUT/name/'images'; d.mkdir(parents=True,exist_ok=True); (OUT/name/'labels').mkdir(exist_ok=True)
    (OUT/name/'labels'/'classes.txt').write_text('weed\ntrunk\nweed_cluster\n')
    with open(OUT/name/'frames.txt','w') as f:
        for i in S: f.write(rows[i][0]+'\n'); shutil.copy2(I/f'{rows[i][0]}.jpg',d)
    print(name,len(S),'frames:',', '.join(rows[i][0].split('_t')[1] for i in S))
print('min index gap T-C1:',min(abs(a-b) for a in T for b in C1))
