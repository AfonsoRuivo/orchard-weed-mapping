#!/usr/bin/env python3
"""Section 5: baseline priority + stability test.  Variants: tau 0.5 / 2, each weight doubled.
Metrics vs baseline: Jaccard of the top-10 % entities, Spearman rho over the full ranking, zone changes.
Noise floor: baseline recomputed after perturbing entity positions (N(0, sigma) per axis) and areas (lognormal),
compared with the unperturbed baseline.  Random-set null for the Jaccard.
-> tables/priority_<det>.csv (entities with factors + priority), priority_stability.csv, priority_zones_<det>.csv"""
import json, sys
from common import *
from priority_lib import *
RUN = sys.argv[1] if len(sys.argv) > 1 else 'frozen'; SFX = '' if RUN == 'frozen' else '_' + RUN
SIG_POS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25     # m per axis, ~ sigma_loc (see s2)
SIG_AREA = 0.3                                                   # lognormal sigma of area (CV of area over repeated views, s2)
t = load_trunks(); T = t[['xe', 'ye']].values
VARS = {'baseline': {}, 'tau=0.5': dict(tau=0.5), 'tau=2': dict(tau=2.0), 'w_dens x2': dict(w_dens=2.0),
        'w_prox x2': dict(w_prox=2.0), 'w_size x2': dict(w_size=2.0)}
res, rng = [], np.random.default_rng(0)
for det in ('yolo', 'colour'):
    if RUN == 'frozen': e = pd.read_csv(TAB / f'entities_{det}.csv')
    else:
        from entities import entities; from sigma_lib import load_weeds_run; e = entities(load_weeds_run(det, RUN), 0.4)[0]
    xy, size = e[['x', 'y']].values, e.area_med.values; n = len(e)
    base = priority(xy, size, T); bZ, hot = zones(xy, base['pri'])
    out = e.copy()
    for k in ('pri', 'd_trunk', 'prox', 'dens', 'dens_n', 'size_n'): out[k if k != 'pri' else 'priority'] = base[k]
    out['rank'] = (-base['pri']).argsort().argsort(); out['in_zone'] = 0
    for zi, m in enumerate(bZ, 1): out.loc[m, 'in_zone'] = zi
    out['top10'] = out['rank'] < len(out) * 0.1; out.to_csv(TAB / f'priority_{det}{SFX}.csv', index=False)
    zrows = [dict(zone=zi, n=len(m), cx=xy[m, 0].mean(), cy=xy[m, 1].mean(), priority_sum=base['pri'][m].sum(),
                  priority_mean=base['pri'][m].mean(), area_m2=size[m].sum()) for zi, m in enumerate(bZ, 1)]
    pd.DataFrame(zrows).to_csv(TAB / f'priority_zones_{det}{SFX}.csv', index=False)
    fac = {k: float(spearmanr(base['pri'], base[k])[0]) for k in ('prox', 'dens_n', 'size_n')}
    nl = random_null_jaccard(n)
    print(det, 'N', n, 'zones', len(bZ), 'zone members', [len(m) for m in bZ], 'hot', len(hot), 'rho(priority, factor)', fac, 'null J', nl)
    for name, kw in VARS.items():
        if name == 'baseline': continue
        v = priority(xy, size, T, **kw); vZ, _ = zones(xy, v['pri'], **{k: kw[k] for k in kw if k.startswith('zone')})
        res.append(dict(det=det, n=n, variant=name, **compare(base['pri'], v['pri'], bZ, vZ), n_zones_base=len(bZ), null_J_mean=nl[0], null_J_p95=nl[1]))
    # noise floor: perturbed baseline vs baseline
    P = []
    for _ in range(300):
        xy2 = xy + rng.normal(0, SIG_POS, xy.shape); s2 = size * np.exp(rng.normal(0, SIG_AREA, n))
        v = priority(xy2, s2, T); vZ, _ = zones(xy2, v['pri']); P.append(compare(base['pri'], v['pri'], bZ, vZ))
    P = pd.DataFrame(P)
    res.append(dict(det=det, n=n, variant=f'noise floor (pos {SIG_POS} m, area {SIG_AREA})', jaccard_top10=P.jaccard_top10.mean(),
                    spearman=P.spearman.mean(), n_zones=P.n_zones.mean(), zones_matched=P.zones_matched.mean(),
                    zone_members_jaccard=P.zone_members_jaccard.mean(), n_zones_base=len(bZ), null_J_mean=nl[0], null_J_p95=nl[1],
                    jaccard_sd=P.jaccard_top10.std(), spearman_sd=P.spearman.std(), jaccard_p05=P.jaccard_top10.quantile(.05),
                    spearman_p05=P.spearman.quantile(.05)))
R = pd.DataFrame(res); R.to_csv(TAB / f'priority_stability{SFX}.csv', index=False); print(R.round(3).to_string())
