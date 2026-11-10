import numpy as np
from scipy import stats
ARC = np.pi/180/60
import sys
r = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else 'v2_mc_szsim_results_fixed.npz'))
NB = dict(det5=85913.4, M3e13=4598099.4, M1e14=396387.8, det4=132126.5)
NS = r['Ntot_det5'].shape[0]; print("skies:", NS)
for k in ['det5', 'det4', 'M3e13', 'M1e14']:
    v = r[f'Ntot_{k}'][:, 0].astype(float)
    print(f"N {k}: MC {v.mean():.1f}+-{v.std(ddof=1)/np.sqrt(NS):.1f} analytic {NB[k]:.1f} z={(v.mean()-NB[k])/(v.std(ddof=1)/np.sqrt(NS)):+.2f} var/mean={v.var(ddof=1)/v.mean():.2f}")
mu = lambda N, t: N*np.sin(t*ARC/2)**2
def pois_vec(m, nmax=6):
    p = stats.poisson.pmf(np.arange(nmax), m); return np.append(p, stats.poisson.sf(nmax-1, m))
allz = []
for key in sorted(k for k in r if k.startswith('rand|') or k.startswith('palm|')):
    parts = key.split('|'); pop = parts[-2]; t = float(parts[-1])
    a = r[key]  # (NS, 7)
    ncen = 100000 if parts[0] == 'rand' else r['ncen|' + '|'.join(parts[1:])][:, 0]
    ncen_tot = ncen*NS if np.isscalar(ncen) else ncen.sum()
    m = mu(NB[pop], t); pa = pois_vec(m)
    mean = a.mean(0); es = a.std(0, ddof=1)/np.sqrt(NS); eb = np.sqrt(pa*(1-pa)/ncen_tot)
    e = np.maximum(es, eb)
    ok = pa*ncen_tot > 20
    z = (mean-pa)/e
    allz += list(z[ok])
    chi2 = (z[ok]**2).sum(); dof = ok.sum()
    msg = " ".join(f"P{n}:{mean[n]:.4g}/{pa[n]:.4g}({z[n]:+.1f})" for n in range(4) if ok[n])
    print(f"{key:28s} mu={m:.4g} chi2={chi2:.1f}/{dof} p={stats.chi2.sf(chi2, dof):.2f} | {msg}")
allz = np.array(allz)
print(f"ALL cells: n={len(allz)} rms z={np.sqrt((allz**2).mean()):.3f} mean={allz.mean():+.3f} frac|z|>2={np.mean(np.abs(allz)>2):.3f} max|z|={np.abs(allz).max():.2f}")
# P(>=1), P(>=2) for det5 centred on det5 (table (a)) and M1e14, M3e13 (b),(c)
print("\nReady table check (q>5):")
for pop, lab in [('det5', 'a'), ('M1e14', 'b'), ('M3e13', 'c')]:
    for t in [1., 2., 5., 10.]:
        a = r[f'palm|det5|{pop}|{t}']; n = r[f'ncen|det5|{pop}|{t}'][:, 0].sum()
        for kmin in [1, 2]:
            v = a[:, kmin:].sum(1); m = mu(NB[pop], t); pa = stats.poisson.sf(kmin-1, m)
            e = max(v.std(ddof=1)/np.sqrt(NS), np.sqrt(pa*(1-pa)/n))
            print(f" ({lab}) {t}' P>={kmin}: analytic {pa:.4e} MC {v.mean():.4e}+-{e:.1e} z={(v.mean()-pa)/e:+.2f}")
# pairs / triples / geq n
print("\nSurvey-wide det5:")
p3 = 1-3*np.sqrt(3)/(4*np.pi); N = NB['det5']
for t in [0.5, 1., 2., 5., 10.]:
    m = mu(N, t); v = r[f'pairs|{t}'][:, 0].astype(float); E = N*m/2; V = E*(1+2*m)
    print(f" pairs<{t}': E={E:.1f} MC={v.mean():.1f}+-{np.sqrt(V/NS):.1f} z={(v.mean()-E)/np.sqrt(V/NS):+.2f} Var_MC/Var_th={v.var(ddof=1)/V:.2f}+-{np.sqrt(2/(NS-1)):.2f}")
for t in [2., 5., 10.]:
    m = mu(N, t); v = r[f'triples|{t}'][:, 0].astype(float); E = N*m*m*p3/6
    e = max(v.std(ddof=1), np.sqrt(E))/np.sqrt(NS)
    print(f" triples<{t}': E={E:.3f} MC={v.mean():.3f}+-{e:.3f} z={(v.mean()-E)/e:+.2f}")
    g = r[f'geqn|{t}'].astype(float)
    for n in range(1, 6):
        Eg = N*stats.poisson.sf(n-1, m); vg = g[:, n-1]
        if Eg < 0.5: continue
        eg = max(vg.std(ddof=1), np.sqrt(Eg))/np.sqrt(NS)
        print(f"   #det5 with >={n} nbrs <{t}': E={Eg:.2f} MC={vg.mean():.2f}+-{eg:.2f} z={(vg.mean()-Eg)/eg:+.2f}")
