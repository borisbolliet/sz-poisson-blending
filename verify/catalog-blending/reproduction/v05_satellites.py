"""V05: high-statistics independent check of the satellite fraction (single detection with a neighbour B
whose contribution to A's filter x_B >= q_A).  Only halos with q > 0.5 are needed (x_B <= q_B by
Cauchy-Schwarz).  Periodic 10x10 deg patches; exact Hankel rho for candidate pairs."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import indep_model as im
from v02_torus_mc import rho_pairs, pdist, L

QC = [4.0, 5.0, 6.0, 7.0]
grid = im.hmf_grid()
ze, lme, dens = grid
cc = im.cell_counts(*grid)
# pre-select cells: only halos that can reach q>0.5 matter (q<=Ymed e^{4 sigma}/sigma); sample all, cheap enough
n_p = int(sys.argv[1]); s0 = int(sys.argv[2])
acc = {qc: [] for qc in QC}
t0 = time.time()
for p in range(n_p):
    rng = np.random.default_rng(s0 + p)
    z, M, Y, th = im.sample_halos(rng, L * L, *grid)
    q = Y / im.sigmaY_fast(th)
    s = q > 0.5
    z, M, Y, th, q = z[s], M[s], Y[s], th[s], q[s]
    X = rng.random(len(q)) * L; Yp = rng.random(len(q)) * L
    n = rng.standard_normal(len(q))
    o = np.argsort(-q)
    PA, PB, PS = [], [], []
    for r, a in enumerate(o[1:], 1):
        bs = o[:r]
        d = pdist(X[a], Yp[a], X[bs], Yp[bs]); ok = d < 5 * th[a] + 5 * th[bs] + 6.0
        if ok.any():
            PA.append(np.full(ok.sum(), a)); PB.append(bs[ok]); PS.append(d[ok])
    PA = np.concatenate(PA); PB = np.concatenate(PB); PS = np.concatenate(PS)
    x = rho_pairs(th[PA], th[PB], PS) * Y[PB] / im.sigmaY_fast(th[PA])
    xm13 = np.zeros(len(q)); np.maximum.at(xm13, PA, x)
    m14 = M[PB] >= 1e14
    xm14 = np.zeros(len(q)); np.maximum.at(xm14, PA[m14], x[m14])
    for qc in QC:
        single = q + n > qc
        acc[qc].append([single.sum(), (single & (xm13 >= q)).sum(), (single & (xm14 >= q)).sum()])
    if p % 20 == 0:
        print(f"patch {p} {time.time()-t0:.0f}s", flush=True)
an13 = {4: 0.00949, 5: 0.00595, 6: 0.00436, 7: 0.00341}; an14 = {4: 0.00923, 5: 0.00590, 6: 0.00434, 7: 0.00340}
for qc in QC:
    a = np.array(acc[qc], float)
    def jk(i):
        r = a[:, i].sum() / a[:, 0].sum()
        rj = (a[:, i].sum() - a[:, i]) / (a[:, 0].sum() - a[:, 0]); k = len(a)
        return r, np.sqrt((k - 1) / k * ((rj - rj.mean()) ** 2).sum())
    r13, e13 = jk(1); r14, e14 = jk(2)
    print(f"q_cut={qc}: N_single={a[:,0].sum():.0f} ({a[:,0].sum()/(len(a)*100):.3f}/deg2); sat(floor1e13) {r13:.5f}±{e13:.5f} vs claimed {an13[qc]} ({(r13-an13[qc])/e13:+.1f}σ); "
          f"sat(floor1e14) {r14:.5f}±{e14:.5f} vs {an14[qc]} ({(r14-an14[qc])/e14:+.1f}σ)")
