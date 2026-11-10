"""V08: my satellite logic (candidates q_B>=q_A, my exact-ish rho_pairs) applied to the analysis' stored sky 1,
q_cut = 5 and 7, to separate code differences from population/geometry differences."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.spatial import cKDTree
import indep_model as im
from v02_torus_mc import rho_pairs
from szsim import sampler
src = sys.argv[1] if len(sys.argv) > 1 else "theirs:1"
kind, sd = src.split(":"); sd = int(sd)
if kind == "theirs":
    cat = sampler.load_catalog(f"/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed{sd}.npz",
                               fields=["ra", "dec", "Y500", "theta500", "q", "q_obs"])
    s = cat["q"] > 2.0
    C = {k: cat[k][s].astype(float) for k in cat if k != "meta"}
    del cat
else:  # my own full-sky sampler, keep q>2
    ze, lme, dens = im.hmf_grid(); cc = im.cell_counts(ze, lme, dens).ravel()
    rng = np.random.default_rng(sd); ntot = rng.poisson(cc.sum() * 4 * np.pi / im.ARCMIN ** 2)
    cdf = np.cumsum(cc); cdf /= cdf[-1]; nm = len(lme) - 1
    parts = []
    for a0 in range(0, ntot, 3_000_000):
        n = min(3_000_000, ntot - a0)
        k = np.searchsorted(cdf, rng.random(n)); iz, imm = k // nm, k % nm
        z = ze[iz] + rng.random(n) * (ze[iz + 1] - ze[iz]); M = np.exp(lme[imm] + rng.random(n) * (lme[imm + 1] - lme[imm]))
        th = im.theta500(M, z); Y = im.Ymed(M, z) * np.exp(im.SIG_LNY * rng.standard_normal(n)); q = Y / im.sigmaY_fast(th)
        qo = q + rng.standard_normal(n); ra = 360 * rng.random(n); dec = np.degrees(np.arcsin(2 * rng.random(n) - 1))
        s = q > 2.0
        parts.append(np.column_stack([ra[s], dec[s], Y[s], th[s], q[s], qo[s]]))
    P = np.concatenate(parts)
    C = dict(ra=P[:, 0], dec=P[:, 1], Y500=P[:, 2], theta500=P[:, 3], q=P[:, 4], q_obs=P[:, 5])
ra, dec = np.radians(C["ra"]), np.radians(C["dec"])
v = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
tr = cKDTree(v)
sigA = im.sigmaY_fast(C["theta500"])
for qc in (5.0, 7.0):
    A = np.nonzero(C["q_obs"] > qc)[0]
    nsingle_all = None
    xm = np.zeros(len(A))
    R = 5 * C["theta500"][A] + 5 * C["theta500"].max() + 6
    PA, PB, PS = [], [], []
    for k, a in enumerate(A):
        nb = np.array(tr.query_ball_point(v[a], 2 * np.sin(0.5 * R[k] * im.ARCMIN)))
        nb = nb[(nb != a) & (C["q"][nb] >= C["q"][a])]
        if len(nb) == 0:
            continue
        sep = 2 * np.arcsin(np.clip(np.linalg.norm(v[nb] - v[a], axis=1) / 2, 0, 1)) / im.ARCMIN
        ok = sep < 5 * C["theta500"][a] + 5 * C["theta500"][nb] + 6
        PA.append(np.full(ok.sum(), k)); PB.append(nb[ok]); PS.append(sep[ok])
    PA = np.concatenate(PA); PB = np.concatenate(PB); PS = np.concatenate(PS)
    x = rho_pairs(C["theta500"][A][PA], C["theta500"][PB], PS) * C["Y500"][PB] / sigA[A][PA]
    np.maximum.at(xm, PA, x)
    sat = (xm >= C["q"][A]).sum()
    print(f"{src} q_cut={qc}: singles with q>2: {len(A)}; my-logic satellites {sat} ({sat/len(A):.5f}); pairs {len(PA)}", flush=True)
