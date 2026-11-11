"""Poisson realisations of the blended (catalogue P(D)) data vector, SO-like 40% sky, q_cut=5, fitted with the naive
model: does the mean recovered shift equal the Asimov bias, and does the scatter equal the Fisher sigma?
Also realisations of the naive expectation itself (no blending) as the null.  Configs a, b, c (as c05; config d is checked with emcee in c06).
Output: data_cosmo/poisson_real.npz
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL
from c05_fits import CONFIGS

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 150
t0 = time.time()
model = CL.NaiveModel(5.0); fsky = 0.4
N0 = model.counts(CL.FID); Dc = np.load(f"{CL.DC}/truth_catalogue.npz")["D_q5"]
rng = np.random.default_rng(11)
out = {}
for cn in ("a", "b", "c"):
    free, pri = CONFIGS[cn]; fi = [CL.PNAMES.index(p) for p in free]
    Ft = CL.Fitter(model, N0 * fsky, free, fsky=fsky, priors=pri).fisher(CL.FID[fi])
    sig, _ = CL.derived_sigma(Ft, fi)
    for dn, dv in (("self", N0), ("cat", N0 + Dc)):
        xa, _, _ = CL.Fitter(model, dv * fsky, free, fsky=fsky, priors=pri).fit()
        res = []
        for r in range(NR):
            dr = rng.poisson(dv * fsky).astype(float)
            # in config d the (1-b) prior mean is also drawn (an external WL measurement scatters too)
            pr = {k: (rng.normal(m_, s_), s_) for k, (m_, s_) in pri.items()}
            x, L, it = CL.Fitter(model, dr, free, fsky=fsky, priors=pr).fit()
            res.append(x)
        res = np.array(res)
        S8 = CL.S8_of(res[:, 0], res[:, 1]); S8t = CL.S8_of(*CL.FID[:2]); S8a = CL.S8_of(xa[0], xa[1])
        out[f"{cn}_{dn}"] = res
        print(f"config {cn} data {dn} ({NR} realisations, {time.time()-t0:.0f}s):")
        for k, p in enumerate(free + ["S8"]):
            v = res[:, k] if p != "S8" else S8
            tr = CL.FID[CL.PNAMES.index(p)] if p != "S8" else S8t
            asv = xa[k] if p != "S8" else S8a
            print(f"   {p:5s}: mean shift {(v.mean()-tr)/sig[p]:+.2f} +- {v.std(ddof=1)/sig[p]/np.sqrt(NR):.2f} sigma_F "
                  f"(Asimov {(asv-tr)/sig[p]:+.2f}); scatter / Fisher sigma = {v.std(ddof=1)/sig[p]:.3f} "
                  f"(+- {1/np.sqrt(2*(NR-1)):.3f}); median shift {(np.median(v)-tr)/sig[p]:+.2f}")
np.savez(f"{CL.DC}/poisson_real.npz", **out)
print(f"total {time.time()-t0:.0f}s")
