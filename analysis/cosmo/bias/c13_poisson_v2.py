"""[REPAIR v2 of c07] Poisson realisations with the converged model (cosmolib2): 40% sky, q_cut=5.
For configs a, b, d: realisations of the naive expectation (null) and of the blended catalogue expectation (v2), each
fitted by ML with the naive model.  Tests: scatter / Fisher sigma (Fisher validity), mean shift vs Asimov bias.
Config d: the (1-b) prior mean is redrawn per realisation (an external WL calibration scatters too); fits are
Fisher scoring + bounded L-BFGS-B polish.  Also: self-data fits started 2 sigma away from the truth (non-trivial sanity).
Output: data_cosmo/poisson_real_v2.npz
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL, cosmolib2 as C2
from c05_fits import CONFIGS

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CFGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["a", "b", "d"]
t0 = time.time()
model = C2.SmoothModel(5.0); fsky = 0.4
N0 = model.counts(CL.FID); Dc = np.load(f"{CL.DC}/truth_catalogue_v2.npz")["D_q5"]
rng = np.random.default_rng(11)
out = {}
for cn in CFGS:
    free, pri = CONFIGS[cn]; fi = [CL.PNAMES.index(p) for p in free]
    Ft = C2.Fitter2(model, N0 * fsky, free, fsky=fsky, priors=pri).fisher(CL.FID[fi])
    sig, Cov = CL.derived_sigma(Ft, fi)
    # non-trivial self-fit: start 2 sigma off in every free parameter (alternating sign)
    x0 = CL.FID[fi] + 2 * np.sqrt(np.diag(Cov)) * np.array([1, -1, 1, -1, 1][:len(fi)])
    x0 = np.clip(x0, [CL.BOUNDS[p][0] for p in free], [CL.BOUNDS[p][1] for p in free])
    xs, _, it = C2.Fitter2(model, N0 * fsky, free, fsky=fsky, priors=pri).fit(x0=x0, tol=1e-4, polish=True)
    print(f"config {cn}: self-fit from 2-sigma offset start -> (x-truth)/sigma = "
          f"{np.round((xs - CL.FID[fi]) / np.sqrt(np.diag(Cov)), 4)} [it {it}]")
    pol = cn not in ("a", "b")
    for dn, dv in (("self", N0), ("cat", N0 + Dc)):
        xa, _, _ = C2.Fitter2(model, dv * fsky, free, fsky=fsky, priors=pri).fit(polish=True)
        res = []
        for r in range(NR):
            dr = rng.poisson(dv * fsky).astype(float)
            pr = {k: (rng.normal(m_, s_), s_) for k, (m_, s_) in pri.items()}
            x, L, it = C2.Fitter2(model, dr, free, fsky=fsky, priors=pr).fit(tol=1e-3, polish=pol)
            res.append(x)
        res = np.array(res)
        S8 = CL.S8_of(res[:, 0], res[:, 1]); S8t = CL.S8_of(*CL.FID[:2]); S8a = CL.S8_of(xa[0], xa[1])
        out[f"{cn}_{dn}"] = res
        print(f"config {cn} free={free} priors={pri} data {dn} ({NR} realisations, {time.time()-t0:.0f}s):")
        for k, p in enumerate(free + ["S8"]):
            v = res[:, k] if p != "S8" else S8
            tr = CL.FID[CL.PNAMES.index(p)] if p != "S8" else S8t
            asv = xa[k] if p != "S8" else S8a
            print(f"   {p:5s}: mean shift {(v.mean()-tr)/sig[p]:+.2f} +- {v.std(ddof=1)/sig[p]/np.sqrt(NR):.2f} sigma_F "
                  f"(Asimov {(asv-tr)/sig[p]:+.2f}); scatter / Fisher sigma = {v.std(ddof=1)/sig[p]:.3f} "
                  f"(+- {1/np.sqrt(2*(NR-1)):.3f}); median shift {(np.median(v)-tr)/sig[p]:+.2f}  [sigma_F {sig[p]:.5f}]", flush=True)
np.savez(f"{CL.DC}/poisson_real_v2{'_' + ''.join(CFGS) if len(sys.argv) > 2 else ''}.npz", **out)
print(f"total {time.time()-t0:.0f}s")
