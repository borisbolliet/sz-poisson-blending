"""Config b (Om, s8, 1-b free), SO40, q>5, Asimov 'self' data: profile likelihood in s8 for MY converged model and
for cosmolib's NaiveModel. Compare the Delta chi2 = 1 half-width with each model's own small-step Fisher sigma(s8),
and the cosmolib Fisher vs finite-difference step size."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import vmodel as V, cosmolib as CL
fsky = 0.4; Mo = CL.NaiveModel(5.0)
models = {"mine": lambda t: V.counts(t, 5.0, fsky), "cosmolib": lambda t: Mo.counts(t, fsky)}
def dev(mu, d):
    return 2 * np.sum(mu - d + d * np.log(d / mu))
# cosmolib Fisher vs step size (config b)
mu = Mo.counts(CL.FID, fsky).ravel()
for fac in [0.5, 1, 4, 10, 25, 50]:
    st = np.array([4e-4, 4e-4, 1e-3]) * fac
    J = []
    for i in range(3):
        tp = CL.FID.copy(); tm = CL.FID.copy(); tp[i] += st[i]; tm[i] -= st[i]
        J.append((Mo.counts(tp, fsky) - Mo.counts(tm, fsky)).ravel() / (2 * st[i]))
    J = np.array(J).T; C = np.linalg.inv(J.T @ (J / mu[:, None]))
    g = np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1], 0])
    print(f"cosmolib config-b Fisher, steps x{fac:>4}: sig(Om,s8,omb) {np.round(np.sqrt(np.diag(C)),5)} S8 {np.sqrt(g@C@g):.5f}", flush=True)
for name, f in models.items():
    d = f(V.FID)
    out = []
    for ds8 in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
        best = np.inf
        for x0 in ([0, 0], [-ds8 * 0.8, ds8 * 4]):
            def obj(u):
                t = V.FID.copy(); t[1] += ds8; t[0] += u[0]; t[2] += u[1]
                return dev(f(t), d)
            r = minimize(obj, np.array(x0, float), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=800))
            best = min(best, r.fun)
        out.append((ds8, best))
        print(f"{name}: s8 = fid{ds8:+.3f}: profile Delta chi2 = {best:.3f} -> implied sigma(s8) = {abs(ds8)/np.sqrt(best):.5f}", flush=True)
