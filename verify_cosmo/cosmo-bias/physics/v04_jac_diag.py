"""Which Jacobian column (s8 or 1-b) drives the config-b Fisher difference; per-bin dlnN/d(1-b) comparison."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import vmodel as V, cosmolib as CL
fsky = 0.4; Mo = CL.NaiveModel(5.0)
fm = lambda t: V.counts(t, 5.0, fsky); ft = lambda t: Mo.counts(t, fsky)
def jcol(fun, fid, i, st):
    tp = fid.copy(); tm = fid.copy(); tp[i] += st; tm[i] -= st
    return ((fun(tp) - fun(tm)) / (2 * st)).ravel()
mu_m = fm(V.FID).ravel(); mu_t = ft(CL.FID).ravel()
Jm = np.stack([jcol(fm, V.FID, i, s) for i, s in [(0, 2e-3), (1, 5e-4), (2, 2e-3)]], 1)
Jt = np.stack([jcol(ft, CL.FID, i, s) for i, s in [(0, 2e-3), (1, 4e-4), (2, 1e-3)]], 1)
g = np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1], 0])
def sg(J, mu):
    F = J.T @ (J / mu[:, None]); C = np.linalg.inv(F); return np.round(np.sqrt(np.diag(C)), 5), round(float(np.sqrt(g @ C @ g)), 5)
for lab, cols in [("all mine", "mmm"), ("all theirs", "ttt"), ("their s8 only", "mtm"), ("their omb only", "mmt"), ("my s8 only", "tmt"), ("my omb only", "ttm")]:
    J = np.stack([(Jm if c == "m" else Jt)[:, k] for k, c in enumerate(cols)], 1)
    print(f"{lab:15s}: {sg(J, mu_t)}")
sel = mu_t > 10
for k, n in enumerate(["Om", "s8", "omb"]):
    rm, rt = (Jm[:, k] / mu_m).reshape(16, 9), (Jt[:, k] / mu_t).reshape(16, 9)
    print(f"dlnN/d{n}: mine-theirs per q bin (z-summed weights):", np.round((Jm[:, k].reshape(16, 9).sum(0) / mu_m.reshape(16, 9).sum(0)) - (Jt[:, k].reshape(16, 9).sum(0) / mu_t.reshape(16, 9).sum(0)), 4))
    print(f"   per z bin:", np.round((Jm[:, k].reshape(16, 9).sum(1) / mu_m.reshape(16, 9).sum(1)) - (Jt[:, k].reshape(16, 9).sum(1) / mu_t.reshape(16, 9).sum(1)), 4))
    print(f"   theirs per z bin:", np.round(Jt[:, k].reshape(16, 9).sum(1) / mu_t.reshape(16, 9).sum(1), 3))
