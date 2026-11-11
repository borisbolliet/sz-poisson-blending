"""Diagnose Fisher differences between my model (vmodel) and cosmolib for config b; Jacobian comparison."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import vmodel as V
import cosmolib as CL
qc = 5.0; fsky = 0.4
Mo = CL.NaiveModel(qc)
N_their = Mo.counts(CL.FID, fsky); N_mine = V.counts(V.FID, qc, fsky)
r = N_mine / N_their - 1
i = np.unravel_index(np.argmax(np.abs(r) * (N_their > 4)), r.shape)
print("worst bin (mu>4) mine/their-1:", r[i], "at z bin", i[0], "q bin", i[1], "N", N_their[i])
print("per q bin mine/their-1:", np.round(N_mine.sum(0) / N_their.sum(0) - 1, 4))
print("per z bin mine/their-1:", np.round(N_mine.sum(1) / N_their.sum(1) - 1, 4))
free = [0, 1, 2]
for name, jfun, mu in [("mine", lambda st: None, N_mine), ("theirs", None, N_their)]:
    pass
def Jmine(steps):
    out = []
    for i, st in zip(free, steps):
        tp = V.FID.copy(); tm = V.FID.copy(); tp[i] += st; tm[i] -= st
        out.append(((V.counts(tp, qc, fsky) - V.counts(tm, qc, fsky)) / (2 * st)).ravel())
    return np.array(out).T
def Jtheirs(steps):
    out = []
    for i, st in zip(free, steps):
        tp = CL.FID.copy(); tm = CL.FID.copy(); tp[i] += st; tm[i] -= st
        out.append(((Mo.counts(tp, fsky) - Mo.counts(tm, fsky)) / (2 * st)).ravel())
    return np.array(out).T
def sig(J, mu):
    F = J.T @ (J / mu.ravel()[:, None]); C = np.linalg.inv(F)
    g = np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1], 0])
    return np.sqrt(np.diag(C)), np.sqrt(g @ C @ g), F
for steps in ([5e-4, 5e-4, 2e-3], [4e-4, 4e-4, 1e-3], [2e-3, 2e-3, 5e-3]):
    Jm, Jt = Jmine(steps), Jtheirs(steps)
    sm, sSm, Fm = sig(Jm, N_mine); st, sSt, Ft = sig(Jt, N_their)
    print(f"steps {steps}: mine sig {np.round(sm,5)} S8 {sSm:.5f} | theirs sig {np.round(st,5)} S8 {sSt:.5f}")
    for k in range(3):
        print(f"   dlnN/dp_{V.NAMES[free[k]]} total: mine {Jm[:,k].sum()/N_mine.sum():+.4f} theirs {Jt[:,k].sum()/N_their.sum():+.4f}; "
              f"max|diff| / max|J| = {np.abs(Jm[:,k]-Jt[:,k]).max()/np.abs(Jt[:,k]).max():.3e}")
    # correlation matrix (theirs)
    C = np.linalg.inv(Ft); print("   their corr:", np.round(C / np.sqrt(np.outer(np.diag(C), np.diag(C))), 5)[np.triu_indices(3, 1)])
    C = np.linalg.inv(Fm); print("   my corr:   ", np.round(C / np.sqrt(np.outer(np.diag(C), np.diag(C))), 5)[np.triu_indices(3, 1)])
# cross: their Jacobian with my mu and vice versa
st_, sSt_, _ = sig(Jtheirs([4e-4, 4e-4, 1e-3]), N_mine)
print("their J with my mu:", np.round(st_, 5), sSt_)
