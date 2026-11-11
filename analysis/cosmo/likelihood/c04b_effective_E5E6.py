"""Part 2 of the repaired c04 (c04_effective.py was stopped at the 30-min limit inside E5 after E0-E4; its log is
c04_effective_E0-E4.log).  E5 skew-normal (single start as in the original c04; the verifier found Delta chi2 43.0 from
4 starts) and E6 = repaired E3 (total variance 1 + v0 + v1 L, may drop below 1) + P_dom thinning, multi-start."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, Nb = C["ZE"], C["QE"], C["Nb"]
nz = len(ZE) - 1
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
Pdom, th_n = K["Pdom"], K["th"]
pop = cc.population(cc.FID, zedges=ZE)
lt2 = np.log(pop["th"] / 2.0)


def show(name, N, pars):
    for fs in (1.0, 0.4):
        P = cc.pull_sigma(N, Nb, fs)
        print(f"{name} pars={np.round(pars, 4)} f_sky={fs}: Delta chi2 {np.sum(P**2):.1f} (max|pull| {np.abs(P).max():.2f})")
    print(f"     q-marginal residual [%]: {np.array2string(100*(N.sum(0)/Nb.sum(0)-1), precision=2)}", flush=True)


def fit(f, starts, maxfev=1500):
    best = None
    for s in starts:
        r = minimize(lambda p: cc.cash(Nb, f(p)), s, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxfev=maxfev, adaptive=True))
        print(f"     start {np.round(s, 3)} -> Cash {r.fun:.3f} pars {np.round(r.x, 4)}", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best


f5 = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0], skew_alpha=p[2], omega=abs(p[1]))
r = fit(f5, [[-0.3, 0.5, 3.0]], maxfev=600); show("E5 skew-normal constant (b, omega, alpha)", f5(r.x), r.x)
ip = RegularGridInterpolator((np.log(th_n), cc.QNODES), Pdom, bounds_error=False, fill_value=None)
pd_pts = np.clip(ip(np.column_stack([np.clip(np.log(pop["th"]), np.log(th_n[0]), np.log(th_n[-1])),
                                     np.clip(pop["q"], cc.QNODES[0], cc.QNODES[-1])])), 0, 1)
popd = dict(pop, w=pop["w"] * pd_pts)
f6 = lambda p: cc.counts_gauss(popd, QE, nz, shift=p[0] + p[1] * lt2, var=np.maximum(1 + p[2] + p[3] * lt2, 0.05))
r = fit(f6, [[0.1066, -0.093, -0.6133, -0.254], [0.0, 0.0, 0.0, 0.0]]); show("E6 = E3 (sub-unit variance allowed) + P_dom thinning", f6(r.x), r.x)
