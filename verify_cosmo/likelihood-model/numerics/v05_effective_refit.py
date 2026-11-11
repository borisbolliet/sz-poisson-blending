"""Re-fit the c04 effective approximations (fixed fiducial cosmology and scaling relation) with other starting points
and other (equally cheap, 4-parameter) theta-dependent Gaussian forms, to test whether c04's E3 (Delta chi2 17.7) and
E5 (skew-normal, 43.0) are the best those families can do.  Objective: Cash vs the blended counts; reported as in c04:
Delta chi2 = sum (N - N_b)^2 / N_b (full sky) and x0.4 for 40% sky.
 F_c04 : c04 E3 form   q + b0 + b1 L + N(0, 1 + s0^2 exp(p L)),       L = ln(th/2')
 F_c05 : c05 eff form  q + b0 + b1 L + N(0, max(1 + v0 + v1 L, 0.05))  (variance may drop below 1)
 F_exp : q + b0 + b1 L + N(0, exp(a + c L))                            (positive, may be < 1)
 SN    : skew-normal, parameterised by (b, log omega, alpha) with several starts"""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, Nb, Nn = C["ZE"], C["QE"], C["Nb"], C["Nn"]
nz = len(ZE) - 1
pop = cc.population(cc.FID, zedges=ZE)
L = np.log(pop["th"] / 2.0)
dchi = lambda N: (np.sum((N - Nb) ** 2 / Nb), np.abs((N - Nb) / np.sqrt(Nb)).max())
forms = {
    "F_c04": lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * L, var=1 + p[2] ** 2 * np.exp(np.clip(p[3] * L, -50, 50))),
    "F_c05": lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * L, var=np.maximum(1 + p[2] + p[3] * L, 0.05)),
    "F_exp": lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * L, var=np.exp(np.clip(p[2] + p[3] * L, -20, 20))),
}
starts = {"F_c04": [[0, 0, 0.4, 1.0], [-0.06, -0.09, 0.3, 0.5], [-0.05, -0.1, 0.5, 2.0], [0.0, -0.15, 0.2, 0.0]],
          "F_c05": [[0, 0, 0, 0], [0.04, -0.14, -0.5, -0.16], [-0.05, -0.1, 0.2, 0.2]],
          "F_exp": [[0, 0, 0, 0], [0.04, -0.14, -0.7, -0.2], [-0.05, -0.1, 0.2, 0.2]]}
for name, f in forms.items():
    best = None
    for s in starts[name]:
        r = minimize(lambda p: cc.cash(Nb, f(p)), s, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=6000, maxfev=6000))
        r = minimize(lambda p: cc.cash(Nb, f(p)), r.x, method="Powell", options=dict(xtol=1e-6, ftol=1e-10))
        d = dchi(f(r.x))
        print(f"{name} start {s}: pars {np.round(r.x, 4)} Cash {r.fun:.2f} | Delta chi2 full {d[0]:.1f} (max|pull| {d[1]:.2f}), 40% {0.4*d[0]:.1f}", flush=True)
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    N = f(best[0])
    if name != "F_c04":
        lt = np.log(np.array([0.25, 1, 2, 5, 10, 20, 50]) / 2)
        v = np.maximum(1 + best[0][2] + best[0][3] * lt, 0.05) if name == "F_c05" else np.exp(best[0][2] + best[0][3] * lt)
        print(f"   {name} best: kernel variance at th = 0.25,1,2,5,10,20,50': {np.round(v, 3)}; q-marginal resid [%] "
              f"{np.array2string(100*(N.sum(0)/Nb.sum(0)-1), precision=2)}")
# skew-normal with parameterisation (b, ln omega, alpha)
best = None
for s in ([-0.1, np.log(0.3), 2.0], [-0.3, np.log(0.5), 5.0], [0.0, np.log(0.2), -2.0], [-0.2, np.log(0.4), 1.0]):
    f5 = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0], skew_alpha=p[2], omega=np.exp(p[1]))
    r = minimize(lambda p: cc.cash(Nb, f5(p)), s, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=3000))
    d = dchi(f5(r.x))
    print(f"SN start {np.round(s,3)}: pars (b, omega, alpha) = ({r.x[0]:.4f}, {np.exp(r.x[1]):.4f}, {r.x[2]:.3f}) Cash {r.fun:.2f} | Delta chi2 full {d[0]:.1f}", flush=True)
print(f"reference: naive Delta chi2 {dchi(Nn)[0]:.1f}")
