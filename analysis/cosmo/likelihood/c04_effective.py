"""Can blending be absorbed into an effective scatter + bias of the observable-mass relation?
Target: blended dN/dz/dq_obs (12 z x 10 q bins, q_obs > 5, fiducial cosmology, full sky).
Each approximation keeps the naive machinery (q_obs = q + n) and adds an extra term; free parameters are fitted
by minimising the Poisson deviance against the blended counts.  Residuals are quoted as Delta chi2 =
sum (N_eff - N_b)^2 / N_b for a full-sky and a 40%-sky survey (Delta chi2 scales with f_sky).
  E0 naive (0 par)
  E1 Gaussian in q: q_obs = q + b + N(0, 1 + s^2)                         (2 par: b, s)
  E2 lognormal in Y: ln Y shift + extra sigma_lnY (i.e. re-fit logA, sigma_lnY of the scaling relation) (2 par)
  E3c Gaussian with theta-dependent bias and EXTRA width (variance >= 1 forced): b = b0 + b1 L, var = 1 + s0^2 (th/2')^p (4 par)
  E3 Gaussian with theta-dependent bias and total width, variance allowed below 1 (c05 'eff' form):
     q_obs = q + b0 + b1 L + N(0, max(1 + v0 + v1 L, 0.05)),  L = ln(th/2')                    (4 par)
  E3x same with a smooth positive variance exp(a + c L)                                         (4 par)
  [repair: the original E3 = E3c collapsed to s0 -> 0 (degenerate corner); 4-parameter fits now use several starts
   and a Nelder-Mead + Powell polish]
  E4 'Gaussianised P(D)' (0 par): b(th) = median of (delta - mu), s(th) = half the 16-84% range of delta, from the P(D)
  E5 skew-normal, constant: q_obs = q + b + n + SN(0, omega, alpha)       (3 par)
  E6 E3 (sub-unit variance allowed) + dominance loss: N x (1 - f_dom) with f_dom = 1 - P_dom(th, q) taken from the P(D)   (4 par + 0)"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import json
import numpy as np
from scipy.optimize import minimize
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, Nb, Nn = C["ZE"], C["QE"], C["Nb"], C["Nn"]
nz = len(ZE) - 1
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
G, Pdom, th_n = K["G"], K["Pdom"], K["th"]
pop = cc.population(cc.FID, zedges=ZE)
lt2 = np.log(pop["th"] / 2.0)


def stats(N):
    d = {}
    for fs in (1.0, 0.4):
        P = cc.pull_sigma(N, Nb, fs)
        d[fs] = (np.sum(P ** 2), np.abs(P).max())
    return d


res = {}


def show(name, N, npar, pars=None):
    s = stats(N)
    res[name] = dict(npar=npar, pars=None if pars is None else list(map(float, pars)), dchi2_full=s[1.0][0], dchi2_40=s[0.4][0],
                     maxpull_full=s[1.0][1], tot=float(N.sum() / Nb.sum() - 1))
    print(f"{name}: npar={npar} pars={None if pars is None else np.round(pars, 4)} | total N/N_b-1 = {100*(N.sum()/Nb.sum()-1):+.2f}% |"
          f" Delta chi2 full sky {s[1.0][0]:.1f} (max|pull| {s[1.0][1]:.2f}), 40% sky {s[0.4][0]:.1f} (max|pull| {s[0.4][1]:.2f})")
    zq = 100 * (N / Nb - 1)
    print(f"     residual N_eff/N_b - 1 [%], q marginal: {np.array2string(100*(N.sum(0)/Nb.sum(0)-1), precision=2)}")
    print(f"                                 z marginal: {np.array2string(100*(N.sum(1)/Nb.sum(1)-1), precision=2)}")
    print(f"                                 lowest q bin (5-6) per z: {np.array2string(zq[:, 0], precision=1)}")


show("E0 naive", Nn, 0)

def fit(f, x0, starts=None):
    best = None
    for s in ([x0] + (starts or [])):
        r = minimize(lambda p: cc.cash(Nb, f(p)), s, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxfev=1500, adaptive=True))
        r = minimize(lambda p: cc.cash(Nb, f(p)), r.x, method="Powell", options=dict(xtol=1e-5, ftol=1e-8, maxfev=1500))
        if starts:
            print(f"     start {np.round(s, 3)} -> Cash {r.fun:.3f} pars {np.round(r.x, 4)}")
        if best is None or r.fun < best.fun:
            best = r
    return best
# E1
f1 = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0], var=1 + p[1] ** 2)
r = fit(f1, [0.0, 0.3]); show("E1 Gaussian (b, s)", f1(r.x), 2, r.x)
# E2 lognormal: refit logA and sigma_lnY
hmf = cc.hmf_grid(cc.FID)
def f2(p):
    sr = dict(cc.SR_FID, logA=cc.SR_FID["logA"] + p[0], sigma_lnY=np.sqrt(cc.SR_FID["sigma_lnY"] ** 2 + p[1] ** 2))
    return cc.counts_gauss(cc.population(cc.FID, sr=sr, zedges=ZE, hmf=hmf), QE, nz)
r = fit(f2, [0.0, 0.05]); show("E2 lognormal (dlog10A, extra sigma_lnY)", f2(r.x), 2, r.x)
# E3c (original c04 E3: extra variance >= 0)
f3c = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * lt2, var=1 + p[2] ** 2 * np.exp(np.clip(p[3] * lt2, -50, 50)))
r = fit(f3c, [0.0, 0.0, 0.4, 1.0], [[-0.06, -0.09, 0.3, 0.5], [0.0, -0.15, 0.2, 0.0]])
show("E3c Gaussian theta-dependent, variance >= 1 forced (b0, b1, s0, p)", f3c(r.x), 4, r.x)
# E3 (variance may drop below 1, clamp at 0.05 as in c05)
f3 = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * lt2, var=np.maximum(1 + p[2] + p[3] * lt2, 0.05))
r3 = fit(f3, [0.0, 0.0, 0.0, 0.0], [[0.04, -0.14, -0.5, -0.16], [-0.05, -0.1, 0.2, 0.2]])
show("E3 Gaussian theta-dependent, total variance 1+v0+v1 L (b0, b1, v0, v1)", f3(r3.x), 4, r3.x)
vt = np.log(np.array([0.25, 1, 2, 5, 10, 20, 50]) / 2)
print(f"     E3 kernel variance at th=0.25,1,2,5,10,20,50': {np.round(np.maximum(1 + r3.x[2] + r3.x[3] * vt, 0.05), 3)}")
# E3x (smooth positive variance)
f3x = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0] + p[1] * lt2, var=np.exp(np.clip(p[2] + p[3] * lt2, -20, 20)))
r = fit(f3x, [0.0, 0.0, 0.0, 0.0], [[0.11, -0.11, -0.97, -0.45], [-0.05, -0.1, 0.2, 0.2]])
show("E3x Gaussian theta-dependent, variance exp(a + c L) (b0, b1, a, c)", f3x(r.x), 4, r.x)
print(f"     E3x kernel variance at th=0.25,1,2,5,10,20,50': {np.round(np.exp(r.x[2] + r.x[3] * vt), 3)}")
# E4 Gaussianised P(D), zero free parameters: percentiles of y = n + delta - mu at the q=3000 node -> remove unit noise
kk = len(cc.QNODES) - 1
pc = np.array([np.interp([0.16, 0.5, 0.84], G[i, kk] / Pdom[i, kk], cc.TGRID) for i in range(len(th_n))])
med = pc[:, 1]; s2 = np.maximum(((pc[:, 2] - pc[:, 0]) / 2) ** 2 - 1.0, 0.0)
print("E4 inputs: theta nodes", np.round(th_n[::6], 2), " median(y)", np.round(med[::6], 3), " s_c", np.round(np.sqrt(s2[::6]), 3))
b4 = np.interp(np.log(pop["th"]), np.log(th_n), med); v4 = 1 + np.interp(np.log(pop["th"]), np.log(th_n), s2)
show("E4 Gaussianised P(D) (no free parameter)", cc.counts_gauss(pop, QE, nz, shift=b4, var=v4), 0)
# E5 skew-normal constant
f5 = lambda p: cc.counts_gauss(pop, QE, nz, shift=p[0], skew_alpha=p[2], omega=abs(p[1]))
r = fit(f5, [-0.3, 0.5, 3.0], [[-0.1, 0.3, 2.0], [-0.15, 0.2, 0.8]]); show("E5 skew-normal constant (b, omega, alpha)", f5(r.x), 3, r.x)
# E6 E3 + dominance thinning from P_dom (interpolated in ln theta and q)
from scipy.interpolate import RegularGridInterpolator
ip = RegularGridInterpolator((np.log(th_n), cc.QNODES), Pdom, bounds_error=False, fill_value=None)
pd_pts = np.clip(ip(np.column_stack([np.clip(np.log(pop["th"]), np.log(th_n[0]), np.log(th_n[-1])),
                                     np.clip(pop["q"], cc.QNODES[0], cc.QNODES[-1])])), 0, 1)
popd = dict(pop, w=pop["w"] * pd_pts)
f6 = lambda p: cc.counts_gauss(popd, QE, nz, shift=p[0] + p[1] * lt2, var=np.maximum(1 + p[2] + p[3] * lt2, 0.05))
r = fit(f6, r3.x, [[0.0, 0.0, 0.0, 0.0]]); show("E6 = E3 + P_dom thinning", f6(r.x), 4, r.x)
json.dump(res, open(f"{cc.OUT}/effective_fits.json", "w"), indent=1)
