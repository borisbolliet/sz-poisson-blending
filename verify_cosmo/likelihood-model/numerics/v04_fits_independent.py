"""Independent re-fit of the c05 Asimov bias tests with a different optimiser and different error estimate.
Data: blended counts at the fiducial (data_cosmo/counts_fid.npz, Nb).  Models: cclib naive / eff (population and
naive counts verified independently in v01).
 - optimiser: scipy Powell then L-BFGS-B on -2lnL = Cash + Gaussian priors, parameters scaled by their step
 - errors: (a) inverse numerical Hessian of -2lnL/2 at the minimum (observed information, independent of c05's
   Gauss-Newton Fisher), (b) expected Fisher at the FIDUCIAL (truth) point -> bias/sigma_fid
 - linear-response prediction of the bias: dp = F^-1 J^T (D/mu - 1) at the fiducial (first-order check)
usage: v04_fits_independent.py SETUP KIND FSKY"""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, NB = C["ZE"], C["QE"], C["Nb"]
nz = len(ZE) - 1
FIDP = {"Om0": (cc.FID["Om0"], 0.004, None), "sigma8": (cc.FID["sigma8"], 0.004, None), "one_minus_b": (0.8, 0.004, 0.04),
        "alpha": (1.79, 0.01, 0.08), "beta_E": (2 / 3, 0.05, 0.5), "sigma_lnY": (0.173, 0.01, 0.023),
        "b0": (0, 0.01, None), "b1": (0, 0.01, None), "v0": (0, 0.01, None), "v1": (0, 0.01, None)}
SET = {"S0": ["Om0", "sigma8"], "S1": ["Om0", "sigma8", "one_minus_b"],
       "S2": ["Om0", "sigma8", "one_minus_b", "alpha", "beta_E", "sigma_lnY"]}
setup, kind, fsky = sys.argv[1], sys.argv[2], float(sys.argv[3])
names = SET[setup] + (["b0", "b1", "v0", "v1"] if kind == "eff" else [])
p_fid = np.array([FIDP[n][0] for n in names]); st = np.array([FIDP[n][1] for n in names])
pri = np.array([FIDP[n][2] or np.inf for n in names])


def model(p):
    P = {n: FIDP[n][0] for n in FIDP}; P.update(dict(zip(names, p)))
    pop = cc.population(dict(cc.FID, Om0=P["Om0"], sigma8=P["sigma8"]),
                        sr=dict(cc.SR_FID, one_minus_b=P["one_minus_b"], alpha=P["alpha"], beta_E=P["beta_E"],
                                sigma_lnY=abs(P["sigma_lnY"])), zedges=ZE)
    if kind == "naive":
        return cc.counts_gauss(pop, QE, nz) * fsky
    lt2 = np.log(pop["th"] / 2.0)
    return cc.counts_gauss(pop, QE, nz, shift=P["b0"] + P["b1"] * lt2, var=np.maximum(1 + P["v0"] + P["v1"] * lt2, 0.05)) * fsky


D = NB * fsky
nev = [0]


def m2lnL(u):
    p = p_fid + u * st
    nev[0] += 1
    return cc.cash(D, model(p)) + np.sum(((p - p_fid) / pri) ** 2)


t0 = time.time()
r = minimize(m2lnL, np.zeros(len(names)), method="Powell", options=dict(xtol=1e-4, ftol=1e-10, maxfev=20000))
r2 = minimize(m2lnL, r.x, method="L-BFGS-B", options=dict(ftol=1e-13, gtol=1e-8, eps=1e-3))
u = r2.x if r2.fun <= r.fun else r.x
pbest = p_fid + u * st
print(f"[{setup} {kind} fsky={fsky}] free: {names}; Powell -2lnL {r.fun:.4f}, L-BFGS-B {r2.fun:.4f}; evals {nev[0]}; {time.time()-t0:.0f}s")
# numerical Hessian of -2lnL in u (step h=0.5 step-units)
h = 0.5; n = len(u); H = np.zeros((n, n)); f0 = m2lnL(u)
for a in range(n):
    for b in range(a, n):
        ea = np.zeros(n); eb = np.zeros(n); ea[a] = h; eb[b] = h
        if a == b:
            H[a, a] = (m2lnL(u + ea) - 2 * f0 + m2lnL(u - ea)) / h ** 2
        else:
            H[a, b] = H[b, a] = (m2lnL(u + ea + eb) - m2lnL(u + ea - eb) - m2lnL(u - ea + eb) + m2lnL(u - ea - eb)) / (4 * h * h)
cov_obs = np.linalg.inv(H / 2) * np.outer(st, st)
sig_obs = np.sqrt(np.diag(cov_obs))


def fisher(p):
    mu = model(p).ravel()
    J = np.empty((mu.size, n))
    for j in range(n):
        dp = np.zeros(n); dp[j] = st[j]
        J[:, j] = (model(p + dp).ravel() - model(p - dp).ravel()) / (2 * st[j])
    Pinv = np.diag(np.where(np.isfinite(pri), 1 / pri ** 2, 0))
    return (J / mu[:, None]).T @ J + Pinv, J, mu


F_fid, J_fid, mu_fid = fisher(p_fid)
sig_fid = np.sqrt(np.diag(np.linalg.inv(F_fid)))
F_b, _, _ = fisher(pbest)
sig_b = np.sqrt(np.diag(np.linalg.inv(F_b)))
dp_lin = np.linalg.solve(F_fid, J_fid.T @ (D.ravel() / mu_fid - 1))
bias = pbest - p_fid
print(f"   best fit          : {np.round(pbest, 4)}")
print(f"   sigma obs-Hessian : {np.round(sig_obs, 4)}")
print(f"   sigma Fisher@best : {np.round(sig_b, 4)}")
print(f"   sigma Fisher@fid  : {np.round(sig_fid, 4)}")
print(f"   bias/sigma (obs Hessian)  : {np.round(bias/sig_obs, 2)}")
print(f"   bias/sigma (Fisher@best)  : {np.round(bias/sig_b, 2)}")
print(f"   bias/sigma (Fisher@fid)   : {np.round(bias/sig_fid, 2)}")
print(f"   linear-response bias/sigma(Fisher@fid): {np.round(dp_lin/sig_fid, 2)}")
s8 = pbest[1] * (pbest[0] / 0.3) ** 0.3; s8f = p_fid[1] * (p_fid[0] / 0.3) ** 0.3
jac = np.zeros(n); jac[0] = 0.3 * s8 / pbest[0]; jac[1] = s8 / pbest[1]
print(f"   S8cl bias/sigma (Fisher@best) {(s8-s8f)/np.sqrt(jac@np.linalg.inv(F_b)@jac):+.2f}, (Fisher@fid) {(s8-s8f)/np.sqrt(jac@np.linalg.inv(F_fid)@jac):+.2f}; -2lnL_min = {m2lnL(u):.2f} (Cash only {cc.cash(D, model(pbest)):.2f})")
if kind == "eff":
    lt = np.log(np.array([0.25, 1, 2, 5, 10, 20, 50]) / 2)
    print(f"   eff variance 1+v0+v1 ln(th/2') at th=0.25,1,2,5,10,20,50': {np.round(1 + pbest[-2] + pbest[-1]*lt, 3)} (clamped at 0.05)")
