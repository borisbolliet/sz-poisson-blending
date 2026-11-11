"""Cosmological-parameter bias from ignoring blending in an SZ cluster-count likelihood.

Asimov test: 'data' = expected blended counts (fiducial Planck18 cosmology, P(D) convolution model, map mean
removed, dominance rule) in 12 z x 10 q_obs bins, q_obs > 5.  Fit with
   naive model  : q_obs = q + n, lognormal Y-M scatter  (the standard likelihood)
   naive+eff    : + extra Gaussian term q_obs = q + b0 + b1 ln(th/2') + N(0, 1 + v0 + v1 ln(th/2')),
                  4 free 'confusion' parameters, no prior (does NOT use the P(D))
   blended      : the P(D) convolution kernel (G fixed at the fiducial) -> closure test, bias must be ~0
Poisson likelihood, Fisher-scoring (Gauss-Newton) maximisation with numerical Jacobians.  Survey = full sky
(f_sky = 1) and f_sky = 0.4 (all counts x f_sky).  Bias in units of the marginalised 1-sigma statistical error of
the same fit (from the Fisher matrix at the best fit, priors included).
Free parameters per setup are printed explicitly.  Cosmology: flat LCDM, h, Omega_b, n_s fixed."""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, NB_FID, NN_FID = C["ZE"], C["QE"], C["Nb"], C["Nn"]
nz = len(ZE) - 1
G = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]

PAR = {  # name: (fiducial, step, prior sigma or None)
    "Om0": (cc.FID["Om0"], 0.004, None),
    "sigma8": (cc.FID["sigma8"], 0.004, None),
    "one_minus_b": (0.8, 0.004, 0.04),
    "alpha": (1.79, 0.01, 0.08),
    "beta_E": (2.0 / 3.0, 0.05, 0.50),
    "sigma_lnY": (0.173, 0.01, 0.023),
    "b0": (0.0, 0.01, None), "b1": (0.0, 0.01, None), "v0": (0.0, 0.01, None), "v1": (0.0, 0.01, None),
}


def model(pv, names, kind, data_kind=None):
    P = {n: PAR[n][0] for n in PAR}
    P.update(dict(zip(names, pv)))
    cp = dict(cc.FID, Om0=P["Om0"], sigma8=P["sigma8"])
    sr = dict(cc.SR_FID, one_minus_b=P["one_minus_b"], alpha=P["alpha"], beta_E=P["beta_E"], sigma_lnY=abs(P["sigma_lnY"]))
    pop = cc.population(cp, sr=sr, zedges=ZE)
    if kind == "naive":
        return cc.counts_gauss(pop, QE, nz)
    if kind == "eff":
        lt2 = np.log(pop["th"] / 2.0)
        return cc.counts_gauss(pop, QE, nz, shift=P["b0"] + P["b1"] * lt2, var=np.maximum(1 + P["v0"] + P["v1"] * lt2, 0.05))
    if kind == "blended":
        return cc.counts_blended(G, pop, QE, nz)


def fit(D, names, kind, fsky, p0=None, niter=12, verbose=False):
    p = np.array([PAR[n][0] for n in names] if p0 is None else p0, float)
    pri = np.array([PAR[n][2] if PAR[n][2] is not None else np.inf for n in names])
    pc = np.array([PAR[n][0] for n in names])
    Pinv = np.diag(np.where(np.isfinite(pri), 1 / pri ** 2, 0.0))
    st = np.array([PAR[n][1] for n in names])
    Ds = D.ravel() * fsky
    lam = 1e-3
    for it in range(niter):
        mu = model(p, names, kind).ravel() * fsky
        J = np.empty((len(mu), len(p)))
        for j in range(len(p)):
            dp = np.zeros(len(p)); dp[j] = st[j]
            J[:, j] = (model(p + dp, names, kind).ravel() - model(p - dp, names, kind).ravel()) * fsky / (2 * st[j])
        F = (J / mu[:, None]).T @ J + Pinv
        g = J.T @ (Ds / mu - 1) - Pinv @ (p - pc)
        step = np.linalg.solve(F + lam * np.diag(np.diag(F)), g)
        c_old = cc.cash(Ds, mu) + np.sum(((p - pc) / pri) ** 2)
        for _ in range(8):
            pn = p + step
            c_new = cc.cash(Ds, model(pn, names, kind).ravel() * fsky) + np.sum(((pn - pc) / pri) ** 2)
            if c_new <= c_old + 1e-9:
                break
            step *= 0.5
        p = pn
        if verbose:
            print(f"      it {it}: -2lnL = {c_new:.4f}, p = {np.round(p, 4)}")
        if np.max(np.abs(step) / np.sqrt(np.diag(np.linalg.inv(F)))) < 1e-3:
            break
    # Fisher at the best fit (expected counts at the best fit)
    mu = model(p, names, kind).ravel() * fsky
    J = np.empty((len(mu), len(p)))
    for j in range(len(p)):
        dp = np.zeros(len(p)); dp[j] = st[j]
        J[:, j] = (model(p + dp, names, kind).ravel() - model(p - dp, names, kind).ravel()) * fsky / (2 * st[j])
    F = (J / mu[:, None]).T @ J + Pinv
    cov = np.linalg.inv(F)
    chi2 = cc.cash(Ds, mu)
    return p, cov, chi2


SETUPS = [
    ("S0: Om0, sigma8 (scaling relation fixed at truth)", ["Om0", "sigma8"]),
    ("S1: Om0, sigma8, (1-b) [prior 0.80+-0.04]", ["Om0", "sigma8", "one_minus_b"]),
    ("S2: Om0, sigma8, (1-b)[+-0.04], alpha[+-0.08], beta_E[+-0.5], sigma_lnY[+-0.023]",
     ["Om0", "sigma8", "one_minus_b", "alpha", "beta_E", "sigma_lnY"]),
]
CONF = ["b0", "b1", "v0", "v1"]   # q_obs = q + b0 + b1 ln(th/2') + N(0, 1 + v0 + v1 ln(th/2'))
out = {}
t0 = time.time()
FSKY = [float(a) for a in sys.argv[1:]] or [1.0, 0.4]
for fsky in FSKY:
    for lab, names in SETUPS:
        for kind, extra in (("naive", []), ("eff", CONF), ("blended", [])):
            nm = names + extra
            p0 = [PAR[n][0] for n in names] + ([0.0] * len(extra))
            p, cov, chi2 = fit(NB_FID, nm, kind, fsky, p0=p0)
            sig = np.sqrt(np.diag(cov))
            bias = (p - np.array([PAR[n][0] for n in nm])) / sig
            # 2D significance of the (Om0, sigma8) shift with the marginalised covariance
            d2 = p[:2] - np.array([PAR["Om0"][0], PAR["sigma8"][0]])
            chi2_2d = float(d2 @ np.linalg.solve(cov[:2, :2], d2))
            # S8-like combination sigma8 (Om0/0.3)^0.3 (cluster-count degeneracy direction)
            jac = np.zeros(len(p)); s8 = p[1] * (p[0] / 0.3) ** 0.3
            jac[0] = 0.3 * s8 / p[0]; jac[1] = s8 / p[1]
            s8_f = PAR["sigma8"][0] * (PAR["Om0"][0] / 0.3) ** 0.3
            sig_s8 = np.sqrt(jac @ cov @ jac)
            key = f"fsky{fsky}|{lab}|{kind}"
            out[key] = dict(free=nm, best=p.tolist(), sigma=sig.tolist(), bias_sigma=bias.tolist(), chi2_min=chi2,
                            shift2d_chi2=chi2_2d, S8cl_bias_sigma=(s8 - s8_f) / sig_s8, S8cl_sigma=sig_s8)
            print(f"[f_sky={fsky}] {lab} | fit model = {kind}{' (+4 free confusion params b0,b1,v0,v1)' if kind=='eff' else ''}"
                  f"  ({time.time()-t0:.0f}s)")
            print(f"   free: {nm}")
            print(f"   best-fit: {np.round(p, 4)}")
            print(f"   sigma   : {np.round(sig, 4)}")
            print(f"   bias/sigma: {np.round(bias, 2)} | (Om0,sigma8) 2D shift chi2 = {chi2_2d:.2f} | "
                  f"S8cl=sigma8(Om0/0.3)^0.3 bias = {(s8-s8_f)/sig_s8:+.2f} sigma (sigma {sig_s8:.4f}) | -2lnL_min = {chi2:.1f} (120 bins)", flush=True)
json.dump(out, open(f"{cc.OUT}/cosmo_bias_fsky{'_'.join(map(str, FSKY))}.json", "w"), indent=1)
