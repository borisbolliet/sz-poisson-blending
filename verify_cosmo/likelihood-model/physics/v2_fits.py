"""V2: independent Asimov bias fits.  Data = blended counts (my population + c01 kernel G), model = naive or
naive+eff (4 free confusion params, no prior).  Optimiser: scipy least_squares (trust region) on Poisson deviance
residuals + prior residuals (different algorithm from c05's damped Gauss-Newton).  Fisher at best fit AND at truth;
linear-response (Fisher) bias prediction for comparison with the non-linear fit."""
import sys, time, json
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2d, ncx2
import vlib as v

ND = v.nodes(nzper=6, nsub=4, nM=160, ngh=20)
PAR = {"Om0": (0.3111, 0.003, None), "sigma8": (0.8102, 0.003, None), "omb": (0.8, 0.005, 0.04),
       "alpha": (1.79, 0.01, 0.08), "beta_E": (2 / 3, 0.04, 0.5), "slnY": (0.173, 0.008, 0.023),
       "b0": (0.0, 0.02, None), "b1": (0.0, 0.02, None), "v0": (0.0, 0.02, None), "v1": (0.0, 0.02, None)}
SETUPS = {"S0": ["Om0", "sigma8"], "S1": ["Om0", "sigma8", "omb"], "S2": ["Om0", "sigma8", "omb", "alpha", "beta_E", "slnY"]}
CONF = ["b0", "b1", "v0", "v1"]
NEVAL = [0]


def model(p, names, kind):
    NEVAL[0] += 1
    P = {k: PAR[k][0] for k in PAR}; P.update(dict(zip(names, p)))
    pop = v.population(Om0=P["Om0"], sigma8=P["sigma8"], omb=P["omb"], alpha=P["alpha"], beta_E=P["beta_E"],
                       slnY=abs(P["slnY"]), nd=ND)
    if kind == "naive":
        return v.counts_gauss(pop).ravel()
    lt = np.log(pop["th"] / 2.0)
    return v.counts_gauss(pop, shift=P["b0"] + P["b1"] * lt, var=np.maximum(1 + P["v0"] + P["v1"] * lt, 0.05)).ravel()


def jac_mu(p, names, kind, fs):
    J = np.empty((120, len(p)))
    for j, n in enumerate(names):
        d = np.zeros(len(p)); d[j] = PAR[n][1]
        J[:, j] = (model(p + d, names, kind) - model(p - d, names, kind)) * fs / (2 * PAR[n][1])
    return J


def prior_vecs(names):
    pc = np.array([PAR[n][0] for n in names]); ps = np.array([PAR[n][2] or np.inf for n in names])
    return pc, ps


def dev_res(D, mu):
    dev = 2 * (mu - D + np.where(D > 0, D * np.log(np.where(D > 0, D, 1) / mu), 0))
    r = np.sign(D - mu) * np.sqrt(np.maximum(dev, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        drdmu = np.where(np.abs(D - mu) > 1e-8 * mu + 1e-12, np.sign(D - mu) * (1 - D / mu) / np.sqrt(np.maximum(dev, 1e-300)), -1 / np.sqrt(mu))
    return r, drdmu


def fit(D, names, kind, fs, p0=None):
    pc, ps = prior_vecs(names)
    fin = np.isfinite(ps)
    cache = {}

    def mu_of(p):
        k = tuple(np.round(p, 12))
        if k not in cache:
            cache.clear(); cache[k] = model(p, names, kind) * fs
        return cache[k]

    def res(p):
        mu = mu_of(p); r, _ = dev_res(D, mu)
        return np.concatenate([r, ((p - pc) / ps)[fin]])

    def jac(p):
        mu = mu_of(p); _, drdmu = dev_res(D, mu)
        J = jac_mu(p, names, kind, fs)
        Jp = np.zeros((fin.sum(), len(p))); Jp[np.arange(fin.sum()), np.where(fin)[0]] = 1 / ps[fin]
        return np.vstack([drdmu[:, None] * J, Jp])

    x0 = pc.copy() if p0 is None else np.array(p0, float)
    sol = least_squares(res, x0, jac=jac, method="trf", x_scale="jac", xtol=1e-10, ftol=1e-12, gtol=1e-10, max_nfev=60)
    p = sol.x
    mu = model(p, names, kind) * fs
    J = jac_mu(p, names, kind, fs)
    F = (J / mu[:, None]).T @ J + np.diag(np.where(fin, 1 / ps ** 2, 0))
    return p, F, v.cash(D, mu) + np.sum(((p - pc) / ps)[fin] ** 2), sol.nfev, sol.status


if __name__ == "__main__":
    t0 = time.time()
    popf = v.population(nd=ND)
    NB = v.counts_blended(popf).ravel(); NN = v.counts_gauss(popf).ravel()
    print(f"[{time.time()-t0:.0f}s] N_blend {NB.sum():.1f}, N_naive {NN.sum():.1f} (this quadrature: {popf['q'].size:,} pts)")
    jobs = [(1.0, "S0", "naive"), (1.0, "S1", "naive"), (1.0, "S2", "naive"), (0.4, "S0", "naive"), (0.4, "S1", "naive"),
            (0.4, "S2", "naive"), (1.0, "S0", "eff"), (1.0, "S1", "eff"), (1.0, "S2", "eff"), (0.4, "S2", "eff")]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" in sys.argv[1:]]
    out = {}
    for fs, S, kind in jobs:
        names = SETUPS[S] + (CONF if kind == "eff" else [])
        D = NB * fs
        pc, ps = prior_vecs(names)
        NEVAL[0] = 0
        p, F, c2, nfev, st = fit(D, names, kind, fs)
        cov = np.linalg.inv(F); sig = np.sqrt(np.diag(cov))
        # Fisher at truth, linear-response bias prediction
        mu0 = model(pc, names, kind) * fs; J0 = jac_mu(pc, names, kind, fs)
        fin = np.isfinite(ps)
        F0 = (J0 / mu0[:, None]).T @ J0 + np.diag(np.where(fin, 1 / ps ** 2, 0))
        dlin = np.linalg.solve(F0, J0.T @ (D / mu0 - 1))
        sig0 = np.sqrt(np.diag(np.linalg.inv(F0)))
        b = (p - pc) / sig
        d2 = (p - pc)[:2]; c2d = float(d2 @ np.linalg.solve(cov[:2, :2], d2))
        s8 = p[1] * (p[0] / 0.3) ** 0.3; s8f = pc[1] * (pc[0] / 0.3) ** 0.3
        jj = np.zeros(len(p)); jj[0] = 0.3 * s8 / p[0]; jj[1] = s8 / p[1]
        bS8 = (s8 - s8f) / np.sqrt(jj @ cov @ jj)
        dof = 120 - len(p)
        # expected GoF for a noisy realisation: chi2 ~ noncentral chi2(dof, lambda = -2lnL_min of the Asimov fit)
        pmean = chi2d.sf(dof + c2, dof)
        power = ncx2.sf(chi2d.isf(0.05, dof), dof, c2)
        print(f"[{time.time()-t0:.0f}s] f_sky={fs} {S} fit={kind} free={names} (status {st}, nfev {nfev}, model evals {NEVAL[0]})")
        print(f"   best      {np.round(p, 4)}")
        print(f"   sigma(bf) {np.round(sig, 4)} | sigma(truth) {np.round(sig0, 4)}")
        print(f"   bias/sigma (nonlinear fit) {np.round(b, 2)}")
        print(f"   bias/sigma (linear Fisher at truth) {np.round(dlin / sig0, 2)}")
        print(f"   2D(Om,s8) shift chi2 {c2d:.2f} | S8cl bias {bS8:+.2f} sigma | -2lnL_min {c2:.2f} | "
              f"noisy-data GoF: P(chi2>dof+lam) under null {pmean:.3f}, power at p<0.05 = {power:.2f}")
        out[f"{fs}|{S}|{kind}"] = dict(names=names, best=p.tolist(), sigma=sig.tolist(), sigma_truth=sig0.tolist(),
                                       bias=b.tolist(), bias_lin=(dlin / sig0).tolist(), chi2=c2, S8=bS8, c2d=c2d)
        json.dump(out, open("/private/tmp/claude-501/-Users-boris-GitHub/0935154e-1cde-4eab-ab64-4515ca671a83/scratchpad/v2_fits_%s.json" % ("all" if len(sys.argv) == 1 else "sub"), "w"), indent=1)
