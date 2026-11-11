"""Repair of c05: cosmological-parameter bias from ignoring blending, with a robust optimiser and robust error bars.

Changes w.r.t. c05 (verifier disputes):
  * optimiser: Levenberg-Marquardt on -2lnL_tot = Poisson deviance + prior chi2, a step is accepted ONLY if -2lnL_tot
    decreases (c05 accepted the last halved step even if it increased), convergence on |Delta(-2lnL)| < 1e-6;
    for the 'eff' fits several starting points + a Nelder-Mead polish, and every local minimum found is printed.
  * reported -2lnL: total (Poisson + prior) AND Poisson part separately.
  * errors: (a) sigma_true = marginalised Fisher error of the CORRECT (blended) likelihood at the truth, same free
    parameters and priors -> the statistical 1-sigma error of the survey; primary bias unit.
    (b) sigma_own = Fisher error of the fitted model at its own best fit (c05 convention), for comparison.
    (c) Delta(-2lnL_tot) between the truth cosmology (Om0, sigma8 fixed at truth, all other free parameters profiled)
        and the global minimum: implementation-free 2D measure; equivalent Gaussian 2D significance quoted.
    (d) for S2 naive: 1D profile likelihood in sigma8, Om0 and S8cl-direction-free check (profile of sigma8 and Om0).
  * goodness of fit: for a noisy realisation the Poisson deviance at the best fit ~ noncentral chi2(dof, lambda)
    with lambda ~ the Asimov Poisson deviance; we print the power to reject at p < 0.05 and the median p-value.
  * eff model in two forms: 'eff'    q + b0 + b1 L + N(0, max(1 + v0 + v1 L, 0.05))   (c05 form, clamp)
                            'effexp' q + b0 + b1 L + N(0, exp(a + c L))                 (smooth, positive)
    L = ln(theta500/2'); 4 free confusion parameters, no prior.
Data: Asimov blended counts (12 z x 10 q bins, q_obs>5), fiducial Planck18; h, Omega_b, n_s fixed; Tinker08.
usage: c05b_cosmo_bias_v2.py <fsky> [setups e.g. S0,S1,S2] [kinds e.g. naive,eff,effexp]"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2, chi2 as chi2d
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, NB_FID = C["ZE"], C["QE"], C["Nb"]
nz = len(ZE) - 1
G = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]

PAR = {  # name: (truth, Jacobian step, prior sigma or None)
    "Om0": (cc.FID["Om0"], 0.002, None), "sigma8": (cc.FID["sigma8"], 0.002, None),
    "one_minus_b": (0.8, 0.004, 0.04), "alpha": (1.79, 0.01, 0.08), "beta_E": (2.0 / 3.0, 0.05, 0.50),
    "sigma_lnY": (0.173, 0.005, 0.023),
    "b0": (0.0, 0.01, None), "b1": (0.0, 0.01, None), "v0": (0.0, 0.01, None), "v1": (0.0, 0.01, None),
    "a": (0.0, 0.01, None), "c": (0.0, 0.01, None),
}
PC = {n: PAR[n][0] for n in PAR}       # prior centres (= truth unless PRIOR_OMB is set, see c08_prior_centre.py)
if os.environ.get("PRIOR_OMB"):
    PC["one_minus_b"] = float(os.environ["PRIOR_OMB"])
    print(f"NOTE: (1-b) prior centred at {PC['one_minus_b']} (truth 0.8), as a blended-observable WL calibration would give")
_HMF = {}


def hmf(Om, s8):
    k = (round(Om, 12), round(s8, 12))
    if k not in _HMF:
        if len(_HMF) > 60:
            _HMF.clear()
        _HMF[k] = cc.hmf_grid(dict(cc.FID, Om0=Om, sigma8=s8))
    return _HMF[k]


def model(pv, names, kind):
    P = {n: PAR[n][0] for n in PAR}
    P.update(dict(zip(names, pv)))
    sr = dict(cc.SR_FID, one_minus_b=P["one_minus_b"], alpha=P["alpha"], beta_E=P["beta_E"], sigma_lnY=abs(P["sigma_lnY"]))
    cp = dict(cc.FID, Om0=P["Om0"], sigma8=P["sigma8"])
    pop = cc.population(cp, sr=sr, zedges=ZE, hmf=hmf(P["Om0"], P["sigma8"]))
    L = np.log(pop["th"] / 2.0)
    if kind == "naive":
        return cc.counts_gauss(pop, QE, nz)
    if kind == "eff":
        return cc.counts_gauss(pop, QE, nz, shift=P["b0"] + P["b1"] * L, var=np.maximum(1 + P["v0"] + P["v1"] * L, 0.05))
    if kind == "effexp":
        return cc.counts_gauss(pop, QE, nz, shift=P["b0"] + P["b1"] * L, var=np.exp(np.clip(P["a"] + P["c"] * L, -20, 20)))
    if kind == "blended":
        return cc.counts_blended(G, pop, QE, nz)


class Fit:
    def __init__(self, names, kind, fsky, fixed=None):
        self.names, self.kind, self.fsky = names, kind, fsky
        self.fixed = fixed or {}
        self.free = [n for n in names if n not in self.fixed]
        self.pc = np.array([PC[n] for n in self.free])
        pri = np.array([PAR[n][2] if PAR[n][2] is not None else np.inf for n in self.free])
        self.pri = pri
        self.Pinv = np.diag(np.where(np.isfinite(pri), 1 / pri ** 2, 0.0))
        self.st = np.array([PAR[n][1] for n in self.free])
        self.D = NB_FID.ravel() * fsky

    def mu(self, p):
        d = dict(self.fixed); d.update(dict(zip(self.free, p)))
        return model([d[n] for n in self.names], self.names, self.kind).ravel() * self.fsky

    def prior(self, p):
        return float(np.sum(((p - self.pc) / self.pri) ** 2))

    def m2l(self, p):
        return cc.cash(self.D, self.mu(p)) + self.prior(p)

    def jac(self, p):
        J = np.empty((self.D.size, len(p)))
        for j in range(len(p)):
            dp = np.zeros(len(p)); dp[j] = self.st[j]
            J[:, j] = (self.mu(p + dp) - self.mu(p - dp)) / (2 * self.st[j])
        return J

    def fisher(self, p):
        mu = self.mu(p); J = self.jac(p)
        return (J / mu[:, None]).T @ J + self.Pinv

    def lm(self, p0, maxit=80, tol=1e-6):
        p = np.array(p0, float); f = self.m2l(p); lam = 1e-3
        for it in range(maxit):
            mu = self.mu(p); J = self.jac(p)
            F = (J / mu[:, None]).T @ J + self.Pinv
            g = J.T @ (self.D / mu - 1) - self.Pinv @ (p - self.pc)
            acc = False
            for _ in range(12):
                step = np.linalg.solve(F + lam * np.diag(np.diag(F)), g)
                fn = self.m2l(p + step)
                if np.isfinite(fn) and fn < f:
                    acc = True; break
                lam *= 10
            if not acc:
                break
            df = f - fn; p = p + step; f = fn; lam = max(lam / 10, 1e-7)
            if df < tol:
                break
        return p, f

    def minimise(self, starts, polish=False):
        found = [self.lm(s)[::-1] for s in starts]      # (f, p) per start, LM only
        found.sort(key=lambda t: t[0])
        if polish:                                       # Nelder-Mead polish of the best LM minimum, then LM again
            f, p = found[0]
            r = minimize(self.m2l, p, method="Nelder-Mead",
                         options=dict(xatol=1e-6, fatol=1e-7, maxfev=60 * len(p), adaptive=True))
            if r.fun < f:
                p2, f2 = self.lm(r.x)
                found.insert(0, (f2, p2) if f2 < r.fun else (r.fun, r.x))
        return found


SETUPS = {
    "S0": ("S0: free Om0, sigma8 (scaling relation fixed at truth)", ["Om0", "sigma8"]),
    "S1": ("S1: free Om0, sigma8, (1-b) [prior 0.80+-0.04]", ["Om0", "sigma8", "one_minus_b"]),
    "S2": ("S2: free Om0, sigma8, (1-b)[+-0.04], alpha[+-0.08], beta_E[+-0.5], sigma_lnY[+-0.023]",
           ["Om0", "sigma8", "one_minus_b", "alpha", "beta_E", "sigma_lnY"]),
}
CONF = {"naive": [], "eff": ["b0", "b1", "v0", "v1"], "effexp": ["b0", "b1", "a", "c"], "blended": []}
CSTARTS = {"naive": [[]], "blended": [[]],
           "eff": [[0, 0, 0, 0], [0.04, -0.14, -0.5, -0.16], [0.107, -0.093, -0.613, -0.254], [-0.10, -0.20, -0.14, 0.10]],
           "effexp": [[0, 0, 0, 0], [0.11, -0.11, -0.97, -0.45], [-0.05, -0.15, -0.3, 0.1]]}


def s8cl(p):
    return p[1] * (p[0] / 0.3) ** 0.3


def s8cl_sig(p, cov):
    j = np.zeros(len(p)); j[0] = 0.3 * s8cl(p) / p[0]; j[1] = s8cl(p) / p[1]
    return float(np.sqrt(j @ cov[:len(p), :len(p)] @ j))


fsky = float(sys.argv[1])
SEL = (sys.argv[2] if len(sys.argv) > 2 else "S0,S1,S2").split(",")
KINDS = (sys.argv[3] if len(sys.argv) > 3 else "naive,eff,effexp").split(",")
t0 = time.time()
out = {}
for sk in SEL:
    lab, names = SETUPS[sk]
    truth = np.array([PAR[n][0] for n in names])
    # (a) statistical error: Fisher of the correct (blended) model at the truth
    FB = Fit(names, "blended", fsky)
    covT = np.linalg.inv(FB.fisher(truth)); sigT = np.sqrt(np.diag(covT)); sigT_s8cl = s8cl_sig(truth, covT)
    print(f"\n[f_sky={fsky}] {lab}\n   statistical error (blended Fisher at truth): sigma = {np.round(sigT, 4)}, "
          f"sigma(S8cl) = {sigT_s8cl:.4f}; blended closure -2lnL(truth) = {FB.m2l(truth):.2e}", flush=True)
    for kind in KINDS:
        nm = names + CONF[kind]
        F = Fit(nm, kind, fsky)
        starts = [list(truth) + list(c) for c in CSTARTS[kind]]
        found = F.minimise(starts, polish=kind.startswith("eff"))
        fbest, p = found[0]
        cash_p = cc.cash(F.D, F.mu(p))
        covO = np.linalg.inv(F.fisher(p)); sigO = np.sqrt(np.diag(covO))
        bT = (p[:len(names)] - truth) / sigT; bO = (p - np.array([PAR[n][0] for n in nm])) / sigO
        sb = s8cl(p) - s8cl(truth)
        # (c) profile at the truth cosmology
        Ft = Fit(nm, kind, fsky, fixed=dict(Om0=truth[0], sigma8=truth[1]))
        if Ft.free:
            st = [[x for n, x in zip(nm, s0) if n not in ("Om0", "sigma8")] for s0 in starts]
            ft = Ft.minimise(st, polish=kind.startswith("eff"))[0][0]
        else:
            ft = Ft.m2l(np.zeros(0))
        dtr = ft - fbest
        nsig2d = np.sqrt(chi2d.isf(chi2d.sf(max(dtr, 0), 2), 1))
        dof = F.D.size - sum(PAR[n][2] is None for n in nm)
        crit = chi2d.isf(0.05, dof)
        power = ncx2.sf(crit, dof, max(cash_p, 1e-9)); pmed = chi2d.sf(ncx2.median(dof, max(cash_p, 1e-9)), dof)
        key = f"fsky{fsky}|{sk}|{kind}"
        out[key] = dict(free=nm, best=p.tolist(), m2l_tot=fbest, m2l_poisson=cash_p, sigma_true=sigT.tolist(),
                        bias_true=bT.tolist(), sigma_own=sigO.tolist(), bias_own=bO.tolist(),
                        S8cl_bias_true=sb / sigT_s8cl, S8cl_bias_own=sb / s8cl_sig(p, covO),
                        dm2l_truth=dtr, nsig2d_truth=float(nsig2d), gof_power=float(power), gof_pmed=float(pmed),
                        minima=[(float(f), q.tolist()) for f, q in found])
        print(f"   fit = {kind}{' (+4 free confusion params, no prior)' if kind.startswith('eff') else ''} ({time.time()-t0:.0f}s)")
        print(f"      free: {nm}")
        print(f"      best fit: {np.round(p, 4)}")
        print(f"      -2lnL_min total = {fbest:.3f} (Poisson part {cash_p:.3f}, prior part {fbest-cash_p:.3f})")
        print(f"      bias/sigma_true (Om0, sigma8, ...): {np.round(bT, 2)} | S8cl bias {sb/sigT_s8cl:+.2f} sigma_true")
        print(f"      sigma_own (Fisher at own best fit): {np.round(sigO, 4)}")
        print(f"      bias/sigma_own: {np.round(bO, 2)} | S8cl bias {sb/s8cl_sig(p, covO):+.2f} sigma_own")
        print(f"      truth cosmology (Om0, sigma8 fixed, rest profiled): Delta(-2lnL) = {dtr:.3f} -> {nsig2d:.2f} sigma (2 dof)")
        print(f"      GOF for a noisy realisation: dof {dof}, lambda {cash_p:.1f}: power to reject at p<0.05 = {power:.2f}, median p = {pmed:.3f}")
        if len(found) > 1:
            for f_, q_ in found:
                bq = (q_[:len(names)] - truth) / sigT
                print(f"         local min: -2lnL {f_:.3f}  (Om0, s8) bias/sigma_true ({bq[0]:+.2f}, {bq[1]:+.2f})  conf {np.round(q_[len(names):], 3)}")
        if kind.startswith("eff"):
            lt = np.log(np.array([0.25, 1, 2, 5, 10, 20, 50]) / 2)
            c4 = p[len(names):]
            v = np.maximum(1 + c4[2] + c4[3] * lt, 0.05) if kind == "eff" else np.exp(c4[2] + c4[3] * lt)
            print(f"      kernel variance at th=0.25,1,2,5,10,20,50': {np.round(v, 3)}")
        if kind == "naive" and sk == "S2":
            # (d) 1D profile likelihoods in sigma8 and Om0 around the naive best fit
            for j, pn in ((1, "sigma8"), (0, "Om0")):
                grid = p[j] + sigO[j] * np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
                prof = []
                for gv in grid:
                    Fp = Fit(nm, kind, fsky, fixed={pn: gv})
                    s0 = [x for n, x in zip(nm, p) if n != pn]
                    prof.append(Fp.lm(s0)[1])
                prof = np.array(prof) - fbest
                # 1-sigma interval from Delta = 1 (interpolate on each side)
                lo = np.interp(1.0, prof[:6][::-1], grid[:6][::-1]); hi = np.interp(1.0, prof[5:], grid[5:])
                sp = 0.5 * (hi - lo)
                tv = truth[j]
                dtrue = np.interp(tv, grid, prof) if grid[0] <= tv <= grid[-1] else np.nan
                print(f"      profile {pn}: Delta(-2lnL) at {np.round((grid-p[j])/sigO[j],1)} sigma_own = {np.round(prof, 2)}")
                print(f"         1-sigma interval [{lo:.4f}, {hi:.4f}] (half-width {sp:.4f}, lower {p[j]-lo:.4f}, upper {hi-p[j]:.4f});"
                      f" bias/half-width {(p[j]-tv)/sp:+.2f}; Delta(-2lnL) at truth {dtrue:.2f} -> {np.sign(p[j]-tv)*np.sqrt(max(dtrue,0)):+.2f} sigma (profile)")
                out[key][f"profile_{pn}"] = dict(grid=grid.tolist(), d=prof.tolist(), half=sp, at_truth=float(dtrue))
        sys.stdout.flush()
json.dump(out, open(f"{cc.OUT}/cosmo_bias_v2{'_prior'+os.environ['PRIOR_OMB'] if os.environ.get('PRIOR_OMB') else ''}_fsky{fsky}_{'_'.join(SEL)}_{'_'.join(KINDS)}.json", "w"), indent=1)
print(f"\ndone {time.time()-t0:.0f}s")
