"""Step 2 library: blended detection kernels in cosmology bins, MMF count models, truth vectors and fits.

Everything here is NEW code; it reuses (imports, never modifies) step 1's pdlib1 (signed compound-Poisson P(D)),
blendlib (rho lookup, neighbour population), szsim (SZModel, MMFModel, HMF table) and the cosmology-bias code
cosmolib / cosmolib2 (HMF grid, smooth (z, q_obs) count model, Poisson fitter).

Conventions (as step 1 / s09 and Q3 / c03):
  q_A   = Y_A / sigma_Y(theta_A)            true SNR of the primary in units of the filter noise (incl. Y scatter)
  delta = sum_B rho_AB Y_B / sigma_A        compound-Poisson neighbour sum (all halos M500 > 1e13, fiducial sky)
  q_b   = q_A + n + delta - mu              n ~ N(0,1); mu = untruncated mean of delta (c = 1; ~0 for the MMF)
  A is counted only if it is dominant: every neighbour has x_B < fdom q_A (truncated compound Poisson, P_dom).
  Kernel K(theta_A, q_A; e) = P(dominant and q_b > e).
'map-sigma' variants: q_obs = Yhat / sigma_eff with sigma_eff = r(theta) sigma_Y fixed on the fiducial sky,
  so  K'(theta, q'; e) = P(dominant at q_A = r q', q_A + n + delta - mu > r e),   q' = Y / sigma_eff,
  and the naive model is q' + N(0,1).
"""
import os, sys
import numpy as np
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
for _p in (f"{ROOT}/analysis/realistic/step1", f"{ROOT}/analysis/blending", f"{ROOT}/analysis/cosmo/bias", ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pdlib1 as pl                                # noqa: E402
import blendlib as bl                              # noqa: E402
import cosmolib as CL                              # noqa: E402
import cosmolib2 as C2                             # noqa: E402
from szsim import signal as sg, cosmo_hmf as ch, instrument as ins   # noqa: E402

DR = f"{ROOT}/data_real"
D2 = f"{ROOT}/data_real/step2"
FIG2 = f"{ROOT}/figures/realistic"
CLS = (0, 1, 2)                                    # neighbour classes: all halos M500 > 1e13
RKEYS = ("Wp", "WXp", "WX2p", "Wn", "WXn", "WX2n", "capp", "capn")
# cosmology q_obs bin edges: those of cosmolib.default_bins(5) and (6), plus 4 and 7 (= c03's EDGES)
EDGES = np.unique(np.concatenate([CL.default_bins(5.0)[1][:-1], CL.default_bins(6.0)[1][:-1], [4.0, 7.0]]))
# q_A grid of c03 / c11 (production) and a 2x finer grid (convergence test)
QA = np.concatenate([np.arange(0.05, 30.0, 0.05), np.geomspace(30.0, 3000.0, 466)])
QA_FINE = np.concatenate([np.arange(0.025, 30.0, 0.025), np.geomspace(30.0, 3000.0, 931)])
# low-pressure population: (1-b) = 0.6 instead of 0.8 for every halo -> Y x (0.6/0.8)^alpha
OMB_LOW = 0.6
C_LOW = (OMB_LOW / 0.8) ** 1.79
FID = CL.FID.copy()
FID_LOW = np.array([0.3111, 0.8102, OMB_LOW, 1.79, 0.173])
S8_FID = CL.S8_of(FID[0], FID[1])


def s8_from_S8(S8, om=FID[0]):
    return S8 / np.sqrt(om / 0.3)


# ============================================================================ rates
def load_s09_rates(run):
    """Step-1 (s09) rate histograms per theta_A node (fiducial sky, M500 > 1e13), plus th, sigA."""
    d = np.load(f"{DR}/pd_tables_{run}.npz")
    return d["th"], d["sigA"], rates_from_npz(d)


def rates_from_npz(d):
    """per-node rate dicts from an npz holding R_<key> arrays (nodes, 3, nbin).  [memory fix after p03's first run:
    each d[...] access re-reads the whole array, and d[...][i] keeps it alive, so every node used to pin 8 full arrays
    (p03 parent peak footprint 12.6 GB); now each array is read once and the nodes share it]"""
    A = {k: np.array(d[f"R_{k}"]) for k in RKEYS}
    return [{k: A[k][i] for k in RKEYS} for i in range(A["Wp"].shape[0])]


def load_prefix_white_rates():
    """data/blending/analytic.npz (b03; pre-fix rho_at kernel, unsigned bins) in the signed-rate format."""
    d = np.load(f"{ROOT}/data/blending/analytic.npz")
    A = {k: np.array(d[k]) for k in ("W", "WX", "WX2", "CAP", "CAPX")}          # read each array once (memory fix)
    R = []
    for i in range(A["W"].shape[0]):
        z = np.zeros_like(A["W"][i])
        capp = np.stack([A["CAP"][i], A["CAPX"][i], np.zeros(3)], -1)        # x^2 of the cap not stored (unused)
        R.append(dict(Wp=A["W"][i], WXp=A["WX"][i], WX2p=A["WX2"][i], Wn=z, WXn=z.copy(), WX2n=z.copy(),
                      capp=capp, capn=np.zeros((3, 3))))
    return d["th"], d["sigA"], R, d["mu_exact"].sum(1)


def scale_rates(o, c):
    """Rates of the same population with every Y_B multiplied by c (x -> c x): W same, WX x c, WX2 x c^2."""
    out = {}
    for k in ("Wp", "Wn"):
        out[k] = o[k].copy()
    for k in ("WXp", "WXn"):
        out[k] = o[k] * c
    for k in ("WX2p", "WX2n"):
        out[k] = o[k] * c * c
    for k in ("capp", "capn"):
        a = o[k].copy(); a[:, 1] *= c; a[:, 2] *= c * c; out[k] = a
    return out


def moments(R):
    mu = np.array([pl.moments(o, CLS)[0] for o in R]); var = np.array([pl.moments(o, CLS)[1] for o in R])
    return mu, var


def fixed_extra(th, var_hist, var_pars, lo=0.25, hi=80.0):
    """Gaussian variance beyond the kernel support = Parseval - histogram (>= 0), set to 0 outside theta_A = lo..hi
    (the arbiter's fix of s06b/s09: s05 tabulates Parseval only for 0.25-80')."""
    e = np.clip(var_pars - var_hist, 0, None)
    e[(th < lo) | (th > hi)] = 0.0
    return e


# ============================================================================ populations
def hmf_factor_s8(s8_new, z, lnM):
    """d2N(sigma8 = s8_new) / d2N(fiducial) at fixed Om (colossus planck18, Tinker08 500c), exactly as c01:
    sigma(M, z) -> s sigma(M, z), so the ratio is f_T08(s sigma; z) / f_T08(sigma; z)."""
    from colossus.lss import mass_function as mf, peaks
    c = ch.get_cosmology()
    s = s8_new / c.sigma8
    R = peaks.lagrangianR(np.exp(lnM) * c.h)
    F = np.empty((len(z), len(lnM)))
    for iz, zz in enumerate(z):
        sig = c.sigma(R, float(zz))
        F[iz] = mf.modelTinker08(sig * s, float(zz), "500c") / mf.modelTinker08(sig, float(zz), "500c")
    return F


def neighbour_population_f(F=None, zsub=1, msub=6, ngh=16, model=None, Yscale=1.0):
    """Copy of blendlib.neighbour_population with d2N multiplied by F (subsampled (z, lnM) grid) and Y x Yscale.
    F = None, Yscale = 1 reproduces blendlib.neighbour_population exactly."""
    m = model or sg.SZModel()
    T = ch.default_table()
    z = T.z[::zsub]; lnM = T.lnM[::msub]
    d2N = T.d2N[::zsub, ::msub] * (1.0 if F is None else F)
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= 0.5; wz[-1] *= 0.5
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    Z, LM = np.meshgrid(z, lnM, indexing="ij")
    M = np.exp(LM)
    th = m.theta500(M, Z)
    Ym = m.Y500_med(M, Z) * Yscale
    out = [[], [], [], []]
    for c in range(len(bl.MASS_EDGES) - 1):
        wM = T._weights(lnM, np.log(bl.MASS_EDGES[c]), np.log(bl.MASS_EDGES[c + 1]))
        W = d2N * wz[:, None] * wM[None, :] / bl.ARCMIN2_FULLSKY
        sel = (W > 0).ravel()
        out[0].append(np.repeat(th.ravel()[sel], ngh))
        out[1].append((Ym.ravel()[sel][:, None] * np.exp(m.sigma_lnY * x)[None, :]).ravel())
        out[2].append(np.full(sel.sum() * ngh, c, np.int8))
        out[3].append((W.ravel()[sel][:, None] * wg[None, :]).ravel())
    return tuple(np.concatenate(o) for o in out)


def population_grid(zsub=1, msub=6):
    T = ch.default_table()
    return T.z[::zsub], T.lnM[::msub]


def parseval_var(mmf, th_nodes, pop, mask=None):
    """Confusion variance / sigma_A^2 (Parseval, s05 method) at th_nodes for the population (or a subset)."""
    thB, YB, cB, wB = pop
    if mask is not None:
        thB, YB, wB = thB[mask], YB[mask], wB[mask]
    Cyy = ins.poisson_Cyy(mmf.inst.ell, thB, YB, wB)
    return mmf.confusion_var(th_nodes, Cyy) / mmf.sigma_Y_exact(th_nodes) ** 2, Cyy


# ============================================================================ kernels
_PD = {}


def pds():
    if not _PD:
        _PD["small"] = pl.PDs(N=2 ** 18, dd=0.001, x_small=0.002, off=60.0)
        _PD["big"] = pl.PDs(N=2 ** 19, dd=0.002, x_small=0.004, off=60.0)
    return _PD["small"], _PD["big"]


def kernel_node(o, mu, extra, qa, thr, fdom=1.0):
    """K[k, j] = P(dominant and q_A + n + delta - mu > thr[j]) at q_A = qa[k] (signed P(D), convention c = 1).
    The unit Gaussian noise and the Gaussian extra variance are added inside the FFT (as c03); dominance:
    positive neighbour contributions x >= fdom q_A excluded (P_dom = exp(-Lambda(x >= fdom q_A)), x >= 500 always).
    Returns K (nq, ne) and P_dom (nq)."""
    ps, pb = pds()
    K = np.zeros((len(qa), len(thr))); Pd = np.zeros(len(qa))
    thr = np.asarray(thr, float)
    for k, q in enumerate(qa):
        xcut = min(fdom * q, pl.X_CAP)
        W, X, X2, lam_above = pl.signed_arrays(o, CLS, xcut)
        Pdom = np.exp(-lam_above)
        pd_ = ps if xcut < 20 else pb
        p = pd_.pdf(W, X, X2, extra_var=extra + 1.0)
        p /= p.sum()
        cdf = np.cumsum(p)
        t = thr - q + mu
        K[k] = Pdom * (1.0 - np.interp(t, pd_.grid + 0.5 * pd_.dd, cdf, left=0.0, right=1.0))
        Pd[k] = Pdom
    return K, Pd


def robust_halfwidth(o, mu, extra, pd_un=None):
    """(84th - 16th percentile)/2 of n + delta - mu, untruncated P(D) (as s06b), and the 2.3/16/50/84/97.7% quantiles."""
    pd_un = pd_un or pl.PDs(N=2 ** 20, dd=0.001, x_small=0.002, off=60.0)
    QL = [0.02275, 0.15866, 0.5, 0.84134, 0.97725]
    W, X, X2, lam_cap = pl.signed_arrays(o, CLS)
    p = pd_un.pdf(W, X, X2, extra_var=extra + 1.0)
    cdf = np.cumsum(p) * np.exp(-lam_cap)
    qs = np.interp(QL, cdf, pd_un.grid + 0.5 * pd_un.dd) - mu
    return 0.5 * (qs[3] - qs[1]), qs


# ============================================================================ count models
class LnSigma:
    """ln sigma(theta500) [arcmin^2] for SmoothModel.nodes: the step-1 sigma_Y spline of an SZModel/MMFModel
    (log-log cubic spline on 300 nodes 1e-3..3e3'), optionally times r(theta) (map-sigma variants), r linear in
    ln theta between nodes and constant beyond them."""

    def __init__(self, model, r_th=None, r_val=None):
        model.sigma_Y(1.0)
        self.tg = model._sigY_tab[0]; self.sp = model._sigY_spline
        self.lth = None if r_th is None else np.log(np.asarray(r_th, float))
        self.lr = None if r_val is None else np.log(np.asarray(r_val, float))

    def ln(self, theta):
        theta = np.asarray(theta, float)
        v = self.sp(np.log(np.clip(theta, self.tg[0], self.tg[-1])))
        if self.lr is not None:
            v = v + np.interp(np.log(theta), self.lth, self.lr)
        return v


_MODELS = {}


def sz_model(kind):
    """kind: 'white' (SZModel, the old filter) or 'baseline' / 'goal' (MMFModel, SO LAT, lmin = 100)."""
    if kind not in _MODELS:
        _MODELS[kind] = sg.SZModel() if kind == "white" else pl.make_model(("so", kind, 100.0))
    return _MODELS[kind]


class SmoothModelS(C2.SmoothModel):
    """cosmolib2.SmoothModel with a different sigma(theta500) (MMF, or sigma_eff for the map-sigma variants)."""

    def __init__(self, qcut, lnsig, grid=None, **kw):
        super().__init__(qcut=qcut, grid=grid, **kw)
        self.sigY = lnsig


class FixedCorrModel:
    """r02's 'recompute once' model: M(theta) = C_fid(z, q) x N_naive(theta)."""

    def __init__(self, base, Cfid):
        self.base, self.C = base, Cfid
        self.G = base.G

    def counts(self, th, fsky=1.0):
        return self.C * self.base.counts(th, fsky)


def bin_tables(M, K, QAg, E, slnY, qmin=None):
    """c11's table builder: K (NT, nQA, NE) = P(q_obs > e) on the q_A grid -> P(q_obs bin | ln q_true) on the fine
    ln q grid of the SmoothModel M, convolved with the lognormal scatter slnY.  Returns (NT, nlq, nqb).
    qmin: zero the kernel for q_true < qmin before differencing (excludes those primaries)."""
    sel = np.array([int(np.argmin(np.abs(E - e))) for e in M.qed[:-1]])
    assert np.allclose(E[sel], M.qed[:-1])
    q = np.clip(np.exp(M.lq), QAg[0], QAg[-1])
    keep = None if qmin is None else (np.exp(M.lq) >= qmin)
    out = []
    for i in range(K.shape[0]):
        C = np.stack([np.interp(q, QAg, K[i][:, j]) for j in sel], 0)
        if keep is not None:
            C = C * keep[None, :]
        C = np.concatenate([C, np.zeros((1, C.shape[1]))], 0)
        out.append(C2.smooth_lnq(C[:-1] - C[1:], slnY, M.h).T)
    return np.stack(out)


def naive_kernel(QAg, E, NT):
    return np.broadcast_to(ndtr(QAg[:, None] - E[None, :]), (NT, len(QAg), len(E)))


def delta_counts(M, Kb, th, QAg, E, truth, qmin=None):
    """c11 route: Nb, Nn (same interpolation) and N0 = M.counts(truth); full sky."""
    lth = np.log(th)
    assert np.allclose(np.diff(lth), lth[1] - lth[0])
    Pb = bin_tables(M, Kb, QAg, E, truth[4], qmin)
    Pn1 = bin_tables(M, naive_kernel(QAg, E, 1), QAg, E, truth[4], qmin)
    Pn = np.broadcast_to(Pn1, (len(th),) + Pn1.shape[1:])
    Nb = M.counts_table(truth, Pb, lth); Nn = M.counts_table(truth, Pn, lth)
    return Nb, Nn, M.counts(truth)


# ============================================================================ fits
CONFIGS = {
    "a": (["Om", "s8"], None),
    "b": (["Om", "s8", "omb"], None),
    "c": (["Om", "s8", "omb", "slnY"], None),
    "d": (["Om", "s8", "omb", "alpha", "slnY"], 0.04),
    "d3": (["Om", "s8", "omb", "alpha", "slnY"], 0.03),
    "d5": (["Om", "s8", "omb", "alpha", "slnY"], 0.05),
    "e": (["Om", "s8", "omb", "alpha", "slnY"], None),
}


def config(cn, truth):
    """free parameters and priors of config cn; the (1-b) prior is centred on the truth with a relative width
    (4% of 0.8 = 0.032 as in c12; 4% of 0.6 = 0.024 for the low-pressure population)."""
    free, rel = CONFIGS[cn]
    pri = {} if rel is None else {"omb": (float(truth[2]), rel * float(truth[2]))}
    return free, pri


class Fitter3(C2.Fitter2):
    """cosmolib2.Fitter2 with a configurable truth (fixed parameters, start, polish start) instead of cosmolib.FID."""

    def __init__(self, model, data, free, fsky=1.0, priors=None, mask=None, truth=None, step_mult=1.0):
        super().__init__(model, data, free, fsky=fsky, priors=priors, mask=mask, step_mult=step_mult)
        self.truth = np.array(FID if truth is None else truth, float)

    def full(self, x):
        th = self.truth.copy(); th[self.free] = x
        return th

    def fit(self, x0=None, tol=1e-2, maxit=60, verbose=False, polish=False):
        """identical to cosmolib2.Fitter2.fit with cosmolib.FID replaced by self.truth"""
        from scipy.optimize import minimize
        lo, hi = self._lohi()
        x = self.truth[self.free].copy() if x0 is None else np.array(x0, float)
        x = np.clip(x, lo, hi); L = self.lnL(x); it = 0
        for it in range(maxit):
            mu = self.mu(x)[self.mask]; J = self.jac(x); d = self.d[self.mask]
            g = J.T @ (d / mu - 1); F = J.T @ (J / mu[:, None])
            th = self.full(x)
            for p, (m_, s_) in self.priors.items():
                if CL.PNAMES.index(p) in self.free:
                    k = self.free.index(CL.PNAMES.index(p)); F[k, k] += 1 / s_ ** 2; g[k] -= (th[CL.PNAMES.index(p)] - m_) / s_ ** 2
            dx = np.linalg.lstsq(F, g, rcond=1e-12)[0]
            lam = 1.0
            while lam > 1e-4:
                xn = np.clip(x + lam * dx, lo, hi); Ln = self.lnL(xn)
                if Ln >= L - 1e-10:
                    break
                lam *= 0.5
            dec = np.sqrt(max((xn - x) @ F @ (xn - x), 0.0))
            x, L = xn, max(Ln, L) if Ln >= L - 1e-10 else L
            if verbose:
                print(f"   it {it}: x={x} lnL={L:.6f} step(sigma)={dec:.2e}")
            if dec < tol:
                break
        self.polish_gain = 0.0; self.polish_shift = 0.0; self.at_bound = []
        if polish:
            sc = np.sqrt(np.diag(np.linalg.inv(self.fisher(self.truth[self.free]))))
            best_x, best_L, L_fs = x, L, L
            for start in (x, self.truth[self.free]):
                ref = start.copy()
                r = minimize(lambda u: -self.lnL(ref + u * sc), np.zeros(len(x)), method="L-BFGS-B",
                             bounds=list(zip((lo - ref) / sc, (hi - ref) / sc)),
                             options=dict(eps=1e-4, ftol=1e-13, gtol=1e-7, maxiter=500))
                if -r.fun > best_L:
                    best_x, best_L = ref + r.x * sc, -r.fun
            self.polish_gain = float(best_L - L_fs); self.polish_shift = float(np.abs((best_x - x) / sc).max())
            x, L = best_x, best_L
        self.at_bound = [CL.PNAMES[i] for k, i in enumerate(self.free)
                         if x[k] <= lo[k] + 1e-3 * self.step[i] or x[k] >= hi[k] - 1e-3 * self.step[i]]
        return x, L, it


def derived_sigma3(F, fi, truth):
    """marginal sigmas of the free parameters and of S8 (gradient at the truth)."""
    C = np.linalg.inv(F)
    out = {CL.PNAMES[i]: float(np.sqrt(C[k, k])) for k, i in enumerate(fi)}
    gS = None
    if 0 in fi and 1 in fi:
        a, b = fi.index(0), fi.index(1)
        S8 = CL.S8_of(truth[0], truth[1])
        gS = np.zeros(len(fi)); gS[a] = 0.5 * S8 / truth[0]; gS[b] = S8 / truth[1]
        out["S8"] = float(np.sqrt(gS @ C @ gS))
    return out, C, gS


def run_fit(model, data, cn, truth, fsky, polish=True, x0=None):
    """Asimov ML fit of `model` to `data` (full sky) at fsky; returns a dict of biases (in Fisher sigma at the truth),
    deviance, goodness-of-fit power and optimiser diagnostics."""
    from scipy.stats import chi2, ncx2
    free, pri = config(cn, truth)
    fi = [CL.PNAMES.index(p) for p in free]
    F_ = Fitter3(model, data * fsky, free, fsky=fsky, priors=pri, truth=truth)
    xt = truth[fi]
    Ft = F_.fisher(xt)
    sig, Cov, gS = derived_sigma3(Ft, fi, truth)
    x, L, nit = F_.fit(x0=x0, polish=polish)
    th = F_.full(x)
    dev = F_.deviance(x)
    ndof = data.size - len(free)
    crit = chi2.ppf(0.95, ndof)
    power = float(ncx2.sf(crit, ndof, dev)) if dev > 1e-9 else 0.05
    mu = F_.mu(xt).ravel(); J = F_.jac(xt)
    xlin = np.linalg.solve(Ft, J.T @ (((data * fsky).ravel() - mu) / mu))
    S8t = CL.S8_of(truth[0], truth[1]); S8f = CL.S8_of(th[0], th[1])
    r = dict(config=cn, fsky=fsky, free=free, priors={k: list(v) for k, v in pri.items()}, nit=int(nit), dev=float(dev),
             ndof=int(ndof), power=power, lnL=float(L), polish_gain=float(F_.polish_gain),
             polish_shift=float(F_.polish_shift), at_bound=list(F_.at_bound), theta=[float(v) for v in th],
             truth=[float(v) for v in truth])
    for k, i in enumerate(fi):
        p = CL.PNAMES[i]
        r[f"d_{p}"] = float(th[i] - truth[i]); r[f"s_{p}"] = sig[p]; r[f"b_{p}"] = float((th[i] - truth[i]) / sig[p])
        r[f"lin_{p}"] = float(xlin[k] / sig[p])
    r["d_S8"] = float(S8f - S8t); r["s_S8"] = sig["S8"]; r["b_S8"] = float((S8f - S8t) / sig["S8"])
    r["lin_S8"] = float((gS @ xlin) / sig["S8"])
    return r
