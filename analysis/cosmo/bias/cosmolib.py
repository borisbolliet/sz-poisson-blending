"""Binned Poisson cluster-count likelihood in (z, q_obs) for the cosmology-bias test.

Parameters (vector order PNAMES): Om, s8, omb = (1-b), alpha (Y-M slope), slnY (lognormal scatter).
Fixed: H0, Ob0, ns (colossus planck18), logA=-0.19, beta_E=2/3, M_pivot=6e14 Msun, beam 1.4', noise 4e-6 arcmin,
no h70 factor, Arnaud profile -> sigma_Y(theta500) from szsim.

Naive (single-cluster) model, as in szsim.SZModel.p_det:
    q0 = Y500_med(M,z) / sigma_Y(theta500),  q = q0 exp(slnY x), x ~ N(0,1),  q_obs = q + N(0,1)
    mu[zb, qb] = fsky * int dz dlnM d2N/dzdlnM(Om,s8) P(q_obs in qb | q0, slnY)
The cluster z is the true z of the halo (spectroscopic/photometric z of the counterpart).
"""
import os
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
DC = f"{ROOT}/data_cosmo"
ARCMIN = np.pi / 180.0 / 60.0
PNAMES = ["Om", "s8", "omb", "alpha", "slnY"]
FID = np.array([0.3111, 0.8102, 0.8, 1.79, 0.173])
BOUNDS = {"Om": (0.2005, 0.4495), "s8": (0.6205, 1.0195), "omb": (0.3, 1.5), "alpha": (1.0, 2.6), "slnY": (1e-3, 1.0)}
STEP = np.array([4e-4, 4e-4, 1e-3, 2e-3, 1e-3])       # finite-difference steps for the Jacobian
LABEL = {"Om": r"$\Omega_m$", "s8": r"$\sigma_8$", "omb": r"$1-b$", "alpha": r"$\alpha$", "slnY": r"$\sigma_{\ln Y}$",
         "S8": r"$S_8$"}


class HMFGrid:
    """Tensor cubic B-spline interpolation of ln d2N/dz/dlnM over (Om, s8); 1D spline of ln E, ln D_A over Om."""

    def __init__(self, path=f"{DC}/hmf_grid.npz"):
        d = np.load(path)
        self.OM, self.S8, self.Z, self.LNM = d["OM"], d["S8"], d["Z"], d["LNM"]
        self.rhoc0 = float(d["rhoc0"]); self.s8fid = float(d["s8fid"])
        lnN = d["lnN"]                                   # (nOm, nS8, nz, nM)
        nO, nS = lnN.shape[:2]; rest = lnN.shape[2:]
        sp = make_interp_spline(self.OM, lnN.reshape(nO, -1), k=3, axis=0)
        self.tO, c1 = sp.t, sp.c                         # (nO, nS*nz*nM)
        sp2 = make_interp_spline(self.S8, c1.reshape(nO, nS, -1), k=3, axis=1)
        self.tS = sp2.t
        c2 = np.moveaxis(sp2.c, 0, 1)          # scipy BSpline stores the spline axis (s8) first: -> (nO, nS, ...)
        self.C = np.ascontiguousarray(c2).reshape(nO, nS, *rest)
        self.spE = make_interp_spline(self.OM, d["lnE"], k=3, axis=0)
        self.spDA = make_interp_spline(self.OM, d["lnDA"], k=3, axis=0)

    @staticmethod
    def _basis(x, t, n):
        b = BSpline.design_matrix(np.atleast_1d(x), t, 3).toarray()[0]
        nz = np.flatnonzero(b)
        return nz, b[nz]

    def in_range(self, om, s8):
        return self.OM[0] <= om <= self.OM[-1] and self.S8[0] <= s8 <= self.S8[-1]

    def eval(self, om, s8):
        io, bo = self._basis(om, self.tO, len(self.OM))
        is_, bs = self._basis(s8, self.tS, len(self.S8))
        Csub = self.C[io][:, is_]                                       # (4,4,nz,nM)
        lnN = np.einsum("a,b,abzm->zm", bo, bs, Csub)
        return lnN, self.spE(om), self.spDA(om)


def trap_weights(x, lo, hi):
    """Trapezoid weights on nodes x (regular) for the integral over [lo, hi] (lo, hi must be nodes)."""
    w = np.zeros_like(x)
    i0 = int(np.argmin(np.abs(x - lo))); i1 = int(np.argmin(np.abs(x - hi)))
    dx = x[1] - x[0]
    w[i0:i1 + 1] = dx; w[i0] *= 0.5; w[i1] *= 0.5
    return w


def default_bins(qcut):
    zed = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    qed = np.concatenate([np.geomspace(qcut, 100.0, 9), [np.inf]])
    return zed, qed


class SigmaY:
    def __init__(self):
        from szsim import signal as sg
        m = sg.SZModel()
        m.sigma_Y(1.0)
        self.tg = m._sigY_tab[0]; self.sp = m._sigY_spline

    def ln(self, theta):
        return self.sp(np.log(np.clip(theta, self.tg[0], self.tg[-1])))


class NaiveModel:
    def __init__(self, qcut=5.0, zedges=None, qedges=None, grid=None, ngh=40):
        zd, qd = default_bins(qcut)
        self.zed = zd if zedges is None else zedges
        self.qed = qd if qedges is None else qedges
        self.qcut = qcut
        self.G = grid or HMFGrid()
        self.Z, self.LNM = self.G.Z, self.G.LNM
        self.M = np.exp(self.LNM)
        self.Wz = np.stack([trap_weights(self.Z, a, b) for a, b in zip(self.zed[:-1], self.zed[1:])])   # (nzb, nz)
        dl = self.LNM[1] - self.LNM[0]
        self.wM = np.full(self.LNM.size, dl); self.wM[0] *= 0.5; self.wM[-1] *= 0.5
        self.sigY = SigmaY()
        self.gx, gw = np.polynomial.hermite_e.hermegauss(ngh); self.gw = gw / gw.sum()
        self.lq = np.linspace(np.log(1e-3), np.log(2e4), 3001)
        self._Pcache = {}
        self.nzb, self.nqb = len(self.zed) - 1, len(self.qed) - 1

    # ---- P(q_obs in bin | q0, slnY) tabulated on ln q0
    def Ptab(self, slnY):
        key = round(float(slnY), 12)
        if key not in self._Pcache:
            q = np.exp(self.lq)[:, None] * np.exp(slnY * self.gx)[None, :]           # (nl, ngh)
            cdf = np.stack([ndtr(q - e) if np.isfinite(e) else np.zeros_like(q) for e in self.qed], -1)  # P(q_obs>e)
            pb = (cdf[..., :-1] - cdf[..., 1:])                                       # (nl, ngh, nqb)
            if len(self._Pcache) > 50:
                self._Pcache.clear()
            self._Pcache[key] = np.einsum("lgj,g->lj", pb, self.gw)
        return self._Pcache[key]

    def nodes(self, th):
        """Per-(z, M) node: ln d2N/dz/dlnM, theta500 [arcmin], ln q0 (median-Y mean SNR)."""
        om, s8, omb, alpha, slnY = th
        lnN, lnE, lnDA = self.G.eval(om, s8)
        lnE = lnE[:, None]; lnDA = lnDA[:, None]; lnM = self.LNM[None, :]
        lnY = (np.log(1e-4) - 0.19 * np.log(10) + (2.0 / 3.0) * lnE + alpha * (np.log(omb) + lnM - np.log(6e14))
               - 2 * lnDA - 2 * np.log(ARCMIN))
        lnR = (np.log(3.0) + lnM - np.log(4 * np.pi * 500 * self.G.rhoc0) - 2 * lnE) / 3.0
        th500 = np.exp(lnR - lnDA) / ARCMIN
        lnq0 = lnY - self.sigY.ln(th500)
        return lnN, th500, lnq0

    def counts(self, th, fsky=1.0):
        lnN, th500, lnq0 = self.nodes(th)
        P = self.Ptab(th[4])
        f = (lnq0 - self.lq[0]) / (self.lq[1] - self.lq[0])
        f = np.clip(f, 0, len(self.lq) - 1.000001); i0 = f.astype(int); w = f - i0
        Pn = P[i0] * (1 - w)[..., None] + P[i0 + 1] * w[..., None]                    # (nz, nM, nqb)
        Wn = np.exp(lnN) * self.wM[None, :]
        return fsky * np.einsum("bz,zm,zmj->bj", self.Wz, Wn, Pn)

    def counts_kernel(self, th, kernel, fsky=1.0):
        """Counts with a generic detection kernel K(theta500, q_true) -> (..., nqb) (q_true includes scatter)."""
        lnN, th500, lnq0 = self.nodes(th)
        Wn = np.exp(lnN) * self.wM[None, :]
        out = np.zeros((self.nzb, self.nqb))
        for g, wg in zip(self.gx, self.gw):
            q = np.exp(lnq0 + th[4] * g)
            out += wg * np.einsum("bz,zm,zmj->bj", self.Wz, Wn, kernel(th500, q))
        return fsky * out


# ------------------------------------------------------------------ likelihood tools
class Fitter:
    """Poisson likelihood lnL = sum d ln mu - mu (+ Gaussian priors) over the free parameters."""

    def __init__(self, model, data, free, fsky=1.0, priors=None, mask=None):
        self.m, self.d, self.fsky = model, np.asarray(data), fsky
        self.free = [PNAMES.index(p) for p in free]
        self.priors = priors or {}          # name -> (mean, sigma)
        self.mask = np.ones_like(self.d, bool) if mask is None else mask

    def full(self, x):
        th = FID.copy(); th[self.free] = x
        return th

    def mu(self, x):
        return self.m.counts(self.full(x), self.fsky)

    def lnprior(self, x):
        th = self.full(x); s = 0.0
        for p, (m_, s_) in self.priors.items():
            s += -0.5 * ((th[PNAMES.index(p)] - m_) / s_) ** 2
        return s

    def lnL(self, x):
        th = self.full(x)
        if not self.m.G.in_range(th[0], th[1]) or th[2] <= 0 or th[4] <= 0:
            return -np.inf
        mu = self.mu(x)[self.mask]; d = self.d[self.mask]
        # relative to the saturated model (= -deviance/2): same maximum, no loss of float precision
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0, d * np.log(mu / np.where(d > 0, d, 1.0)), 0.0)
        return float(np.sum(t - mu + d)) + self.lnprior(x)

    def jac(self, x):
        J = []
        for k, i in enumerate(self.free):
            e = np.zeros(len(x)); e[k] = STEP[i]
            J.append((self.mu(x + e) - self.mu(x - e)) / (2 * STEP[i]))
        return np.stack(J, -1)[self.mask]                    # (nbin, nfree)

    def fisher(self, x):
        mu = self.mu(x)[self.mask]; J = self.jac(x)
        F = J.T @ (J / mu[:, None])
        for p, (m_, s_) in self.priors.items():
            if PNAMES.index(p) in self.free:
                k = self.free.index(PNAMES.index(p)); F[k, k] += 1 / s_ ** 2
        return F

    def fit(self, x0=None, tol=1e-2, maxit=60, verbose=False, polish=False):
        """Fisher scoring with step halving (Poisson: exact Newton for the expected Hessian)."""
        x = FID[self.free].copy() if x0 is None else np.array(x0, float)
        L = self.lnL(x)
        for it in range(maxit):
            mu = self.mu(x)[self.mask]; J = self.jac(x); d = self.d[self.mask]
            g = J.T @ (d / mu - 1)
            F = J.T @ (J / mu[:, None])
            th = self.full(x)
            for p, (m_, s_) in self.priors.items():
                if PNAMES.index(p) in self.free:
                    k = self.free.index(PNAMES.index(p)); F[k, k] += 1 / s_ ** 2; g[k] -= (th[PNAMES.index(p)] - m_) / s_ ** 2
            dx = np.linalg.solve(F, g)
            lam = 1.0
            while lam > 1e-4:
                xn = x + lam * dx; Ln = self.lnL(xn)
                if Ln >= L - 1e-10:
                    break
                lam *= 0.5
            x, Lold, L = xn, L, Ln
            dec = np.sqrt(dx @ F @ dx)
            if verbose:
                print(f"   it {it}: x={x} lnL={L:.6f} step(sigma)={dec:.2e}")
            if dec < tol:
                break
        if polish:      # bounded L-BFGS-B from the Fisher-scoring solution AND from the truth; keep the best
            from scipy.optimize import minimize
            sc = np.sqrt(np.diag(np.linalg.inv(self.fisher(FID[self.free]))))
            lo = np.array([BOUNDS[PNAMES[i]][0] for i in self.free]); hi = np.array([BOUNDS[PNAMES[i]][1] for i in self.free])
            best_x, best_L, L_fs = x, L, L
            for start in (np.clip(x, lo, hi), FID[self.free]):
                ref = start.copy()
                r = minimize(lambda u: -self.lnL(ref + u * sc), np.zeros(len(x)), method="L-BFGS-B",
                             bounds=list(zip((lo - ref) / sc, (hi - ref) / sc)),
                             options=dict(eps=1e-5, ftol=1e-13, gtol=1e-7, maxiter=500))
                if -r.fun > best_L:
                    best_x, best_L = ref + r.x * sc, -r.fun
            self.polish_gain = float(best_L - L_fs); self.polish_shift = float(np.abs((best_x - x) / sc).max())
            x, L = best_x, best_L
            self.at_bound = [PNAMES[i] for k, i in enumerate(self.free) if x[k] <= lo[k] + 1e-3 * sc[k] or x[k] >= hi[k] - 1e-3 * sc[k]]
        return x, L, it

    def deviance(self, x):
        mu = self.mu(x)[self.mask]; d = self.d[self.mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0, d * np.log(d / mu), 0.0)
        return float(2 * np.sum(mu - d + t))


def S8_of(om, s8):
    return s8 * np.sqrt(om / 0.3)


def derived_sigma(F, free):
    """Marginal sigmas of free params + S8 (if Om, s8 free) from a Fisher matrix."""
    C = np.linalg.inv(F)
    out = {PNAMES[i]: np.sqrt(C[k, k]) for k, i in enumerate(free)}
    if 0 in free and 1 in free:
        a, b = free.index(0), free.index(1)
        om, s8 = FID[0], FID[1]
        g = np.zeros(len(free)); g[a] = 0.5 * S8_of(om, s8) / om; g[b] = S8_of(om, s8) / s8
        out["S8"] = np.sqrt(g @ C @ g)
    return out, C
