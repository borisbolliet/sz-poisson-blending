"""Converged (smooth) version of the binned (z, q_obs) count model of cosmolib.py.  [repair, Nov 2026]

Bugs fixed relative to cosmolib.NaiveModel (found by verification):
  1. Lognormal Y scatter was integrated with 40 Gauss-Hermite nodes.  For q >~ 10 the integrand Phi(q0 e^{s x} - e)
     is a step of width 1/(q s) << GH node spacing, so P(q_obs in bin | q0) was a staircase in ln q0.  Summed over
     the coarse lnM grid this made mu(theta) jagged at sub-sigma scale along the Om-s8-(1-b)-alpha degeneracy.
     -> Here the scatter is a Gaussian CONVOLUTION in ln q0 on a fine uniform grid (h = 0.002), with cell-integrated
        kernel weights (exact delta function for s -> 0).  P(bin | ln q0) is then smooth on scale max(s, 1/q) >> h.
  2. lnM integration on the 121-node HMF grid (d lnM = 0.058, i.e. 0.10 in ln q) is refined by a factor MSUB
     (cubic spline of ln d2N/dz/dlnM in lnM).
  3. Finite-difference steps for the Jacobian are increased (STEP2) so that they are >> the ln q0 grid spacing; the
     Fisher matrix is checked to be step-independent (c10_smooth_model_check.py).
Everything else (cosmology, Y-M relation, sigma_Y, bins, HMF grid, priors, likelihood) is identical to cosmolib.
"""
import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline

import cosmolib as CL
from cosmolib import PNAMES, FID, BOUNDS, S8_of, derived_sigma, ROOT, DC, LABEL, default_bins  # noqa: F401

STEP2 = np.array([2e-3, 2e-3, 6e-3, 5e-3, 5e-3])   # Om, s8, (1-b), alpha, slnY  (~0.2-0.5 of the 40%-sky sigma)
LQ_LO, LQ_HI, HLQ = np.log(1e-3), np.log(2e4), 0.002


def gauss_cell_kernel(s, h, nsig=8.0):
    """Weights w_k = P(x in cell k) for x ~ N(0, s), cells [(k-1/2)h, (k+1/2)h]; exact delta for s -> 0."""
    K = max(1, int(np.ceil(nsig * max(s, h) / h)))
    k = np.arange(-K, K + 1)
    if s <= 0:
        w = (k == 0).astype(float)
    else:
        w = ndtr((k + 0.5) * h / s) - ndtr((k - 0.5) * h / s)
    return w / w.sum(), K


def smooth_lnq(f, s, h=HLQ):
    """Convolve f(lq) (..., nlq) [uniform grid, spacing h] with N(0, s) in lq; edges padded with the end values."""
    w, K = gauss_cell_kernel(s, h)
    fp = np.concatenate([np.repeat(f[..., :1], K, -1), f, np.repeat(f[..., -1:], K, -1)], -1)
    shp = [1] * (f.ndim - 1) + [len(w)]
    return fftconvolve(fp, w[::-1].reshape(shp), mode="valid", axes=-1)


class SmoothModel(CL.NaiveModel):
    def __init__(self, qcut=5.0, msub=4, h=HLQ, grid=None, zedges=None, qedges=None):
        super().__init__(qcut=qcut, grid=grid, zedges=zedges, qedges=qedges)
        n = self.LNM.size
        self.LNMf = np.linspace(self.LNM[0], self.LNM[-1], (n - 1) * msub + 1)
        self.Mf = np.exp(self.LNMf)
        # cubic-spline interpolation matrix lnN(LNMf) = B @ lnN(LNM)
        self.B = CubicSpline(self.LNM, np.eye(n), axis=0)(self.LNMf)            # (nMf, nM)
        dl = self.LNMf[1] - self.LNMf[0]
        self.wM = np.full(self.LNMf.size, dl); self.wM[0] *= 0.5; self.wM[-1] *= 0.5
        self.h = h
        self.lq = np.arange(LQ_LO, LQ_HI + 0.5 * h, h)
        q = np.exp(self.lq)
        cdf = np.stack([ndtr(q - e) if np.isfinite(e) else np.zeros_like(q) for e in self.qed], 0)
        self.Praw = cdf[:-1] - cdf[1:]                                            # (nqb, nlq), no scatter
        self._Pcache = {}

    def Ptab(self, slnY):
        key = round(float(slnY), 12)
        if key not in self._Pcache:
            if len(self._Pcache) > 50:
                self._Pcache.clear()
            self._Pcache[key] = np.ascontiguousarray(smooth_lnq(self.Praw, slnY, self.h).T)   # (nlq, nqb)
        return self._Pcache[key]

    def nodes(self, th):
        om, s8, omb, alpha, slnY = th
        lnN, lnE, lnDA = self.G.eval(om, s8)
        lnN = lnN @ self.B.T                                                       # (nz, nMf)
        lnE = lnE[:, None]; lnDA = lnDA[:, None]; lnM = self.LNMf[None, :]
        lnY = (np.log(1e-4) - 0.19 * np.log(10) + (2.0 / 3.0) * lnE + alpha * (np.log(omb) + lnM - np.log(6e14))
               - 2 * lnDA - 2 * np.log(CL.ARCMIN))
        lnR = (np.log(3.0) + lnM - np.log(4 * np.pi * 500 * self.G.rhoc0) - 2 * lnE) / 3.0
        th500 = np.exp(lnR - lnDA) / CL.ARCMIN
        lnq0 = lnY - self.sigY.ln(th500)
        return lnN, th500, lnq0

    def _interp(self, P, lnq0):
        f = (lnq0 - self.lq[0]) / self.h
        f = np.clip(f, 0, len(self.lq) - 1.000001); i0 = f.astype(int); w = (f - i0)[..., None]
        return P[i0] * (1 - w) + P[i0 + 1] * w

    def counts(self, th, fsky=1.0):
        lnN, th500, lnq0 = self.nodes(th)
        Pn = self._interp(self.Ptab(th[4]), lnq0)                                # (nz, nMf, nqb)
        Wn = np.exp(lnN) * self.wM[None, :]
        return fsky * np.einsum("bz,zm,zmj->bj", self.Wz, Wn, Pn, optimize=True)

    def counts_table(self, th, Ptheta, lth, fsky=1.0):
        """Counts with a theta-dependent table Ptheta[(ntheta, nlq, nqb)] (scatter already convolved), linear in ln theta."""
        lnN, th500, lnq0 = self.nodes(th)
        fa = np.clip((np.log(th500) - lth[0]) / (lth[1] - lth[0]), 0, len(lth) - 1.000001)
        ia = fa.astype(int); wa = (fa - ia)[..., None]
        f = np.clip((lnq0 - self.lq[0]) / self.h, 0, len(self.lq) - 1.000001); i0 = f.astype(int); w = (f - i0)[..., None]
        Pn = ((Ptheta[ia, i0] * (1 - w) + Ptheta[ia, i0 + 1] * w) * (1 - wa)
              + (Ptheta[ia + 1, i0] * (1 - w) + Ptheta[ia + 1, i0 + 1] * w) * wa)
        Wn = np.exp(lnN) * self.wM[None, :]
        return fsky * np.einsum("bz,zm,zmj->bj", self.Wz, Wn, Pn, optimize=True)


class Fitter2(CL.Fitter):
    """cosmolib.Fitter with step sizes STEP2 (or a multiple) for the Jacobian."""

    def __init__(self, *a, step_mult=1.0, **k):
        super().__init__(*a, **k)
        self.step = STEP2 * step_mult

    def jac(self, x):
        J = []
        for k, i in enumerate(self.free):
            e = np.zeros(len(x)); e[k] = self.step[i]
            J.append((self.mu(x + e) - self.mu(x - e)) / (2 * self.step[i]))
        return np.stack(J, -1)[self.mask]

    def lnL(self, x):
        th = self.full(x)
        if not self.m.G.in_range(th[0], th[1]) or th[2] <= 0 or th[4] < 0 or not (0.5 < th[3] < 3.5):
            return -np.inf
        mu = self.mu(x)[self.mask]; d = self.d[self.mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0, d * np.log(mu / np.where(d > 0, d, 1.0)), 0.0)
        return float(np.sum(t - mu + d)) + self.lnprior(x)

    # ---- robust fitting near the slnY >= 0 boundary (mu is even in slnY, so dmu/dslnY -> 0 at slnY = 0)
    def _lohi(self):
        lo = np.array([BOUNDS[PNAMES[i]][0] for i in self.free]); hi = np.array([BOUNDS[PNAMES[i]][1] for i in self.free])
        return lo, hi

    def jac(self, x):
        lo, hi = self._lohi()
        J = []
        for k, i in enumerate(self.free):
            h = self.step[i]
            a, b = x.copy(), x.copy()
            a[k] = max(x[k] - h, lo[k]); b[k] = min(x[k] + h, hi[k])       # one-sided at a bound
            J.append((self.mu(b) - self.mu(a)) / (b[k] - a[k]))
        return np.stack(J, -1)[self.mask]

    def fit(self, x0=None, tol=1e-2, maxit=60, verbose=False, polish=False):
        """Fisher scoring with bound projection and step halving, then (optionally) bounded L-BFGS-B polish
        started from the scoring solution and from the truth (best lnL kept)."""
        from scipy.optimize import minimize
        lo, hi = self._lohi()
        x = FID[self.free].copy() if x0 is None else np.array(x0, float)
        x = np.clip(x, lo, hi); L = self.lnL(x); it = 0
        for it in range(maxit):
            mu = self.mu(x)[self.mask]; J = self.jac(x); d = self.d[self.mask]
            g = J.T @ (d / mu - 1); F = J.T @ (J / mu[:, None])
            th = self.full(x)
            for p, (m_, s_) in self.priors.items():
                if PNAMES.index(p) in self.free:
                    k = self.free.index(PNAMES.index(p)); F[k, k] += 1 / s_ ** 2; g[k] -= (th[PNAMES.index(p)] - m_) / s_ ** 2
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
            sc = np.sqrt(np.diag(np.linalg.inv(self.fisher(FID[self.free]))))
            best_x, best_L, L_fs = x, L, L
            for start in (x, FID[self.free]):
                ref = start.copy()
                r = minimize(lambda u: -self.lnL(ref + u * sc), np.zeros(len(x)), method="L-BFGS-B",
                             bounds=list(zip((lo - ref) / sc, (hi - ref) / sc)),
                             options=dict(eps=1e-4, ftol=1e-13, gtol=1e-7, maxiter=500))
                if -r.fun > best_L:
                    best_x, best_L = ref + r.x * sc, -r.fun
            self.polish_gain = float(best_L - L_fs); self.polish_shift = float(np.abs((best_x - x) / sc).max())
            x, L = best_x, best_L
        self.at_bound = [PNAMES[i] for k, i in enumerate(self.free)
                         if x[k] <= lo[k] + 1e-3 * self.step[i] or x[k] >= hi[k] - 1e-3 * self.step[i]]
        return x, L, it
