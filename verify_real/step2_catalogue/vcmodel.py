"""My own binned (z, q_obs) count model and Poisson-likelihood fitter (lens 'catalogue', step-2 verification).

Model (naive, no blending):  q_med = Y_med(M, z; Om, (1-b), alpha) / sigma(theta500),  ln q = ln q_med + slnY x,
q_obs = q + N(0, 1);  mu[z bin, q bin] = fsky * sum_{z, M nodes} w d2N/dz/dlnM(Om, s8) P(q_obs in bin | q_med, slnY).
  * d2N: out/hmf_grid_mine.npz (vc02: direct colossus at every (Om, s8) node), tensor not-a-knot cubic spline weights;
  * ln E, ln D_A: 1D cubic spline in Om; rho_c0 fixed (h fixed);
  * sigma(theta500): the sigma_Y spline of SZModel (white) or MMFModel (SO; sigma_Y only), times r(theta) for the
    map-sigma variants (the variant's definition, linear in ln theta);
  * P(q_obs > e | ln q_med, slnY): Phi(e^u - e) on a fine u = ln q grid (h = 0.001), convolved with N(0, slnY) by
    'hat-function-exact' weights w_k = int phi_s(t) Lambda(t/h - k) dt (exact for the piecewise-linear interpolant,
    a delta function for slnY -> 0), FFT; linear interpolation in u.
Fitter: Nelder-Mead on the exact Poisson lnL (+ Gaussian prior) in Fisher-whitened coordinates, restarted until
stable, then a Newton polish with the finite-difference Hessian of lnL (not Fisher scoring).  Linearised estimator
dtheta = F^-1 J^T (d - mu)/mu with my own finite-difference Jacobian.
"""
import os, sys
import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vclib as V

ARC = np.pi / 180 / 60
LOGY0 = np.log(1e-4) - 0.19 * np.log(10.0)
BOUNDS = {"Om": (0.2505, 0.3745), "s8": (0.7405, 0.8795), "omb": (0.3, 1.5), "alpha": (1.0, 2.6), "slnY": (1e-3, 1.0)}

_G = {}


def grid():
    """my HMF grid (vc02); VC_GRID=wide selects the sigma8 0.69-0.90 version (used only by vc09)."""
    if not _G:
        d = np.load(f"{HERE}/out/hmf_grid_mine{'_wide' if os.environ.get('VC_GRID') == 'wide' else ''}.npz")
        _G.update({k: d[k] for k in d.files})
        _G["Lf"] = _G["L"].reshape(len(_G["OM"]) * len(_G["S8"]), -1).astype(np.float64)     # 320 MB
        _G["spE"] = CubicSpline(_G["OM"], _G["LE"], axis=0)
        _G["spD"] = CubicSpline(_G["OM"], _G["LD"], axis=0)
        _G["wO"] = CubicSpline(_G["OM"], np.eye(len(_G["OM"])))
        _G["wS"] = CubicSpline(_G["S8"], np.eye(len(_G["S8"])))
    return _G


def hat_gauss_weights(s, h, nsig=9.0):
    """w_k = int phi_s(t) Lambda(t/h - k) dt, k = -K..K (exact convolution of the piecewise-linear interpolant)."""
    K = int(np.ceil(nsig * max(s, h) / h)) + 2
    k = np.arange(-K, K + 1)
    if s <= 0:
        return (k == 0).astype(float), K
    t = np.arange(-K - 1, K + 2) * h                     # node positions t_{k-1}, t_k, t_{k+1}
    F = ndtr(t / s)
    G = -s * s * np.exp(-0.5 * (t / s) ** 2) / (s * np.sqrt(2 * np.pi))     # antiderivative of t phi_s(t)
    i = np.arange(1, 2 * K + 2)                          # index of t_k in t for k = -K..K
    left = (G[i] - G[i - 1] - t[i - 1] * (F[i] - F[i - 1])) / h
    right = (t[i + 1] * (F[i + 1] - F[i]) - (G[i + 1] - G[i])) / h
    w = left + right
    return w / w.sum(), K


class CountModel:
    """mu(theta) in 16 z bins x 9 q_obs bins (q_cut .. 100 geometric, + q > 100)."""
    H = 0.001
    U = np.arange(np.log(1e-3), np.log(3e4) + 0.0005, 0.001)

    def __init__(self, filt="baseline", qcut=5.0, r=None):
        G = grid()
        self.qcut = qcut
        self.qed = np.geomspace(qcut, 100.0, 9)
        self.smod = V.sigma_model(filt)
        self.r = r                                        # (theta nodes, r nodes) or None
        self.ZN, self.ZW, self.ZB, self.LNM, self.WM = G["ZN"], G["ZW"], G["ZB"], G["LNM"], G["WM"]
        self.nz = len(V.ZED) - 1
        self.Zsum = np.zeros((self.nz, len(self.ZN)))
        self.Zsum[self.ZB, np.arange(len(self.ZN))] = self.ZW
        g = ndtr(np.exp(self.U)[None, :] - self.qed[:, None])           # (9, nU) P(q_obs > e | q), no scatter
        self.g = g
        self._cache = {}

    def table(self, slnY):
        key = round(float(slnY), 13)
        T = self._cache.get(key)
        if T is None:
            w, K = hat_gauss_weights(float(slnY), self.H)
            gp = np.concatenate([np.repeat(self.g[:, :1], K, 1), self.g, np.repeat(self.g[:, -1:], K, 1)], 1)
            T = fftconvolve(gp, w[None, :], mode="valid", axes=1)          # (9, nU)
            P = np.empty_like(T)
            P[:-1] = T[:-1] - T[1:]; P[-1] = T[-1]
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = T = np.ascontiguousarray(P)
        return T

    def nodes(self, th):
        om, s8, omb, alpha, slnY = th
        G = grid()
        wO = G["wO"](om); wS = G["wS"](s8)
        lnN = (np.outer(wO, wS).ravel() @ G["Lf"]).reshape(len(self.ZN), len(self.LNM))
        lnE = G["spE"](om)[:, None]; lnD = G["spD"](om)[:, None]
        lnM = self.LNM[None, :]
        lnY = LOGY0 + (2.0 / 3.0) * lnE + alpha * (np.log(omb) + lnM - np.log(6e14)) - 2 * lnD - 2 * np.log(ARC)
        lnR = (np.log(3.0) + lnM - np.log(4 * np.pi * 500 * float(G["rhoc0"])) - 2 * lnE) / 3.0
        th500 = np.exp(lnR - lnD) / ARC
        lsig = np.log(self.smod.sigma_Y(th500))
        if self.r is not None:
            lsig = lsig + np.log(np.interp(np.log(th500), np.log(self.r[0]), self.r[1]))
        return lnN, lnY - lsig

    def counts(self, th, fsky=1.0):
        lnN, u = self.nodes(th)
        P = self.table(th[4])
        f = np.clip((u - self.U[0]) / self.H, 0, len(self.U) - 1.000001)
        i0 = f.astype(np.int64); w = f - i0
        Pn = P[:, i0] * (1 - w) + P[:, i0 + 1] * w                       # (9, nz, nM)
        A = np.exp(lnN) * self.WM[None, :]
        perz = np.einsum("jzm,zm->zj", Pn, A)                             # (nZN, 9)
        return fsky * (self.Zsum @ perz)                                  # (16, 9)


def S8_of(om, s8):
    return s8 * np.sqrt(om / 0.3)


CONFIGS = {"a": (["Om", "s8"], None), "b": (["Om", "s8", "omb"], None), "c": (["Om", "s8", "omb", "slnY"], None),
           "d": (["Om", "s8", "omb", "alpha", "slnY"], 0.04), "e": (["Om", "s8", "omb", "alpha", "slnY"], None)}


class Fit:
    """Poisson likelihood of data d (already scaled to the survey) under model.counts(theta) * fsky."""

    def __init__(self, model, data, cfg, truth, fsky):
        self.m, self.d, self.truth, self.fsky = model, np.asarray(data, float), np.asarray(truth, float), fsky
        free, rel = CONFIGS[cfg]
        self.cfg = cfg
        self.free = [V.PN.index(p) for p in free]
        self.prior = None if rel is None else (2, float(truth[2]), rel * float(truth[2]))
        G = grid()
        bnd = dict(BOUNDS, Om=(float(G["OM"][0]) + 5e-4, float(G["OM"][-1]) - 5e-4), s8=(float(G["S8"][0]) + 5e-4, float(G["S8"][-1]) - 5e-4))
        self.lo = np.array([bnd[V.PN[i]][0] for i in self.free]); self.hi = np.array([bnd[V.PN[i]][1] for i in self.free])
        self.nev = 0

    def full(self, x):
        th = self.truth.copy(); th[self.free] = x
        return th

    def mu(self, x):
        self.nev += 1
        return self.m.counts(self.full(x), self.fsky)

    def lnL(self, x):
        x = np.asarray(x, float)
        if np.any(x < self.lo) or np.any(x > self.hi):
            return -np.inf
        mu = self.mu(x); d = self.d
        t = np.where(d > 0, d * np.log(np.where(d > 0, mu / np.where(d > 0, d, 1.0), 1.0)), 0.0)
        L = float(np.sum(t - mu + d))
        if self.prior is not None:
            k, m_, s_ = self.prior
            L -= 0.5 * ((self.full(x)[k] - m_) / s_) ** 2
        return L

    def jac(self, x, steps):
        J = []
        for k in range(len(x)):
            e = np.zeros(len(x)); e[k] = steps[k]
            a = np.clip(x - e, self.lo, self.hi); b = np.clip(x + e, self.lo, self.hi)
            J.append(((self.mu(b) - self.mu(a)) / (b[k] - a[k])).ravel())
        return np.stack(J, 1)

    def fisher(self, x, steps):
        mu = self.mu(x).ravel(); J = self.jac(x, steps)
        F = J.T @ (J / mu[:, None])
        if self.prior is not None:
            k = self.free.index(self.prior[0]); F[k, k] += 1.0 / self.prior[2] ** 2
        return F, J, mu

    def sigmas(self, F):
        C = np.linalg.inv(F)
        s = np.sqrt(np.diag(C))
        a, b = self.free.index(0), self.free.index(1)
        S8 = S8_of(self.truth[0], self.truth[1])
        g = np.zeros(len(self.free)); g[a] = 0.5 * S8 / self.truth[0]; g[b] = S8 / self.truth[1]
        return s, float(np.sqrt(g @ C @ g)), g, C

    def linearised(self, steps):
        xt = self.truth[self.free]
        F, J, mu = self.fisher(xt, steps)
        dx = np.linalg.solve(F, J.T @ ((self.d.ravel() - mu) / mu))
        return dx, F, J, mu

    def fit_nm(self, F, x0=None, tol=1e-4, rounds=4, maxfev=3000, steps=None):
        """Nelder-Mead in whitened coordinates, restarted until the solution stops moving.  Round 1 is whitened with
        the Fisher matrix at the truth (F); later rounds with the Fisher matrix at the current best point (so the
        simplex is adapted to the curved degeneracies of the 5-parameter fits).  Outside the parameter box the
        objective is continued smoothly: -lnL(clipped x) + 1e3 sum((x - clip)/sigma)^2 (no hard wall); the clipped point
        is returned.  self.nm_info records (evaluations, converged flag, distance moved in sigma) per round."""
        from scipy.optimize import minimize
        xt = self.truth[self.free]
        sc = np.sqrt(np.diag(np.linalg.inv(F)))
        x_best = xt.copy() if x0 is None else np.clip(np.asarray(x0, float), self.lo, self.hi)

        def obj_factory(center, Fw):
            Lc = np.linalg.cholesky(Fw)
            Linv_T = np.linalg.inv(Lc).T
            to_x = lambda y: center + Linv_T @ y

            def f(y):
                x = to_x(y)
                xc = np.clip(x, self.lo, self.hi)
                return -self.lnL(xc) + 1e3 * float(np.sum(((x - xc) / sc) ** 2))
            return f, to_x
        info = []
        Fw = F
        f_best = None
        for r in range(rounds):
            f, to_x = obj_factory(x_best, Fw)
            y0 = np.zeros(len(xt))
            f0 = f(y0)
            f_best = f0 if f_best is None else min(f_best, f0)
            res = minimize(f, y0, method="Nelder-Mead",
                           options=dict(xatol=tol, fatol=1e-8, maxiter=maxfev, maxfev=maxfev, adaptive=True,
                                        initial_simplex=np.vstack([y0, 0.3 * np.eye(len(y0))])))
            moved = float(np.linalg.norm(res.x))
            info.append((int(res.nfev), bool(res.success), moved))
            if res.fun <= f0:
                x_best = np.clip(to_x(res.x), self.lo, self.hi)
            if moved < 10 * tol:
                break
            # re-whiten at the new best point (Fisher of the model there, incl. the prior)
            if steps is not None:
                try:
                    Fn, _, _ = self.fisher(x_best, steps)
                    np.linalg.cholesky(Fn)
                    Fw = Fn
                except np.linalg.LinAlgError:
                    Fw = F
        self.nm_info = info
        return x_best, self.lnL(x_best)

    def newton_polish(self, x, h_sig, iters=6):
        """Newton steps with the finite-difference Hessian of lnL; parameters at a bound are held fixed."""
        x = np.array(x, float)
        free_k = [k for k in range(len(x)) if (x[k] - self.lo[k]) > 3 * h_sig[k] and (self.hi[k] - x[k]) > 3 * h_sig[k]]
        L0 = self.lnL(x)
        for it in range(iters):
            n = len(free_k)
            if n == 0:
                break
            g = np.zeros(n); Hs = np.zeros((n, n))
            f0 = L0
            for a_, ka in enumerate(free_k):
                ea = np.zeros(len(x)); ea[ka] = h_sig[ka]
                fp, fm = self.lnL(x + ea), self.lnL(x - ea)
                g[a_] = (fp - fm) / (2 * h_sig[ka]); Hs[a_, a_] = (fp - 2 * f0 + fm) / h_sig[ka] ** 2
                for b_ in range(a_):
                    kb = free_k[b_]
                    eb = np.zeros(len(x)); eb[kb] = h_sig[kb]
                    Hs[a_, b_] = Hs[b_, a_] = (self.lnL(x + ea + eb) - self.lnL(x + ea - eb) - self.lnL(x - ea + eb)
                                               + self.lnL(x - ea - eb)) / (4 * h_sig[ka] * h_sig[kb])
            step = np.linalg.solve(-Hs, g)
            xn = x.copy(); xn[free_k] += step
            Ln = self.lnL(xn)
            if not np.isfinite(Ln) or Ln < L0 - 1e-9:
                break
            x, L0 = xn, Ln
            if np.max(np.abs(step / h_sig[free_k])) < 1e-3:
                break
        return x, L0

    def deviance(self, x):
        mu = self.mu(x); d = self.d
        t = np.where(d > 0, d * np.log(np.where(d > 0, d / mu, 1.0)), 0.0)
        return float(2 * np.sum(mu - d + t))
