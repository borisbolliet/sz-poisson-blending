"""Repair library for the cosmology-dependence / likelihood study (analysis/cosmo/blend_dependence).

Fixes relative to c03_analysis.py:
  * naive_exact(): naive single-cluster counts in (z, q_obs) bins by direct integration
    (Gauss-Legendre in z inside each z bin, so no node straddles a bin edge; analytic Phi(q - edge) for the
    q_obs bins, so no interpolation in q).  Replaces cosmolib.primary_binned/naive_counts for the likelihood.
    Background chi(z) by cumulative trapezoid of 1/E(z) (2e-8 agreement with colossus, ~100x faster).
  * blended truth = C(z,q; theta) x naive_exact(theta), where C = N_blended/N_naive is taken from the stored
    cosmolib runs (data_cosmo/c02_grid.npz, c02_extra.npz).  C is a ratio computed with one consistent
    binning, so the binning error of cosmolib cancels to first order.
  * fit(): Poisson maximum likelihood by Fisher scoring with step-halving line search on the exact Poisson
    deviance, run to convergence (Newton decrement < tol) and reporting convergence; multi-start wrapper.
  * fits are done directly at each f_sky (no sqrt(f_sky) rescaling), priors centred on the truth.
"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blend_dependence"))
import numpy as np
from scipy.special import ndtr
from scipy.integrate import cumulative_trapezoid
import cosmolib as cl
from c02_grid import S8, OM, OMB, SL, FIDP, SIG_P18
from colossus.cosmology import cosmology

D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
PN = ["sigma8", "Om", "1-b", "sigma_lnY"]
NODES = [S8, OM, OMB, SL]
ZE, QE = cl.ZEDGES, cl.QEDGES
H_STEP = np.array([1e-3, 1e-3, 4e-3, 6e-3])     # FD half-steps for the fit Jacobian (~0.1-0.2 Fisher sigma)


LNK = np.linspace(np.log(1e-5), np.log(2e3), 6001)
_KK = np.exp(LNK)
_WK = np.full(len(LNK), (LNK[1] - LNK[0]) / 3.0); _WK[1:-1:2] *= 4; _WK[2:-1:2] *= 2   # Simpson


def _tophat(x):
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    dW = (9 * x * np.cos(x) + 3 * (x ** 2 - 3) * np.sin(x)) / x ** 4
    return W, dW


_T08 = dict(D=np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200.]),
            A=np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260]),
            a=np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66]),
            b=np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41]),
            c=np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44]))


def tinker08_f(sigma, z, Delta_m):
    """Tinker08 f(sigma) exactly as colossus.lss.mass_function.modelTinker08, but WITHOUT rounding Delta_m
    to an integer (colossus rounds, which makes f piecewise constant in Om: r01_validate.log)."""
    A0, a0, b0, c0 = (np.interp(Delta_m, _T08["D"], _T08[k]) for k in "Aabc")
    alpha = 10 ** (-(0.75 / np.log10(Delta_m / 75.0)) ** 1.2)
    A = A0 * (1.0 + z) ** -0.14; a = a0 * (1.0 + z) ** -0.06; b = b0 * (1.0 + z) ** -alpha
    return A * ((sigma / b) ** -a + 1.0) * np.exp(-c0 / sigma ** 2)


class FastCosmo(cl.Cosmo):
    """Same cosmology as cosmolib.Cosmo, but (i) chi(z) by cumulative trapezoid, (ii) d2N from a smooth
    Tinker08 HMF: sigma(R) and dln sigma/dln R by fixed-grid Simpson integration of the colossus
    Eisenstein-Hu (1998) P(k) (colossus's default model), normalised to sigma8; colossus growth factor and
    Tinker08 f(sigma; z, Delta_m=500 rho_c/rho_m) without the integer rounding of Delta_m done by colossus.  colossus rounds Delta_m to an integer inside Tinker08, which makes
    ln dN/dlnM jump by ~6e-3 in steps as Om changes (r01_validate.log): finite-difference derivatives in Om
    are then noisy and the fitted minimum becomes optimiser-dependent."""

    def __init__(self, sigma8, Om):
        FastCosmo._n += 1
        p = dict(cl.P18); p["sigma8"] = float(sigma8); p["Om0"] = float(Om)
        self.c = cosmology.setCosmology(f"rep{os.getpid()}_{FastCosmo._n}", **p, persistence="")
        self.h = self.c.h
        zg = np.linspace(0.0, 4.0, 8001)
        chi = cumulative_trapezoid(1.0 / self.c.Ez(zg), zg, initial=0.0) * (cl.C_KMS / 100.0)   # Mpc/h
        self._chi = (zg, chi / self.h)
        from colossus.cosmology import power_spectrum as ps
        T = ps.modelEisenstein98(_KK, self.c.h, self.c.Om0, self.c.Ob0, self.c.Tcmb0)
        self._Pk = _KK ** self.c.ns * T ** 2                       # unnormalised, k in h/Mpc
        s2_8 = self._sig2(np.array([8.0]))[0][0]
        self._norm = float(sigma8) ** 2 / s2_8
        self.rho_m0 = self.c.rho_m(0.0) * 1e9                      # Msun h^2 / Mpc^3

    def _sig2(self, R):
        x = np.outer(R, _KK)
        W, dW = _tophat(x)
        base = _WK * _KK ** 3 * self._Pk / (2 * np.pi ** 2)          # dk -> k dlnk
        s2 = (base * W ** 2).sum(1)
        ds2 = (base * 2 * W * dW * x).sum(1)                         # d sigma^2 / d ln R
        return s2, ds2

    def d2N(self, z, lnM):
        from colossus.lss import mass_function
        cosmology.setCurrent(self.c)
        Mh = np.exp(lnM) * self.h                                    # Msun/h
        R = (3 * Mh / (4 * np.pi * self.rho_m0)) ** (1 / 3)
        s2, ds2 = self._sig2(R)
        s2 *= self._norm; ds2 *= self._norm
        dlns_dlnM = ds2 / (2 * s2) / 3.0
        D = self.c.growthFactor(np.asarray(z, float))
        out = np.empty((len(z), len(lnM)))
        for i, zz in enumerate(z):
            sig = np.sqrt(s2) * D[i]
            f = tinker08_f(sig, float(zz), 500.0 * self.c.rho_c(zz) / self.c.rho_m(zz))
            out[i] = f * self.rho_m0 / Mh * np.abs(dlns_dlnM) * self.h ** 3
        return out * (self.dVdzdO(z) * 4 * np.pi)[:, None]


def naive_exact(theta, ngl=12, nM=500, ngh=32, zedges=ZE, qedges=QE, Mmin=1e13, Mmax=1e16):
    """full-sky naive counts N[zbin, qbin] for theta = (sigma8, Om, 1-b, sigma_lnY)."""
    s8, Om, omb, slnY = map(float, theta)
    co = FastCosmo(s8, Om); sz = cl.SZ(co, omb, slnY)
    _, _, m0 = cl.rho_tab()
    xg, wgl = np.polynomial.legendre.leggauss(ngl)
    zs, wz, zb = [], [], []
    for b in range(len(zedges) - 1):
        a, c = zedges[b], zedges[b + 1]
        zs.append(0.5 * (a + c) + 0.5 * (c - a) * xg); wz.append(0.5 * (c - a) * wgl); zb.append(np.full(ngl, b))
    zs, wz, zb = np.concatenate(zs), np.concatenate(wz), np.concatenate(zb)
    lnM = np.linspace(np.log(Mmin), np.log(Mmax), nM); wM = cl.trap_w(lnM); M = np.exp(lnM)
    d2N = co.d2N(zs, lnM)                                   # (nz, nM)
    x, wh = np.polynomial.hermite_e.hermegauss(ngh); wh = wh / wh.sum()
    out = np.zeros((len(zedges) - 1, len(qedges) - 1))
    fin = np.isfinite(qedges)
    for i, zz in enumerate(zs):
        th = sz.theta500(M, zz)
        q0 = sz.Y500_med(M, zz) / m0.sigma_Y(th)
        q = q0[:, None] * np.exp(slnY * x)[None, :]          # (nM, ngh)
        w = (d2N[i] * wz[i] * wM)[:, None] * wh[None, :]
        Phi = np.zeros(q.shape + (len(qedges),))
        Phi[..., fin] = ndtr(q[..., None] - qedges[fin])
        P = Phi[..., :-1] - Phi[..., 1:]
        out[zb[i]] += np.einsum("mg,mge->e", w, P)
    return out


# ----------------------------------------------------------------------------- stored correction C(z,q; theta)
G = dict(np.load(f"{D}/c02_grid.npz"))
E = dict(np.load(f"{D}/c02_extra.npz"))


def _C(S, j):
    Nn = S["Nzq_n"][j]
    return np.where(Nn > 0, S["Nzq_b"][j] / np.where(Nn > 0, Nn, 1), 1.0)


def find(p):
    """return stored C(z,q) at parameter point p (grid or extra)."""
    key = tuple(np.round(p, 6))
    for S in (G, E):
        ks = [tuple(np.round(q, 6)) for q in S["params"]]
        if key in ks:
            return _C(S, ks.index(key))
    raise KeyError(p)


C_FID = find(FIDP)
_idx = np.array([[NODES[a].index(round(float(p[a]), 6)) for a in range(4)] for p in G["params"]])
_lnCg = np.zeros((3, 3, 3, 3) + C_FID.shape)
for _j in range(len(G["params"])):
    _lnCg[tuple(_idx[_j])] = np.log(_C(G, _j))


def _lag(x, nodes):
    x0, x1, x2 = nodes
    return np.array([(x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2)), (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2)),
                     (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))])


def emu_C(p):
    L = [_lag(float(p[a]), NODES[a]) for a in range(4)]
    return np.exp(np.einsum("a,b,c,d,abcd...->...", *L, _lnCg))


# ----------------------------------------------------------------------------- likelihood
_cache = {}


def naive(th):
    k = tuple(np.round(np.asarray(th, float), 10))
    if k not in _cache:
        if len(_cache) > 4000:
            _cache.clear()
        _cache[k] = naive_exact(th)
    return _cache[k]


def deviance(Dd, M, fsky, mask=None):
    """Poisson -2 ln L ratio (saturated), = sum 2 [M - D + D ln(D/M)], scaled to fsky."""
    Dd, M = Dd * fsky, M * fsky
    if mask is not None:
        Dd, M = Dd[mask], M[mask]
    t = M - Dd + np.where(Dd > 0, Dd * np.log(np.where(Dd > 0, Dd, 1) / M), 0.0)
    return 2 * t.sum()


def chi2_tot(model, Dd, th, th0, free, pri, fsky, mask=None):
    c = deviance(Dd, model(th), fsky, mask)
    for a in free:
        if a in pri:
            c += ((th[a] - th0[a]) / pri[a]) ** 2
    return c


def jac(model, th, free):
    J = []
    for a in free:
        tp, tm = th.copy(), th.copy(); tp[a] += H_STEP[a]; tm[a] -= H_STEP[a]
        J.append((model(tp) - model(tm)) / (2 * H_STEP[a]))
    return np.array(J)


def fisher_from(J, M, free, pri, fsky, mask=None):
    m = (M > 1e-3) if mask is None else (mask & (M > 1e-3))
    F = fsky * np.einsum("ak,bk->ab", J[:, m], J[:, m] / M[m])
    for ii, a in enumerate(free):
        if a in pri:
            F[ii, ii] += 1 / pri[a] ** 2
    return F


def fit(model, Dd, th0, free, pri, fsky, start=None, mask=None, tol=1e-8, maxit=60):
    """Poisson ML fit (Fisher scoring + step halving on the exact deviance).  Returns (theta, chi2, info)."""
    th = np.array(th0 if start is None else start, float)
    th0 = np.array(th0, float)
    free = list(free)
    c_old = chi2_tot(model, Dd, th, th0, free, pri, fsky, mask)
    dec = np.inf
    for it in range(maxit):
        M = model(th)
        J = jac(model, th, free)
        m = (M > 1e-3) if mask is None else (mask & (M > 1e-3))
        F = fsky * np.einsum("ak,bk->ab", J[:, m], J[:, m] / M[m])
        g = fsky * np.einsum("ak,k->a", J[:, m], (Dd[m] - M[m]) / M[m])
        for ii, a in enumerate(free):
            if a in pri:
                F[ii, ii] += 1 / pri[a] ** 2
                g[ii] -= (th[a] - th0[a]) / pri[a] ** 2
        step = np.linalg.solve(F, g)
        dec = float(g @ step)                      # Newton decrement (expected chi2 drop x 2)
        if dec < tol:
            break
        lam = 1.0
        while True:
            tn = th.copy(); tn[free] += lam * step
            c_new = chi2_tot(model, Dd, tn, th0, free, pri, fsky, mask)
            if c_new <= c_old + 1e-10 or lam < 1e-3:
                break
            lam *= 0.5
        th, c_old = tn, c_new
    return th, c_old, dict(it=it + 1, dec=dec, conv=dec < tol)
