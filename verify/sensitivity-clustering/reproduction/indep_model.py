"""Independent re-implementation of the baseline model (does NOT import szsim).

Own: comoving distance integral, rho_crit, Y-M relation, theta500, Arnaud GNFW projection (Gauss-Legendre),
Hankel transform, matched-filter sigma_Y, cross response table, P_det, selection integral (Gauss-Legendre
in z and lnM, colossus HMF called directly), mean sky y.
Units: M500 in Msun (h-free), distances in Mpc, angles in arcmin, Y in arcmin^2.
"""
import os
import numpy as np
from scipy.special import j0, ndtr
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from colossus.cosmology import cosmology
from colossus.lss import mass_function

cosmo = cosmology.setCosmology("planck18")
h = cosmo.h
ARC = np.pi / 180 / 60
C_KMS = 299792.458
DH = C_KMS / cosmo.H0  # Mpc
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------- background
_zf = np.linspace(0, 4.0, 40001)
_chif = DH * cumulative_trapezoid(1.0 / cosmo.Ez(_zf), _zf, initial=0.0)


def Ez(z):
    return cosmo.Ez(np.asarray(z, float))


def chi(z):
    return np.interp(np.asarray(z, float), _zf, _chif)


def DA(z):
    return chi(z) / (1 + np.asarray(z, float))


def rho_c(z):  # Msun / Mpc^3 physical
    return 2.77536627e11 * h ** 2 * Ez(z) ** 2


def dVdzdO(z):
    return DH / Ez(z) * chi(z) ** 2


def dndlnM(M, z, model="tinker08"):
    return mass_function.massFunction(np.asarray(M, float) * h, float(z), mdef="500c", model=model, q_out="dndlnM") * h ** 3


def R500(M, z):
    return (3 * np.asarray(M, float) / (4 * np.pi * 500 * rho_c(z))) ** (1 / 3)


def theta500(M, z):
    return R500(M, z) / DA(z) / ARC


def Ymed(M, z):
    """median Y500 [arcmin^2] (Planck 2013 XX, no h70 factor)."""
    YDA2 = 1e-4 * 10 ** -0.19 * Ez(z) ** (2 / 3) * (0.8 * np.asarray(M, float) / 6e14) ** 1.79
    return YDA2 / DA(z) ** 2 / ARC ** 2


# ---------------------------------------------------------------- Arnaud profile
P0, C500, GAM, ALP, BET = 8.403, 1.177, 0.3081, 1.0510, 5.4905
X5 = 5.0


def P(x):
    cx = C500 * x
    return 1.0 / (cx ** GAM * (1 + cx ** ALP) ** ((BET - GAM) / ALP))


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def Ysph(X):
    # substitution x = X t^3 removes the r^-0.31 cusp
    t, w = _gl(0, 1, 400)
    x = X * t ** 3
    return np.sum(w * 4 * np.pi * x * x * P(x) * 3 * X * t ** 2)


RATIO = Ysph(X5) / Ysph(1.0)


def proj(s):
    """projected unit-Y500 shape f(s), s = theta/theta500; int 2 pi s f ds = RATIO."""
    s = np.atleast_1d(s).astype(float)
    v, wv = _gl(0, 1, 300)
    L = np.sqrt(np.clip(X5 ** 2 - s ** 2, 0, None))
    l = L[:, None] * v[None, :] ** 3
    jac = L[:, None] * 3 * v[None, :] ** 2
    r = np.sqrt(s[:, None] ** 2 + l ** 2)
    return 2 * np.sum(wv * P(r) * jac, axis=1) / Ysph(1.0)


def _hankel_table():
    fn = os.path.join(CACHE, "g_table.npz")
    if os.path.exists(fn):
        d = np.load(fn); return d["u"], d["g"]
    # s = X5 (1 - (1-t)^2) clusters nodes at the sqrt edge; t panels with GL nodes
    edges = np.concatenate([[0], np.geomspace(1e-6, 1e-2, 30), np.linspace(1e-2, 1, 1500)[1:]])
    T, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        t, w = _gl(a, b, 16); T.append(t); W.append(w)
    t = np.concatenate(T); w = np.concatenate(W)
    s = X5 * (1 - (1 - t) ** 2); ds = X5 * 2 * (1 - t)
    f = proj(s)
    ww = 2 * np.pi * s * f * ds * w
    u = np.concatenate([[0], np.geomspace(1e-3, 4e3, 6000)])
    g = np.array([np.sum(ww * j0(uu * s)) for uu in u])
    np.savez(fn, u=u, g=g)
    return u, g


U_G, G_G = _hankel_table()
_gspl = CubicSpline(U_G, G_G)


def g(u):
    u = np.asarray(u, float)
    return np.where(u > U_G[-1], 0.0, _gspl(np.clip(u, 0, U_G[-1])))


class Instrument:
    def __init__(self, beam=1.4, noise=4e-6):
        self.beam, self.noise = beam, noise
        self.sb = beam * ARC / np.sqrt(8 * np.log(2))
        self.Nw_sr = (noise * ARC) ** 2
        self.lmax = 9.0 / self.sb
        # GL nodes in l
        edges = np.linspace(0, self.lmax, 801)
        L, Wl = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            x, w = _gl(a, b, 24); L.append(x); Wl.append(w)
        self.l = np.concatenate(L); self.wl = np.concatenate(Wl)
        self.B2 = np.exp(-(self.l * self.sb) ** 2)
        tg = np.geomspace(1e-3, 300, 250)
        self._sig = CubicSpline(np.log(tg), np.log([self._sigY(t) for t in tg]))

    def _sigY(self, th):
        gg = g(self.l * th * ARC)
        I = np.sum(self.wl * self.l * self.B2 * gg * gg) / (2 * np.pi) / self.Nw_sr
        return 1 / np.sqrt(I) / ARC ** 2

    def sigY(self, th):
        return np.exp(self._sig(np.log(np.clip(np.asarray(th, float), 1e-3, 300))))

    def mu_coef(self, th):
        """response (in SNR units) to a uniform y: sigma_Y RATIO y / N_w  -> returns sigma_Y RATIO / N_w[arcmin^2]."""
        return self.sigY(th) * RATIO / self.noise ** 2

    def rho_table(self, th_nodes, sep_nodes):
        fn = os.path.join(CACHE, f"rho_beam{self.beam:.2f}_{len(th_nodes)}_{len(sep_nodes)}.npz")
        if os.path.exists(fn):
            return np.load(fn)["rho"]
        l = self.l; w = self.wl * l * self.B2
        G = g(np.outer(th_nodes * ARC, l))                      # (nth, nl)
        rho = np.empty((len(th_nodes), len(th_nodes), len(sep_nodes)), np.float32)
        J = j0(np.outer(sep_nodes * ARC, l))                     # (nsep, nl)
        for i in range(len(th_nodes)):
            norm = np.sum(w * G[i] ** 2)
            M = (w * G[i])[None, :] * G                          # (nth, nl)
            rho[i] = (M @ J.T) / norm
        TA, TB, S = np.meshgrid(th_nodes, th_nodes, sep_nodes, indexing="ij")
        rho[S > 5 * (TA + TB) + 4 * self.beam] = 0.0
        np.savez(fn, rho=rho)
        return rho


# ---------------------------------------------------------------- selection integral
def p_det(qmean, sig_lnY, qcut, n=40):
    x, w = np.polynomial.hermite_e.hermegauss(n); w = w / w.sum()
    q = np.asarray(qmean)[..., None] * np.exp(sig_lnY * x)
    return (ndtr(q - qcut) * w).sum(-1)


def zM_grid(zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16, nzp=60, nmp=60, k=8):
    ze = np.linspace(zmin, zmax, nzp + 1); me = np.linspace(np.log(Mmin), np.log(Mmax), nmp + 1)
    Z, WZ, LM, WM = [], [], [], []
    for a, b in zip(ze[:-1], ze[1:]):
        x, w = _gl(a, b, k); Z.append(x); WZ.append(w)
    for a, b in zip(me[:-1], me[1:]):
        x, w = _gl(a, b, k); LM.append(x); WM.append(w)
    return np.concatenate(Z), np.concatenate(WZ), np.concatenate(LM), np.concatenate(WM)


_HMF_CACHE = {}


def d2N_grid(model="tinker08", **kw):
    key = (model, tuple(sorted(kw.items())))
    if key in _HMF_CACHE:
        return _HMF_CACHE[key]
    z, wz, lnM, wm = zM_grid(**kw)
    M = np.exp(lnM)
    d2N = np.array([dndlnM(M, zz, model) * dVdzdO(zz) * 4 * np.pi for zz in z])
    _HMF_CACHE[key] = (z, wz, M, wm, d2N)
    return _HMF_CACHE[key]


def N_det(inst, qcut=5.0, sig=0.173, model="tinker08", Mmin=1e13, zmax=3.0, Mfloor=None):
    z, wz, M, wm, d2N = d2N_grid(model, zmax=zmax, Mmin=Mmin)
    tot = 0.0
    for i, zz in enumerate(z):
        q = Ymed(M, zz) / inst.sigY(theta500(M, zz))
        tot += wz[i] * np.sum(wm * d2N[i] * p_det(q, sig, qcut))
    return tot


def ybar(sig=0.173, model="tinker08", Mmin=1e13, zmax=3.0):
    """mean sky y from halos (Y5R500 per steradian)."""
    z, wz, M, wm, d2N = d2N_grid(model, zmax=zmax, Mmin=Mmin)
    tot = 0.0
    for i, zz in enumerate(z):
        tot += wz[i] * np.sum(wm * d2N[i] * Ymed(M, zz)) * np.exp(sig ** 2 / 2)
    return tot * ARC ** 2 * RATIO / (4 * np.pi)


# ---------------------------------------------------------------- own Poisson sampler
def sampler_grid(Mmin=1e13, zmax=3.0, model="tinker08", nz=700, nm=900, Mmax=1e16, zmin=0.01):
    fn = os.path.join(CACHE, f"samp_{model}_{Mmin:.0e}_{zmax}_{nz}_{nm}.npz")
    if os.path.exists(fn):
        d = np.load(fn); return d["z"], d["lnM"], d["d2N"]
    z = np.linspace(zmin, zmax, nz); lnM = np.linspace(np.log(Mmin), np.log(Mmax), nm)
    d2N = np.array([dndlnM(np.exp(lnM), zz, model) * dVdzdO(zz) * 4 * np.pi for zz in z])
    np.savez(fn, z=z, lnM=lnM, d2N=d2N)
    return z, lnM, d2N


def draw(seed, inst, sig=0.173, Mmin=1e13, zmax=3.0, model="tinker08", sinmin=-1.0):
    """Own Poisson draw over the region sin(dec) > sinmin (full sky if -1). Cell weights = 4-corner mean
    (bilinear-integral), uniform position within the cell (cells are fine)."""
    rng = np.random.default_rng(seed)
    z, lnM, d2N = sampler_grid(Mmin, zmax, model)
    dz, dm = z[1] - z[0], lnM[1] - lnM[0]
    cw = 0.25 * (d2N[:-1, :-1] + d2N[1:, :-1] + d2N[:-1, 1:] + d2N[1:, 1:]) * dz * dm
    frac = (1 - sinmin) / 2
    lam = cw.sum() * frac
    N = rng.poisson(lam)
    cdf = np.cumsum(cw.ravel()); cdf /= cdf[-1]
    k = np.searchsorted(cdf, rng.random(N))
    iz, im_ = np.divmod(k, cw.shape[1])
    zz = z[iz] + dz * rng.random(N)
    MM = np.exp(lnM[im_] + dm * rng.random(N))
    ra = 360 * rng.random(N)
    dec = np.degrees(np.arcsin(sinmin + (1 - sinmin) * rng.random(N)))
    th = theta500(MM, zz)
    Y = Ymed(MM, zz) * np.exp(sig * rng.standard_normal(N))
    q = Y / inst.sigY(th)
    n = rng.standard_normal(N)
    return dict(ra=ra, dec=dec, z=zz, M=MM, th=th, Y=Y, q=q, n=n, lam=lam)
