"""Independent verifier library (does NOT import cclib).
Population: colossus Tinker08 M500c directly, Gauss-Legendre in z (per analysis z bin) and lnM,
Gauss-Hermite in lnY.  sigma_Y from szsim (instrument only).  Blended kernel: data_cosmo/G_fid_b03rates.npz
interpolated with my own numpy trilinear code."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from szsim import signal as sg

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
SZ = sg.SZModel()
ARC = np.pi / 180 / 60
CKMS = 299792.458
P18 = dict(Om0=0.3111, Ob0=0.049, H0=67.66, sigma8=0.8102, ns=0.9665)
SR0 = dict(logA=-0.19, alpha=1.79, beta_E=2 / 3, omb=0.8, slnY=0.173)
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
_c = {}


def set_cosmo(Om0, sigma8):
    key = (round(Om0, 12), round(sigma8, 12))
    if key not in _c:
        if len(_c) > 30:
            _c.clear()
        _c[key] = cosmology.Cosmology(name="v", flat=True, Om0=Om0, Ob0=P18["Ob0"], H0=P18["H0"], sigma8=sigma8,
                                      ns=P18["ns"], persistence="", print_warnings=False)
    cosmology.setCurrent(_c[key])
    return _c[key]


def nodes(nzper=6, nsub=6, nM=200, ngh=24, Mmin=1e13, Mmax=1e16):
    """each z bin split into nsub sub-intervals (log-spaced in 1+z), nzper GL nodes each"""
    xz, wz = np.polynomial.legendre.leggauss(nzper)
    z, dz, zb = [], [], []
    for i in range(len(ZE) - 1):
        sub = np.geomspace(ZE[i], ZE[i + 1], nsub + 1)
        for a, b in zip(sub[:-1], sub[1:]):
            z.append(0.5 * (a + b) + 0.5 * (b - a) * xz); dz.append(0.5 * (b - a) * wz); zb.append(np.full(nzper, i))
    xm, wm = np.polynomial.legendre.leggauss(nM)
    la, lb = np.log(Mmin), np.log(Mmax)
    lnM = 0.5 * (la + lb) + 0.5 * (lb - la) * xm; dl = 0.5 * (lb - la) * wm
    xg, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    return dict(z=np.concatenate(z), wz=np.concatenate(dz), zb=np.concatenate(zb), lnM=lnM, wM=dl, xg=xg, wg=wg)


ND = nodes()


def population(Om0=P18["Om0"], sigma8=P18["sigma8"], omb=0.8, alpha=1.79, beta_E=2 / 3, slnY=0.173, logA=-0.19, nd=ND):
    c = set_cosmo(Om0, sigma8)
    h = c.h
    z = nd["z"]; M = np.exp(nd["lnM"])
    chi = c.comovingDistance(0.0, z) / h                 # Mpc
    DA = chi / (1 + z); Ez = c.Ez(z)
    dVdz = 4 * np.pi * CKMS / (100 * h) / Ez * chi ** 2   # full-sky comoving volume per dz [Mpc^3]
    rhoc = c.rho_c(z) * 1e9 * h ** 2                     # Msun/Mpc^3
    n = np.array([mass_function.massFunction(M * h, zz, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 for zz in z])
    w = n * (dVdz * nd["wz"])[:, None] * nd["wM"][None, :]
    R500 = (3 * M[None, :] / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3)
    th = R500 / DA[:, None] / ARC
    Y = 1e-4 * 10 ** logA * Ez[:, None] ** beta_E * (omb * M[None, :] / 6e14) ** alpha / DA[:, None] ** 2 / ARC ** 2
    q0 = Y / SZ.sigma_Y(th)
    q = q0[..., None] * np.exp(slnY * nd["xg"])
    W = w[..., None] * nd["wg"]
    sh = q.shape
    return dict(q=q.ravel(), w=W.ravel(), th=np.broadcast_to(th[..., None], sh).ravel().copy(),
                zb=np.broadcast_to(nd["zb"][:, None, None], sh).ravel().copy(), Ymean_density=None,
                Y=np.broadcast_to(Y[..., None], sh).ravel() * np.exp(slnY * np.broadcast_to(nd["xg"], sh).ravel()))


def bin_counts(P, zb, w, nzb=12):
    """P: (npts, nqbins) probabilities -> (nzb, nqbins) counts"""
    out = np.zeros((nzb, P.shape[1]))
    for j in range(P.shape[1]):
        out[:, j] = np.bincount(zb, weights=w * P[:, j], minlength=nzb)
    return out


def counts_gauss(pop, shift=0.0, var=1.0, qe=QE):
    s = np.sqrt(var)
    e = qe[None, :]
    F = np.where(np.isfinite(e), ndtr((e - pop["q"][:, None] - np.atleast_1d(shift)[:, None] * np.ones((1, 1))) /
                                       (np.atleast_1d(s)[:, None] * np.ones((1, 1)))), 1.0)
    return bin_counts(np.diff(F, axis=1), pop["zb"], pop["w"])


_K = None


def kernel():
    global _K
    if _K is None:
        d = np.load(f"{ROOT}/data_cosmo/G_fid_b03rates.npz")
        _K = dict(G=d["G"], lth=np.log(d["th"]), q=d["qnodes"], t=d["tgrid"])
    return _K


def _lin(x, grid):
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    f = np.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0)
    return i, f


def counts_blended(pop, qe=QE, chunk=200_000):
    K = kernel(); G = K["G"]
    out = np.zeros((12, len(qe) - 1))
    for s in range(0, pop["q"].size, chunk):
        q = pop["q"][s:s + chunk]; w = pop["w"][s:s + chunk]; zb = pop["zb"][s:s + chunk]
        ia, fa = _lin(np.log(pop["th"][s:s + chunk]), K["lth"])
        ik, fk = _lin(q, K["q"])
        F = np.empty((q.size, len(qe)))
        for j, e in enumerate(qe):
            if not np.isfinite(e):
                t = np.full(q.size, K["t"][-1])
            else:
                t = e - q
            it, ft = _lin(t, K["t"])
            below = t < K["t"][0]
            v = 0.0
            for da, wa in ((0, 1 - fa), (1, fa)):
                for dk, wk in ((0, 1 - fk), (1, fk)):
                    g = (1 - ft) * G[ia + da, ik + dk, it] + ft * G[ia + da, ik + dk, it + 1]
                    v = v + wa * wk * g
            F[:, j] = np.where(below, 0.0, v)
        out += bin_counts(np.diff(F, axis=1), zb, w)
    return out


def cash(D, mu):
    D = np.asarray(D, float); mu = np.asarray(mu, float)
    return 2 * np.sum(mu - D + np.where(D > 0, D * np.log(np.where(D > 0, D, 1) / mu), 0.0))
