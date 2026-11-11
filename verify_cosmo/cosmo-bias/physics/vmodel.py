"""Independent naive (single-cluster) SZ count model for verification.

Own implementation: direct colossus Tinker08 500c HMF at each (Om, s8) (no grid / spline / s8-rescaling),
Gauss-Legendre in z inside each z bin, own Y-M relation and theta500, own lognormal-scatter x Gaussian-noise
bin probabilities. Only sigma_Y(theta500) (matched-filter noise, an instrument property) is taken from szsim.
Params th = (Om, s8, omb=(1-b), alpha, slnY).
"""
import functools
import numpy as np
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf

ARCMIN = np.pi / 180 / 60
CKMS = 299792.458
FID = np.array([0.3111, 0.8102, 0.8, 1.79, 0.173])
NAMES = ["Om", "s8", "omb", "alpha", "slnY"]
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def qedges(qcut):
    return np.concatenate([np.geomspace(qcut, 100.0, 9), [np.inf]])


NGL = 10
_x, _w = np.polynomial.legendre.leggauss(NGL)
ZN = np.concatenate([0.5 * (b - a) * _x + 0.5 * (a + b) for a, b in zip(ZED[:-1], ZED[1:])])
ZW = np.concatenate([0.5 * (b - a) * _w for a, b in zip(ZED[:-1], ZED[1:])])
ZB = np.repeat(np.arange(len(ZED) - 1), NGL)
LNM = np.linspace(np.log(1e13), np.log(1e16), 241)
WM = np.full(LNM.size, LNM[1] - LNM[0]); WM[[0, -1]] *= 0.5
GHX, GHW = np.polynomial.hermite_e.hermegauss(60); GHW = GHW / GHW.sum()

from szsim import signal as _sg
_SZ = _sg.SZModel()


@functools.lru_cache(maxsize=4096)
def _cosmo(om, s8):
    c = cosmology.setCosmology("planck18", Om0=om, sigma8=s8, persistence="")
    h = c.h
    M = np.exp(LNM)
    dn = np.stack([mf.massFunction(M * h, float(z), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 for z in ZN])
    chi = c.comovingDistance(0.0, ZN) / h                    # Mpc
    Hz = 100 * h * c.Ez(ZN)
    dVdz = 4 * np.pi * CKMS / Hz * chi ** 2
    DA = chi / (1 + ZN)
    rhoc = c.rho_c(ZN) * 1e9 * h ** 2                        # Msun / Mpc^3
    return dn * dVdz[:, None], c.Ez(ZN), DA, rhoc


def counts(th, qcut=5.0, fsky=1.0, kernel=None):
    om, s8, omb, alpha, slnY = th
    d2N, E, DA, rhoc = _cosmo(round(float(om), 10), round(float(s8), 10))
    M = np.exp(LNM)[None, :]
    Ympc = 1e-4 * 10 ** (-0.19) * E[:, None] ** (2 / 3) * (omb * M / 6e14) ** alpha
    Y = Ympc / DA[:, None] ** 2 / ARCMIN ** 2                # arcmin^2
    R500 = (3 * M / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3)
    th500 = R500 / DA[:, None] / ARCMIN
    q0 = Y / _SZ.sigma_Y(th500)
    e = qedges(qcut)
    P = np.zeros(q0.shape + (len(e) - 1,))
    for x, w in zip(GHX, GHW):
        q = q0 * np.exp(slnY * x)
        cdf = np.stack([ndtr(q - ee) if np.isfinite(ee) else np.zeros_like(q) for ee in e], -1)
        P += w * (cdf[..., :-1] - cdf[..., 1:])
    integ = (d2N * WM[None, :])[..., None] * P               # (nz, nM, nq)
    perz = integ.sum(1) * ZW[:, None]
    out = np.zeros((len(ZED) - 1, len(e) - 1))
    np.add.at(out, ZB, perz)
    return fsky * out


STEPS = np.array([5e-4, 5e-4, 2e-3, 3e-3, 2e-3])


def jac(th, free, qcut=5.0, fsky=1.0):
    J = []
    for i in free:
        tp = th.copy(); tm = th.copy(); tp[i] += STEPS[i]; tm[i] -= STEPS[i]
        J.append(((counts(tp, qcut, fsky) - counts(tm, qcut, fsky)) / (2 * STEPS[i])).ravel())
    return np.array(J).T


def S8(om, s8):
    return s8 * np.sqrt(om / 0.3)
