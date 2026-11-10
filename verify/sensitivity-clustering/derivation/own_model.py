"""Independent minimal SZ model (does NOT import szsim or analysis code).
Units: M in Msun (h-free), distances in Mpc, angles in arcmin, Y in arcmin^2."""
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function

cos = cosmology.setCosmology("planck18")
h = cos.h
ARC = np.pi / 180 / 60
CK = 299792.458

# --- GNFW Arnaud+10
c5, ga, al, be = 1.177, 0.3081, 1.0510, 5.4905
P = lambda x: 1 / ((c5 * x) ** ga * (1 + (c5 * x) ** al) ** ((be - ga) / al))
Ysph = lambda X: quad(lambda x: 4 * np.pi * x * x * P(x), 0, X, limit=500, epsrel=1e-11)[0]
RATIO = Ysph(5.0) / Ysph(1.0)
_norm = Ysph(1.0)


def proj(s):
    if s >= 5: return 0.0
    L = np.sqrt(25 - s * s)
    return 2 * quad(lambda l: P(np.hypot(s, l)), 0, L, limit=500, epsrel=1e-10)[0] / _norm


# projected profile on a grid in s = theta/theta500, Hankel transform g(u)
_s = np.concatenate([np.geomspace(1e-6, 0.05, 400), np.linspace(0.05, 5, 3000)[1:]])
_f = np.array([proj(x) for x in _s])
_u = np.concatenate([[0], np.geomspace(1e-3, 3e3, 3000)])


def _hankel(u):
    # g(u) = 2 pi int s f(s) J0(u s) ds ; f ~ s^{-ga}... integrable near 0; use trapezoid in s
    return np.array([2 * np.pi * np.trapezoid(_s * _f * j0(uu * _s), _s) for uu in u])


_g = _hankel(_u)


def g_of(u):
    return np.interp(u, _u, _g, right=0.0)


def sigmaY(theta500, beam, noise):
    """matched-filter sigma_Y [arcmin^2] for unit-Y500 template at theta500 [arcmin]."""
    sb = beam / np.sqrt(8 * np.log(2))
    l = np.linspace(0, 12 / sb, 40000)
    B2 = np.exp(-(l * sb) ** 2)
    out = []
    for t in np.atleast_1d(theta500):
        integ = l * B2 * g_of(l * t) ** 2 / (2 * np.pi)
        out.append(noise / np.sqrt(np.trapezoid(integ, l)))
    return np.array(out)


def Ez(z): return cos.Ez(z)


def DA(z): return cos.angularDiameterDistance(z) / h


def chi(z): return cos.comovingDistance(0.0, z) / h


def rhoc(z): return cos.rho_c(z) * h ** 2 * 1e9  # Msun/Mpc^3 (colossus rho_c in Msun h^2/kpc^3)


def R500(M, z): return (3 * M / (4 * np.pi * 500 * rhoc(z))) ** (1 / 3)


def theta500(M, z): return R500(M, z) / DA(z) / ARC


def Ymed(M, z):
    Y_Mpc2 = 1e-4 * 10 ** -0.19 * (0.8 * M / 6e14) ** 1.79 * Ez(z) ** (2 / 3)
    return Y_Mpc2 / DA(z) ** 2 / ARC ** 2


def dVdzdO(z): return CK / (cos.H0 * Ez(z)) * chi(z) ** 2


def dndlnM(M, z, model="tinker08"):
    return mass_function.massFunction(M * h, z, mdef="500c", model=model, q_out="dndlnM") * h ** 3


def grid(nz=600, nM=400, zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16, model="tinker08"):
    z = np.linspace(zmin, zmax, nz); lnM = np.linspace(np.log(Mmin), np.log(Mmax), nM)
    M = np.exp(lnM)
    d2N = np.array([dndlnM(M, zz, model) for zz in z]) * (4 * np.pi * dVdzdO(z))[:, None]  # full sky per dz dlnM
    return z, lnM, d2N


def trapw(x):
    w = np.full(len(x), x[1] - x[0]); w[0] *= .5; w[-1] *= .5; return w


def pdet_grid(z, M, beam, noise, sig, qc, nsc=41):
    """P(q_obs > qc) with lognormal Y scatter, q_obs = Y/sigma_Y + N(0,1)."""
    th_grid = np.geomspace(0.01, 200, 500)
    sY_grid = sigmaY(th_grid, beam, noise)
    out = np.empty((len(z), len(M)))
    if sig > 0:
        xg, wg = np.polynomial.hermite_e.hermegauss(nsc); wg = wg / wg.sum()
    for i, zz in enumerate(z):
        th = theta500(M, zz); Y = Ymed(M, zz)
        sY = np.exp(np.interp(np.log(th), np.log(th_grid), np.log(sY_grid)))
        if sig > 0:
            q = Y[:, None] * np.exp(sig * xg[None, :]) / sY[:, None]
            out[i] = (ndtr(q - qc) * wg).sum(1)
        else:
            out[i] = ndtr(Y / sY - qc)
    return out
