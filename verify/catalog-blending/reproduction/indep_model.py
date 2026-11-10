"""Independent re-implementation (no szsim imports) of the baseline model, for verification.
Units: angles arcmin, l in 1/arcmin, Y in arcmin^2, M500c in Msun (h-free), N_w in arcmin^2 (y^2 arcmin^2).
Profile FT via 3D route: g(u) = RATIO * int_0^5 x^2 P(x) j0(u x) dx / int_0^5 x^2 P(x) dx, u = l*theta500
(sphere truncated at 5R500, projected -> 2D FT equals 3D FT at k = l/D_A).  Normalised to Y_sph(R500)=Y500.
"""
import os
import numpy as np
from scipy.special import j0
from scipy.interpolate import CubicSpline
from colossus.cosmology import cosmology
from colossus.lss import mass_function

HERE = os.path.dirname(os.path.abspath(__file__))
ARCMIN = np.pi / 180 / 60
cosmo = cosmology.setCosmology("planck18")
h = cosmo.h
C_KMS = 299792.458
BEAM_FWHM = 1.4
SIG_B = BEAM_FWHM / np.sqrt(8 * np.log(2))
NW = (4e-6) ** 2
SIG_LNY = 0.173
c500, gam, alp, bet = 1.177, 0.3081, 1.0510, 5.4905


def P3(x):
    cx = c500 * x
    return 1 / (cx ** gam * (1 + cx ** alp) ** ((bet - gam) / alp))


# ---- profile FT
_xg, _wg = np.polynomial.legendre.leggauss(400)
def _gl(a, b, n_sub):
    e = np.linspace(a, b, n_sub + 1)
    xs = (0.5 * (e[1:] - e[:-1])[:, None] * _xg[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
    ws = (0.5 * (e[1:] - e[:-1])[:, None] * _wg[None, :]).ravel()
    return xs, ws
# substitution x = t^2 handles the x^-0.31 cusp
_t, _wt = _gl(0.0, np.sqrt(5.0), 100)
_x = _t ** 2; _wx = _wt * 2 * _t
_f = _x ** 2 * P3(_x)
I5 = np.sum(_wx * _f)
_t1, _wt1 = _gl(0.0, 1.0, 20)
I1 = np.sum(_wt1 * 2 * _t1 * (_t1 ** 4) * P3(_t1 ** 2))
RATIO = I5 / I1
U = np.concatenate([[0.0], np.geomspace(1e-4, 2e3, 6000)])
G = np.array([np.sum(_wx * _f * np.sinc(u * _x / np.pi)) for u in U]) / I5 * RATIO
_lu = np.log(U[1:])
_gs = CubicSpline(_lu, G[1:])


def g_of_u(u):
    u = np.asarray(u, float)
    out = np.empty_like(u)
    small = u < 1e-4
    out[small] = RATIO
    big = u > 2e3
    out[big] = 0.0
    mid = ~small & ~big
    out[mid] = _gs(np.log(u[mid]))
    return out


def B2(l):
    return np.exp(-(l * SIG_B) ** 2)


LGRID = np.linspace(0, 12.0, 24001)  # 1/arcmin; B^2(12) = e^-50


def sigmaY(theta):
    th = np.atleast_1d(theta)
    out = np.empty(len(th))
    for i, t in enumerate(th):
        out[i] = np.trapezoid(LGRID * B2(LGRID) * g_of_u(LGRID * t) ** 2, LGRID) / (2 * np.pi) / NW
    return 1 / np.sqrt(out)


def rho(thA, thB, s):
    s = np.atleast_1d(np.asarray(s, float))
    w = LGRID * B2(LGRID) * g_of_u(LGRID * thA) * g_of_u(LGRID * thB)
    I = np.trapezoid(w[None, :] * j0(np.outer(s, LGRID)), LGRID, axis=1) / (2 * np.pi) / NW
    return sigmaY(thA)[0] ** 2 * I


# sigma table (fine) for fast interpolation
_THT = np.geomspace(0.02, 200, 801)
_SIGT = sigmaY(_THT)
def sigmaY_fast(theta):
    return np.exp(np.interp(np.log(theta), np.log(_THT), np.log(_SIGT)))


def DC(theta):
    return sigmaY_fast(theta) ** 2 * RATIO ** 2 / NW


# ---- cosmology / scaling
def Ez(z):
    return cosmo.Ez(z)


_ZT = np.linspace(0, 3.2, 32001)
_CHIT = cosmo.comovingDistance(0.0, _ZT) / h
def DA(z):
    return np.interp(z, _ZT, _CHIT) / (1 + np.asarray(z))  # Mpc


def rhoc(z):
    return cosmo.rho_c(z) * 1e9 * h ** 2  # Msun/Mpc^3


def theta500(M, z):
    R = (3 * M / (4 * np.pi * 500 * rhoc(z))) ** (1 / 3)
    return R / DA(z) / ARCMIN


def Ymed(M, z):
    Ympc = 10 ** -0.19 * 1e-4 * (0.8 * M / 6e14) ** 1.79 * Ez(z) ** (2 / 3)  # Mpc^2
    return Ympc / DA(z) ** 2 / ARCMIN ** 2


# ---- HMF grid: d2N/(dz dlnM dOmega) per arcmin^2, at cell centres
def hmf_grid(nz=1500, nm=600, zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16):
    f = os.path.join(HERE, f"hmf_grid_{nz}_{nm}.npz")
    if os.path.exists(f):
        d = np.load(f); return d["ze"], d["lme"], d["dens"]
    ze = np.linspace(zmin, zmax, nz + 1); lme = np.linspace(np.log(Mmin), np.log(Mmax), nm + 1)
    zc = 0.5 * (ze[1:] + ze[:-1]); lmc = 0.5 * (lme[1:] + lme[:-1])
    Mc = np.exp(lmc)
    dens = np.zeros((nz, nm))
    for i, z in enumerate(zc):
        dn = mass_function.massFunction(Mc * h, z, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3  # Mpc^-3
        chi = cosmo.comovingDistance(0.0, z) / h
        dV = C_KMS / (100 * h * Ez(z)) * chi ** 2  # Mpc^3/sr per dz
        dens[i] = dn * dV * ARCMIN ** 2  # per arcmin^2 per dz per dlnM
    np.savez(f, ze=ze, lme=lme, dens=dens)
    return ze, lme, dens


def cell_counts(ze, lme, dens):
    return dens * np.diff(ze)[:, None] * np.diff(lme)[None, :]  # expected number per arcmin^2 per cell


def sample_halos(rng, area, ze, lme, dens):
    """Poisson sample on an area [arcmin^2]: returns z, M, Y (with scatter), theta500."""
    cc = cell_counts(ze, lme, dens).ravel()
    lam = cc.sum() * area
    n = rng.poisson(lam)
    cdf = np.cumsum(cc); cdf /= cdf[-1]
    k = np.searchsorted(cdf, rng.random(n))
    nm = len(lme) - 1
    iz, im = k // nm, k % nm
    z = ze[iz] + rng.random(n) * (ze[iz + 1] - ze[iz])
    lM = lme[im] + rng.random(n) * (lme[im + 1] - lme[im])
    M = np.exp(lM)
    Y = Ymed(M, z) * np.exp(SIG_LNY * rng.standard_normal(n))
    return z, M, Y, theta500(M, z)
