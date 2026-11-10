"""Independent re-implementation of the baseline model (no szsim imports).

Units: angles in arcmin, Y in arcmin^2, l in arcmin^-1, masses in Msun (h-free), distances Mpc (h-free).
Colossus works in Msun/h and Mpc/h: converted explicitly here.
"""
import numpy as np
from scipy import integrate
from scipy.special import j0
from colossus.cosmology import cosmology
from colossus.lss import mass_function

COSMO = cosmology.setCosmology("planck18")
H = COSMO.h
ARCMIN_PER_RAD = 180 * 60 / np.pi
SR_TO_ARCMIN2 = ARCMIN_PER_RAD ** 2
DEG2_FULLSKY = 4 * np.pi * (180 / np.pi) ** 2
C_KMS = 299792.458

# ---------------- instrument
FWHM = 1.4
SIGB = FWHM / np.sqrt(8 * np.log(2))
NW = (4e-6) ** 2           # arcmin^2 (y^2 arcmin^2)

# ---------------- Arnaud GNFW, truncated at 5 R500
C500, GAM, ALP, BET = 1.177, 0.3081, 1.0510, 5.4905


def P3(x):
    cx = C500 * x
    return cx ** (-GAM) * (1 + cx ** ALP) ** (-(BET - GAM) / ALP)


def Ysph(xmax):
    return integrate.quad(lambda x: 4 * np.pi * x * x * P3(x), 0, xmax, limit=400, points=[1e-3, 0.1, 1])[0]


I1 = Ysph(1.0)
I5 = Ysph(5.0)
RATIO = I5 / I1


def proj(s):
    zmax = np.sqrt(max(25 - s * s, 0.0))
    if zmax == 0:
        return 0.0
    f = lambda z: P3(np.sqrt(s * s + z * z))
    return 2 * integrate.quad(f, 0, zmax, limit=400, points=[min(1e-3, zmax / 2), min(1.0, zmax * 0.9)])[0]


def build_g(nu=6001, umax=3000.0):
    """g(u) = 2pi int s tau(s) J0(u s) ds, tau = unit-Y500 projected template in units theta500^-2; g(0)=RATIO."""
    # s grid: dense; integrand s*tau(s) ~ s^(1-gamma)... finite
    sg = np.concatenate([np.geomspace(1e-6, 1e-2, 400, endpoint=False), np.linspace(1e-2, 5.0, 40000)])
    # compute projected profile on a coarser grid then interpolate in log
    sc = np.concatenate([np.geomspace(1e-6, 0.05, 200, endpoint=False), np.linspace(0.05, 4.9999, 800)])
    pc = np.array([proj(s) for s in sc])
    p = np.exp(np.interp(np.log(sg), np.log(sc), np.log(np.maximum(pc, 1e-300))))
    p[sg > 4.9999] = np.interp(sg[sg > 4.9999], sc, pc)
    tau = p / I1
    norm = np.trapezoid(2 * np.pi * sg * tau, sg)
    u = np.unique(np.concatenate([np.linspace(0, 20, 8001), np.linspace(20, 200, 7201), np.linspace(200, umax, 2801)]))
    nu = u.size
    g = np.empty(nu)
    w = 2 * np.pi * sg * tau
    for a in range(0, nu, 200):
        g[a:a + 200] = np.trapezoid(w[None, :] * j0(np.outer(u[a:a + 200], sg)), sg, axis=1)
    return u, g, norm


_G = None


def gfun(u):
    global _G
    if _G is None:
        import os
        fn = os.path.join(os.path.dirname(__file__), "own_g_table.npz")
        if os.path.exists(fn):
            f = np.load(fn); _G = (f["u"], f["g"])
        else:
            uu, gg, nrm = build_g()
            np.savez(fn, u=uu, g=gg, norm=nrm)
            _G = (uu, gg)
    uu, gg = _G
    return np.interp(u, uu, gg, right=0.0)


# ---------------- matched filter (continuum Hankel)
LGRID = np.linspace(0, 14.0, 14001)        # arcmin^-1 ; B^2 at l=14 is exp(-69)
BL = np.exp(-0.5 * (LGRID * SIGB) ** 2)


def sigmaY(theta):
    theta = np.atleast_1d(theta)
    out = np.empty(theta.size)
    for i, t in enumerate(theta):
        ig = LGRID * BL ** 2 * gfun(LGRID * t) ** 2 / NW / (2 * np.pi)
        out[i] = 1 / np.sqrt(np.trapezoid(ig, LGRID))
    return out


def response(theta_k, theta_b, r):
    """R_k(r; theta_b): Y_hat of the scale theta_k filter at distance r from a unit-Y500 halo of size theta_b."""
    sk = sigmaY(theta_k)[0]
    ig = LGRID * BL ** 2 * gfun(LGRID * theta_k) * gfun(LGRID * theta_b) * j0(LGRID * r) / (2 * np.pi)
    return sk ** 2 / NW * np.trapezoid(ig, LGRID)


# ---------------- cosmology, HMF, scaling relation
def Ez(z):
    return COSMO.Ez(z)


def DA(z):
    return COSMO.angularDiameterDistance(z) / H          # Mpc


def chi(z):
    return COSMO.comovingDistance(0.0, z) / H            # Mpc


def dVdzdOmega(z):
    return C_KMS / (100 * H * Ez(z)) * chi(z) ** 2       # Mpc^3 / sr  (flat)


def dndlnM(M, z):
    """M in Msun -> dn/dlnM in Mpc^-3."""
    return mass_function.massFunction(M * H, z, mdef="500c", model="tinker08", q_out="dndlnM") * H ** 3


def rho_c(z):
    return COSMO.rho_c(z) * H ** 2 * 1e9                 # Msun/Mpc^3


def Y500_med(M, z):
    """arcmin^2."""
    Ympc = 1e-4 * 10 ** (-0.19) * (0.8 * M / 6e14) ** 1.79 * Ez(z) ** (2 / 3) / DA(z) ** 2   # Mpc^2 / DA^2
    return Ympc * SR_TO_ARCMIN2


def theta500(M, z):
    R = (3 * M / (4 * np.pi * 500 * rho_c(z))) ** (1 / 3)
    return R / DA(z) * ARCMIN_PER_RAD


SIG_LNY = 0.173


def hmf_table(zmin=0.01, zmax=3.0, lnMmin=np.log(1e13), lnMmax=np.log(1e16), nz=600, nM=600):
    """d2N/dz/dlnM per arcmin^2 on a grid (cell centres) and cell sizes."""
    ze = np.linspace(zmin, zmax, nz + 1); Me = np.linspace(lnMmin, lnMmax, nM + 1)
    zc = 0.5 * (ze[1:] + ze[:-1]); lc = 0.5 * (Me[1:] + Me[:-1])
    d2 = np.empty((nz, nM))
    for i, z in enumerate(zc):
        d2[i] = dndlnM(np.exp(lc), z) * dVdzdOmega(z) / SR_TO_ARCMIN2
    return zc, lc, d2, ze[1] - ze[0], Me[1] - Me[0]
