"""Independent naive (single-cluster) count model in (z, q_obs) bins and mean-y, for verification.
Written from scratch: colossus HMF (Tinker08, 500c), own Y-M, Gauss-Legendre in z per bin, Simpson in lnM,
Gauss-Hermite in lnY scatter, analytic Gaussian noise.  Only sigma_Y(theta) is taken from szsim (filter noise)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr
from scipy.integrate import simpson
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from szsim import signal as sg

ARCMIN = np.pi / 180 / 60
C_KMS = 299792.458
P18 = dict(flat=True, H0=67.66, Om0=0.3111, Ob0=0.049, sigma8=0.8102, ns=0.9665)
ZE = np.array([0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0, np.inf])
SZM = sg.SZModel()
_n = [0]


def cosmo(s8, Om):
    _n[0] += 1
    p = dict(P18); p["sigma8"] = float(s8); p["Om0"] = float(Om)
    return cosmology.setCosmology(f"v{_n[0]}", **p, persistence="")


def zgrid(ng=10):
    x, w = np.polynomial.legendre.leggauss(ng)
    zs, ws, ib = [], [], []
    for b in range(len(ZE) - 1):
        a, c = ZE[b], ZE[b + 1]
        zs.append(0.5 * (c - a) * x + 0.5 * (c + a)); ws.append(0.5 * (c - a) * w); ib.append(np.full(ng, b))
    return np.concatenate(zs), np.concatenate(ws), np.concatenate(ib)


def counts(s8, Om, omb, slnY, nM=241, Mmin=1e13, Mmax=1e16, ngh=24, ng=10, want_y=False):
    c = cosmo(s8, Om); h = c.h
    z, wz, ib = zgrid(ng)
    lnM = np.linspace(np.log(Mmin), np.log(Mmax), nM); M = np.exp(lnM)
    chi = c.comovingDistance(0.0, z) / h                     # Mpc
    DA = chi / (1 + z)
    E = c.Ez(z)
    dVdzdO = C_KMS / c.H0 / E * chi ** 2                       # Mpc^3/sr
    rhoc = c.rho_c(z) * 1e9 * h ** 2                            # Msun/Mpc^3
    xg, wg = np.polynomial.hermite_e.hermegauss(ngh); wg = wg / wg.sum()
    N = np.zeros((len(ZE) - 1, len(QE) - 1))
    ytot = 0.0
    for i in range(len(z)):
        dn = mass_function.massFunction(M * h, z[i], mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
        d2N = dn * dVdzdO[i] * 4 * np.pi                         # full-sky per dz dlnM
        R = (3 * M / (4 * np.pi * 500 * rhoc[i])) ** (1 / 3)
        th = R / DA[i] / ARCMIN
        Y = 1e-4 * 10 ** -0.19 * E[i] ** (2 / 3) * (omb * M / 6e14) ** 1.79 / DA[i] ** 2 / ARCMIN ** 2
        q0 = Y / SZM.sigma_Y(th)
        q = q0[:, None] * np.exp(slnY * xg)[None, :]            # (nM, ngh)
        Phi = np.array([ndtr(q - e) if np.isfinite(e) else np.zeros_like(q) for e in QE])   # (11, nM, ngh)
        Pb = ((Phi[:-1] - Phi[1:]) * wg).sum(-1)                 # (10, nM)
        N[ib[i]] += wz[i] * simpson(d2N[None, :] * Pb, x=lnM, axis=-1)
        if want_y:
            ytot += wz[i] * simpson(d2N * Y * np.exp(0.5 * slnY ** 2), x=lnM)
    if want_y:
        return N, ytot * sg.RATIO_5R500 / (4 * np.pi / ARCMIN ** 2)   # mean y (Y5R500 per arcmin^2)
    return N
