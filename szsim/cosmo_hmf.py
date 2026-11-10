"""Cosmology + halo mass function tables for the full-sky halo population.

Units (h-free everywhere in the public API):
  * masses M500c in Msun (NOT Msun/h); colossus is called with M*h internally
  * distances in Mpc (physical angular-diameter distance D_A, comoving chi)
  * d2N/dz/dlnM is the FULL-SKY (4 pi sr) number of halos per unit z per unit ln M500c

Model: colossus 'planck18', Tinker08 HMF for mdef='500c' (colossus default
power spectrum = Eisenstein & Hu 1998 transfer function).
"""
import os
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function

C_KMS = 299792.458
FULLSKY_SR = 4.0 * np.pi
ARCMIN = np.pi / 180.0 / 60.0  # rad
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_COSMO = None


def get_cosmology(name="planck18"):
    global _COSMO
    if _COSMO is None or _COSMO.name != name:
        _COSMO = cosmology.setCosmology(name)
    return _COSMO


def h():
    return get_cosmology().h


def Ez(z):
    """E(z) = H(z)/H0."""
    return get_cosmology().Ez(np.asarray(z, dtype=float))


_CHI_SPLINE = None


def chi_Mpc_exact(z):
    """Comoving distance [Mpc] straight from colossus (slow for large arrays)."""
    c = get_cosmology()
    return c.comovingDistance(0.0, np.asarray(z, dtype=float)) / c.h


def chi_Mpc(z):
    """Comoving distance [Mpc]: linear interpolation of colossus on a 1e-4 z grid over [0,4]
    (relative error < 1e-7 for z >= 0.005; verified in analysis/build)."""
    global _CHI_SPLINE
    if _CHI_SPLINE is None:
        zg = np.linspace(0.0, 4.0, 40001)
        _CHI_SPLINE = (zg, chi_Mpc_exact(zg))
    return np.interp(np.asarray(z, dtype=float), *_CHI_SPLINE)


def DA_Mpc(z):
    """Physical angular-diameter distance [Mpc] (flat cosmology)."""
    z = np.asarray(z, dtype=float)
    return chi_Mpc(z) / (1.0 + z)


def rho_crit_Msun_Mpc3(z):
    """Critical density at z [Msun / Mpc^3, physical]."""
    c = get_cosmology()
    # colossus: rho_c in Msun h^2 / kpc^3
    return c.rho_c(np.asarray(z, dtype=float)) * 1e9 * c.h ** 2


def dV_dz_dOmega_Mpc3(z):
    """Comoving volume element dV/dz/dOmega [Mpc^3 / sr]."""
    c = get_cosmology()
    return (C_KMS / c.H0) / Ez(z) * chi_Mpc(z) ** 2


def dndlnM_Mpc3(M_Msun, z, model="tinker08", mdef="500c"):
    """Comoving dn/dlnM [Mpc^-3] at scalar z, M in Msun (h-free)."""
    c = get_cosmology()
    M_h = np.asarray(M_Msun, dtype=float) * c.h
    return mass_function.massFunction(M_h, float(z), mdef=mdef, model=model, q_out="dndlnM") * c.h ** 3


def d2N_dzdlnM_fullsky(M_Msun, z, model="tinker08", mdef="500c"):
    """Full-sky d2N/dz/dlnM at scalar z for array of M (Msun)."""
    return dndlnM_Mpc3(M_Msun, z, model, mdef) * dV_dz_dOmega_Mpc3(z) * FULLSKY_SR


class HMFTable:
    """Tabulated full-sky d2N/dz/dlnM on a regular (z, lnM) grid.

    Attributes
    ----------
    z : (nz,) z nodes (regular, includes zmin and zmax)
    lnM : (nM,) ln(M500c/Msun) nodes (regular)
    d2N : (nz, nM) full-sky d2N/dz/dlnM at the nodes
    """

    def __init__(self, zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16, nz=1201, nM=1201,
                 model="tinker08", mdef="500c", cache=True):
        self.zmin, self.zmax, self.Mmin, self.Mmax = zmin, zmax, Mmin, Mmax
        self.model, self.mdef = model, mdef
        self.z = np.linspace(zmin, zmax, nz)
        self.lnM = np.linspace(np.log(Mmin), np.log(Mmax), nM)
        fn = os.path.join(DATA_DIR, f"hmf_table_{model}_{mdef}_z{zmin}-{zmax}_M{Mmin:.0e}-{Mmax:.0e}_{nz}x{nM}.npz")
        if cache and os.path.exists(fn):
            self.d2N = np.load(fn)["d2N"]
        else:
            M = np.exp(self.lnM)
            self.d2N = np.array([d2N_dzdlnM_fullsky(M, zz, model, mdef) for zz in self.z])
            if cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                np.savez(fn, z=self.z, lnM=self.lnM, d2N=self.d2N)
        self.dz = self.z[1] - self.z[0]
        self.dlnM = self.lnM[1] - self.lnM[0]

    # --- integrals (Simpson on the regular grid; both node counts odd) ---
    @staticmethod
    def _simpson(y, dx, axis=-1):
        from scipy.integrate import simpson
        return simpson(y, dx=dx, axis=axis)

    def dN_dz(self, Mmin=None, Mmax=None):
        """Full-sky dN/dz on self.z for Mmin <= M500 < Mmax (Msun)."""
        w = self._mass_weights(Mmin, Mmax)
        return (self.d2N * w[None, :]).sum(axis=1)

    def N_total(self, Mmin=None, Mmax=None, zmin=None, zmax=None):
        """Expected full-sky number of halos in the given M500 (Msun) and z range."""
        wM = self._mass_weights(Mmin, Mmax)
        wz = self._weights(self.z, zmin if zmin is not None else self.zmin,
                           zmax if zmax is not None else self.zmax)
        return float(wz @ self.d2N @ wM)

    def _mass_weights(self, Mmin, Mmax):
        lo = np.log(Mmin) if Mmin is not None else self.lnM[0]
        hi = np.log(Mmax) if Mmax is not None else self.lnM[-1]
        return self._weights(self.lnM, lo, hi)

    @staticmethod
    def _weights(x, lo, hi):
        """Quadrature weights for integral over [lo,hi] of a function sampled on regular x.
        Trapezoid with exact partial-cell clipping (function linearly interpolated)."""
        dx = x[1] - x[0]
        lo = max(lo, x[0]); hi = min(hi, x[-1])
        w = np.zeros_like(x)
        if hi <= lo:
            return w
        # integrate hat functions over [lo, hi]
        for i in range(len(x) - 1):
            a, b = x[i], x[i + 1]
            l, r = max(a, lo), min(b, hi)
            if r <= l:
                continue
            # hat i decreasing on [a,b]: (b-t)/dx ; hat i+1 increasing: (t-a)/dx
            w[i] += ((b - l) ** 2 - (b - r) ** 2) / (2 * dx)
            w[i + 1] += ((r - a) ** 2 - (l - a) ** 2) / (2 * dx)
        return w

    def N_above(self, Mmin):
        return self.N_total(Mmin=Mmin)


_DEFAULT_TABLE = None


def default_table():
    """Shared default table: z in [0.01,3], M500c in [1e13,1e16] Msun."""
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = HMFTable()
    return _DEFAULT_TABLE
