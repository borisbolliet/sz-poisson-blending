"""Independent re-implementation of the naive (single-cluster) dN/dz/dq_obs used by analysis/cosmo/likelihood.
Different code path: astropy distances (not colossus), own volume element, own z/M grid (trapezoid, finer in z),
lognormal scatter integrated with 64-node Gauss-Hermite, Phi via scipy.  Only the Tinker08 dn/dlnM call
(colossus) and sigma_Y(theta) (szsim instrument) are shared.
Checks: (1) fiducial counts vs data_cosmo/counts_fid.npz Nn (12 z x 10 q bins)
        (2) cosmology propagation: ratio N(cosmo)/N(fid) at (Om0+0.02) and (sigma8+0.03), mine vs cclib."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.special import ndtr
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from colossus.cosmology import cosmology
from colossus.lss import mass_function
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc
from szsim import signal as sg

SZ = sg.SZModel()
ARC = np.pi / 180 / 60
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 1e9])


def counts(Om0, s8, h=0.6766, Ob0=0.049, ns=0.9665, nz_per=40, nM=400):
    cosmology.setCosmology("tmp", dict(flat=True, H0=100 * h, Om0=Om0, Ob0=Ob0, sigma8=s8, ns=ns), persistence="")
    ap = FlatLambdaCDM(H0=100 * h, Om0=Om0, Tcmb0=2.7255 * u.K, Neff=3.046, m_nu=0 * u.eV, Ob0=Ob0)
    lnM = np.linspace(np.log(1e13), np.log(1e16), nM)
    M = np.exp(lnM)
    xg, wg = np.polynomial.hermite_e.hermegauss(64); wg = wg / wg.sum()
    out = np.zeros((len(ZE) - 1, len(QE) - 1))
    for iz in range(len(ZE) - 1):
        z = np.linspace(ZE[iz], ZE[iz + 1], nz_per + 1)
        wz = np.full(z.size, z[1] - z[0]); wz[[0, -1]] *= 0.5
        for zz, wzz in zip(z, wz):
            DA = ap.angular_diameter_distance(zz).to_value(u.Mpc)
            chi = ap.comoving_distance(zz).to_value(u.Mpc)
            Ez = ap.efunc(zz)
            dVdz = 4 * np.pi * (299792.458 / (100 * h)) * chi ** 2 / Ez          # Mpc^3, full sky
            rhoc = ap.critical_density(zz).to_value(u.Msun / u.Mpc ** 3)
            dndlnM = mass_function.massFunction(M * h, zz, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
            th = (3 * M / (4 * np.pi * 500 * rhoc)) ** (1 / 3) / DA / ARC
            Y = 1e-4 * 10 ** (-0.19) * Ez ** (2 / 3) * (0.8 * M / 6e14) ** 1.79 / DA ** 2 / ARC ** 2
            q0 = Y / SZ.sigma_Y(th)
            q = q0[:, None] * np.exp(0.173 * xg)[None, :]
            P = np.stack([ndtr(QE[j + 1] - q) * 0 + (ndtr(q - QE[j]) - ndtr(q - QE[j + 1])) for j in range(len(QE) - 1)])
            P = (P * wg).sum(-1)                                                 # (nq, nM)
            wM = np.full(nM, lnM[1] - lnM[0]); wM[[0, -1]] *= 0.5
            out[iz] += wzz * dVdz * (P * (dndlnM * wM)[None, :]).sum(1)
    return out


if __name__ == "__main__":
    C = np.load(f"{cc.OUT}/counts_fid.npz")
    Nn = C["Nn"]
    mine = counts(cc.FID["Om0"], cc.FID["sigma8"])
    print(f"fiducial naive N(q>5): mine {mine.sum():,.1f}  cclib(counts_fid.npz) {Nn.sum():,.1f}  rel {mine.sum()/Nn.sum()-1:+.2e}")
    r = mine / Nn - 1
    print(f"   per-bin rel diff: max |.| {np.abs(r).max():.2e} (bins with N>5: {np.abs(r[Nn>5]).max():.2e}); z-marginal rel {np.array2string(mine.sum(1)/Nn.sum(1)-1, precision=5)}")
    print(f"   q-marginal rel {np.array2string(mine.sum(0)/Nn.sum(0)-1, precision=5)}")
    for lab, (om, s8) in {"Om+0.02": (cc.FID["Om0"] + 0.02, cc.FID["sigma8"]), "s8+0.03": (cc.FID["Om0"], cc.FID["sigma8"] + 0.03)}.items():
        mi = counts(om, s8)
        pop = cc.population(dict(cc.FID, Om0=om, sigma8=s8), zedges=ZE)
        cl = cc.counts_gauss(pop, C["QE"], len(ZE) - 1)
        print(f"{lab}: N/N_fid mine {mi.sum()/mine.sum():.5f}  cclib {cl.sum()/Nn.sum():.5f}; per-bin ratio-of-ratios max |.-1| "
              f"{np.abs((mi/mine)/(cl/Nn)-1)[Nn>5].max():.2e}")
