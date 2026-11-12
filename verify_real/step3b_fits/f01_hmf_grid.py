"""f01 (step-3b verification, lens 'fits'): my own HMF grid for the count model, wider than the step-2 verifier's.

Adapted from verify_real/step2_catalogue/vc02_hmf_grid.py (written independently of the step-2 build).  Direct colossus
calls (planck18 with Om0 and sigma8 set; Tinker08, 500c) at EVERY (Om, sigma8) node, on my own integration nodes:
  z: 6 Gauss-Legendre nodes inside each of the 16 cosmology z bins (96 nodes);
  ln M500: 553 uniform nodes on [ln 1e13, ln 1e16] (Simpson weights);
  Om: 0.24..0.40 step 0.005 (33), sigma8: 0.74..0.90 step 0.005 (33)  [step 3b's fits reach Om 0.24-0.37, s8 0.77-0.865].
Stored: ln d2N/dz/dlnM (full sky, h-free), ln E(z), ln D_A(z) [Mpc], rho_c0.
Checks: (1) fiducial vs szsim's d2N_dzdlnM_fullsky (the sampler of the tiles); (2) the tensor cubic-spline interpolation
at off-node points vs direct colossus.  Output: out/hmf_grid_f01.npz
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from scipy.interpolate import CubicSpline

t0 = time.time()
C_KMS = 299792.458
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
OM = np.round(np.arange(0.24, 0.40001, 0.005), 6)
S8 = np.round(np.arange(0.74, 0.90001, 0.005), 6)
xg, wg = np.polynomial.legendre.leggauss(6)
ZN = np.concatenate([0.5 * (b - a) * xg + 0.5 * (b + a) for a, b in zip(ZED[:-1], ZED[1:])])
ZW = np.concatenate([0.5 * (b - a) * wg for a, b in zip(ZED[:-1], ZED[1:])])
ZB = np.repeat(np.arange(len(ZED) - 1), 6)
LNM = np.linspace(np.log(1e13), np.log(1e16), 553)
h = LNM[1] - LNM[0]
WM = np.full(len(LNM), 2.0); WM[1:-1:2] = 4.0; WM[0] = WM[-1] = 1.0; WM *= h / 3.0       # Simpson


def lnN_at(om, s8, z=ZN, lnM=LNM):
    c = cosmology.setCosmology("planck18", Om0=float(om), sigma8=float(s8), persistence="")
    hh = c.h
    M = np.exp(lnM)
    chi = c.comovingDistance(0.0, z) / hh
    dV = (C_KMS / c.H0) / c.Ez(z) * chi ** 2 * 4 * np.pi
    out = np.empty((len(z), len(lnM)))
    for i, zz in enumerate(z):
        dn = mf.massFunction(M * hh, float(zz), mdef="500c", model="tinker08", q_out="dndlnM") * hh ** 3
        out[i] = np.log(dn * dV[i])
    return out, np.log(c.Ez(z)), np.log(chi / (1 + z)), c.rho_c(0.0) * 1e9 * hh ** 2


fn = f"{HERE}/out/hmf_grid_f01.npz"
L = np.zeros((len(OM), len(S8), len(ZN), len(LNM)), np.float32)
LE = np.zeros((len(OM), len(ZN))); LD = np.zeros((len(OM), len(ZN)))
for i, om in enumerate(OM):
    for j, s8 in enumerate(S8):
        L[i, j], e, d, rhoc0 = lnN_at(om, s8)
        if j == 0:
            LE[i], LD[i] = e, d
    if i % 5 == 0:
        print(f"  Om node {i+1}/{len(OM)} [{time.time()-t0:.0f}s]", flush=True)
np.savez(fn, OM=OM, S8=S8, ZN=ZN, ZW=ZW, ZB=ZB, LNM=LNM, WM=WM, L=L, LE=LE, LD=LD, rhoc0=rhoc0)
print(f"grid {L.shape} built in {time.time()-t0:.0f}s; rho_c0 = {rhoc0:.6e} Msun/Mpc^3")

from szsim import cosmo_hmf as ch
print("(1) fiducial: direct colossus at my nodes (Om 0.3111, s8 0.8102) vs szsim d2N_dzdlnM_fullsky:")
ln_fid, _, _, _ = lnN_at(0.3111, 0.8102)
for zz in (0.05, 0.5, 1.0, 2.0):
    i = int(np.argmin(np.abs(ZN - zz)))
    ref = ch.d2N_dzdlnM_fullsky(np.exp(LNM), ZN[i])
    sel = np.exp(ln_fid[i]) > 1e-3
    print(f"    z = {ZN[i]:.4f}: max |rel| (d2N > 1e-3) = {np.abs(np.exp(ln_fid[i, sel]) / ref[sel] - 1).max():.1e}")


def wts(nodes, x):
    return CubicSpline(nodes, np.eye(len(nodes)))(x)


print("(2) tensor not-a-knot cubic spline at off-node points vs direct colossus (nodes with d2N > 1e-2):")
for om, s8 in ((0.3111, 0.8102), (0.2963, 0.7891), (0.3297, 0.8456), (0.2457, 0.7712), (0.3689, 0.8647), (0.3912, 0.8873)):
    lnI = np.tensordot(np.outer(wts(OM, om), wts(S8, s8)), L.astype(np.float64), axes=([0, 1], [0, 1]))
    lnD, _, _, _ = lnN_at(om, s8)
    sel = lnD > np.log(1e-2)
    r = np.exp(lnI - lnD)[sel] - 1
    wN = np.exp(lnD)[sel]
    print(f"    Om={om:.4f} s8={s8:.4f}: max |rel| {np.abs(r).max():.1e}; count-weighted mean |rel| {np.sum(wN*np.abs(r))/wN.sum():.1e}")
print(f"total {time.time()-t0:.0f}s")
