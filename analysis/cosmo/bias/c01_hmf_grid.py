"""Cosmology grid for the cluster-count likelihood: Tinker08 M500c d2N/dz/dlnM (full sky, h-free), E(z), D_A(z)
on a grid of (Omega_m, sigma8), all other parameters fixed at colossus planck18 (H0, Ob0, ns, flat, relspecies).

For each Omega_m node the colossus mass function is evaluated once at the fiducial sigma8; other sigma8 values
follow EXACTLY from sigma(M,z) -> s sigma(M,z) (s = sigma8/sigma8_fid; dln sigma/dlnM unchanged):
    dn/dlnM(s) = dn/dlnM(fid) * f_T08(s sigma; z, 500c) / f_T08(sigma; z, 500c).
Checked at the end against direct colossus calls at off-node (Om, s8) points.
Output: data_cosmo/hmf_grid.npz
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

OUT = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo/hmf_grid.npz"
OM = np.round(np.arange(0.20, 0.4501, 0.0125), 5)          # 21 nodes
S8 = np.round(np.arange(0.62, 1.0201, 0.02), 5)            # 21 nodes
Z = np.round(0.01 + 0.01 * np.arange(300), 5)              # 0.01 .. 3.00
LNM = np.linspace(np.log(1e13), np.log(1e16), 121)
C_KMS = 299792.458


def one(om, z=Z, lnM=LNM, s8list=S8):
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function as mf, peaks
    c = cosmology.setCosmology("planck18", Om0=float(om), persistence="")
    h = c.h; s8f = c.sigma8
    M = np.exp(lnM)
    out = np.zeros((len(s8list), len(z), len(lnM)))
    chi = c.comovingDistance(0.0, z) / h                     # Mpc
    Ez = c.Ez(z)
    dV = (C_KMS / c.H0) / Ez * chi ** 2 * 4 * np.pi          # full-sky dV/dz [Mpc^3]
    for iz, zz in enumerate(z):
        dn = mf.massFunction(M * h, float(zz), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
        R = peaks.lagrangianR(M * h)
        sig = c.sigma(R, float(zz))
        f0 = mf.modelTinker08(sig, float(zz), "500c")
        for ib, s8 in enumerate(s8list):
            f1 = mf.modelTinker08(sig * s8 / s8f, float(zz), "500c")
            out[ib, iz] = dn * f1 / f0 * dV[iz]
    return om, np.log(out), np.log(Ez), np.log(chi / (1 + z)), c.rho_c(0.0) * 1e9 * h ** 2, s8f


def direct(om, s8, zs=(0.1, 0.5, 1.0, 2.0)):
    from colossus.cosmology import cosmology
    from colossus.lss import mass_function as mf
    c = cosmology.setCosmology("planck18", Om0=float(om), sigma8=float(s8), persistence="")
    h = c.h; M = np.exp(LNM)
    res = []
    for zz in zs:
        chi = c.comovingDistance(0.0, zz) / h
        dV = (C_KMS / c.H0) / c.Ez(zz) * chi ** 2 * 4 * np.pi
        res.append(mf.massFunction(M * h, float(zz), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 * dV)
    return np.array(res)


if __name__ == "__main__":
    t0 = time.time()
    if not os.path.exists(OUT):
        with Pool(3) as p:
            rows = p.map(one, OM)
        lnN = np.stack([r[1] for r in rows]); lnE = np.stack([r[2] for r in rows]); lnDA = np.stack([r[3] for r in rows])
        rhoc0 = rows[0][4]; s8fid = rows[0][5]
        np.savez(OUT, OM=OM, S8=S8, Z=Z, LNM=LNM, lnN=lnN, lnE=lnE, lnDA=lnDA, rhoc0=rhoc0, s8fid=s8fid)
        print(f"grid built in {time.time()-t0:.0f}s: lnN {lnN.shape}, rho_c0 = {rhoc0:.6e} Msun/Mpc^3, s8_fid = {s8fid}")
    # ---- checks: (1) spline interpolation at off-node points vs direct colossus; (2) fiducial vs szsim table
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cosmolib as CL
    G = CL.HMFGrid()
    zs = (0.1, 0.5, 1.0, 2.0)
    for om, s8 in [(0.3111, 0.8102), (0.2731, 0.7737), (0.3567, 0.8519), (0.2263, 0.9411)]:
        lnN, _, _ = G.eval(om, s8)
        iz = [int(round((zz - 0.01) / 0.01)) for zz in zs]
        dd = direct(om, s8, zs)
        M = np.exp(LNM)
        for k, zz in enumerate(zs):
            sel = (M > 5e13) & (dd[k] > 1e-3)        # where d2N/dz/dlnM > 1e-3 (relevant halos)
            r = np.exp(lnN[iz[k], sel]) / dd[k, sel] - 1
            print(f"Om={om:.4f} s8={s8:.4f} z={zz}: grid-spline / direct colossus - 1: max|.| = {np.abs(r).max():.2e} "
                  f"over {sel.sum()} masses (M up to {M[sel].max():.2e})")
    from szsim import cosmo_hmf as ch
    T = ch.default_table()
    lnN, lnE, lnDA = G.eval(0.3111, 0.8102)
    for zz in (0.1, 0.5, 1.0, 2.0):
        iz = int(round((zz - 0.01) / 0.01))
        ref = ch.d2N_dzdlnM_fullsky(np.exp(LNM), zz)
        print(f"fiducial z={zz}: grid vs szsim d2N max|rel| (M<5e15) = "
              f"{np.abs(np.exp(lnN[iz]) / ref - 1)[np.exp(LNM) < 5e15].max():.2e}; "
              f"E rel {np.exp(lnE[iz]) / ch.Ez(zz) - 1:.1e}; DA rel {np.exp(lnDA[iz]) / ch.DA_Mpc(zz) - 1:.1e}")
    print(f"total {time.time()-t0:.0f}s")
