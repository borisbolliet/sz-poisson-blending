"""Independent re-fit of c06 (b): data = blended counts at the shifted truth with the kernel G recomputed at the truth
(data_cosmo/G_<truth>_coarse.npz, saved by c06); fit model = population at the trial cosmology with G held at the
fiducial (G_fid_coarse.npz).  Free: Om0, sigma8 (scaling relation fixed).  Optimiser: scipy Nelder-Mead on Cash;
error: observed Hessian.  Also (a) the count change at the truth from holding G at the fiducial."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz"); ZE, QE = C["ZE"], C["QE"]; nz = len(ZE) - 1
Gf = np.load(f"{cc.OUT}/G_fid_coarse.npz")["G"]
for name, cp in (("s8+0.03", dict(cc.FID, sigma8=cc.FID["sigma8"] + 0.03)), ("Om+0.02", dict(cc.FID, Om0=cc.FID["Om0"] + 0.02))):
    Gt = np.load(f"{cc.OUT}/G_{name}_coarse.npz")["G"]
    pop = cc.population(cp, zedges=ZE)
    Dt = cc.counts_blended(Gt, pop, QE, nz); Df = cc.counts_blended(Gf, pop, QE, nz); Dn = cc.counts_gauss(pop, QE, nz)
    print(f"truth {name}: N_b(G_true) {Dt.sum():,.1f}, N_b(G_fid) {Df.sum():,.1f} ({100*(Df.sum()/Dt.sum()-1):+.3f}%), "
          f"N_b/N_n-1: true G {100*(Dt.sum()/Dn.sum()-1):+.3f}%, fid G {100*(Df.sum()/Dn.sum()-1):+.3f}%; Delta chi2 full {np.sum((Df-Dt)**2/Dt):.2f}")
    pt = np.array([cp["Om0"], cp["sigma8"]]); st = np.array([0.001, 0.001])
    m = lambda p: cc.counts_blended(Gf, cc.population(dict(cc.FID, Om0=p[0], sigma8=p[1]), zedges=ZE), QE, nz)
    for fs in (1.0, 0.4):
        f = lambda u: cc.cash(Dt * fs, m(pt + u * st) * fs)
        r = minimize(f, [0, 0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
        u = r.x; h = 0.5; f0 = f(u); H = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                ea = np.eye(2)[a] * h; eb = np.eye(2)[b] * h
                H[a, b] = (f(u + ea + eb) - f(u + ea - eb) - f(u - ea + eb) + f(u - ea - eb)) / (4 * h * h)
        sig = np.sqrt(np.diag(np.linalg.inv(H / 2))) * st
        print(f"   f_sky={fs}: best {np.round(pt + u*st, 4)} truth {np.round(pt, 4)} sigma {np.round(sig, 4)} bias/sigma {np.round(u*st/sig, 2)}; Cash_min {r.fun:.2f}", flush=True)
