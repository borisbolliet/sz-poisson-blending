"""Independent chi2_min (Asimov deviance) and GOF power for config a and b, cat and map_clean, SO40 and full sky,
with the direct-colossus model (Gauss-Newton ML with safe-step Jacobians); compare with c09_table.log."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.stats import chi2, ncx2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import cosmolib as CL
import v01_indep_model as V

tc = np.load(f"{CL.DC}/truth_catalogue.npz"); tm = np.load(f"{CL.DC}/truth_maps.npz")
sc = float(tm["fullsky_deg2"] / tm["area_deg2"])
D = {"cat": tc["D_q5"], "map_clean": (tm["clean5"] - tm["iso5"]).sum(0) * sc}
mu_f = V.counts(V.FID)


def dev(d, mu):
    t = np.where(d > 0, d * np.log(np.where(d > 0, d, 1) / mu), 0.0)
    return 2 * np.sum(mu - d + t)


for cn, free in (("a", ["Om", "s8"]), ("b", ["Om", "s8", "omb"])):
    for dn in ("cat", "map_clean"):
        for fsky in (0.4, 1.0):
            d = np.clip(mu_f + D[dn], 0, None).ravel() * fsky
            x = np.array([V.FID[k] for k in free])
            for it in range(4):
                p = V.pvec(x, free); mu = V.counts(p).ravel() * fsky
                J = V.jac(free, 5.0, base=p).reshape(-1, len(free)) * fsky
                F = J.T @ (J / mu[:, None]); dx = np.linalg.solve(F, J.T @ (d / mu - 1)); x = x + dx
            mu = V.counts(V.pvec(x, free)).ravel() * fsky
            c2 = dev(d, mu); ndof = d.size - len(free)
            pw = ncx2.sf(chi2.ppf(0.95, ndof), ndof, c2)
            print(f"indep cfg {cn} {dn:9s} fsky {fsky}: chi2_min {c2:.1f} (ndof {ndof}), power {pw:.2f}; x={np.round(x,5)}", flush=True)
