"""V01: independent sigma_Y, rho, DC response, ybar, mu vs the analysis (table + analytic.npz)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["NUMBA_NUM_THREADS"] = "2"
import numpy as np
import indep_model as im
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/blending")
import blendlib as bl
from szsim import signal as sg

m = sg.SZModel(); tab = bl.load_rho_table()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
th = np.array([0.25, 0.5, 1, 3, 10, 30])
my = im.sigmaY(th); lib = m.sigma_Y(th)
print("sigma_Y mine:", my); print("sigma_Y lib :", lib); print("rel diff", my / lib - 1)
print("DC response mine [arcmin^2] at 0.25,1,3,10,30:", np.round(im.DC(np.array([0.25, 1, 3, 10, 30])), 3))
# rho comparison at random points
rng = np.random.default_rng(424242)
errs = []; worst = None
for k in range(80):
    a = float(np.exp(rng.uniform(np.log(0.26), np.log(40)))); b = float(np.exp(rng.uniform(np.log(0.26), np.log(40))))
    S = 5 * a + 5 * b + tab.R_BEAM
    s = np.sort(rng.uniform(0, 1, 10) ** 2 * S)
    mine = im.rho(a, b, s); r0 = im.rho(a, b, [0.0])[0]
    theirs = np.array([bl.rho_interp_py(tab, a, b, x) for x in s])
    e = np.max(np.abs(mine - theirs)) / r0
    errs.append(e)
errs = np.array(errs)
print(f"rho table vs my independent rho (80 pairs x 10 s): max|diff|/rho0 median {np.median(errs):.2e}, 95% {np.percentile(errs,95):.2e}, max {errs.max():.2e}")
for a, b, s in ((3, 0.5, 0), (2, 2, 2), (1, 1, 0), (0.5, 0.5, 1.0)):
    print(f"   rho({a},{b},{s}) mine {im.rho(a,b,[s])[0]:.4f}  table {bl.rho_interp_py(tab,a,b,s):.4f}  lib exact {m.cross_response(a,b,[s])[0]:.4f}")
# small-theta clipping: halos with theta < 0.25' are clipped to the first table node
for a, b, s in ((1.0, 0.1, 0.0), (1.0, 0.15, 1.0), (0.3, 0.1, 0.5)):
    print(f"   clip test rho({a},{b},{s}) mine {im.rho(a,b,[s])[0]:.5f}  table {bl.rho_interp_py(tab,a,b,s):.5f}")
# ybar
ze, lme, dens = im.hmf_grid()
cc = im.cell_counts(ze, lme, dens)
zc = 0.5 * (ze[1:] + ze[:-1]); Mc = np.exp(0.5 * (lme[1:] + lme[:-1]))
Z, MM = np.meshgrid(zc, Mc, indexing="ij")
Y = im.Ymed(MM, Z) * np.exp(0.5 * im.SIG_LNY ** 2)
TH = im.theta500(MM, Z)
cls = np.searchsorted([1e13, 3e13, 1e14], MM, side="right") - 1
yb = np.array([(cc * Y)[cls == c].sum() for c in range(3)])
nfs = cc.sum() * 4 * np.pi / im.ARCMIN ** 2
print(f"full-sky N(M>1e13) mine {nfs:,.0f}; N(>1e14) {cc[MM>1e14].sum()*4*np.pi/im.ARCMIN**2:,.0f}")
print("ybar per class mine", yb, "total", yb.sum(), "| analysis", A["ybar"], A["ybar"].sum(), "rel", yb / A["ybar"] - 1)
print(f"min theta500 of halos (z<=3, M>=1e13): {TH.min():.3f}'; fraction of halos with theta<0.25': {cc[TH<0.25].sum()/cc.sum():.3f}")
for t in (0.5237, 1.10, 2.30, 4.153, 8.70, 18.23):
    i = int(np.argmin(np.abs(A["th"] - t)))
    tt = A["th"][i]
    mu = yb * im.DC(tt) / im.sigmaY_fast(tt)
    print(f"  thA={tt:.3f}': mu mine {np.round(mu,4)} tot {mu.sum():.4f} | analysis mu_exact {np.round(A['mu_exact'][i],4)} tot {A['mu_exact'][i].sum():.4f}")
