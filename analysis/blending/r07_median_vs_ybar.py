"""[repair after verification] Are the 8-sky floor-1e13/3e13 medians of delta (which sit 0.0-0.4% below the
fine-grid analytic) low because these skies are low in the bright (M>1e14) Y-sum?  Regress the per-sky median
on the per-sky ratio ybar_MC/ybar_an of class M>1e14 and read off the intercept at ratio = 1."""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg
m = sg.SZModel()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
files = sorted(glob.glob(os.path.join(bl.DATA, "mc_sky*.npz")), key=lambda f: int(f.split("sky")[1][:-4]))
rthA = np.load(files[0])["rand_thA"]
yr = np.array([np.load(f)["ybar"][2] / A["ybar"][2] for f in files])
print("per-sky ybar(M>1e14) ratio:", np.round(yr, 4))
zs = []
for i, thA in enumerate(rthA):
    inode = int(np.argmin(np.abs(A["th"] - thA))); sA = float(m.sigma_Y(thA))
    for fi, fl in enumerate(A["FLOORS"]):
        cl = list(bl.FLOORS[fl])
        med = np.array([np.median(np.load(f)[f"rand_D_{i}"][:, cl].sum(1) / sA) for f in files])
        X = np.column_stack([np.ones(len(yr)), yr - 1])
        coef, res, *_ = np.linalg.lstsq(X, med, rcond=None)
        dof = len(yr) - 2; s2 = np.sum((med - X @ coef) ** 2) / dof
        cov = s2 * np.linalg.inv(X.T @ X); e0 = np.sqrt(cov[0, 0])
        an = A["QUANT"][fi, inode, 1]; z = (coef[0] - an) / e0; zs.append(z)
        print(f"  thA={thA:.3f}' floor {fl}: pooled-sky mean of medians {med.mean():.5f}; intercept at ybar ratio 1: "
              f"{coef[0]:.5f}±{e0:.5f} (slope {coef[1]:+.4f} per unit ratio) vs fine analytic {an:.5f} ({z:+.2f}σ)")
zs = np.array(zs)
print(f"15 intercept tests: rms z {np.sqrt(np.mean(zs**2)):.2f}, max|z| {np.abs(zs).max():.2f}")
