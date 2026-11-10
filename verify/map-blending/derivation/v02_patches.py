"""C1: independent equal-area patch cut of the stored full-sky catalogues; counts, var/mean, patch ybar; check the
catalogue's q and Y scatter against our own model."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import own_model as om

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
lam_full = {1e13: 29295478.0, 3e13: None, 1e14: 396387.0}   # from our own v01 midpoint integral
counts = {m: [] for m in (1e13, 3e13, 1e14, 3e14)}
qcounts = {q: [] for q in (4, 5, 6, 7)}
ybar = []; ybar_comp = []
thg = np.geomspace(1e-3, 200, 400); sYg = om.sigmaY(thg)
rng = np.random.default_rng(5)
for seed in (1, 2, 3):
    with np.load(f"{ROOT}/data/catalogs/cat_M1e13_seed{seed}.npz") as f:
        dec = f["dec"].astype(np.float64)
        Yc = np.degrees(np.sin(np.radians(dec)))
        sel = np.abs(Yc) < 5.0
        del dec
        X = f["ra"][sel].astype(np.float64)
        d = {k: f[k][sel].astype(np.float64) for k in ("z", "M500", "Y500", "theta500", "q")}
    Yc = Yc[sel]
    print(f"seed {seed}: band halos {sel.sum()}  (expected {29295478*10*360/om.DEG2_FULLSKY:.0f})")
    # spot-check catalogue quantities against own model on a random subsample
    ii = rng.choice(X.size, 20000, replace=False)
    Ym = om.Y500_med(d["M500"][ii], d["z"][ii]); th = om.theta500(d["M500"][ii], d["z"][ii])
    lnres = np.log(d["Y500"][ii] / Ym)
    qown = d["Y500"][ii] / np.exp(np.interp(np.log(d["theta500"][ii]), np.log(thg), np.log(sYg)))
    print(f"   theta500 cat/own-1: max|.|={np.abs(d['theta500'][ii]/th-1).max():.2e};  ln(Y/Ymed): mean {lnres.mean():+.4f} std {lnres.std():.4f} (0.173);"
          f"  q cat/own-1: median {np.median(d['q'][ii]/qown-1):+.2e} max|.| {np.abs(d['q'][ii]/qown-1).max():.2e}")
    ip = np.floor(X / 10.0).astype(int)
    for p in range(36):
        m = ip == p
        for Mm in counts:
            counts[Mm].append((d["M500"][m] >= Mm).sum())
        for qc in qcounts:
            qcounts[qc].append((d["q"][m] > qc).sum())
        ybar.append((d["Y500"][m].sum()) * om.RATIO / (600.0 ** 2))
        ybar_comp.append([d["Y500"][m & (d["M500"] >= lo) & (d["M500"] < hi)].sum() * om.RATIO / 600.0 ** 2
                          for lo, hi in [(1e13, 3e13), (3e13, 1e14), (1e14, 1e20)]])
    del X, d, Yc, ip
n = 108
print("\nper-patch counts (100 deg^2, 108 patches)")
exp = {1e13: 71014.26, 3e13: 11146.08, 1e14: 960.87, 3e14: 52.943}   # library N_above x fsky; 1e13,1e14 also reproduced by own integral
for Mm, c in counts.items():
    c = np.array(c, float)
    mu = c.mean(); lam = exp[Mm]
    print(f"M>{Mm:.0e}: mean {mu:.3f}  expected {lam:.3f}  z={(mu-lam)/np.sqrt(lam/n):+.2f}  var/mean {c.var(ddof=1)/mu:.3f} +- {np.sqrt(2/(n-1)):.3f}")
for qc, c in qcounts.items():
    c = np.array(c, float)
    print(f"q_true>{qc}: {c.mean()/100:.4f} +- {c.std(ddof=1)/np.sqrt(n)/100:.4f} per deg^2")
yb = np.array(ybar); yc = np.array(ybar_comp)
print(f"\npatch ybar: patch0 {yb[0]:.4e} comps {yc[0]}; over 108 patches mean {yb.mean():.4e} median {np.median(yb):.4e} "
      f"16-84% {np.percentile(yb,16):.3e}-{np.percentile(yb,84):.3e};  analytic 6.4486e-07")
print("component means:", yc.mean(0), " analytic 1.6043e-07 2.1694e-07 2.6748e-07")
