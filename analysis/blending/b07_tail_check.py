"""Bright-tail check of P(D) with many skies: random points, neighbours M>1e14 only (cheap skies).
The 8 full M>1e13 skies sit ~2 sigma low in the M>1e14 Y-sum, and the tail P(delta>t>~5) is set by
the same rare bright clusters, so test the tail on 60 independent floor-1e14 skies instead.
Errors: sky-to-sky scatter / sqrt(N_sky)."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import sampler, signal as sg

A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
tab = bl.load_rho_table()
m = sg.SZModel()
T_TEST = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
thAs = [float(tab.th[np.argmin(np.abs(np.log(tab.th / t)))]) for t in (0.5, 1.0, 2.0, 4.0)]
NR = 400_000
fi = list(A["FLOORS"]).index("1e14")
t0 = time.time()
res = {t: [] for t in thAs}
ybar2 = []
seeds = list(range(101, 111)) + list(range(3001, 3051))
for isky, s in enumerate(seeds):
    if s <= 110:
        cat = sampler.load_catalog(f"/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e14_seed{s}.npz",
                                   fields=["ra", "dec", "Y500", "theta500"])
    else:
        cat = sampler.draw_catalog(s, Mmin=1e14, fields=("ra", "dec", "Y500", "theta500"))
    n = len(cat["ra"])
    ybar2.append(cat["Y500"].astype(np.float64).sum() / bl.ARCMIN2_FULLSKY)
    cls = np.full(n, 2, np.int8); orig = np.arange(n, dtype=np.int32)
    zs = [bl.ZoneSet(cat["ra"][sel], cat["dec"][sel], cat["Y500"][sel], cat["theta500"][sel], cls[sel], orig[sel])
          for sel in (cat["theta500"] <= 10, cat["theta500"] > 10)]
    rng = np.random.default_rng(77_000 + s)
    for thA in thAs:
        ra = 360 * rng.random(NR); dec = np.degrees(np.arcsin(2 * rng.random(NR) - 1))
        D, _, _ = bl.neighbour_sums(zs, tab, ra, dec, np.full(NR, thA), np.full(NR, -1))
        dl = D[:, 2] / float(m.sigma_Y(thA))
        res[thA].append([(dl > t).mean() for t in T_TEST] + [dl.mean()])
    if isky % 10 == 0:
        print(f"sky {s} done {time.time()-t0:.1f}s", flush=True)
ybar2 = np.array(ybar2)
print(f"\n{len(seeds)} floor-1e14 skies: ybar(M>1e14) = {ybar2.mean():.5e} ± {ybar2.std(ddof=1)/np.sqrt(len(seeds)):.1e} "
      f"(analytic {A['ybar'][2]:.5e}, {(ybar2.mean()-A['ybar'][2])/(ybar2.std(ddof=1)/np.sqrt(len(seeds))):+.2f}σ)")
zs_all = []
for thA in thAs:
    r = np.array(res[thA]); mean = r.mean(0); err = r.std(0, ddof=1) / np.sqrt(len(r))
    inode = int(np.argmin(np.abs(tab.th - thA)))
    an = np.interp(T_TEST, A["CCDF_T"], A["CCDF"][fi, inode])
    mu_an = A["mu_exact"][inode, 2]
    s = f" thA={thA:.3f}': mean {mean[-1]:.4f}±{err[-1]:.4f} vs {mu_an:.4f} ({(mean[-1]-mu_an)/err[-1]:+.2f}σ); "
    zs_all.append((mean[-1] - mu_an) / err[-1])
    for j, t in enumerate(T_TEST):
        z = (mean[j] - an[j]) / err[j]
        zs_all.append(z)
        s += f"P(>{t:g}) {mean[j]:.4e}±{err[j]:.1e}|{an[j]:.4e} ({z:+.1f}) "
    print(s)
zs_all = np.array(zs_all)
print(f"tail tests: {len(zs_all)} z-scores, rms {np.sqrt(np.mean(zs_all**2)):.2f}, max|z| {np.abs(zs_all).max():.2f}; "
      f"total {time.time()-t0:.1f}s")
