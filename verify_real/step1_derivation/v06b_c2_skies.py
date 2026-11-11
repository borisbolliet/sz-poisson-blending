"""v06b: the M500 > 1e14 (class c2) part of the confusion variance is dominated by rare objects, so one sky
(v06) is not a precise estimate of the Campbell EXPECTATION.  Here: 300 fresh M500 > 1e14 skies (szsim.sampler,
seeds 9000-9299), the same per-halo catalogue sum with my I_A(theta_B) tables (confusion_mine.npz), mean +- error
of the mean and the sky-to-sky distribution, compared with the builder's c2 numbers (expectation over the HMF).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from scipy.interpolate import CubicSpline
from szsim import sampler

t0 = time.time()
A_SKY = 4 * np.pi / V.AM ** 2
C = np.load(os.path.join(HERE, "confusion_mine.npz"))
BV = np.load(os.path.join(V.ROOT, "data_real", "confusion_var.npz"))
THA, THB = C["thA"], C["thB"]
show = [int(np.argmin(np.abs(THA - s))) for s in (0.52, 0.95, 1.98, 4.15, 7.5, 15.72, 32.94)]
modes = ("white", "baseline", "goal")
sp = {m: [CubicSpline(np.log(THB), np.log(C[f"{m}_I"][i])) for i in show] for m in modes}
NS = 300
v = {m: np.zeros((NS, len(show))) for m in modes}
for k in range(NS):
    cat = sampler.draw_catalog(9000 + k, Mmin=1e14, fields=("Y500", "theta500"))
    Y2 = cat["Y500"].astype(np.float64) ** 2; lt = np.log(cat["theta500"].astype(np.float64))
    lt = np.clip(lt, np.log(THB[0]), np.log(THB[-1]))
    for m in modes:
        sA = C[f"{m}_sA"]
        for j, i in enumerate(show):
            v[m][k, j] = np.sum(Y2 * np.exp(sp[m][j](lt))) / A_SKY / sA[i] ** 2
    if k % 50 == 0:
        print(f"   sky {k} ({len(Y2):,} halos) {time.time()-t0:.0f}s", flush=True)
print(f"{NS} skies done {time.time()-t0:.0f}s")
for m in modes:
    mean = v[m].mean(0); err = v[m].std(0, ddof=1) / np.sqrt(NS)
    med = np.median(v[m], 0); p16, p84 = np.percentile(v[m], [16, 84], 0)
    b = BV[f"{m}_var_c2"][show]
    print(f"\n[{m}] c2 (M>1e14) var_conf/sigma_A^2 at thA = " + " ".join(f"{THA[i]:.2f}" for i in show))
    print("   MC mean   " + " ".join(f"{x:9.4g}" for x in mean))
    print("   +- err    " + " ".join(f"{x:9.2g}" for x in err))
    print("   builder   " + " ".join(f"{x:9.4g}" for x in b))
    print("   pull      " + " ".join(f"{(x-y)/e:+9.2f}" for x, y, e in zip(mean, b, err)))
    print("   mean/bldr " + " ".join(f"{x/y:9.4f}" for x, y in zip(mean, b)))
    print("   median/mean (typical sky)  " + " ".join(f"{x/y:6.3f}" for x, y in zip(med, mean)))
    print("   16-84% sky range / mean    " + " ".join(f"{a/y:.2f}-{c/y:.2f}" for a, c, y in zip(p16, p84, mean)))
    # single-sky v06 value (seed 20261111) as a percentile of the distribution
    one = C[f"{m}_c2"][show]
    print("   v06 sky (seed 20261111) percentile: " + " ".join(f"{(v[m][:, j] < one[j]).mean()*100:5.1f}" for j in range(len(show))))
print(f"total {time.time()-t0:.0f}s")
