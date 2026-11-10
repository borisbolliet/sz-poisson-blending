"""REPAIR (C1): patch counts of all 324 patches (9 independent full skies) vs full-sky intensity x 100 deg^2,
var/mean per mass threshold, and a dispersion test on 412 equal-area 10 deg x (10 deg in Y) cells covering a fresh
full sky (seed 4, redrawn) to test whether the sampler's positions are Poisson (var/mean = 1)."""
import numpy as np
from scipy.stats import chi2 as chi2d
from szsim import sampler
D = "/Users/boris/GitHub/sz-poisson-blending/data/q2_maps"
fs = 100.0 / (4 * np.pi * (180 / np.pi) ** 2)
M = {}
with np.load(f"{D}/patches_10deg_M1e13.npz") as f:
    for p in range(108): M[p] = f[f"p{p}_M500"]
with np.load(f"{D}/patches_10deg_M1e13_set2.npz") as f:
    for p in range(108, 324): M[p] = f[f"p{p}_M500"]
for name, P in [("S1", range(108)), ("S2", range(108, 324)), ("ALL", range(324))]:
    for lab, Mmin in [("M>1e13", 1e13), ("M>3e13", 3e13), ("M>1e14", 1e14), ("M>3e14", 3e14)]:
        n = np.array([(M[p] >= Mmin).sum() for p in P], float); lam = sampler.expected_count(Mmin=Mmin) * fs
        vm = n.var(ddof=1) / lam; e = np.sqrt(2 / (len(n) - 1))
        print(f"{name:3s} {lab}: mean {n.mean():.2f} vs {lam:.2f} ({(n.mean()-lam)/np.sqrt(lam/len(n)):+.2f}s)  var/lambda {vm:.3f}+-{e:.3f} ({(vm-1)/e:+.2f}s)")
# full-sky cell test on a fresh sky: 36 ra-columns x 18 equal-area rows of 10 deg in Y=(180/pi) sin dec (|Y|<57.3) -> use |Y|<50
cat = sampler.draw_catalog(4, Mmin=1e13, fields=("ra", "dec", "M500"))
Y = (180 / np.pi) * np.sin(np.radians(cat["dec"].astype(np.float64)))
s = np.abs(Y) < 50
ix = np.floor(cat["ra"][s] / 10.0).astype(int); iy = np.floor((Y[s] + 50) / 10.0).astype(int)
for lab, Mmin in [("M>1e13", 1e13), ("M>1e14", 1e14)]:
    m = cat["M500"][s] >= Mmin
    H = np.bincount(ix[m] * 10 + iy[m], minlength=360).astype(float)
    lam = sampler.expected_count(Mmin=Mmin) * fs
    c2 = ((H - lam) ** 2 / lam).sum()
    print(f"fresh sky seed 4, 360 equal-area 100 deg^2 cells (|Y|<50), {lab}: mean {H.mean():.1f} vs {lam:.1f}, "
          f"var/lambda {H.var(ddof=1)/lam:.3f}+-{np.sqrt(2/359):.3f}, chi2 {c2:.0f}/360 (p={chi2d.sf(c2,360):.3f})")
