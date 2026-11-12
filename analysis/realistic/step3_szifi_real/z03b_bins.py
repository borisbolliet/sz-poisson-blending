"""z03b: NaMaster's binned spectra ('none', ones mask) against a plain l0 <= l < lf periodogram of the same maps, from the
arrays z03_decouple.py stored (data_real/step3_szifi_real/z03_decouple.npz): how many in-band bins differ, by how much,
below and above l = 1000, and how well the other bins agree.
Usage: python3 -I -B z03b_bins.py   (numpy only)
"""
import os
import numpy as np

d = np.load("/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/z03_decouple.npz")
l0, none, own = d["l0"], d["none"], d["own"]
dd = np.arange(6)
allb = (l0 >= 72) & (l0 < 20000)
dev = np.max(np.abs(none[:, dd, dd] / own[:, dd, dd] - 1), axis=1)
bad = allb & (dev > 1e-6)
print(f"in-band bins (lower edge 72 ... 19944): {allb.sum()}; bins where NaMaster's 'none' differs from the periodogram by "
      f"> 1e-6: {bad.sum()}")
for lo, hi in ((72, 1000), (1000, 3000), (3000, 20000)):
    m = bad & (l0 >= lo) & (l0 < hi)
    print(f"  lower edge {lo}-{hi}: {m.sum()} differing bins, largest |none/periodogram - 1| = {dev[m].max() if m.any() else 0:.4f}, "
          f"median {np.median(dev[m]) if m.any() else 0:.4f}")
ok = allb & ~bad
print(f"  the other {ok.sum()} in-band bins: largest |none/periodogram - 1| = {dev[ok].max():.1e}")
