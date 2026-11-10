"""REPAIR (C1): over-dispersion test of the sampler positions. Skies 4..9 redrawn (identical to the S2 skies);
equal-area cells 10 deg (ra) x 10 deg (Y=(180/pi) sin dec) for |Y|<50 (360 cells/sky) and the S2 band cells |Y|<5 (36/sky).
var/lambda with lambda = full-sky intensity x 100 deg^2 (known, so no ddof)."""
import numpy as np
from szsim import sampler
fs = 100.0 / (4 * np.pi * (180 / np.pi) ** 2)
lam = {m: sampler.expected_count(Mmin=m) * fs for m in (1e13, 1e14)}
cells = {m: [] for m in lam}; band = {m: [] for m in lam}
for seed in range(4, 10):
    cat = sampler.draw_catalog(seed, Mmin=1e13, fields=("ra", "dec", "M500"))
    Y = (180 / np.pi) * np.sin(np.radians(cat["dec"].astype(np.float64)))
    for m in lam:
        sm = cat["M500"] >= m
        s = sm & (np.abs(Y) < 50)
        H = np.bincount((np.floor((cat["ra"][s] % 360.0) / 10).astype(int) % 36) * 10 + np.floor((Y[s] + 50) / 10).astype(int), minlength=360)
        cells[m].append(H)
        s = sm & (np.abs(Y) < 5)
        band[m].append(np.bincount((np.floor((cat["ra"][s] % 360.0) / 10).astype(int) % 36), minlength=36))
    del cat, Y
for m in lam:
    for name, arr in [("|Y|<50 cells", np.concatenate(cells[m])), ("|Y|<5 band ", np.concatenate(band[m]))]:
        r = ((arr - lam[m]) ** 2).mean() / lam[m]; e = np.sqrt(2 / arr.size)
        print(f"M>{m:.0e} {name}: n_cells {arr.size}  mean {arr.mean():.1f} vs {lam[m]:.1f}  <(N-lam)^2>/lam = {r:.3f}+-{e:.3f} ({(r-1)/e:+.2f}s)")
    rows = np.concatenate([c.reshape(36, 10) for c in cells[m]], 0)
    r_row = ((rows - lam[m]) ** 2).mean(0) / lam[m]
    print(f"   per Y-row (10 rows, Y from -50 to 50): " + " ".join(f"{x:.2f}" for x in r_row) + f"  (each +-{np.sqrt(2/rows.shape[0]):.2f})")
