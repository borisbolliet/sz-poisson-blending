"""Own cut of the stored full-sky catalogues into 108 equal-area 10x10 deg patches (x=ra-10p, y=(180/pi)sin dec+5, arcmin),
saved for running OUR pipeline on the SAME halos as the analysis (only noise differs)."""
import numpy as np
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
out = {}
for seed in (1, 2, 3):
    with np.load(f"{ROOT}/data/catalogs/cat_M1e13_seed{seed}.npz") as f:
        Yc = np.degrees(np.sin(np.radians(f["dec"].astype(np.float64))))
        sel = np.abs(Yc) < 5.0
        X = f["ra"][sel].astype(np.float64); Yc = Yc[sel]
        d = {k: f[k][sel] for k in ("z", "M500", "Y500", "theta500", "q")}
    ip = np.floor(X / 10.0).astype(int)
    for p in range(36):
        m = ip == p; pid = (seed - 1) * 36 + p
        out[f"p{pid}_x"] = ((X[m] - 10 * p) * 60).astype(np.float64)
        out[f"p{pid}_y"] = ((Yc[m] + 5) * 60).astype(np.float64)
        for k in d:
            out[f"p{pid}_{k}"] = d[k][m]
    print("seed", seed, sel.sum(), flush=True)
np.savez("/Users/boris/GitHub/sz-poisson-blending/verify/map-blending/derivation/band_patches.npz", **out)
