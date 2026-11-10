"""REPAIR: 216 additional independent patches (pids 108..323) from 6 NEW full-sky Poisson realisations (seeds 4..9),
cut exactly as in extract_patches.py (X=ra, Y=(180/pi) sin dec, |Y|<5 deg, 36 patches per sky).  Noise seeds in
q2lib.process_patch are 10000+pid, so they do not overlap with the original 108 patches either.
Only the band is kept (full-sky arrays are discarded after the cut)."""
import numpy as np, time, sys
from szsim import sampler, cosmo_hmf as ch

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
L = 10.0
FIELDS = ["z", "M500", "Y500", "theta500", "q"]
out = {}
for seed in range(4, 10):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, Mmin=1e13, fields=("ra", "dec", "z", "M500", "Y500", "theta500", "q"))
    Y = (180 / np.pi) * np.sin(np.radians(cat["dec"].astype(np.float64)))
    sel = np.abs(Y) < L / 2
    d = {"X": cat["ra"][sel].astype(np.float64), "Y": Y[sel]}
    for k in FIELDS:
        d[k] = cat[k][sel]
    N = cat["meta"]["N"]; Nexp = cat["meta"]["N_expected"]
    del cat, Y
    ip = np.floor(d["X"] / L).astype(int)
    for p in range(36):
        m = ip == p
        pid = 108 + (seed - 4) * 36 + p
        out[pid] = dict(x=(d["X"][m] - p * L).astype(np.float32), y=(d["Y"][m] + L / 2).astype(np.float32),
                        **{k: d[k][m] for k in FIELDS})
    print(f"seed {seed}: N_fullsky {N} (expected {Nexp:.0f}, {(N-Nexp)/np.sqrt(Nexp):+.2f} sigma)  band halos {sel.sum()}  t={time.time()-t0:.1f}s", flush=True)

np.savez(f"{ROOT}/data/q2_maps/patches_10deg_M1e13_set2.npz",
         **{f"p{pid}_{k}": v for pid, dd in out.items() for k, v in dd.items()})
fsky_patch = L * L / (4 * np.pi * (180 / np.pi) ** 2)
for lab, Mmin in [("M>1e13", 1e13), ("M>3e13", 3e13), ("M>1e14", 1e14), ("M>3e14", 3e14)]:
    n = np.array([(dd["M500"] >= Mmin).sum() for dd in out.values()], float)
    lam = sampler.expected_count(Mmin=Mmin) * fsky_patch
    print(f"{lab}: mean/patch {n.mean():.3f} expected {lam:.3f} ({(n.mean()-lam)/np.sqrt(lam/len(n)):+.2f} sigma)  var/mean {n.var(ddof=1)/n.mean():.3f}")
