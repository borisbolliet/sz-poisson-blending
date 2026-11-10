"""Cut flat, exactly equal-area 10x10 deg patches out of the stored full-sky M>1e13 catalogues.

Mapping (Lambert cylindrical equal-area about the equator):
    X [deg] = ra,   Y [deg] = (180/pi) sin(dec)
dX dY = (180/pi)^2 d(ra_rad) d(sin dec) = solid angle in deg^2, so a rectangle of 100 deg^2 in (X,Y)
contains exactly the halos of a 100 deg^2 region of the sphere: a thinned Poisson process with the
full-sky intensity x 100 deg^2.  The equatorial band |Y|<5 deg (|dec|<5.006 deg) is split in 36
non-overlapping 10 deg wide patches per sky, i.e. 108 independent patches from seeds 1,2,3.
Local shape distortion of the mapping is 1/cos(dec) in X and cos(dec) in Y: <= 0.38% at the band edge.
Each patch is later treated as a periodic torus (FFT), which keeps the process exactly homogeneous Poisson.
"""
import numpy as np, json, os, time
from szsim import sampler

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
L = 10.0
FIELDS = ["z", "M500", "Y500", "theta500", "q"]
out = {}
for seed in (1, 2, 3):
    t0 = time.time()
    path = f"{ROOT}/data/catalogs/cat_M1e13_seed{seed}.npz"
    with np.load(path) as f:
        dec = f["dec"]
        Y = (180 / np.pi) * np.sin(np.radians(dec.astype(np.float64)))
        sel = np.abs(Y) < L / 2
        del dec
        ra = f["ra"][sel].astype(np.float64)
        d = {"X": ra, "Y": Y[sel]}
        del Y
        for k in FIELDS:
            d[k] = f[k][sel]
    ip = np.floor(d["X"] / L).astype(int)
    for p in range(36):
        m = ip == p
        pid = (seed - 1) * 36 + p
        out[pid] = dict(seed=seed, x=(d["X"][m] - p * L).astype(np.float32), y=(d["Y"][m] + L / 2).astype(np.float32),
                        **{k: d[k][m] for k in FIELDS})
    print(f"seed {seed}: band halos {sel.sum()}  t={time.time()-t0:.1f}s", flush=True)

np.savez(f"{ROOT}/data/q2_maps/patches_10deg_M1e13.npz",
         **{f"p{pid}_{k}": v for pid, dd in out.items() for k, v in dd.items() if k != "seed"})

# ---- check patch counts vs full-sky surface density
from szsim import cosmo_hmf as ch
T = ch.default_table()
fsky_patch = L * L / (4 * np.pi * (180 / np.pi) ** 2)
print(f"patch area {L*L} deg^2 = fsky {fsky_patch:.6e};  N_patch = {len(out)}")
res = {}
for lab, Mmin, qmin in [("M>1e13", 1e13, None), ("M>3e13", 3e13, None), ("M>1e14", 1e14, None), ("M>3e14", 3e14, None),
                        ("q>5 (true mean q)", 1e13, 5.0)]:
    n = np.array([((dd["M500"] >= Mmin) & ((dd["q"] >= qmin) if qmin else True)).sum() for dd in out.values()], float)
    if qmin is None:
        lam = sampler.expected_count(Mmin=Mmin) * fsky_patch
    else:
        lam = None
    mu, err = n.mean(), n.std(ddof=1) / np.sqrt(len(n))
    vm = n.var(ddof=1) / n.mean()
    vm_err = np.sqrt(2.0 / (len(n) - 1))
    if lam is not None:
        z = (mu - lam) / np.sqrt(lam / len(n))
        print(f"{lab:18s}: mean/patch {mu:12.3f} +- {np.sqrt(lam/len(n)):.3f}  expected {lam:12.3f}  offset {z:+.2f} sigma;  var/mean {vm:.3f} +- {vm_err:.3f}")
    else:
        print(f"{lab:18s}: mean/patch {mu:12.3f} +- {err:.3f};  var/mean {vm:.3f} +- {vm_err:.3f}")
    res[lab] = dict(mean=mu, expected=lam, var_over_mean=vm)
json.dump(res, open(f"{ROOT}/data/q2_maps/patch_counts.json", "w"), indent=1)
