"""V4 (C1): recount the equal-area patches straight from the full-sky catalogues (own code), compare with an
independent HMF integral (own colossus grid), and extend the test to all 11 equal-area latitude bands
(36 x 10 x 3 = 1080 other patches) to test the var/mean = 1.24 +- 0.14 of the equatorial band at M>1e13."""
import numpy as np
import indep_lib as I

g = I.hmf_grid()
dz = np.diff(g["ze"]); dl = np.diff(g["le"])
cell = g["dens"] * dz[:, None] * dl[None, :]
A_sr = 100 * (np.pi / 180) ** 2
lamM = {M: (cell * (np.exp(g["lc"]) >= M)[None, :]).sum() * A_sr for M in (1e13, 3e13, 1e14, 3e14)}
# bin-edge correction: thresholds fall inside bins -> use exact fractional bin weights
def lam_above(M):
    le = g["le"]; frac = np.clip((le[1:] - np.log(M)) / (le[1:] - le[:-1]), 0, 1)
    return (cell * frac[None, :]).sum() * A_sr
lamM = {M: lam_above(M) for M in (1e13, 3e13, 1e14, 3e14)}
print("independent lambda per 100 deg^2:", {f"{k:.0e}": round(v, 3) for k, v in lamM.items()},
      " (claimed 71014.3, 11146.1, 960.9, 52.94)")
bands = [(-55 + 10 * b, -45 + 10 * b) for b in range(11) if b != 5]      # Y in [-55,55), |Y|<57.3 allowed
cnt = {M: [] for M in lamM}; eq = {M: [] for M in lamM}
for seed in (1, 2, 3):
    with np.load(f"/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed{seed}.npz") as f:
        Y = (180 / np.pi) * np.sin(np.radians(f["dec"].astype(np.float64)))
        X = f["ra"].astype(np.float64); M = f["M500"]
    ix = np.floor(X / 10).astype(int)
    for lo, hi in bands + [(-5, 5)]:
        s = (Y >= lo) & (Y < hi)
        for Mm in lamM:
            c = np.bincount(ix[s & (M >= Mm)], minlength=36)[:36]
            (eq if (lo, hi) == (-5, 5) else cnt)[Mm].append(c)
    del Y, X, M, ix
for Mm, lam in lamM.items():
    e = np.concatenate(eq[Mm]); a = np.concatenate(cnt[Mm])
    for lab, c in (("equatorial 108", e), ("10 other bands 1080", a)):
        n = c.size
        print(f"M>{Mm:.0e} {lab:20s}: mean {c.mean():11.3f}  expected {lam:11.3f}  ({(c.mean()-lam)/np.sqrt(lam/n):+.2f} sigma)"
              f"  var/mean {c.var(ddof=1)/c.mean():.3f} +- {np.sqrt(2/(n-1)):.3f}")
