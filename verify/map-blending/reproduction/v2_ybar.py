"""V2: analytic mean tSZ ybar (own HMF integration) per mass slice vs the claimed patch-0 values (C4), plus the
patch-to-patch distribution of ybar in the claim's 108 patches (from their patch_out means) and in mine."""
import glob
import numpy as np
from scipy.special import ndtr
import indep_lib as I

g = I.hmf_grid()
dz = np.diff(g["ze"]); dl = np.diff(g["le"])
Z, LM = np.meshgrid(g["zc"], g["lc"], indexing="ij")
Ym = I.Ymed(np.exp(LM), Z)
SR_PER_ARCMIN2 = I.ARC ** 2
w = g["dens"] * dz[:, None] * dl[None, :]            # number per sr per cell
for lo, hi in [(1e13, 3e13), (3e13, 1e14), (1e14, 1e16), (1e13, 1e16)]:
    s = (np.exp(LM) >= lo) & (np.exp(LM) < hi)
    ybar = (w * Ym * np.exp(I.SIGLNY ** 2 / 2) * I.RATIO * s).sum() * SR_PER_ARCMIN2
    print(f"analytic ybar  M in [{lo:.0e},{hi:.0e}): {ybar:.4e}")
# claimed patches
P = "/Users/boris/GitHub/sz-poisson-blending/data/q2_maps/patch_out"
m = np.array([np.load(f"{P}/patch_{p:03d}.npz")["means"][:3] for p in range(108)])
for j, lab in enumerate(["1e13-3e13", "3e13-1e14", ">=1e14"]):
    print(f"claimed pipeline 108 patches, painted mean {lab}: mean {m[:, j].mean():.4e}  median {np.median(m[:, j]):.4e}  min {m[:, j].min():.3e} max {m[:, j].max():.3e}")
tot = m.sum(1)
print(f"claimed patches total ybar: mean {tot.mean():.4e} median {np.median(tot):.4e}; patch 0 = {tot[0]:.4e}")
mine = [np.load(f) for f in sorted(glob.glob(f"{I.OUT}/patches/p*.npz"))]
yA = np.array([o["ybarA"] for o in mine]); yC = np.array([o["ybarC"] for o in mine])
print(f"independent patches ({len(mine)}): ybar[1e13,1e14) mean {yA.mean():.4e}; ybar[>=1e14] mean {yC.mean():.4e} median {np.median(yC):.4e} max {yC.max():.3e}")
