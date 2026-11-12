"""w03b: analyst noise levels from w03 (measured on the simulated filtered maps) as map-sigma ratios r(theta) at the
build's 48 nodes, for w04's extra map-sigma variants:
   r_<run>_std  : plain std of the filtered map (noise + all halos), pooled over the patches;
   r_<run>_clip : 3-sigma-clipped std (pooled);
   r_<run>_rob  : robust (16-84%) half-width (pooled)  [the build's r_ii is the P(D) version of this];
each divided by the measured noise-only std (so that r = 1 without halos), linear in ln theta between the 12 measured
nodes (0.25-40.5') and constant beyond.  Output: out/w03_rvariants.npz
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
d = np.load(os.path.join(HERE, "out", "w03_noise.npz"))
out = {}
lines = []
for run in ("baseline", "goal"):
    res = d[f"{run}_res"]                               # columns: th, std, rob, clip, noise std, r_i, r_ii, ...
    th48 = np.load(f"{ROOT}/data_real/pd_tables_{run}.npz")["th"]
    lt = np.log(res[:, 0])
    for nm, col in (("std", 1), ("clip", 3), ("rob", 2)):
        r12 = res[:, col] / res[:, 4]
        r48 = np.interp(np.log(th48), lt, r12)
        out[f"r_{run}_{nm}"] = r48
        lines.append(f"{run:8s} {nm:4s}: r at the 12 nodes {np.round(r12, 4).tolist()}; 48-node range {r48.min():.4f}-{r48.max():.4f}")
np.savez(os.path.join(HERE, "out", "w03_rvariants.npz"), **out)
with open(os.path.join(HERE, "w03b_rvariants.log"), "w") as f:
    for s in lines:
        print(s); f.write(s + "\n")
