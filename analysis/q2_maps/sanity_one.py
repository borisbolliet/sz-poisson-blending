"""Sanity checks on one processed patch: exact signal decomposition and noise level."""
import numpy as np, sys
import q2lib as L
pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
o = dict(np.load(f"{L.DATA}/patch_out/patch_{pid:03d}.npz"))
d = L.load_patch(pid); tab = L.build_tables(); tr = L.Truth(d, tab)
sig = o["sig"]; ybar = o["means"][:3].sum()
print("patch", pid, "N halos", d["x"].size, " ybar (all halos) = %.4e" % ybar, " component means", o["means"])
print("mean-removal term ybar*kappa_k/sigma_k:", np.array2string(ybar * L.kappa_mean(sig) / sig, precision=3))
print("noise std / sigma_k:", np.array2string(np.sqrt(o["st_var_N"]) / sig, precision=4))
# decomposition check at the candidate peaks of V13
c, v = o["cand_V13"], o["candv_V13"]
rel = []
for j in np.argsort(-v[:, 0])[:300]:
    iy, ix, k = c[j]
    x0, y0 = (ix + 0.5) * L.PIX, (iy + 0.5) * L.PIX
    I, C = tr.contributions(x0, y0, k)
    pred = C.sum() - ybar * L.kappa_mean(sig[k])
    rel.append((pred - v[j, 1]) / sig[k])
rel = np.array(rel)
print(f"decomposition check (300 brightest V13 candidates): (sum_i c_i - ybar*kappa - S_map)/sigma_k: "
      f"mean {rel.mean():+.2e}  rms {rel.std():.2e}  max|.| {np.abs(rel).max():.2e}")
