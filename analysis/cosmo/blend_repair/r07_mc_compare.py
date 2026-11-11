"""Compare the independent direct-MC (r05_mc_blend.py, 2e6 primaries, seed 41) with cosmolib (c02 grid) at sigma8=0.75/0.81/0.87."""
import numpy as np
G = dict(np.load("/Users/boris/GitHub/sz-poisson-blending/data_cosmo/c02_grid.npz"))
ks = [tuple(np.round(p, 6)) for p in G["params"]]
mc = {}
for s in (0.75, 0.81, 0.87):
    M = np.load(f"/Users/boris/GitHub/sz-poisson-blending/data_cosmo/r05_mc_{s}_0.31_0.8_0.17_s41.npz")
    j = ks.index((s, 0.31, 0.8, 0.17)); mc[s] = M
    for k in ("ratio", "B2_10", "B3", "lost", "ybias"):
        v, e = M[k]; c = G[k][j]
        print(f"sigma8={s}: {k:6s} MC {v:.5f} +- {e:.5f}  cosmolib {c:.5f}  pull {(c-v)/e:+.1f}")
for k in ("ybias", "ratio", "B2_10", "lost"):
    sl_mc = np.log(mc[0.87][k][0] / mc[0.75][k][0]) / np.log(0.87 / 0.75)
    j0, j1 = ks.index((0.75, 0.31, 0.8, 0.17)), ks.index((0.87, 0.31, 0.8, 0.17))
    sl_cl = np.log(G[k][j1] / G[k][j0]) / np.log(0.87 / 0.75)
    print(f"grid-end log slope in sigma8, {k}: MC {sl_mc:.2f}, cosmolib {sl_cl:.2f}")
# per-bin C(z,q) at the fiducial: MC vs cosmolib
M = mc[0.81]; j = ks.index((0.81, 0.31, 0.8, 0.17))
Ccl = G["Nzq_b"][j] / G["Nzq_n"][j]
Cb, Cs = M["Cb"], M["Cs"]; Cmc = Cb.sum(0) / Cs.sum(0)
err = (Cb / Cs).std(0, ddof=1) / np.sqrt(Cb.shape[0])
pull = (Ccl - Cmc) / err
print("fiducial C(z,q): cosmolib - MC pulls; q<50 columns: max |pull| %.1f, rms %.2f" % (np.abs(pull[:, :-1]).max(), np.sqrt((pull[:, :-1] ** 2).mean())))
print("  q>50 column: cosmolib", np.round(Ccl[:, -1], 4), "\n               MC      ", np.round(Cmc[:, -1], 4), "\n               err     ", np.round(err[:, -1], 4))
print("  C(z=0.01-0.25,q=5-6): cosmolib %.4f MC %.4f +- %.4f" % (Ccl[0, 0], Cmc[0, 0], err[0, 0]))
