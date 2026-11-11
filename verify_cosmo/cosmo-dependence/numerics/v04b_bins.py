"""Verifier: worst (z,q) bins, claimed C vs independent MC; and chi2 impact of the differences on the likelihood."""
import numpy as np
G = dict(np.load("/Users/boris/GitHub/sz-poisson-blending/data_cosmo/c02_grid.npz"))
for p in [(0.81, 0.31, 0.8, 0.17), (0.87, 0.31, 0.8, 0.17)]:
    j = int(np.abs(G["params"] - np.array(p)).max(1).argmin())
    V = np.load(f"v01_{p[0]}_{p[1]}_{p[2]}_{p[3]}_s11.npz"); Cb, Cs = V["Cb"], V["Cs"]
    C = Cb.sum(0) / Cs.sum(0)
    with np.errstate(all="ignore"):
        e = (Cb / Cs).std(0, ddof=1) / np.sqrt(Cb.shape[0])
    gC = G["Nzq_b"][j] / G["Nzq_n"][j]; N = G["Nzq_n"][j]
    pull = np.where(N > 20, (gC - C) / e, 0)
    print(f"== {p}: worst bins (zbin,qbin): claimed C, MC C +- err, N_naive")
    for k in np.argsort(-np.abs(pull).ravel())[:8]:
        z, q = divmod(k, 10)
        print(f"   ({z},{q}) {gC[z,q]:.4f} {C[z,q]:.4f}+-{e[z,q]:.4f} pull {pull[z,q]:+.1f} N={N[z,q]:.0f}")
    d = (gC - C) * N; s = N > 20
    print(f"   chi2 of (claimed - MC) count difference, full sky, bins N>20 (includes MC noise): {np.sum(d[s]**2 / (gC*N)[s]):.2f}; "
          f"MC-noise-only expectation {np.sum((e*N)[s]**2/(gC*N)[s]):.2f}")
