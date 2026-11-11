"""Verifier: compare independent direct-MC blending (v01) with the claimed cosmolib grid (c02_grid.npz)."""
import numpy as np, glob
G = dict(np.load("/Users/boris/GitHub/sz-poisson-blending/data_cosmo/c02_grid.npz"))
pts = [(0.81, 0.31, 0.8, 0.17), (0.75, 0.31, 0.8, 0.17), (0.87, 0.31, 0.8, 0.17), (0.81, 0.35, 0.8, 0.17), (0.81, 0.31, 0.9, 0.17)]
ref = {}
for p in pts:
    j = int(np.abs(G["params"] - np.array(p)).max(1).argmin())
    V = np.load(f"v01_{p[0]}_{p[1]}_{p[2]}_{p[3]}_s11.npz")
    Cb, Cs = V["Cb"], V["Cs"]; nb = Cb.shape[0]
    print(f"\n== {p}")
    for k in ["ratio", "B2_10", "B3", "lost"]:
        v, e = V[k]; g = G[k][j]
        print(f"  {k:6s} MC {v:.5f}+-{e:.5f}  claimed {g:.5f}  pull {(g-v)/e:+.1f}")
    print(f"  N_single MC(prim.) {V['N_single_mc'][0]:.0f}  analytic-indep {float(V['Nsingle']):.0f}  claimed {G['N_single'][j]:.0f}")
    Cq = Cb.sum((0, 1)) / Cs.sum((0, 1)); eq = (Cb.sum(1) / Cs.sum(1)).std(0, ddof=1) / np.sqrt(nb)
    gq = G["Nzq_b"][j].sum(0) / G["Nzq_n"][j].sum(0)
    print("  C(q) pulls:", np.round((gq - Cq) / eq, 1))
    print("  C(q) claimed-MC:", np.round(gq - Cq, 4))
    C = Cb.sum(0) / Cs.sum(0); e = (Cb / Cs).std(0, ddof=1) / np.sqrt(nb)
    gC = G["Nzq_b"][j] / G["Nzq_n"][j]
    ok = G["Nzq_n"][j] > 20
    pull = ((gC - C) / e)[ok]
    print(f"  per-bin (N>20, {ok.sum()} bins): chi2 {np.sum(pull**2):.1f}, max |pull| {np.abs(pull).max():.1f}")
    ref[p] = (Cq, eq, gq, C, gC)
# slopes of total ratio
r = {p: float(np.load(f"v01_{p[0]}_{p[1]}_{p[2]}_{p[3]}_s11.npz")["ratio"][0]) for p in pts}
print("\nlog slope of ratio wrt sigma8 (0.75->0.87): MC %.3f, claimed grid -0.188" % (np.log(r[pts[2]] / r[pts[1]]) / np.log(0.87 / 0.75)))
# change of C(q=5-6) between sigma8 0.75 and 0.87
print("C(q=5-6): MC %.4f/%.4f/%.4f vs claimed 0.9845/0.9640/0.9340" % (ref[pts[1]][0][0], ref[pts[0]][0][0], ref[pts[2]][0][0]))
# change of C between fid and other points, in MC vs claimed (this is what drives fixed-correction bias)
for p in pts[1:]:
    Cq, eq, gq = ref[p][:3]; Cq0, eq0, gq0 = ref[pts[0]][:3]
    print(f"Delta C(q) vs fid at {p}: MC {np.round(Cq-Cq0,4)}\n                               claimed {np.round(gq-gq0,4)}")
