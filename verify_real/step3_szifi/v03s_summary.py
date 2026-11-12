"""v03s: summary of v03 (S/N maps with the build's covariance, SZiFi conventions vs the build) over the 10 tiles per
noise level.  Reads out/v03_MODE_PID.npz.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TH = np.geomspace(0.5, 15.0, 15)
for mode in ("baseline", "goal"):
    R = [dict(np.load(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith(f"v03_{mode}_")]
    E = R[0]["edges"]
    print(f"\n=== {mode}: {len(R)} tiles, the build's iteration-1 covariance given to SZiFi-convention filtering")
    print("  theta [']                          " + " ".join(f"{t:6.2f}" for t in TH))
    em = np.max([r["edge_max"] for r in R], axis=0)      # (15, nb)
    er = np.sqrt(np.mean([r["edge_rms"] ** 2 for r in R], axis=0))
    print("  (i) SZiFi's linear 'same' convolution vs periodic, identical template (the build's): |dq| by distance from the edge")
    for b in range(len(E) - 1):
        print(f"      {E[b]:5.1f}-{E[b + 1]:5.1f}'  max " + " ".join(f"{x:6.3f}" for x in em[:, b]))
        print(f"      {'':12s}  rms " + " ".join(f"{x:6.3f}" for x in er[:, b]))
    print("  (ii) SZiFi's sampled template vs the build's, periodic, same covariance, >= 30' from the edges:")
    tr = np.sqrt(np.mean([r["tmpl_rms"] ** 2 for r in R], axis=0))
    tm = np.max([r["tmpl_max"] for r in R], axis=0)
    tc = np.mean([r["tmpl_corr"] for r in R], axis=0)
    print("      rms dq       " + " ".join(f"{x:6.3f}" for x in tr))
    print("      max |dq|     " + " ".join(f"{x:6.3f}" for x in tm))
    print("      correlation  " + " ".join(f"{x:6.4f}" for x in tc))
    print("  (iii) both (SZiFi template, linear convolution) vs the build, >= 30' from the edges:")
    print("      rms dq       " + " ".join(f"{x:6.3f}" for x in np.sqrt(np.mean([r['tot_rms'] ** 2 for r in R], axis=0))))
    print("      max |dq|     " + " ".join(f"{x:6.3f}" for x in np.max([r['tot_max_in'] for r in R], axis=0)))
    qd = np.concatenate([r["q_det"][r["det_sel"]] for r in R])
    qb = np.concatenate([r["q_build"][r["det_sel"]] for r in R])
    jj = np.concatenate([r["det_j"][r["det_sel"]] for r in R])
    d = qd - qb
    print(f"  (iv) at the build's {qb.size} final detections with q > 4.5 (same pixel and scale, periodic, SZiFi template):"
          f" q_SZiFi - q_build mean {d.mean():+.4f}, std {d.std():.4f}, min {d.min():+.3f}, max {d.max():+.3f}")
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15)):
        m = (jj >= lo) & (jj < hi)
        if m.any():
            print(f"      scales {TH[lo]:.2f}-{TH[hi - 1]:.2f}': N {m.sum():5d}, mean {d[m].mean():+.4f}, std {d[m].std():.4f}, "
                  f"relative mean {np.mean(d[m] / qb[m]):+.4f}")
    q5 = qb > 5
    print(f"      build q > 5: {q5.sum()}; of these with q_SZiFi < 5: {(qd[q5] < 5).sum()}; build 4.5 < q <= 5 with "
          f"q_SZiFi > 5: {((qb <= 5) & (qd > 5)).sum()}")
    sb = np.array([r["sig_b"] for r in R]); ss = np.array([r["sig_s"] for r in R])
    print("  sigma ratio SZiFi template (y0 units) / build template (sr units), relative to tile 0: spread over tiles "
          f"max {np.max(np.abs((ss / sb) / (ss[0] / sb[0]) - 1)):.1e} (the template effect on sigma is tile-independent)")
