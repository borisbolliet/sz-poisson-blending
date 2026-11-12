"""Step 3b arbitration / a07: claim V4 (why dense injection tiles select fewer clusters).

The build says the dense tiles' own SZ raises sigma_finder (+0.2-0.4%) and so they select fewer faint clusters (-0.67 /
-0.53% above q = 4).  The sims verifier finds the same q > 4 loss with SPARSE injections, whose sigma does not change,
and traces it to the injection tile's filter.  Test with the build's own data:
  (1) b03's sparse injections (<= 10 clusters per tile, 22,680 per level): painted end to end (sp_q_none) vs the same
      clusters inserted virtually on the halo-free filter (sp_qv), paired, above q = 4, 4.5, 5, 6; tile bootstrap; and
      the sigma change of the sparse injection tiles (sp_sig vs the halo-free sig) at 0.5', 4.45', 15'.
  (2) b04's dense tiles: P (painted), Vd (virtual, dense tile's filter), Vn (virtual, halo-free filter), paired, same
      thresholds; sigma change.
Usage: python a07_sparse_dense.py
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np

TH = (4.0, 4.5, 5.0, 6.0)


def paired(t, a, b, nt, W):
    """sum over clusters of 1[a > e] - 1[b > e], relative to sum 1[b > e], per threshold, with tile bootstrap."""
    out = []
    for e in TH:
        x = A.per_tile(t, (a > e).astype(float), nt); y = A.per_tile(t, (b > e).astype(float), nt)
        v, err, _ = A.ratio_boot(x, y, W)
        out.append((e, int(x.sum()), int(y.sum()), v, err))
    return out


if __name__ == "__main__":
    for mode in ("baseline", "goal"):
        print(f"=================== {mode} ===================", flush=True)
        T, P, Vv, ds = [], [], [], []
        for p in range(324):
            with np.load(os.path.join(A.OUT, "virtual", "prod", f"v_{mode}_{p:03d}.npz")) as d:
                T.append(np.full(d["sp_q_none"].size, p)); P.append(d["sp_q_none"]); Vv.append(d["sp_qv"])
                ds.append(d["sp_sig"] / d["sig"][None, :] - 1)
        T, P, Vv = np.concatenate(T), np.concatenate(P), np.concatenate(Vv)
        ds = np.concatenate(ds)
        W = A.boot_w(324, 2000, 21)
        print(f"(1) sparse injections, {P.size} clusters on 324 halo-free skies (7 sets of <= 10 per sky); sigma of the injection "
              f"tile / halo-free tile - 1 at 0.5' / 4.45' / 15': " + " / ".join(f"{100 * ds[:, j].mean():+.3f}%" for j in (0, 9, 14)), flush=True)
        for e, a, b, v, err in paired(T, P, Vv, 324, W):
            print(f"    q_obs > {e:.1f}: painted {a}, virtual (halo-free filter) {b}: painted / virtual - 1 = {100 * v:+.2f} +- {100 * err:.2f}%", flush=True)
        fs = sorted(glob.glob(os.path.join(A.OUT, "dense", f"d_{mode}_*.npz")))
        T, Pd, Vd, Vn, sd = [], [], [], [], []
        for k, fn in enumerate(fs):
            with np.load(fn) as d:
                T.append(np.full(d["P"].size, k)); Pd.append(d["P"]); Vd.append(d["Vd"]); Vn.append(d["Vn"])
                sd.append(d["sig_d"] / d["sig_n"] - 1)
        T, Pd, Vd, Vn, sd = np.concatenate(T), np.concatenate(Pd), np.concatenate(Vd), np.concatenate(Vn), np.array(sd)
        W = A.boot_w(len(fs), 2000, 22)
        print(f"(2) dense injections, {Pd.size} clusters on {len(fs)} halo-free skies; sigma dense / halo-free - 1 at 0.5' / 4.45' / 15': "
              + " / ".join(f"{100 * sd[:, j].mean():+.3f}%" for j in (0, 9, 14)), flush=True)
        for lab, a, b in (("P / Vn", Pd, Vn), ("Vd / Vn", Vd, Vn), ("P / Vd", Pd, Vd)):
            print(f"    {lab}: " + "; ".join(f"q > {e:.1f}: {100 * v:+.2f} +- {100 * err:.2f}%" for e, _, _, v, err in paired(T, a, b, len(fs), W)), flush=True)
    print(f"peak RSS {A.peak_rss_gb():.2f} GB")
