"""Step 3 / t08: small supporting numbers for RESULTS.md.
 (1) the tiles: halos per tile, theta500 range, halos with q_true > 5 (truth tables);
 (2) the isolated search (s3lib.decompose): where the window maximum lies relative to the halo, for halos with
     q_alone > 4.5, and how often those are credited by the finder (all 324 tiles, both noise levels)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                   # noqa: E402
F = S.F
t0 = time.time()
nh, thmin, thmax, n5 = [], [], [], {m: [] for m in S.MODES}
for pid in range(324):
    t = np.load(os.path.join(S.OUT, "truth", f"truth_{pid:03d}.npz"))
    nh.append(t["x"].size); thmin.append(t["theta500"].min()); thmax.append(t["theta500"].max())
    for m in S.MODES:
        n5[m].append((t["q_" + m] > 5).sum())
nh = np.array(nh)
print(f"(1) 324 tiles: halos per tile {nh.min()}-{nh.max()} (mean {nh.mean():.0f}); theta500 {min(thmin):.3f}'-{max(thmax):.1f}'; "
      f"halos with q_true > 5 per tile: baseline {np.mean(n5['baseline']):.1f}, goal {np.mean(n5['goal']):.1f}")
print("(2) isolated search: offset of the window maximum from the halo's pixel, and the credited fraction")
for mode in S.MODES:
    off_all, qa_all, cr_all = [], [], []
    for pid in range(324):
        r = dict(np.load(os.path.join(S.OUT, "tiles", f"tile_{pid:03d}.npz")))
        q, ex, ti = r[mode + "_it1_q"], r[mode + "_d_top_ex"], r[mode + "_d_top_idx"]
        sig, j = r[mode + "_it1_sig"], r[mode + "_it1_j"]
        with np.errstate(invalid="ignore"):
            real = ex[:, 0] / sig[j] >= 2
        cred = np.unique(ti[(q > 5) & real, 0])
        fidx, qa = r[mode + "_f_idx"], r[mode + "_f_q_alone"]
        off = np.hypot(r[mode + "_f_q_alone_dy"], r[mode + "_f_q_alone_dx"]) * F.PIX
        m = qa > 4.5
        off_all.append(off[m]); qa_all.append(qa[m]); cr_all.append(np.isin(fidx[m], cred))
    off, qa, cr = np.concatenate(off_all), np.concatenate(qa_all), np.concatenate(cr_all)
    print(f"  {mode}:")
    for lo, hi in ((4.5, 5), (5, 5.5), (5.5, 6), (6, 8), (8, 1e9)):
        m = (qa > lo) & (qa <= hi)
        far = m & (off > 1.0)
        print(f"    q_alone {lo}-{hi if hi < 1e8 else 'inf'}: N {m.sum():6d}; maximum more than 1' away {100 * np.mean(off[m] > 1):5.2f}%, "
              f"more than 2' {100 * np.mean(off[m] > 2):5.2f}%; credited {100 * cr[m].mean():5.1f}% "
              f"(maximum within 1': {100 * cr[m & (off <= 1)].mean():5.1f}%; beyond: {100 * cr[far].mean() if far.any() else float('nan'):5.1f}%)")
    m5 = qa > 5
    print(f"    all q_alone > 5: maximum within 1' of the halo for {100 * np.mean(off[m5] <= 1):.2f}%")
print(f"done [{time.time() - t0:.0f} s]")
