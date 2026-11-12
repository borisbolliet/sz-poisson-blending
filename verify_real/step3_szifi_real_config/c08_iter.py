"""c08 (test d): claim 2 (without inpainting SZiFi's iteration re-estimates exactly the iteration-0 covariance) and the
effect of l_sep, read directly from SZiFi's own internal covariance (its binned 6x6 spectra recorded at each noise
iteration, after its inpainting correction) on realistic tiles 0 and 33 (baseline).

(1) S0g (inpaint False), edge buffer 0 as the build's S0: iteration-1 spectra, sigma and catalogue against
    iteration 0, bit for bit.
(2) S1g (l_sep 3000) and S2g (l_sep 12000) start from the same iteration 0, so they mask the same discs and inpaint
    the same maps; SZiFi divides the iteration-1 spectra by the unmasked fraction f only in bins with l > l_sep.
    Expected: spec1(S1) / spec1(S2) = 1/f exactly for 3000 < l <= 12000 and 1 elsewhere.
(3) f recomputed with our own code from SZiFi's iteration-0 catalogue: discs of radius 3 theta_est (no minimum, no
    maximum) around each q > 4 detection's pixel centre, pixels with centre distance <= radius; against SZiFi's own
    cluster mask (mean and number of zero pixels, recorded in the run).
(4) The size of the effect: auto spectra after / before the iteration, averaged over channels, in l bands.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c08_iter.py
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import vc_model as M                                 # noqa: E402

RUND = os.path.join(HERE, "out", "runs")


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


print("(1) claim 2: S0 (no inpainting), iteration 1 vs iteration 0")
for pid in (0, 33):
    fn = os.path.join(RUND, f"S0g_real_baseline_{pid:03d}_b0.npz")
    if not os.path.exists(fn):
        print(f"  tile {pid}: not run")
        continue
    r = load(fn)
    same_spec = np.array_equal(r["spec0"], r["spec1"])
    same_sig = np.array_equal(r["it0_sigvec"], r["it1_sigvec"])
    same_cat = all(np.array_equal(r[f"it0_{k}"], r[f"it1_{k}"]) for k in ("q", "y0", "i", "j", "jt"))
    print(f"  tile {pid:3d} baseline: noise iterations run {int(r['n_it'])}; cluster mask built (f = {r['fsky'].tolist()}, "
          f"{r['n_masked_pix'].tolist()} pixels masked); spectra it1 == it0: {same_spec}; sigma_vec identical: {same_sig}; "
          f"catalogue identical: {same_cat} ({r['it0_q'].size} / {r['it1_q'].size} detections)")

print("\n(2)-(4) l_sep: S1g (3000) vs S2g (12000)")
for pid in (0, 33):
    f1, f2 = os.path.join(RUND, f"S1g_real_baseline_{pid:03d}.npz"), os.path.join(RUND, f"S2g_real_baseline_{pid:03d}.npz")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        print(f"  tile {pid}: pair not complete")
        continue
    a, b = load(f1), load(f2)
    e = a["spec1_ell"]
    same0 = np.array_equal(a["spec0"], b["spec0"]) and all(np.array_equal(a[f"it0_{k}"], b[f"it0_{k}"]) for k in ("q", "i", "j", "jt"))
    f = float(a["fsky"][0])
    rat = a["spec1"] / b["spec1"]
    mid = (e > 3000) & (e <= 12000)
    rest = ~mid & (e < 20016)
    print(f"  tile {pid:3d}: iteration 0 identical in S1 and S2: {same0}; f recorded S1 {f:.6f}, S2 {float(b['fsky'][0]):.6f}")
    print(f"    spec1(S1)/spec1(S2) x f - 1 for 3000 < l <= 12000 ({mid.sum()} bins): max |.| {np.abs(rat[mid] * f - 1).max():.1e}; "
          f"spec1(S1)/spec1(S2) - 1 elsewhere below 20016 ({rest.sum()} bins): max |.| {np.abs(rat[rest] - 1).max():.1e}")
    # (3) our own cluster mask from SZiFi's iteration-0 catalogue
    m = np.ones((M.N, M.N), bool)
    ii, jj = np.arange(M.N)[:, None], np.arange(M.N)[None, :]
    sel = a["it0_q"] > 4.0
    for i0, j0, th in zip(a["it0_i"][sel], a["it0_j"][sel], a["it0_theta"][sel]):
        rad = 3.0 * th * M.ARC
        m &= ((ii - i0) ** 2 + (jj - j0) ** 2) * M.DX ** 2 > rad ** 2
    print(f"    (3) our disc mask from SZiFi's {sel.sum()} iteration-0 detections: f = {m.mean():.6f}, {(~m).sum()} pixels masked; "
          f"SZiFi's mask: f = {f:.6f}, {int(a['n_masked_pix'][0])} pixels masked")
    for lab, r in (("S1 (l_sep 3000) ", a), ("S2 (l_sep 12000)", b)):
        d0 = np.array([r["spec0"][k].diagonal() for k in range(len(e))])
        d1 = np.array([r["spec1"][k].diagonal() for k in range(len(e))])
        rr = d1 / d0
        out = []
        for lo, hi in ((100, 3000), (3000, 6000), (6000, 12000), (12000, 20000)):
            s = (e > lo) & (e <= hi)
            out.append(f"{lo}-{hi}: {rr[s].mean():.3f}")
        print(f"    (4) {lab}: auto spectra it1/it0 (mean over channels and bins) " + "; ".join(out))
