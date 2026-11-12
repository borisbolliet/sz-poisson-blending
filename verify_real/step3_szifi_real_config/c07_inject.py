"""c07 (test b): isolated clusters of known Y500 and theta500 (c02, own painting) recovered by SZiFi itself run through
our own driver and loader.  Tests frequency order, SED, units, beams, map orientation and SZiFi's position convention
end to end.

Truth (out/inject_truth.npz): index position (py, px) on axis 0 / axis 1, always half-integer; theta500 on SZiFi's
scales; y0 = Y500 f(1e-8) / theta500^2 (the central Compton y that SZiFi's y0 estimates, its template being normalised
to y(1e-8 R500)).
Expectation for SZiFi's catalogue (our reading of its code):
  * position: the S/N sampled at pixel n belongs to the position n - 1/2 (zero-padded 'same' convolution with a template
    centred at N/2 - 1/2), so the cluster at p = n - 1/2 peaks exactly at pixel n = p + 1/2 on both axes;
  * scale: the injected scale (theta500 on the grid), +- 1 step with noise;
  * y0 = kappa(theta) y0_true, with kappa the response of SZiFi's filter (pixel-sampled template, beams
    exp(-sigma^2 l(l+1)/2), its own covariance = the run's recorded spectra, its l range) to our painted cluster
    (band-limited, beams exp(-sigma^2 l^2/2)), per unit y0 (vc_model.filter_quantities);
  * q = kappa y0_true / sigma_SZiFi(theta500) + N(0, 1), plus a small positive bias from the search over pixels and
    scales.
Matching: the highest-q detection within 2 pixels (1') of the expected pixel.  Hypotheses for the orientation are
compared by the number of exact pixel matches: as expected (i = py + 1/2, j = px + 1/2), axes swapped, axis 0
flipped, no half-pixel shift (i = py - 1/2, j = px - 1/2).
Runs (out/runs/): S1g_inj_* (SZiFi's iteration defaults), S3ig_inj_* (SO example with an isotropic covariance;
kappa with its l_max,1d square cut and l range), S3it0g_inj_* (SO example, mixed_gaussian, iteration 0) and the
4-channel S3it0g run (channels 93-280 GHz, as in SZiFi's SO example).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c07_inject.py
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import vc_model as M                                 # noqa: E402

RUND = os.path.join(HERE, "out", "runs")
T = dict(np.load(os.path.join(HERE, "out", "inject_truth.npz")))
TH = np.geomspace(0.5, 15.0, 15)
SED = M.tsz_sed_uK(M.FREQS_GHZ)
t0 = time.time()
TILES = [("baseline", 0), ("baseline", 33), ("goal", 0), ("goal", 33)]


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def sigma_scales(r, it):
    s = r[f"it{it}_sigvec"].copy()
    m = r[f"it{it}_jt"] == 0
    s[0] = np.median(r[f"it{it}_y0"][m] / r[f"it{it}_q"][m]) if m.any() else np.nan
    return s


def match(r, it, py, px, mode="expected"):
    i, j, q = r[f"it{it}_i"], r[f"it{it}_j"], r[f"it{it}_q"]
    if mode == "expected":
        ey, ex = py + 0.5, px + 0.5
    elif mode == "swapped":
        ey, ex = px + 0.5, py + 0.5
    elif mode == "flipped0":
        ey, ex = (M.N - 1 - py) + 0.5, px + 0.5
    elif mode == "noshift":
        ey, ex = py - 0.5, px - 0.5
    idx = np.full(py.size, -1)
    exact = 0
    for h in range(py.size):
        d2 = (i - ey[h]) ** 2 + (j - ex[h]) ** 2
        c = np.flatnonzero(d2 <= 4.0)
        if c.size:
            idx[h] = c[np.argmax(q[c])]
            exact += int(i[idx[h]] == ey[h] and j[idx[h]] == ex[h])
    return idx, exact


def report(cfg, it, ncov=None, lrange=(100., 20000.), header=""):
    print(f"\n{cfg}, iteration {it}{header}")
    allq = []
    for mode, pid in TILES:
        tag = f"{cfg}_inj_{mode}_{pid:03d}"
        fn = os.path.join(RUND, tag + ".npz")
        if not os.path.exists(fn):
            print(f"  {tag}: not run")
            continue
        r = load(fn)
        if int(r["n_it"]) <= it:
            continue
        key = f"{mode}_{pid:03d}"
        py, px, jt, y0t = T[key + "_py"], T[key + "_px"], T[key + "_jtheta"], T[key + "_y0"]
        sz = sigma_scales(r, it)
        _, kap = M.filter_quantities(r[f"spec{it}_ell"], r[f"spec{it}"], SED, M.FWHM, thetas=TH, lrange=lrange,
                                     n_cov=ncov)
        hyp = {h: match(r, it, py, px, h)[1] for h in ("expected", "swapped", "flipped0", "noshift")}
        idx, _ = match(r, it, py, px)
        ok = idx >= 0
        di = r[f"it{it}_i"][idx[ok]] - py[ok]
        dj = r[f"it{it}_j"][idx[ok]] - px[ok]
        ds = r[f"it{it}_jt"][idx[ok]] - jt[ok]
        qd = r[f"it{it}_q"][idx[ok]]
        yd = r[f"it{it}_y0"][idx[ok]]
        qe = kap[jt[ok]] * y0t[ok] / sz[jt[ok]]
        allq.append((qd - qe, yd / (kap[jt[ok]] * y0t[ok]), ds, jt[ok], qe))
        for h in np.flatnonzero(~ok):
            d2 = (r[f"it{it}_i"] - py[h] - 0.5) ** 2 + (r[f"it{it}_j"] - px[h] - 0.5) ** 2
            k = int(np.argmin(d2)) if d2.size else -1
            print(f"    not matched: cluster at ({py[h]}, {px[h]}), theta500 {TH[jt[h]]:.2f}', q_exp "
                  f"{kap[jt[h]] * y0t[h] / sz[jt[h]]:.1f}; nearest detection {np.sqrt(d2[k]) * M.PIXA:.1f}' away (q {r[f'it{it}_q'][k]:.1f}, "
                  f"scale {TH[r[f'it{it}_jt'][k]]:.2f}')")
        print(f"  {tag}: {r[f'it{it}_q'].size} detections; injected clusters matched within 1': {ok.sum()} of {py.size}; "
              f"exact pixel matches by hypothesis {hyp}")
        if ok.any():
            print(f"    offset of the reported pixel from the true position (rows, columns): mean ({di.mean():+.3f}, {dj.mean():+.3f}), "
                  f"fraction exactly (+0.5, +0.5): {np.mean((di == 0.5) & (dj == 0.5)):.3f}; scale step: "
                  f"{dict(zip(*np.unique(ds, return_counts=True)))}")
            print(f"    q - q_exp: mean {np.mean(qd - qe):+.3f} +- {np.std(qd - qe) / np.sqrt(ok.sum()):.3f}, sd {np.std(qd - qe):.3f}; "
                  f"y0 / (kappa y0_true): mean {np.mean(yd / (kap[jt[ok]] * y0t[ok])):.4f}, sd {np.std(yd / (kap[jt[ok]] * y0t[ok])):.4f}; "
                  f"q_exp {qe.min():.1f}..{qe.max():.1f}")
    if allq:
        dq = np.concatenate([a[0] for a in allq])
        yr = np.concatenate([a[1] for a in allq])
        ds = np.concatenate([a[2] for a in allq])
        js = np.concatenate([a[3] for a in allq])
        qe = np.concatenate([a[4] for a in allq])
        print(f"  ALL: {dq.size} matched clusters; q - q_exp mean {dq.mean():+.3f} +- {dq.std() / np.sqrt(dq.size):.3f}, sd {dq.std():.3f}; "
              f"y0 / (kappa y0_true) mean {yr.mean():.4f} +- {yr.std() / np.sqrt(yr.size):.4f}; same scale {np.mean(ds == 0):.2f}, "
              f"within one step {np.mean(np.abs(ds) <= 1):.2f}, mean scale step {ds.mean():+.2f}")
        s0 = ds == 0
        print(f"  detected at the injected scale ({s0.sum()}): q - q_exp mean {dq[s0].mean():+.3f} +- {dq[s0].std() / np.sqrt(s0.sum()):.3f}, "
              f"sd {dq[s0].std():.3f}; y0 / (kappa y0_true) mean {yr[s0].mean():.4f} +- {yr[s0].std() / np.sqrt(s0.sum()):.4f}, "
              f"sd {yr[s0].std():.3f}, median {np.median(yr[s0]):.4f}  (y0 at another scale is the central y of another template)")
        for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 13)):
            s = s0 & (js >= lo) & (js < hi)
            if s.sum() > 1:
                print(f"    same scale, true scales {TH[lo]:.2f}'-{TH[hi - 1]:.2f}' ({s.sum()}): y0 ratio {yr[s].mean():.4f} +- "
                      f"{yr[s].std() / np.sqrt(s.sum()):.4f}; q - q_exp {dq[s].mean():+.3f} +- {dq[s].std() / np.sqrt(s.sum()):.3f}")
        for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 13)):
            s = (js >= lo) & (js < hi)
            if s.any():
                print(f"    true scales {TH[lo]:.2f}'-{TH[hi - 1]:.2f}' ({s.sum()}): q - q_exp {dq[s].mean():+.3f} +- {dq[s].std() / np.sqrt(s.sum()):.3f}; "
                      f"y0 ratio {yr[s].mean():.4f} +- {yr[s].std() / np.sqrt(s.sum()):.4f}; mean scale step {ds[s].mean():+.2f}")
        for lo, hi in ((0, 10), (10, 20), (20, 100)):
            s = (qe >= lo) & (qe < hi)
            if s.any():
                print(f"    q_exp {lo}-{hi} ({s.sum()}): q - q_exp {dq[s].mean():+.3f} +- {dq[s].std() / np.sqrt(s.sum()):.3f}")


print("kappa (SZiFi's y0 per true unit y0 for our painted clusters) and sigma come from each run's own spectra.")
for it in (0, 1):
    report("S1g", it)
for it in (0, 1):
    report("S3ig", it, ncov=556, lrange=(1., 300000.), header=" (kappa with the 556^2 covariance grid and l range [1, 300000])")

print("\nS3 as configured (mixed_gaussian), iteration 0: the breakdown and what happens to the injected clusters")
print("  (q_exp uses kappa from the run's isotropic spectra (recorded; the mixed covariance only reshapes them in 2D),")
print("   the channels of the run, the 556^2 grid and l range [1, 300000], and SZiFi's own sigma of that run)")
for mode, pid in TILES:
    for suf, ch, lab in (("", [0, 1, 2, 3, 4, 5], "6 channels"), ("_ch2345", [2, 3, 4, 5], "4 channels 93-280 GHz, SZiFi's SO example set")):
        tag = f"S3it0g_inj_{mode}_{pid:03d}{suf}"
        fn = os.path.join(RUND, tag + ".npz")
        if not os.path.exists(fn):
            continue
        r = load(fn)
        q = r["it0_q"]
        inner = (r["it0_i"] >= 18) & (r["it0_i"] <= 1181) & (r["it0_j"] >= 18) & (r["it0_j"] <= 1181)
        key = f"{mode}_{pid:03d}"
        py, px, jt, y0t = T[key + "_py"], T[key + "_px"], T[key + "_jtheta"], T[key + "_y0"]
        idx, exact = match(r, 0, py, px)
        ok = idx >= 0
        sz = sigma_scales(r, 0)
        _, kap = M.filter_quantities(r["spec0_ell"], r["spec0"], SED[ch], M.FWHM[ch], thetas=TH, lrange=(1., 300000.),
                                     n_cov=556, paint_fwhm=M.FWHM[ch])
        qe = kap[jt] * y0t / sz[jt]
        msg = ""
        if ok.any():
            dq = q[idx[ok]] - qe[ok]
            msg = (f"; matched clusters: q - q_exp median {np.median(dq):+.2f}, mean {dq.mean():+.2f}, sd {dq.std():.2f}; "
                   f"q / q_exp median {np.median(q[idx[ok]] / qe[ok]):.3f}")
        print(f"  {tag} ({lab}): {q.size} peaks with q > 4 ({inner.sum()} at >= 9' from the edges), median q {np.median(q):.3g}, "
              f"q > 5: {(q > 5).sum()}, peaks by scale {np.bincount(r['it0_jt'], minlength=15).tolist()}; "
              f"injected clusters with a peak within 1': {ok.sum()} of {py.size}, exact pixel {exact}" + msg)
print(f"\ndone [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
