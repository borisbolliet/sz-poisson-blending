"""c06 (test c): SZiFi's sigma(theta) recomputed with our own code (vc_model.filter_quantities) from the tile's binned
spectra, against the sigma SZiFi reports (sigma_vec at scales 1-14; at scale 0, where 'maxima_lomem' leaves 0,
sigma = y0 / q of SZiFi's own detections at that scale).

For every run given (default: our S1g runs on realistic tiles 0 and 33 at both noise levels, and our S1g runs on the
injected tiles):
  (A) from SZiFi's own recorded spectra (iteration 0, and iteration 1 = its inpainting-corrected spectra);
  (B) from our own binned periodogram of the map: the float32 map band-limited to SZiFi's l range as SZiFi does
      (FFT, zero l < 100 or l > 20000, inverse FFT, real part, cast to float32), periodogram
      Re(F_i F_j*) dx^2 / N^2 averaged over the modes with l0 <= l < lf of each bin (bins of width 4 pi / (N dx) = 72,
      first bin dropped), compared bin by bin with SZiFi's spectra, then sigma from it.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c06_sigma.py [RUN_TAG ...]
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                   # noqa: E402
import vc_model as M                                 # noqa: E402

RUND = os.path.join(HERE, "out", "runs")
MAPD = os.path.join(HERE, "out", "maps")
TAGS = sys.argv[1:] or ["S1g_real_baseline_000", "S1g_real_baseline_033", "S1g_real_goal_000", "S1g_real_goal_033",
                        "S1g_inj_baseline_000", "S1g_inj_baseline_033", "S1g_inj_goal_000", "S1g_inj_goal_033"]
t0 = time.time()
SED = M.tsz_sed_uK(M.FREQS_GHZ)
TH_G = np.geomspace(0.5, 15.0, 15)


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def own_spectra(map_name, lrange=(100., 20000.)):
    m = np.load(os.path.join(MAPD, map_name + ".npy"))
    ell = M.szifi_ell()
    out_band = (ell < lrange[0]) | (ell > lrange[1])
    F = []
    for c in range(6):
        f = np.fft.fft2(m[:, :, c])
        f[out_band] = 0.
        x = np.asarray(np.fft.ifft2(f).real, dtype=np.float32)          # SZiFi's filter_tmap, float32 maps
        F.append(np.fft.fft2(x.astype(np.float64)))
    del m
    mm = np.rint(np.fft.fftfreq(M.N) * M.N).astype(np.int64)
    n2 = mm[:, None] ** 2 + mm[None, :] ** 2
    isq = np.floor(np.sqrt(n2.astype(float))).astype(np.int64)
    isq -= (isq * isq > n2)
    isq += ((isq + 1) * (isq + 1) <= n2)
    b = (isq // 2).ravel()                              # bin of width 72 = 2 x 36: floor(sqrt(n2) / 2), exact
    keep = b < 300
    cnt = np.bincount(b[keep], minlength=300)
    S = np.zeros((300, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            p = (F[i] * np.conj(F[j])).real.ravel() * (M.DX * M.DX / M.N ** 2)
            S[:, i, j] = S[:, j, i] = np.bincount(b[keep], weights=p[keep], minlength=300) / np.maximum(cnt, 1)
    return S[1:], cnt[1:]


print(f"c06: {len(TAGS)} runs")
_mm = np.rint(np.fft.fftfreq(M.N) * M.N).astype(np.int64)
_n2 = (_mm[:, None] ** 2 + _mm[None, :] ** 2).ravel()
_l = 36.0 * np.sqrt(_n2.astype(float))
for lo, hi in ((72., 144.), (19944., 20016.)):
    inbin = (_l >= lo) & (_l < hi)
    inband = inbin & (_l >= 100.) & (_l <= 20000.)
    print(f"  bin {lo:.0f}-{hi:.0f}: {inbin.sum()} modes, {inband.sum()} inside l 100-20000: SZiFi's bin average of a band-limited"
          f" map is diluted by {inband.sum() / inbin.sum():.4f} relative to an in-band-only average (exact mode count)")
for tag in TAGS:
    fn = os.path.join(RUND, tag + ".npz")
    if not os.path.exists(fn):
        print(f"\n{tag}: not run")
        continue
    r = load(fn)
    th = TH_G if tag.startswith(("S1g", "S0g", "S2g")) else M.THETAS
    print(f"\n{tag}  [{time.time() - t0:.0f} s]")
    for it in range(int(r["n_it"])):
        q, y0, jt = r[f"it{it}_q"], r[f"it{it}_y0"], r[f"it{it}_jt"]
        sz = r[f"it{it}_sigvec"].copy()
        s0 = y0[jt == 0] / q[jt == 0]
        sz[0] = np.median(s0) if s0.size else np.nan
        sig, _ = M.filter_quantities(r[f"spec{it}_ell"], r[f"spec{it}"], SED, M.FWHM, thetas=th, want_kappa=False)
        rel = sig / sz - 1
        print(f"  (A) it{it}: ours from SZiFi's recorded spectra / SZiFi's sigma - 1, scales 0..14:")
        print("      " + " ".join(f"{x:+.1e}" for x in rel)
              + f"   [scale 0 from y0/q of {s0.size} detections, spread of y0/q {np.ptp(s0) / np.median(s0) if s0.size else 0:.1e}]")
        print(f"      max |.| over scales 1-14: {np.abs(rel[1:]).max():.1e}; at scale 0: {abs(rel[0]):.1e}")
        if it == 0:
            Sown, cnt = own_spectra(r["map_name"].item() if r["map_name"].ndim == 0 else str(r["map_name"]))
            Ssz = r["spec0"]
            e = r["spec0_ell"]
            dg = np.array([Sown[k].diagonal() / Ssz[k].diagonal() for k in range(len(e))])
            inb = (e > 144) & (e < 19980)
            bad = np.flatnonzero(np.abs(dg - 1).max(1) > 1e-6)
            badin = [k for k in bad if inb[k]]
            print(f"  (B) own binned periodogram vs SZiFi's iteration-0 spectra: {len(e)} bins; auto spectra agree to 1e-6 "
                  f"in {len(e) - len(bad)} bins; {len(badin)} in-band bins (144 < l < 19980) differ, max |ratio - 1| "
                  f"{np.abs(dg[badin] - 1).max() if badin else 0:.1e} (median {np.median(np.abs(dg[badin] - 1).max(1)) if badin else 0:.1e}); "
                  f"agreeing in-band bins: max |ratio - 1| {np.abs(dg[[k for k in range(len(e)) if inb[k] and k not in bad]] - 1).max():.1e}")
            big = [k for k in badin if np.abs(dg[k] - 1).max() > 0.01]
            print(f"      in-band bins differing by > 1%: l = {[int(e[k]) for k in big]} (modes per bin {[int(cnt[k]) for k in big]}); "
                  f"differing bins by range: <1000 {sum(e[k] < 1000 for k in badin)}, 1000-3000 {sum(1000 <= e[k] < 3000 for k in badin)}, "
                  f">=3000 {sum(e[k] >= 3000 for k in badin)}")
            lowb = [k for k in badin if e[k] < 1000]
            print(f"      differing in-band bins below l = 1000: {[int(e[k]) for k in lowb]} with ratios "
                  f"{[round(float(dg[k].mean()), 4) for k in lowb]} (modes per bin: {[int(cnt[k]) for k in lowb]})")
            print(f"      band-edge bins: l = 108 ratio {dg[0].mean():.4f}; l = 19980 ratio {dg[np.argmin(abs(e - 19980))].mean():.4f}; "
                  f"l = 20052 (empty, round-off): SZiFi C_00 {Ssz[np.argmin(abs(e - 20052)), 0, 0]:.2e}, ours {Sown[np.argmin(abs(e - 20052)), 0, 0]:.2e}")
            sig2, _ = M.filter_quantities(e, Sown, SED, M.FWHM, thetas=th, want_kappa=False)
            rel2 = sig2 / sz - 1
            print(f"  (B) it0: ours from our own spectra / SZiFi's sigma - 1, scales 0..14:")
            print("      " + " ".join(f"{x:+.1e}" for x in rel2) + f"   max |.| (1-14) {np.abs(rel2[1:]).max():.1e}")
print(f"\ndone [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
