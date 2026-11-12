"""x01 (indep lens), check (a): own MMF S/N maps on the build's tiles with the build's own covariances.

For build tiles PIDS and both noise levels the six channel maps are re-created with own code (vlib3): the build's
CMB and noise draws (same seeds, same draw order; full complex FFT instead of rfft) and the tSZ by own NUFFT synthesis
(scheme 'mix2' = the build's 2-node template mixture, to test the filter at machine precision; scheme 'lag4' = own
4-point template, to show the size of the template approximation).  Then, with OWN filter code:
  (1) covariance: own estimate of the unmasked covariance vs the build's it0_C; own whitened masked covariance (mask
      from the build's it0 q > 4 detections) vs the build's it1_C;
  (2) sigma(theta) from the build's C (own lattice sum) vs the build's stored sigma;
  (3) S/N cube with the build's it0_C and it1_C -> own peak finder -> catalogue vs the build's it0 / it1 catalogue;
  (4) filtered CMB, noise and tSZ at the build's final detections vs the build's d_S_cmb, d_S_noise, d_S_tsz;
  (5) a different numerical route: per-channel inverse FFTs summed in real space (full maps), and a direct real-space
      circular convolution at single pixels (kernel . shifted map), both vs the combined route;
  (6) A1 / A3: own sigma with the analytic per-mode covariance vs MMFModel.sigma_Y_exact (step-1 and smooth template).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.fft as sfft
import vlib3 as V

PIDS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 137, 290]
log = V.Tee(os.path.join(V.HERE, "x01_check_a.log"))
t00 = time.time()
G = V.grid()
P2 = V.Painter(G, eps=1e-9, scheme="mix2")
P4 = V.Painter(G, eps=1e-9, scheme="lag4")
log(f"x01: check (a) on build tiles {PIDS}; nm = {G.nm} in-band modes (full plane), {G.bins_used.size} bins")

# ---------------------------------------------------------------- (6) analytic covariance: A1, A3
log("\n(6) sigma(theta) with the analytic per-mode covariance (own lattice sum) vs step 1's MMFModel.sigma_Y_exact")
for mode in V.MODES:
    sex = V.sigma_Y_exact(mode, V.THETAS)
    fl = V.MMF(G, mode=mode, analytic=True, template="lin")
    fc = V.MMF(G, mode=mode, analytic=True, template="cub")
    r = fl.sigma_am / sex - 1
    rc = fc.sigma_am / sex - 1
    log(f"  {mode}: step-1 template: max |sigma/exact - 1| = {np.abs(r).max():.1e} (at theta {V.THETAS[np.abs(r).argmax()]:.2f}');"
        f" smooth template: {rc.min():+.1e} .. {rc.max():+.1e}")
    log(f"     sigma [arcmin^2] at 0.5, 1.04, 2.15, 4.45, 9.23, 15': " + ", ".join(f"{v:.4e}" for v in fl.sigma_am[[0, 3, 6, 9, 12, 14]]))


def match(detA, detB):
    ka = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(zip(detA["j"], detA["iy"], detA["ix"]))}
    kb = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(zip(detB["j"], detB["iy"], detB["ix"]))}
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka[k] for k in common], np.int64); ib = np.array([kb[k] for k in common], np.int64)
    return ia, ib, len(ka) - len(common), len(kb) - len(common)


summary = []
for pid in PIDS:
    tr = V.build_truth(pid); bt = V.build_tile(pid)
    t0 = time.time()
    T2 = P2.T(tr["y"], tr["x"], tr["Y500"], tr["theta500"])
    T4 = P4.T(tr["y"], tr["x"], tr["Y500"], tr["theta500"])
    Xt2 = V.tsz_channels(T2, G); Xt4 = V.tsz_channels(T4, G)
    del T2, T4
    Xc = V.cmb_build_rng(300000 + pid, G)
    log(f"\n==================== tile {pid}: {tr['y'].size} halos; painted ({time.time() - t0:.1f} s)")
    for mode in V.MODES:
        Xn = V.noise_build_rng(400000 + pid, mode, G)
        X = Xt2 + Xc + Xn
        bu = G.bins_used
        C0b = bt[f"{mode}_it0_C"] * V.RFFT_TO_PHYS
        C1b = bt[f"{mode}_it1_C"] * V.RFFT_TO_PHYS
        nrm = lambda C: np.sqrt(np.einsum("bii->bi", C[bu])[:, :, None] * np.einsum("bii->bi", C[bu])[:, None, :])
        # ---- (1) covariance
        C0m = V.binned_cov(X, G)
        e0 = np.abs(C0m[bu] - C0b[bu]) / nrm(C0b)
        C0m4 = V.binned_cov(Xt4 + Xc + Xn, G)
        e04 = np.abs(C0m4[bu] - C0b[bu]) / nrm(C0b)
        det0b = dict(j=bt[f"{mode}_it0_j"], iy=bt[f"{mode}_it0_iy"], ix=bt[f"{mode}_it0_ix"], q=bt[f"{mode}_it0_q"])
        maskb, nmask = V.cluster_mask(det0b, V.THETAS)
        C1m, fs = V.masked_cov(X, maskb, G, Cref=C0b, method="whitened")
        e1 = np.abs(C1m[bu] - C1b[bu]) / nrm(C1b)
        log(f"--- {mode}")
        log(f"  (1) unmasked covariance, own estimate vs build it0_C: max |dC_ij| / sqrt(C_ii C_jj) = {e0.max():.1e} (median {np.median(e0):.1e});"
            f" with the own 4-point template: {e04.max():.1e}")
        log(f"      whitened masked covariance (mask: build's {nmask} it0 detections q > 4, unmasked fraction {fs:.6f} vs build {float(bt[f'{mode}_it1_fsky']):.6f}),"
            f" own vs build it1_C: max {e1.max():.1e} (median {np.median(e1):.1e})")
        # ---- (2) sigma
        F0 = V.MMF(G, C=C0b); F1 = V.MMF(G, C=C1b)
        s0 = np.abs(F0.sigma_am / bt[f"{mode}_it0_sig"] - 1).max()
        s1 = np.abs(F1.sigma_am / bt[f"{mode}_it1_sig"] - 1).max()
        F1m = V.MMF(G, C=C1m)
        s1m = np.abs(F1m.sigma_am / bt[f"{mode}_it1_sig"] - 1).max()
        log(f"  (2) sigma from the build's C (own code) vs build's stored sigma: it0 max |ratio - 1| = {s0:.1e}, it1 {s1:.1e};"
            f" from the own masked C: {s1m:.1e}")
        # ---- (3) catalogues
        D0 = F0.combine(X)
        det0 = V.detect(F0, D0)
        ia, ib, oa, ob = match(det0, det0b)
        dq0 = np.abs(det0["q"][ia] - det0b["q"][ib]).max() if ia.size else np.nan
        log(f"  (3) it0 catalogue (build's it0_C): own {det0['q'].size} peaks q > 4, build {det0b['q'].size}; matched {ia.size}, own-only {oa}, build-only {ob};"
            f" max |dq| = {dq0:.1e}")
        keep = {}
        D1 = F1.combine(X)
        det1 = V.detect(F1, D1, keep=keep)
        det1b = dict(j=bt[f"{mode}_it1_j"], iy=bt[f"{mode}_it1_iy"], ix=bt[f"{mode}_it1_ix"], q=bt[f"{mode}_it1_q"])
        ia, ib, oa, ob = match(det1, det1b)
        dq1 = np.abs(det1["q"][ia] - det1b["q"][ib]).max() if ia.size else np.nan
        log(f"      it1 catalogue (build's it1_C): own {det1['q'].size}, build {det1b['q'].size}; matched {ia.size}, own-only {oa}, build-only {ob}; max |dq| = {dq1:.1e}")
        # with the own 4-point template sky (independent painting): catalogue differences
        X4 = Xt4 + Xc + Xn
        det14 = V.detect(F1, F1.combine(X4))
        ia4, ib4, oa4, ob4 = match(det14, det1b)
        dq14 = np.abs(det14["q"][ia4] - det1b["q"][ib4])
        log(f"      same with the own-template sky: matched {ia4.size}, own-only {oa4}, build-only {ob4}; |dq| median {np.median(dq14):.1e}, max {dq14.max():.1e}")
        # full pipeline with own covariances (own it0 estimate, own detections -> own mask -> own whitened C)
        R = V.run_finder(X, G)
        iaf, ibf, oaf, obf = match(R["det1"], det1b)
        dqf = np.abs(R["det1"]["q"][iaf] - det1b["q"][ibf]).max()
        log(f"      full own pipeline (own C0, own it0 peaks, own mask, own masked C1): it1 {R['det1']['q'].size} vs build {det1b['q'].size}; "
            f"matched {iaf.size}, own-only {oaf}, build-only {obf}; max |dq| = {dqf:.1e}; sigma it1 vs build max |ratio-1| = "
            f"{np.abs(R['F1'].sigma_am / bt[f'{mode}_it1_sig'] - 1).max():.1e}")
        del R
        # ---- (4) components at the build's detections
        j, iy, ix = det1b["j"], det1b["iy"], det1b["ix"]
        comp = {}
        for name, Xk in (("tsz", Xt2), ("cmb", Xc), ("noise", Xn)):
            Dk = F1.combine(Xk)
            v = np.zeros(j.size)
            for jj in range(15):
                m = j == jj
                if m.any():
                    v[m] = F1.ymap(Dk, jj)[iy[m], ix[m]]
            comp[name] = v
        sd = F1.sigma_am[j]
        out = []
        for name in ("tsz", "cmb", "noise"):
            d = np.abs(comp[name] - bt[f"{mode}_d_S_{name}"]) / sd
            out.append(f"{name} {d.max():.1e}")
        dtot = np.abs(comp["tsz"] + comp["cmb"] + comp["noise"] - bt[f"{mode}_d_S_tot"]) / sd
        log(f"  (4) filtered components at the build's {j.size} final detections, own - build, max |.| in sigma: " + ", ".join(out)
            + f"; total {dtot.max():.1e}")
        # own-template tSZ at the same detections
        Dk = F1.combine(Xt4)
        v4 = np.zeros(j.size)
        for jj in range(15):
            m = j == jj
            if m.any():
                v4[m] = F1.ymap(Dk, jj)[iy[m], ix[m]]
        d4 = (v4 - bt[f"{mode}_d_S_tsz"]) / sd
        log(f"      tSZ with the own 4-point template (exact-size template to <= 2.5e-5 of g(0)) - build: median |.| {np.median(np.abs(d4)):.1e} sigma,"
            f" max {np.abs(d4).max():.1e} sigma; relative to the tSZ value: median {np.median(np.abs(v4 - bt[f'{mode}_d_S_tsz']) / np.abs(bt[f'{mode}_d_S_tsz'])):.1e}")
        # ---- (5) other numerical routes
        dmax = 0.0
        for jj in (0, 7, 14):
            yp = F1.ymap_perchannel(X, jj) / F1.sigma_am[jj]
            dmax = max(dmax, np.abs(yp - keep[jj]).max())
        # direct real-space circular convolution at 12 detection pixels
        sel = np.argsort(-det1b["q"])[:12]
        dd = []
        for s in sel:
            jj = int(j[s])
            gj = F1.gs[jj][G.mshell]
            yv = 0.0
            for i in range(6):
                K = sfft.ifft2(G.to_full(gj * F1.psi[:, i]), workers=1).real * (F1.sigma[jj] ** 2 / V.AR ** 2)
                dmap = sfft.ifft2(G.to_full(X[:, i]), workers=1).real
                # sum_x K(x) d(p - x)
                sh = np.roll(np.roll(dmap[::-1, ::-1], int(iy[s]) + 1, 0), int(ix[s]) + 1, 1)
                yv += np.sum(K * sh)
            dd.append(abs(yv / F1.sigma_am[jj] - det1b["q"][s]))
        log(f"  (5) per-channel inverse FFTs summed in real space vs the combined route: max |dq| over 3 scales x 1.44e6 pixels = {dmax:.1e};"
            f" direct real-space convolution (kernel . shifted map) at the 12 highest-q detections vs the build's q: max |dq| = {max(dd):.1e}")
        summary.append((pid, mode, e0.max(), e1.max(), s0, s1, dq0, dq1, oa + ob, dqf, oaf + obf))
        del X, X4, Xn, keep
    del Xt2, Xt4, Xc

log("\nsummary (tile, mode): C0 err, C1 err, sigma0, sigma1 err, max|dq| it0, it1 (build C), unmatched it1, full own pipeline max|dq|, unmatched")
for s in summary:
    log("  " + f"{s[0]:3d} {s[1]:8s} " + " ".join(f"{v:.1e}" if isinstance(v, float) else str(v) for v in s[2:]))
log(f"\ndone [{time.time() - t00:.0f} s], peak RSS {V.peak_rss_gb():.2f} GB")
