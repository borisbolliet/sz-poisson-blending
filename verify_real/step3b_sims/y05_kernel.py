"""y05 (sims lens): the own no-blending kernel K(theta500, q'; e) from the virtual clusters of y02 (halo-free versions
of the own fresh tiles), its checks, and its comparison with the build's kernel (data_real/step3b/kernel_*.npz).
  (1) placements, credit limit (credited entries with q' < 2), window doubled (first 8 tiles), largest peak offset;
  (2) monotone fix (max |K - Kraw|, effect on N_exp);
  (3) Monte Carlo error of N_exp(q > 5) from the kernel: 100 tile-bootstrap resamples of the kernel (halos fixed),
      and the even/odd-tile half kernels;
  (4) build vs own kernel: K(e = 5) node by node (q' 2-12) with pulls; N_exp of the own tiles' realistic halos with
      the build's kernel vs the own kernel (q > 4, q > 5, per q bin, per coarse z bin);
  (5) halo-free noise level r_free = sigma_finder / sigma_Y at 0.5' / 4.45' / 15'; noise peaks per 100 deg^2.
Usage: python y05_kernel.py METHOD BUILDNAME     (METHOD whitened|szifi; BUILDNAME prod|szifi3000)
Output: out/kernel_<METHOD>_<mode>.npz
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import alib as A
import vlib3b as V

meth = sys.argv[1]; bname = sys.argv[2]
log = V.Tee(os.path.join(A.HERE, f"y05_kernel_{meth}.log"))
t00 = time.time()
for mode in A.MODES:
    recs = A.load_tiles(meth, mode)
    nt = len(recs)
    K, Kraw, kn = A.kernel_from(recs)
    log(f"\n=== [{meth}, {mode}] {nt} tiles; virtual placements {int(kn.sum()):,} ({int(kn.min())}-{int(kn.max())} per theta "
        f"node), each on {A.QG.size} q' values -> {int(kn.sum()) * A.QG.size:,} evaluations ===")
        # (1)
    nb2 = sum(r["k_below2"] for r in recs)
    nq2 = int(np.sum(A.QG < 2.0))
    log(f"(1) credited virtual entries with q' < 2: {nb2} of {int(kn.sum()) * nq2:,} (q' = {A.QG[A.QG < 2].tolist()})")
    wc = [(r["wc_changed"], r["wc_n"]) for r in recs if "wc_n" in r]
    if wc:
        log(f"    window doubled (500 placements x {A.QG.size} q' on {len(wc)} tiles): {sum(a for a, b in wc)} of "
            f"{sum(b for a, b in wc):,} entries changed")
    offm = np.max(np.stack([r["k_offmax"] for r in recs]), 0)
    log("    largest offset of a credited q > 4 peak from its cluster [arcmin] per theta node: "
        + " ".join(f"{th:.2f}:{o:.1f}" for th, o in zip(A.TH_K, offm)))
    i5 = int(np.argmin(np.abs(A.EDGES - 5.0))); i4 = int(np.argmin(np.abs(A.EDGES - 4.0)))
    # (2) monotone fix
    allrows = [(r["th"], r["qp"]) for r in recs]
    th_all = np.concatenate([a for a, b in allrows]); qp_all = np.concatenate([b for a, b in allrows])
    Kh = A.K_at(K, th_all, qp_all); Khr = A.K_at(Kraw, th_all, qp_all)
    log(f"(2) monotone fix: max |K - Kraw| {np.abs(K - Kraw).max():.4f}; N_exp(q > 5) changes by "
        f"{Kh[:, i5].sum() / Khr[:, i5].sum() - 1:+.4%}")
    Nexp5 = Kh[:, i5].sum(); Nexp4 = Kh[:, i4].sum()
    log(f"    N_exp of the own realistic halos ({th_all.size:,} rows with q' >= 1.5): q > 4 {Nexp4:,.1f}, q > 5 {Nexp5:,.1f}")
    # (3) MC error
    KC, KN = A.kernel_stack(recs)
    W = A.boot_weights(nt, 100, 11)
    vals = []
    for b in range(W.shape[0]):
        kc = (W[b] @ KC).reshape(A.TH_K.size, A.QG.size, A.EDGES.size)
        kb = W[b] @ KN
        Kb = np.maximum.accumulate(kc / kb[:, None, None], axis=1)
        vals.append(A.K_at(Kb, th_all, qp_all)[:, [i4, i5]].sum(0))
    vals = np.array(vals)
    mc = vals.std(0) / np.array([Nexp4, Nexp5])
    log(f"(3) kernel MC error on N_exp (100 tile-bootstrap resamples of the kernel, halos fixed): q > 4 {mc[0]:.3%}, "
        f"q > 5 {mc[1]:.3%}")
    ev = [r for r in recs if r["tile"] % 2 == 0]; od = [r for r in recs if r["tile"] % 2 == 1]
    Ke, _, _ = A.kernel_from(ev); Ko, _, _ = A.kernel_from(od)
    ne = A.K_at(Ke, th_all, qp_all)[:, i5].sum(); no = A.K_at(Ko, th_all, qp_all)[:, i5].sum()
    log(f"    even vs odd tiles' half kernels: N_exp(q > 5) differs by {ne / no - 1:+.3%} (expected scatter of a half "
        f"difference ~ 2 x {mc[1]:.3%} = {2 * mc[1]:.3%})")
    # per-node MC error of K(e=5) from the bootstrap at q' = 5
    jq5 = int(np.argmin(np.abs(A.QG - 5.0)))
    kb5 = []
    for b in range(30):
        kc = (W[b] @ KC).reshape(A.TH_K.size, A.QG.size, A.EDGES.size)
        kb5.append(kc[:, :, i5] / (W[b] @ KN)[:, None])
    kb5 = np.array(kb5)
    Kerr = kb5.std(0)                                                     # (25, nq)
    log(f"    MC error of K(e = 5) per node at q' = 5: median {np.median(Kerr[:, jq5]):.4f}")
    # (4) build's kernel
    bk = np.load(os.path.join(A.ROOT, "data_real", "step3b", f"kernel_{bname}_{mode}.npz"))
    bth, bQA, bK = bk["th"], bk["QA"], bk["K"]
    assert np.allclose(bth, A.TH_K, rtol=1e-6), "theta nodes differ"
    bKg = np.stack([np.stack([np.interp(A.QG, bQA, bK[i, :, e]) for e in range(A.EDGES.size)], -1) for i in range(25)])
    sel = (A.QG >= 2.0) & (A.QG <= 12.0)
    bErr = bk["K_err"] if "K_err" in bk.files else None
    log(f"(4) build ({bname}) vs own kernel, K(e = 5), q' = 2-12: per theta node max |dK| (own - build) and pull "
        f"dK / own MC error at the max:")
    pulls = []
    for i in range(25):
        d = (K[i, sel, i5] - bKg[i, sel, i5])
        k = int(np.argmax(np.abs(d)))
        e_ = Kerr[i, sel][k]
        pulls.append(d[k] / max(e_, 1e-9))
        log(f"    theta {A.TH_K[i]:6.2f}': max |dK| {abs(d[k]):.4f} at q' {A.QG[sel][k]:.1f} (own {K[i, sel, i5][k]:.4f}, "
            f"build {bKg[i, sel, i5][k]:.4f}, own MC err {e_:.4f}); mean dK over q' 3-7 "
            f"{np.mean((K[i, :, i5] - bKg[i, :, i5])[(A.QG >= 3) & (A.QG <= 7)]):+.4f}")
    # the build's raw measured counts at its own discrete q' nodes against the own kernel (null hypothesis: the own K;
    # binomial variance K(1-K)/n_build + own MC variance; the build's own K_err is zero where it counted 0 or n)
    if "Kraw" in bk.files and "nodes" in bk.files:
        bn = bk["nodes"]
        qa_i = np.array([int(np.argmin(np.abs(bQA - x))) for x in bn])
        nraw = bk["n"][:, qa_i]
        for e_, ie in ((4.0, i4), (5.0, i5), (6.0, int(np.argmin(np.abs(A.EDGES - 6.0))))):
            sn = (bn >= 2.0) & (bn <= 12.0)
            mk = np.stack([np.interp(bn[sn], A.QG, K[i, :, ie]) for i in range(25)])
            me = (np.stack([np.interp(bn[sn], A.QG, Kerr[i]) for i in range(25)]) if ie == i5
                  else np.sqrt(mk * (1 - mk) / kn[:, None]))
            n = nraw[:, sn]
            v = mk * (1 - mk) / np.maximum(n, 1) + me ** 2
            ok = v > 0
            c = ((bk["Kraw"][:, sn, ie] - mk) ** 2 / np.where(ok, v, 1.0))[ok]
            parts = []
            for lo, hi in ((2, 4), (4, 6), (6, 7.5), (7.5, 12)):
                s_ = (bn >= lo) & (bn < hi)
                mk2 = np.stack([np.interp(bn[s_], A.QG, K[i, :, ie]) for i in range(25)]); n2 = nraw[:, s_]
                obs = (bk["Kraw"][:, s_, ie] * n2).sum(); exp_ = (mk2 * n2).sum(); sd = np.sqrt((mk2 * (1 - mk2) * n2).sum())
                parts.append(f"q' {lo}-{hi}: {obs:.0f} vs {exp_:.1f} ({(obs - exp_) / max(sd, 1e-9):+.1f} sd)")
            log(f"    build's raw kernel counts vs own K, e = {e_:.0f}, q' 2-12, 25 theta nodes: chi2 {c.sum():.1f} for "
                f"{ok.sum()} nodes; credited counts summed over theta, " + "; ".join(parts))
    Khb = A.K_at(bKg, th_all, qp_all)
    for e, ie in ((4.0, i4), (5.0, i5)):
        log(f"    N_exp(q > {e:.0f}) of the own halos with the build's kernel / own kernel - 1 = "
            f"{Khb[:, ie].sum() / Kh[:, ie].sum() - 1:+.3%} (own kernel MC {mc[0 if e == 4 else 1]:.3%}; build's claim "
            f"0.21-0.23%; combined {np.hypot(mc[0 if e == 4 else 1], 0.0022):.3%})")
    zed, qed = A.bins(5.0)
    z_all = np.concatenate([r["z"] for r in recs])
    Pm = A.pbin(Kh, qed); Pb = A.pbin(Khb, qed)
    log("    per q bin (q_cut 5) build / own - 1: " + " ".join(f"{a:+.2%}" for a in Pb.sum(0) / Pm.sum(0) - 1))
    iz = np.clip(np.searchsorted(zed, z_all, side="right") - 1, 0, zed.size - 2)
    cz = []
    for a, b in A.ZC:
        s = (z_all >= a) & (z_all < b)
        cz.append(Pb[s].sum() / Pm[s].sum() - 1)
    log("    per coarse z bin build / own - 1: " + " ".join(f"{v:+.2%}" for v in cz))
    # theta dependence of the comparison at the realistic-halo weights
    # MC errors of these subset comparisons: own kernel bootstrap (vals per subset) and the build's even/odd halves
    subs = [("theta<0.6'", th_all < 0.6), ("0.6-1.2'", (th_all >= 0.6) & (th_all < 1.2)), ("z 1.5-2", (z_all >= 1.5) & (z_all < 2)),
            ("z 2-3", z_all >= 2.0), ("z<0.1", z_all < 0.1)]
    Kbe = np.stack([np.stack([np.interp(A.QG, bQA, bk["K_even"][i, :, e]) for e in range(A.EDGES.size)], -1) for i in range(25)])
    Kbo = np.stack([np.stack([np.interp(A.QG, bQA, bk["K_odd"][i, :, e]) for e in range(A.EDGES.size)], -1) for i in range(25)])
    for name, sb in subs:
        own = Kh[sb, i5].sum()
        bt = []
        for b in range(30):
            kc = (W[b] @ KC).reshape(A.TH_K.size, A.QG.size, A.EDGES.size)
            Kb = np.maximum.accumulate(kc / (W[b] @ KN)[:, None, None], axis=1)
            bt.append(A.K_at(Kb, th_all[sb], qp_all[sb])[:, i5].sum() / own - 1)
        he = A.K_at(Kbe, th_all[sb], qp_all[sb])[:, i5].sum(); ho = A.K_at(Kbo, th_all[sb], qp_all[sb])[:, i5].sum()
        log(f"    subset {name:10s}: N_exp(q>5) own {own:9.1f}; build / own - 1 = {Khb[sb, i5].sum() / own - 1:+.2%}; own MC "
            f"{np.std(bt):.2%}; build's half kernels differ by {he / ho - 1:+.2%} (full-kernel MC ~ half of that)")
    log("    N_exp(q > 5) build / own - 1 by theta500 of the halos: " + " ".join(
        f"[{lo:.2g}-{hi:.2g}'] {Khb[(th_all >= lo) & (th_all < hi), i5].sum() / Kh[(th_all >= lo) & (th_all < hi), i5].sum() - 1:+.2%}"
        for lo, hi in ((0, 0.6), (0.6, 1.2), (1.2, 2.4), (2.4, 4.8), (4.8, 9.6), (9.6, 100))))
    # (5) halo-free noise level and noise peaks
    rf = np.stack([r["rf"] for r in recs])
    n5 = np.array([np.sum(r["noise_q"] > 5) for r in recs]); n45 = np.array([np.sum(r["noise_q"] > 4.5) for r in recs])
    n4 = np.array([r["noise_q"].size for r in recs])
    nc1 = np.array([r["noise_n5_closest1"] for r in recs])
    log(f"(5) halo-free r_free at 0.5' / 4.45' / 15': {rf[:, 0].mean():.4f} / {rf[:, 9].mean():.4f} / {rf[:, 14].mean():.4f} "
        f"(tile scatter {rf[:, 0].std():.4f}); noise peaks per 100 deg^2 (merge none) q > 4 / 4.5 / 5: {n4.mean():.1f} / "
        f"{n45.mean():.2f} / {n5.mean():.3f} +- {n5.std() / np.sqrt(nt):.3f}; q > 5 closest 1': {nc1.mean():.3f}")
    np.savez(os.path.join(A.OUT, f"kernel_{meth}_{mode}.npz"), K=K, Kraw=Kraw, kn=kn, QG=A.QG, th=A.TH_K, EDGES=A.EDGES,
             Kerr5=Kerr, rbar_free=rf.mean(0), ntiles=nt, mc5=mc[1])
log(f"\ny05 done in {time.time() - t00:.0f} s")
