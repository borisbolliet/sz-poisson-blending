"""y06 (sims lens): the own finder's data vector on the own fresh tiles, in step 2's bins (16 z x 9 q_obs, q_cut 5, 6).
  N_det(z, q): credited halos (own credit rule; merge none / closest 1' / fof 10'; '+ no real' = none plus the 'no real
              cluster' entries at their primary's z), at their true z and the q_obs of their highest credited detection;
  N_exp(z, q): sum over the same halos of K(theta500, q'; bin) with the own kernel (y05);
  N_iso(z, q): credited isolated copies (same position, same q', halo-free sky);
  R = N_det / N_exp (the task's estimator), R_p = N_det / N_iso (paired), and a pooled-over-z estimator R_s
  (own version of the build's idea): dK(theta cell, q' cell; e) = <1(q_obs > e) - 1(q_iso > e)> over all halos in the
  cell; here applied to the tiles' own halos (N-route: N_{K+dK} / N_K); the fits (y07) apply it to the count model.
  Errors: 2000 tile-bootstrap resamples (kernel fixed); for R also 200 joint resamples with the kernel recomputed
  from the resampled tiles (the kernel comes from the same tiles' halo-free skies).
Prints the count changes at q > 5 (total, by z, by q), r after the iteration, credited halos per tile, 'no real
cluster' fractions, against the build's claims.  Output: out/datavec_<METHOD>_<mode>.npz.
Usage: python y06_datavec.py METHOD
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import alib as A
import vlib3b as V

meth = sys.argv[1]
log = V.Tee(os.path.join(A.HERE, f"y06_datavec_{meth}.log"))
t00 = time.time()
# q' cells for the pooled estimator
QC = np.r_[1.5, np.arange(2.0, 10.0 + 1e-9, 0.25), np.arange(10.5, 15.0 + 1e-9, 0.5), np.arange(16.0, 30.0 + 1e-9, 1.0),
           35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 3000.0 + 1e-6]
MERGES = ("none", "closest1", "fof10", "keepnr")


def cell_of(th, qp):
    ci = np.clip(np.rint((np.log(th) - A.LTK[0]) / A.DLT).astype(int), 0, A.TH_K.size - 1)
    cq = np.clip(np.searchsorted(QC, qp, side="right") - 1, 0, QC.size - 2)
    return ci * (QC.size - 1) + cq


def dK_grid(Ssum, nsum):
    """(25 * nqc, 19) cell sums -> dK on the kernel grid (25, nq, 19); cells without halos -> 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        dk = np.where(nsum[:, None] > 0, Ssum / np.maximum(nsum[:, None], 1), 0.0).reshape(A.TH_K.size, QC.size - 1, -1)
    cq = np.clip(np.searchsorted(QC, A.QG, side="right") - 1, 0, QC.size - 2)
    out = dk[:, cq, :]
    out[:, A.QG < QC[0], :] = 0.0
    return out


for mode in A.MODES:
    recs = A.load_tiles(meth, mode)
    nt = len(recs)
    kz = np.load(os.path.join(A.OUT, f"kernel_{meth}_{mode}.npz"))
    K = kz["K"]
    log(f"\n=================== [{meth}, {mode}] {nt} tiles ({nt} x 100 deg^2) ===================")
    # ---- finder-level numbers
    r1 = np.stack([r["r1"] for r in recs]); r0 = np.stack([r["r0"] for r in recs])
    log(f"r after the iteration at 0.5' / 4.45' / 15': {r1[:, 0].mean():.4f} / {r1[:, 9].mean():.4f} / {r1[:, 14].mean():.4f} "
        f"(+- {r1[:, 0].std() / np.sqrt(nt):.4f} / {r1[:, 9].std() / np.sqrt(nt):.4f} / {r1[:, 14].std() / np.sqrt(nt):.4f}); "
        f"before: {r0[:, 0].mean():.4f} / {r0[:, 9].mean():.4f} / {r0[:, 14].mean():.4f}")
    nc = {m: np.array([r[f"ncred_{m}"] for r in recs]) for m in ("none", "closest1", "fof10")}
    nd5 = np.array([r["ndet5"] for r in recs]); nnr = np.array([r["nnr_none"] for r in recs])
    log(f"per tile: q > 5 catalogue entries {nd5.mean():.2f} +- {nd5.std() / np.sqrt(nt):.2f}; credited halos merge none "
        f"{nc['none'].mean():.2f} +- {nc['none'].std() / np.sqrt(nt):.2f}, closest 1' {nc['closest1'].mean():.2f}, fof 10' "
        f"{nc['fof10'].mean():.2f}; 'no real cluster' entries {nnr.sum() / nd5.sum():.2%} of all q > 5 entries "
        f"(+- {np.sqrt(nnr.sum()) / nd5.sum():.2%})")
    log(f"merging removes credited halos: closest 1' {1 - nc['closest1'].sum() / nc['none'].sum():.2%}, fof 10' "
        f"{1 - nc['fof10'].sum() / nc['none'].sum():.2%}; credited with q' < 2: {sum(r['ncred_below_none'] for r in recs)}; "
        f"credited halos missing from the stored rows (q' < 1.5): "
        f"{int(nc['none'].sum() - sum(np.isfinite(r['qobs_none']).sum() for r in recs))}")
    nh = np.array([r["nh"] for r in recs])
    lam = V.smp.expected_count(1e13, 1e16, 0.01, 3.0) * V.FSKY_TILE
    log(f"halos per tile {nh.mean():.1f} +- {nh.std() / np.sqrt(nt):.1f} (sampler expectation for 100 deg^2: {lam:.1f})")
    out = {"rbar": r1.mean(0), "ntiles": nt}
    for qcut in (5.0, 6.0):
        s = f"q{qcut:.0f}"
        zed, qed = A.bins(qcut)
        nz, nq = zed.size - 1, qed.size - 1
        Nd = {}
        Nd["none"], Ne, Ni = A.per_tile_counts(recs, K, qcut, "none")
        for m in ("closest1", "fof10"):
            Nd[m], _, _ = A.per_tile_counts(recs, None, qcut, m)
        # + no real cluster entries at their primary's z
        Nnr = np.zeros_like(Ne)
        for i, r in enumerate(recs):
            ok = r["nr_z"] >= 0
            Nnr[i] = A.hist_zq(r["nr_z"][ok], r["nr_q"][ok], zed, qed)
        Nd["keepnr"] = Nd["none"] + Nnr
        # ---- pooled estimator: per-tile cell sums
        ncell = A.TH_K.size * (QC.size - 1)
        St = {m: np.zeros((nt, ncell, A.EDGES.size)) for m in ("none", "closest1", "fof10")}
        nt_c = np.zeros((nt, ncell))
        for i, r in enumerate(recs):
            c = cell_of(r["th"], r["qp"])
            nt_c[i] = np.bincount(c, minlength=ncell)
            iso = (r["qiso"][:, None] > A.EDGES[None, :]).astype(float)
            for m in St:
                det = (r[f"qobs_{m}"][:, None] > A.EDGES[None, :]).astype(float)
                for e in range(A.EDGES.size):
                    St[m][i, :, e] = np.bincount(c, weights=det[:, e] - iso[:, e], minlength=ncell)
        th_all = np.concatenate([r["th"] for r in recs]); qp_all = np.concatenate([r["qp"] for r in recs])
        z_all = np.concatenate([r["z"] for r in recs])
        iz_all = np.clip(np.searchsorted(zed, z_all, side="right") - 1, 0, nz - 1)
        Pk = A.pbin(A.K_at(K, th_all, qp_all), qed)

        def Ns_route(dk):
            P2 = A.pbin(A.K_at(K + dk, th_all, qp_all), qed)
            return np.stack([np.bincount(iz_all, weights=P2[:, b], minlength=nz) for b in range(nq)], 1)
        NK = np.stack([np.bincount(iz_all, weights=Pk[:, b], minlength=nz) for b in range(nq)], 1)
        tot = lambda X: X.sum(0)
        Ne_t = tot(Ne); Ni_t = tot(Ni)
        R = A.ratio(tot(Nd["none"]), Ne_t, Ne_t); Rp = A.ratio(tot(Nd["none"]), Ni_t, Ne_t)
        W = A.boot_weights(nt, 2000, 7)
        F = lambda X: np.tensordot(W, X, axes=(1, 0))                 # (B, nz, nq)
        Ndb, Neb, Nib = F(Nd["none"]), F(Ne), F(Ni)
        Rb = A.ratio(Ndb, Neb, Neb); Rpb = A.ratio(Ndb, Nib, Neb)
        out.update({f"Ndet_{s}": tot(Nd["none"]), f"Nexp_{s}": Ne_t, f"Niso_{s}": Ni_t, f"R_{s}": R, f"Rp_{s}": Rp,
                    f"Rboot_{s}": Rb.astype(np.float32), f"Rpboot_{s}": Rpb.astype(np.float32), f"Nd_tile_{s}": Nd["none"],
                    f"Ne_tile_{s}": Ne, f"Ni_tile_{s}": Ni})
        for m in ("closest1", "fof10", "keepnr"):
            out[f"R_{m}_{s}"] = A.ratio(tot(Nd[m]), Ne_t, Ne_t); out[f"Rp_{m}_{s}"] = A.ratio(tot(Nd[m]), Ni_t, Ne_t)
            out[f"Rpboot_{m}_{s}"] = A.ratio(F(Nd[m]), Nib, Neb).astype(np.float32)
        for m in St:
            out[f"S_{m}_{s}"] = St[m].sum(0); out[f"Stile_{m}_{s}"] = St[m].astype(np.float32)
        out[f"ncell_{s}"] = nt_c.sum(0); out[f"ncell_tile_{s}"] = nt_c
        out["QC"] = QC
        # ---- totals q > qcut
        def tot_change(num, den):
            return num.sum() / den.sum() - 1
        log(f"--- q_cut {qcut:.0f}: N_det {Nd['none'].sum():,.0f}, N_exp {Ne.sum():,.1f}, N_iso {Ni.sum():,.0f}")
        ch = {"R": (tot_change(Nd["none"], Ne), (Ndb.sum((1, 2)) / Neb.sum((1, 2)) - 1).std()),
              "R_p": (tot_change(Nd["none"], Ni), (Ndb.sum((1, 2)) / Nib.sum((1, 2)) - 1).std()),
              "N_iso/N_exp": (tot_change(Ni, Ne), (Nib.sum((1, 2)) / Neb.sum((1, 2)) - 1).std())}
        # R_s, N-route
        dk = dK_grid(St["none"].sum(0), nt_c.sum(0))
        NS = Ns_route(dk)
        Rs_N = NS / np.where(NK > 0, NK, 1)
        rsb = []
        for b in range(200):
            dkb = dK_grid(np.tensordot(W[b], St["none"], axes=(0, 0)), W[b] @ nt_c)
            rsb.append(Ns_route(dkb))
        rsb = np.array(rsb)
        ch["R_s (N-route)"] = (NS.sum() / NK.sum() - 1, (rsb.sum((1, 2)) / NK.sum() - 1).std())
        out[f"Rs_N_{s}"] = Rs_N; out[f"Rs_Nboot_{s}"] = (rsb / np.where(NK > 0, NK, 1)[None]).astype(np.float32)
        for m in ("closest1", "fof10"):
            NSm = Ns_route(dK_grid(St[m].sum(0), nt_c.sum(0)))
            ch[f"R_s {m}"] = (NSm.sum() / NK.sum() - 1, np.nan)
            ch[f"R_p {m}"] = (tot_change(Nd[m], Ni), (F(Nd[m]).sum((1, 2)) / Nib.sum((1, 2)) - 1).std())
            ch[f"R {m}"] = (tot_change(Nd[m], Ne), np.nan)
        ch["R_p + no real"] = (tot_change(Nd["keepnr"], Ni), (F(Nd["keepnr"]).sum((1, 2)) / Nib.sum((1, 2)) - 1).std())
        ch["R + no real"] = (tot_change(Nd["keepnr"], Ne), np.nan)
        for k, (v, e) in ch.items():
            log(f"    count change at q > {qcut:.0f}, {k:16s}: {100 * v:+.2f}% +- {100 * e:.2f}%")
        tl = np.array([r["tile"] for r in recs])
        for lab, sel_ in (("tiles 0-199", tl < 200), ("tiles 200-399", (tl >= 200) & (tl < 400))):
            if sel_.sum() < 20:
                continue
            a_, i_, e_ = Nd["none"][sel_].sum((1, 2)), Ni[sel_].sum((1, 2)), Ne[sel_].sum((1, 2))
            Wh = A.boot_weights(int(sel_.sum()), 1000, 21)
            log(f"    {lab} ({sel_.sum()} tiles): R_p {100 * (a_.sum() / i_.sum() - 1):+.2f}% +- "
                f"{100 * ((Wh @ a_) / (Wh @ i_) - 1).std():.2f}%, R {100 * (a_.sum() / e_.sum() - 1):+.2f}% +- "
                f"{100 * ((Wh @ a_) / (Wh @ e_) - 1).std():.2f}% (tiles only)")
        if qcut == 5.0:
            # joint bootstrap including the kernel's MC noise (R only)
            KC, KN = A.kernel_stack(recs)
            tix = np.concatenate([np.full(r["th"].size, i) for i, r in enumerate(recs)])
            okz = (z_all >= zed[0]) & (z_all < zed[-1])
            jb = []; Rjb = []
            for b in range(200):
                kc = (W[b] @ KC).reshape(A.TH_K.size, A.QG.size, A.EDGES.size)
                Kb = np.maximum.accumulate(kc / (W[b] @ KN)[:, None, None], axis=1)
                Pb_ = A.pbin(A.K_at(Kb, th_all, qp_all), qed) * W[b, tix][:, None]
                Ne_b = np.stack([np.bincount(iz_all[okz], weights=Pb_[okz, k], minlength=nz) for k in range(nq)], 1)
                jb.append(Ndb[b].sum() / Ne_b.sum() - 1)
                Rjb.append(A.ratio(Ndb[b], Ne_b, Ne_b))
            log(f"    R at q > 5 with the kernel resampled jointly (200 resamples): {100 * ch['R'][0]:+.2f}% +- "
                f"{100 * np.std(jb):.2f}%")
            out["R_joint_boot_q5"] = np.array(jb); out["Rjboot_q5"] = np.array(Rjb, np.float32)
            # step 3a's preview selection (pooled t03 + a02 fit, data_real/step3/selection_arbiter.npz) on the own tiles:
            # P = Phi((sqrt(q^2 + b) - 5 - mu) / exp(ls)) per injected size, linear in ln theta (clamped), applied at
            # q = q' r_inj(theta), zero below the credit limit q' = 2
            if meth in ("whitened", "szifi"):
                from scipy.special import ndtr
                sa = np.load(os.path.join(A.ROOT, "data_real", "step3", "selection_arbiter.npz"))
                fits = sa[f"{mode}_fit"]; rinj = sa[f"{mode}_rinj"]; tinj = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
                n3 = []
                for i, r in enumerate(recs):
                    q3 = r["qp"] * np.interp(np.log(r["th"]), np.log(np.geomspace(0.5, 15.0, 15)), rinj)
                    Pm = np.array([ndtr((np.sqrt(np.maximum(q3 * q3 + b, 0.0)) - 5.0 - mu_) / np.exp(ls)) for b, mu_, ls in fits])
                    x = np.clip(np.log(r["th"]), np.log(tinj[0]), np.log(tinj[-1]))
                    ii = np.clip(np.searchsorted(np.log(tinj), x) - 1, 0, 4)
                    ww = (x - np.log(tinj[ii])) / np.log(tinj[ii + 1] / tinj[ii])
                    nn = np.arange(q3.size)
                    p3 = np.where(r["qp"] < 2.0, 0.0, (1 - ww) * Pm[ii, nn] + ww * Pm[ii + 1, nn])
                    n3.append(p3.sum())
                n3 = np.array(n3); ndt = Nd["none"].sum((1, 2)); net = Ne.sum((1, 2))
                v3 = ndt.sum() / n3.sum() - 1; v3b = (W @ ndt) / (W @ n3) - 1
                kv = net.sum() / n3.sum() - 1; kvb = (W @ net) / (W @ n3) - 1
                log(f"    step 3a's pooled injection selection on the own tiles: N_det / N_exp(3a) - 1 = {100 * v3:+.2f}% +- "
                    f"{100 * v3b.std():.2f}% (tiles only); own kernel N_exp / 3a N_exp - 1 = {100 * kv:+.2f}% +- {100 * kvb.std():.2f}%")
            # by z (coarse) and by q
            for name, num, den, numb, denb in (("R", Nd["none"].sum(0), Ne_t, Ndb, Neb), ("R_p", Nd["none"].sum(0), Ni_t, Ndb, Nib)):
                zc = A.zcoarse(num, den, zed) - 1; zcb = A.zcoarse(numb, denb, zed) - 1
                log(f"    {name:4s} by z (0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3): " + ", ".join(
                    f"{100 * a:+.1f} +- {100 * b:.1f}" for a, b in zip(zc, zcb.std(0))) + " %")
                qc_ = num.sum(0) / den.sum(0) - 1; qcb = numb.sum(1) / denb.sum(1) - 1
                log(f"    {name:4s} by q ({', '.join(f'{a:.1f}-{b:.1f}' for a, b in zip(qed[:-2], qed[1:-1]))}, >100): " + ", ".join(
                    f"{100 * a:+.1f} +- {100 * b:.1f}" for a, b in zip(qc_, qcb.std(0))) + " %")
            zc = A.zcoarse(NS, NK, zed) - 1; zcb = np.array([A.zcoarse(x, NK, zed) - 1 for x in rsb])
            log(f"    R_s  by z: " + ", ".join(f"{100 * a:+.1f} +- {100 * b:.1f}" for a, b in zip(zc, zcb.std(0))) + " %")
            qc_ = NS.sum(0) / NK.sum(0) - 1; qcb = rsb.sum(1) / NK.sum(0)[None] - 1
            log(f"    R_s  by q: " + ", ".join(f"{100 * a:+.1f} +- {100 * b:.1f}" for a, b in zip(qc_, qcb.std(0))) + " %")
            # bin-by-bin agreement of the estimators (bins with N_exp >= 10)
            m10 = Ne_t >= 10
            Rsb_ = rsb / np.where(NK > 0, NK, 1)[None]
            ed = (Rpb[:200] - Rsb_).std(0)
            ok = m10 & (ed > 0)
            log(f"    bins with N_exp >= 10: {m10.sum()}; chi2 of R_p - R_s (bootstrap error of the difference, same "
                f"resamples) {np.sum(((Rp - Rs_N)[ok] / ed[ok]) ** 2):.1f} for {ok.sum()} bins; R - R_s: "
                f"{np.sum(((R - Rs_N)[ok] / (Rb[:200] - Rsb_).std(0)[ok]) ** 2):.1f}")
    np.savez_compressed(os.path.join(A.OUT, f"datavec_{meth}_{mode}.npz"), **out)
log(f"\ny06 done in {time.time() - t00:.0f} s")
