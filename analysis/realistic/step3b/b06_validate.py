"""Step 3b / b06: the virtual-cluster kernel K (b05, production finder) against real injections, and the cause of step 3a's
gap between injection tiles and noise-only virtual clusters.

Units: every injected cluster gets q' = Y500 / sigma_finder(theta) of its own tile (the tile's final sigma, s3blib
convention), so all comparisons are in the finder's own units.  'Credited' = a02's / t03's rule (nearest injected
cluster, filtered tSZ >= 2 sigma), the same criterion as the kernel's (own filtered signal >= 2 sigma).
  (1) t03 (38,760 clusters per level) and a02 (155,040): observed credited fractions q_obs > e against the kernel
      K(theta_inj, q'), per size and q' range; tile-bootstrap errors.
  (2) The sparse injections of b03 (<= 10 clusters per tile, 22,680 per level, end to end on the noise-only skies):
      the same comparison; and paired with the same clusters inserted virtually on the same CMB + noise (q_obs painted
      - virtual); merge rules on isolated clusters.
  (3) The dense experiment of b04 (a02's design on the noise-only skies): painted (P), virtual in the dense tile's filter
      (Vd), virtual in the noise-only filter (Vn), all paired.
  (4) What the differences mean for the counts: 'halo-weighted' ratios sum w 1[q_obs > 5] / sum w K, each cluster
      weighted by the density of realistic halos (production, all 324 tiles) at its (theta, q') over the density of the
      test clusters there (bins of 0.25 in q'; theta within a factor 2^(1/2) of the cluster's size), i.e. N_exp(test
      clusters' selection) / N_exp(K) over the realistic halos with q' 1.8-9 (the range of a02 and of the sparse set).
Usage: python b06_validate.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(2)
import numpy as np                                 # noqa: E402
F = B.F
E = B.EDGES
E5 = int(np.argmin(np.abs(E - 5.0)))
rng = np.random.default_rng(606)
QB = np.arange(1.75, 9.0 + 1e-9, 0.25)


def load_inj(mode):
    """t03 and a02 clusters: set, tile, theta, q', q_obs (credited best; -inf if none)."""
    rows = []
    d3 = os.path.join(B.D3, "injections")
    for f in sorted(os.listdir(d3)):
        if f.startswith(f"inj_{mode}_"):
            with np.load(os.path.join(d3, f)) as r:
                sig = r["sig"]; th = float(r["theta_inj"]); Y = r["Y"]
                cred = r["d_S_tsz"] / sig[r["d_j"]] >= 2.0
                best = np.full(Y.size, -np.inf)
                for d in np.flatnonzero(cred):
                    best[r["d_near"][d]] = max(best[r["d_near"][d]], r["d_q"][d])
                rows.append((np.zeros(Y.size, int), np.full(Y.size, int(r["k"])), np.full(Y.size, th),
                             Y / B.sigma_finder(sig, mode, th), best))
    d2 = os.path.join(B.D3, "injections_hi")
    for f in sorted(os.listdir(d2)):
        if f.startswith(f"inj_{mode}_"):
            with np.load(os.path.join(d2, f)) as r:
                sig = r["sig"]; th = float(r["theta_inj"]); q = r["q_inj"]
                Y = q * B.sigma_Y(mode, th)
                rows.append((np.ones(Y.size, int), np.full(Y.size, int(r["k"])), np.full(Y.size, th),
                             Y / B.sigma_finder(sig, mode, th), r["qbest"]))
    return [np.concatenate(a) for a in zip(*rows)]


def halo_density(mode):
    C = np.load(os.path.join(B.OUT, f"halos_{mode}.npz")); V = np.load(os.path.join(B.OUT, f"halos_prod_{mode}.npz"))
    return C["theta500"].astype(float), V["qp"].astype(float)


def weights(th_c, q_c, hth, hq):
    """w_c = halo density / test-cluster density at (theta_c, q'_c), bins of 0.25 in q', theta within 2^(+-1/2)."""
    w = np.zeros(th_c.size)
    for t in np.unique(th_c):
        mc = (th_c == t)
        mh = np.abs(np.log(hth / t)) <= 0.5 * np.log(2.0)
        Hc = np.histogram(hq[mh], QB)[0].astype(float)
        Ic = np.histogram(q_c[mc], QB)[0].astype(float)
        k = np.clip(np.searchsorted(QB, q_c[mc], side="right") - 1, 0, QB.size - 2)
        inside = (q_c[mc] >= QB[0]) & (q_c[mc] < QB[-1])
        w[mc] = np.where(inside & (Ic[k] > 0), Hc[k] / np.maximum(Ic[k], 1), 0.0)
    return w


def weighted_ratio(det, Kc, w, groups, nboot=1000):
    r = np.sum(w * det) / np.sum(w * Kc) - 1
    ug, inv = np.unique(groups, return_inverse=True)
    a = np.bincount(inv, weights=w * det); b = np.bincount(inv, weights=w * Kc)
    bb = rng.integers(0, ug.size, (nboot, ug.size))
    return r, (a[bb].sum(1) / b[bb].sum(1) - 1).std()


RES = {}


def compare(lab, th, qp, qo, tile, Kern, hth, hq, sizes=None, key=None):
    Kc = Kern(th, qp)                                                         # (n, ne)
    det = (qo[:, None] > E[None, :]).astype(float)
    print(f"  {lab}: {th.size} clusters; observed / expected credited (kernel at each cluster's theta, q') - 1, tile bootstrap:")
    for e in (4.0, 5.0, 6.0, 7.0):
        a = int(np.argmin(np.abs(E - e)))
        r, er = weighted_ratio(det[:, a], Kc[:, a], np.ones(th.size), tile)
        print(f"     q_obs > {e:.0f}: {100 * r:+.2f} +- {100 * er:.2f}%  (observed {det[:, a].sum():.0f}, expected {Kc[:, a].sum():.1f})")
    w = weights(th, qp, hth, hq)
    r, er = weighted_ratio(det[:, E5], Kc[:, E5], w, tile)
    print(f"     halo-weighted (N_exp of the realistic halos, q' 1.75-9), q_obs > 5: {100 * r:+.2f} +- {100 * er:.2f}%")
    if key is not None:
        r5, e5_ = weighted_ratio(det[:, E5], Kc[:, E5], np.ones(th.size), tile)
        RES[key] = dict(q5=float(r5), q5_err=float(e5_), halo=float(r), halo_err=float(er))
    rows = []
    for t in (sizes if sizes is not None else np.unique(th)):
        m = np.isclose(th, t)
        line = []
        chi = []
        for lo, hi in ((2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 9.0)):
            mm = m & (qp >= lo) & (qp < hi)
            if mm.sum() < 30:
                continue
            o, x = det[mm, E5].mean(), Kc[mm, E5].mean()
            s = np.sqrt(max(x * (1 - x), 1e-4) / mm.sum())
            line.append(f"{lo:.0f}-{hi:.0f}: {o:.3f}/{x:.3f}"); chi.append((o - x) / s)
        rows.append((t, line, chi))
        print(f"     theta {t:6.3f}': P(q_obs > 5) observed/kernel per q' range: " + "  ".join(line)
              + f"   [pulls {' '.join(f'{c:+.1f}' for c in chi)}]")
    allchi = np.concatenate([np.array(c) for _, _, c in rows]) if rows else np.zeros(0)
    print(f"     chi^2 of the binned comparison: {np.sum(allchi ** 2):.1f} for {allchi.size} bins")
    return w


if __name__ == "__main__":
    t00 = time.time()
    for mode in B.MODES:
        Kern = B.Kernel.load(os.path.join(B.OUT, f"kernel_prod_{mode}.npz"))
        hth, hq = halo_density(mode)
        print(f"==================== {mode}", flush=True)
        # ---- (1) t03, a02
        st, tile, th, qp, qo = load_inj(mode)
        for s_, lab in ((0, "(1a) t03 injection tiles (15% of clusters at q_inj 10.5-30)"), (1, "(1b) a02 injection tiles")):
            m = st == s_
            compare(lab, th[m], qp[m], qo[m], tile[m], Kern, hth, hq, key=f"{mode}_{'t03' if s_ == 0 else 'a02'}")
        m = np.ones(st.size, bool)
        compare("(1c) t03 + a02 pooled", th, qp, qo, tile + 10000 * st, Kern, hth, hq, sizes=[], key=f"{mode}_pooled")
        # ---- (2) sparse injections
        S = {k: [] for k in ("thi", "qp", "q_none", "q_closest1", "q_fof10", "qv", "qt", "pid", "Y", "qv0")}
        for p in range(324):
            with np.load(os.path.join(B.OUT, "virtual", "prod", f"v_{mode}_{p:03d}.npz")) as d:
                for k in S:
                    if k == "pid":
                        S[k].append(np.full(d["sp_qp"].size, p))
                    else:
                        S[k].append(d["sp_" + k])
        S = {k: np.concatenate(v) for k, v in S.items()}
        sth = B.TH_K[S["thi"]]
        compare("(2a) sparse injections, end to end", sth, S["qp"], S["q_none"], S["pid"], Kern, hth, hq, sizes=[], key=f"{mode}_sparse")
        compare("(2b) the same clusters inserted virtually on the noise-only filter", sth, S["qt"], S["qv"], S["pid"], Kern, hth, hq, sizes=[])
        qa, qb = S["q_none"], S["qv"]
        both = np.isfinite(qa) & np.isfinite(qb)
        dq = qa[both] - qb[both]
        same = np.isclose(qa, qb, atol=1e-6) | (np.isinf(qa) & np.isinf(qb))
        print(f"  (2c) paired painted - virtual (same clusters, same CMB + noise): identical q_obs {same.mean():.3f}; credited in both "
              f"{both.sum()}, only painted {int((np.isfinite(qa) & ~np.isfinite(qb)).sum())}, only virtual {int((~np.isfinite(qa) & np.isfinite(qb)).sum())}; "
              f"q_obs > 5: painted {int((qa > 5).sum())}, virtual {int((qb > 5).sum())}, disagree {int(((qa > 5) != (qb > 5)).sum())}", flush=True)
        rel = dq / qb[both]
        print(f"     dq = q_painted - q_virtual: mean {dq.mean():+.4f}, std {dq.std():.4f}, 1/50/99% {np.percentile(dq, [1, 50, 99]).round(4).tolist()}; "
              f"dq/q: mean {rel.mean():+.5f}, std {rel.std():.5f}", flush=True)
        for lo, hi in ((4, 6), (6, 10), (10, 40)):
            mm = (qb[both] >= lo) & (qb[both] < hi)
            print(f"        q_virtual {lo}-{hi}: N {mm.sum()}, mean dq {dq[mm].mean():+.4f}, std {dq[mm].std():.4f}")
        ds = S["qp"] / S["qt"] - 1
        print(f"     sigma of the sparse injection tile vs the noise-only tile (q'_painted / q'_virtual - 1): mean {ds.mean():+.2e}, max |.| {np.abs(ds).max():.1e}")
        for mname in ("closest1", "fof10"):
            a5 = np.where(S["q_none"] > 5, S["q_none"], -np.inf); b5 = np.where(S[f"q_{mname}"] > 5, S[f"q_{mname}"], -np.inf)
            print(f"     merge {mname}: isolated clusters whose credited q > 5 changes: {int(np.sum(~((a5 == b5) | (np.isinf(a5) & np.isinf(b5)))))} of {a5.size}")
        # ---- (3) dense experiment
        Dn = [dict(np.load(os.path.join(B.OUT, "dense", f))) for f in sorted(os.listdir(os.path.join(B.OUT, "dense"))) if f.startswith(f"d_{mode}_")]
        if Dn:
            th_ = np.concatenate([np.full(d["Y"].size, float(d["theta"])) for d in Dn])
            Y = np.concatenate([d["Y"] for d in Dn])
            qpd = np.concatenate([d["Y"] / B.sigma_finder(d["sig_d"], mode, float(d["theta"])) for d in Dn])
            qpn = np.concatenate([d["Y"] / B.sigma_finder(d["sig_n"], mode, float(d["theta"])) for d in Dn])
            P = np.concatenate([d["P"] for d in Dn]); Vd = np.concatenate([d["Vd"] for d in Dn]); Vn = np.concatenate([d["Vn"] for d in Dn])
            tl = np.concatenate([np.full(d["Y"].size, int(d["pid"])) for d in Dn])
            rd = np.array([d["sig_d"] / d["sig_n"] for d in Dn])
            print(f"  (3) dense experiment: {len(Dn)} tiles, {Y.size} clusters; sigma_dense / sigma_noise-only at 0.5' / 4.45' / 15': "
                  + " / ".join(f"{v:+.4f}" for v in rd.mean(0)[[0, 9, 14]] - 1) + f"; masked fraction dense {100 * (1 - np.mean([d['fsky_d'] for d in Dn])):.2f}%, "
                  f"noise-only {100 * (1 - np.mean([d['fsky_n'] for d in Dn])):.2f}%", flush=True)
            compare("(3a) P: painted, end to end (q' in the dense tile's units)", th_, qpd, P, tl, Kern, hth, hq, sizes=[])
            compare("(3b) Vd: virtual, dense tile's filter (same units)", th_, qpd, Vd, tl, Kern, hth, hq, sizes=[])
            compare("(3c) Vn: virtual, noise-only filter (its units)", th_, qpn, Vn, tl, Kern, hth, hq, sizes=[])
            # paired difference of the selection in finder units: Vd (dense filter, q'_d) vs Vn (noise-only filter, q'_n)
            Kd, Kn = Kern(th_, qpd), Kern(th_, qpn)
            ut, inv = np.unique(tl, return_inverse=True)
            bb = rng.integers(0, ut.size, (1000, ut.size))
            for e in (4.0, 5.0):
                a = int(np.argmin(np.abs(E - e)))
                for lab, X_, KX, Y_, KY in (("Vd vs Vn", Vd, Kd, Vn, Kn), ("P vs Vn", P, Kd, Vn, Kn), ("P vs Vd", P, Kd, Vd, Kd)):
                    for wlab, w in (("cluster", np.ones(th_.size)), ("halo", weights(th_, qpd, hth, hq))):
                        num1 = np.bincount(inv, weights=w * (X_ > e)); den1 = np.bincount(inv, weights=w * KX[:, a])
                        num2 = np.bincount(inv, weights=w * (Y_ > e)); den2 = np.bincount(inv, weights=w * KY[:, a])
                        d0 = num1.sum() / den1.sum() - num2.sum() / den2.sum()
                        db = num1[bb].sum(1) / den1[bb].sum(1) - num2[bb].sum(1) / den2[bb].sum(1)
                        print(f"     paired, q_obs > {e:.0f}, {wlab}-weighted: (obs/exp)[{lab.split(' vs ')[0]}] - (obs/exp)[{lab.split(' vs ')[1]}] = "
                              f"{100 * d0:+.2f} +- {100 * db.std():.2f}%")
                        if e == 5.0 and wlab == "halo" and lab == "Vd vs Vn":
                            RES[f"{mode}_dense_VdVn_halo"] = dict(diff=float(d0), err=float(db.std()))
            for lab, a, b in (("P - Vd", P, Vd), ("Vd - Vn", Vd, Vn), ("P - Vn", P, Vn)):
                bb = np.isfinite(a) & np.isfinite(b)
                print(f"     paired {lab}: q_obs > 5 disagree {int(((a > 5) != (b > 5)).sum())} of {a.size} ({int(((a > 5) & ~(b > 5)).sum())} / "
                      f"{int((~(a > 5) & (b > 5)).sum())} one-sided); dq mean {np.mean(a[bb] - b[bb]):+.4f}, std {np.std(a[bb] - b[bb]):.4f}")
    import json
    json.dump(RES, open(os.path.join(B.OUT, "validate.json"), "w"), indent=1)
    print(f"saved validate.json; done [{time.time() - t00:.0f} s]")
