"""Step 3b / b05: the analyst's no-blending selection kernel K(theta500, q'; e) per finder variant and noise level, from
the virtual isolated clusters of b03 (noise-only tiles 0..323, the variant's own finder, both iterations).

K(theta, q'; e) = P(an isolated cluster of size theta and S/N q' = Y500 / sigma_finder(theta) is credited with
q_obs > e), e = step 2's 19 bin edges (data_cosmo/kernel_catalogue.npz EDGES).
  * theta: the 25 nodes TH_K (0.238'-61', every injected size of t03 / a02 is a node); linear in ln theta between them,
    clamped outside (the painter's node mixture at each TH_K is what was inserted).
  * q' <= Q_MODEL = 27.5: the measured fractions at the discrete nodes Q_NODES (1, 1.5, 1.75, 2-10 step 0.125, 10.5-27.5
    step 0.5), binomial Monte Carlo errors; K = 0 at q' = 0 and below the credit limit q' = 2 (the measured fraction there
    must be 0: printed).
  * q' > 27.5: an empirical model from the log-uniform clusters (q' = 25-400) of the same node: each such cluster gives
    lambda = own S/N at its credited peak / q' and eps = q_obs - own S/N; K(q'; e) = mean over them of
    1[lambda q' + eps > e] (the observed efficiency and noise at the peak, rescaled to q').  Checked against the
    measured fractions at q' = 20-27.5 and in log-q' bins of the same clusters.
  * monotone in q' (weighted pool-adjacent-violators per theta node and edge over all q' nodes; it keeps the ordering in
    e).
Checks printed: credit limit; nodes per cell; Monte Carlo error of K; size of the PAVA corrections; model vs measured;
window check (b03: the kernel clusters of tiles 0-7 again with 2x the window); accuracy of the virtual response (b03,
tile 0); half-split convergence (kernels from even and from odd tiles, and from tiles 0-161 and 162-323: N_exp over
the realistic halos, q > 5); kernel bootstrap (NBK resamples of the noise-only tiles: the kernel's Monte Carlo error on
N_exp).  Errors of measured fractions use a pseudo-count (K(1-K) + 1/(4n)) / n, so that fractions of 0 or 1 have one.
Outputs: data_real/step3b/kernel_VARIANT_MODE.npz (th, QA, K, K_err, n, EDGES, model diagnostics; kernels of the even
and odd halves as K_even, K_odd).  Usage: python b05_kernel.py VARIANT [VARIANT ...]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(2)
import numpy as np                                 # noqa: E402
F = B.F
E = B.EDGES
QA_MODEL = np.geomspace(B.Q_MODEL, 3000.0, 400)[1:]
NBK = 60


def load(variant, mode):
    keys = ("k_thi", "k_q", "k_qobs", "k_own")
    acc = {k: [] for k in keys}
    pid = []
    extra = dict(wc_same=0, wc_n=0, wc_maxdq=0.0, acc=[], noise_n=[], r=[])
    for p in range(324):
        fn = os.path.join(B.OUT, "virtual", variant, f"v_{mode}_{p:03d}.npz")
        with np.load(fn) as d:
            for k in keys:
                acc[k].append(d[k])
            pid.append(np.full(d["k_q"].size, p, np.int16))
            if "wc_qobs" in d.files:
                a, b = d["wc_qobs"], d["k_qobs"]
                same = (a == b) | (np.isinf(a) & np.isinf(b))
                extra["wc_same"] += int(same.sum()); extra["wc_n"] += a.size
                fin = np.isfinite(a) & np.isfinite(b)
                if fin.any():
                    extra["wc_maxdq"] = max(extra["wc_maxdq"], float(np.abs(a[fin] - b[fin]).max()))
                extra["wc_changed_det5"] = extra.get("wc_changed_det5", 0) + int(((a > 5) != (b > 5)).sum())
            if "acc_err" in d.files:
                extra["acc"] = d["acc_err"]
            extra["noise_n"].append(d["noise_n"]); extra["r"].append(d["r"])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["pid"] = np.concatenate(pid)
    return out, extra


def model_K(lam, eps, q):
    """K(q'; e) = mean over the pairs of 1[lam q' + eps > e] = fraction of pair thresholds (e - eps) / lam below q'."""
    out = np.zeros((np.size(q), E.size))
    if lam.size == 0:
        return out
    for a, e in enumerate(E):
        thr = np.sort((e - eps) / lam)
        out[:, a] = np.searchsorted(thr, np.asarray(q, float), side="left") / thr.size
    return out


def build(V, sel):
    """Kernel from the virtual clusters V[sel]."""
    thi, q, qo, own = V["k_thi"][sel], V["k_q"][sel].astype(float), V["k_qobs"][sel].astype(float), V["k_own"][sel].astype(float)
    nodes = B.Q_NODES[B.Q_NODES <= B.Q_MODEL + 1e-9]
    QA = np.concatenate([[0.0], nodes, QA_MODEL])
    nK, nn, ne = B.TH_K.size, nodes.size, E.size
    K = np.zeros((nK, QA.size, ne)); n = np.zeros((nK, QA.size)); Kraw = np.zeros((nK, nn, ne))
    lam_eps = []
    for i in range(nK):
        mi = thi == i
        # discrete nodes
        qi, oi = q[mi], qo[mi]
        kk = np.searchsorted(nodes, qi)
        on = (kk < nn) & np.isclose(qi, nodes[np.minimum(kk, nn - 1)])
        cnt = np.bincount(kk[on], minlength=nn).astype(float)
        for a, e in enumerate(E):
            Kraw[i, :, a] = np.bincount(kk[on], weights=(oi[on] > e).astype(float), minlength=nn) / np.maximum(cnt, 1)
        n[i, 1:nn + 1] = cnt
        # model above Q_MODEL from the log-uniform clusters
        lu = (qi >= B.Q_LOGU[0]) & ~np.isclose(qi, np.round(qi * 8) / 8)
        lam = own[mi][lu] / qi[lu]
        eps = oi[lu] - own[mi][lu]
        lam_eps.append((lam, eps, qi[lu], oi[lu]))
        Km = model_K(lam, eps, QA_MODEL)
        n[i, nn + 1:] = lam.size
        Kc = np.concatenate([np.zeros((1, ne)), Kraw[i], Km], 0)
        Kc[QA < B.CRED_MIN] = 0.0                                       # credit limit
        w = np.concatenate([[1e6], np.maximum(cnt, 1.0), np.full(QA_MODEL.size, float(max(lam.size, 1)))])
        for a in range(ne):
            K[i, :, a] = B.pava_increasing(Kc[:, a], w)
    return QA, K, n, Kraw, nodes, lam_eps


def n_exp(Kern, mode, variant, qcut=5.0):
    """N_exp(q_obs > qcut) over the realistic halos (b02 tables) and per step-3a coarse z bin."""
    C = np.load(os.path.join(B.OUT, f"halos_{mode}.npz"))
    V = np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz"))
    th, qp, z = C["theta500"].astype(float), V["qp"].astype(float), C["z"].astype(float)
    m = qp >= B.CRED_MIN
    ei = int(np.argmin(np.abs(E - qcut)))
    k = Kern(th[m], qp[m], ei)[:, 0]
    return k.sum(), np.histogram(z[m], [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0], weights=k)[0]


if __name__ == "__main__":
    t00 = time.time()
    for variant in sys.argv[1:] or list(B.VARIANTS):
        for mode in B.MODES:
            t0 = time.time()
            V, X = load(variant, mode)
            print(f"==================== {variant} / {mode}: {V['k_q'].size} virtual clusters on 324 noise-only tiles "
                  f"({V['k_q'].size / B.TH_K.size:.0f} per theta node)", flush=True)
            r = np.array(X["r"])
            print(f"  noise-only tiles: r = sigma_finder / sigma_Y after the iteration at 0.5' / 4.45' / 15': "
                  + " / ".join(f"{v:.4f}" for v in r.mean(0)[[0, 9, 14]]) + f" (tile-to-tile std {r.std(0)[0]:.4f})")
            nn = np.array(X["noise_n"])                                    # (324, merges, 3 thresholds)
            print("  noise peaks per 100 deg^2 (final catalogue), q > 4 / 4.5 / 5, merge none / closest 1' / fof 10': "
                  + "; ".join(" / ".join(f"{v:.2f}" for v in nn[:, a].sum(0) / 324.0) for a in range(nn.shape[1])))
            QA, K, n, Kraw, nodes, LE = build(V, np.ones(V["k_q"].size, bool))
            nl = nodes.size
            # credit limit
            below = nodes < B.CRED_MIN
            print(f"  credit limit: credited fraction (q_obs > 4) at q' = 1, 1.5, 1.75, max over theta nodes: "
                  + ", ".join(f"{Kraw[:, k, 0].max():.4f}" for k in np.flatnonzero(below))
                  + f" ({int(n[:, 1:nl + 1][:, below].sum())} clusters); at q' = 2: {Kraw[:, np.flatnonzero(np.isclose(nodes, 2.0))[0], 0].max():.4f}")
            cnt = n[:, 1:nl + 1]
            mid = (nodes >= 2) & (nodes <= 10)
            print(f"  clusters per (theta, q') node: q' 2-10: {cnt[:, mid].min():.0f}-{cnt[:, mid].max():.0f} (mean {cnt[:, mid].mean():.0f}); "
                  f"q' 10.5-27.5: {cnt[:, nodes > 10].min():.0f}-{cnt[:, nodes > 10].max():.0f}; log-uniform (model) per theta node: "
                  f"{n[:, -1].min():.0f}-{n[:, -1].max():.0f}")
            e5 = int(np.argmin(np.abs(E - 5.0)))
            err = np.sqrt(np.maximum(Kraw * (1 - Kraw), 0) / np.maximum(cnt, 1)[:, :, None])
            errp = np.sqrt((Kraw * (1 - Kraw) + 0.25 / np.maximum(cnt, 1)[:, :, None]) / np.maximum(cnt, 1)[:, :, None])  # pseudo-count
            print(f"  Monte Carlo error of K(e = 5) at the q' 2-10 nodes: median {np.median(err[:, mid, e5]):.4f}, max {err[:, mid, e5].max():.4f}")
            dp = np.abs(K[:, 1:nl + 1] - np.where(nodes[None, :, None] < B.CRED_MIN, 0.0, Kraw))
            pp = dp / errp
            print(f"  monotone fix (PAVA): max |K - measured| over the measured nodes {dp.max():.4f}; in units of the binomial error "
                  f"(with a pseudo-count of 1/4): max {pp.max():.2f}, rms {np.sqrt(np.mean(pp ** 2)):.2f}; nodes moved by > 2 errors: "
                  f"{int((pp > 2).sum())} of {pp.size}")
            Kmono = K.copy()
            Kmono[:, 1:nl + 1] = np.where(nodes[None, :, None] < B.CRED_MIN, 0.0, Kraw)
            Nm, _ = n_exp(B.Kernel(B.TH_K, QA, K), mode, variant); Nr, _ = n_exp(B.Kernel(B.TH_K, QA, Kmono), mode, variant)
            print(f"  monotone fix: N_exp(q > 5) over the realistic halos with the measured (non-monotone) fractions instead: {100 * (Nr / Nm - 1):+.3f}%")
            # model vs measured
            hi = (nodes >= 20.0) & (nodes <= B.Q_MODEL + 1e-9)
            dev = []
            for i in range(B.TH_K.size):
                lam, eps, ql, ol = LE[i]
                for kq in np.flatnonzero(hi):
                    km = model_K(lam, eps, [nodes[kq]])[0]
                    nm, nn_ = max(lam.size, 1), max(cnt[i, kq], 1)
                    sm = (km * (1 - km) + 0.25 / nm) / nm; sk = (Kraw[i, kq] * (1 - Kraw[i, kq]) + 0.25 / nn_) / nn_
                    dev.append((km - Kraw[i, kq]) / np.sqrt(sm + sk))
            dev = np.array(dev)
            adiff = []
            for i in range(B.TH_K.size):
                lam, eps, ql, ol = LE[i]
                for kq in np.flatnonzero(hi):
                    adiff.append(model_K(lam, eps, [nodes[kq]])[0] - Kraw[i, kq])
            adiff = np.array(adiff)
            print(f"  model (lambda q' + eps) vs measured nodes q' 20-27.5: pulls over all theta nodes and edges (errors with a pseudo-count): "
                  f"rms {np.sqrt(np.mean(dev ** 2)):.2f}, max |pull| {np.abs(dev).max():.2f}, |pull| > 3: {int((np.abs(dev) > 3).sum())} of {dev.size}; "
                  f"model - measured: mean {adiff.mean():+.4f}, max |.| {np.abs(adiff).max():.3f}")
            sc = []
            for i in range(B.TH_K.size):
                lam, eps, ql, ol = LE[i]
                for lo, hi_ in ((25, 40), (40, 70), (70, 120), (120, 220), (220, 400)):
                    mm = (ql >= lo) & (ql < hi_)
                    if mm.sum() < 20:
                        continue
                    pr = model_K(lam, eps, ql[mm])                       # model at every cluster of the bin
                    ob = (ol[mm][:, None] > E[None, :]).astype(float)
                    for a in np.flatnonzero(E >= 22):
                        o, x = ob[:, a].sum(), pr[:, a].sum()
                        v = np.sum(pr[:, a] * (1 - pr[:, a])) + 0.25
                        sc.append(((o - x) / mm.sum(), (o - x) / np.sqrt(v)))
            sc = np.array(sc)
            print(f"  model self-consistency in log-q' bins (25-400; edges >= 22): observed - model fraction: rms {np.sqrt(np.mean(sc[:, 0] ** 2)):.4f}, "
                  f"max |.| {np.abs(sc[:, 0]).max():.3f}; pulls rms {np.sqrt(np.mean(sc[:, 1] ** 2)):.2f}, max |pull| {np.abs(sc[:, 1]).max():.2f} "
                  f"({sc.shape[0]} comparisons)")
            lb = [(np.median(LE[i][0][(LE[i][2] >= a) & (LE[i][2] < b)]), LE[i][1][(LE[i][2] >= a) & (LE[i][2] < b)].mean())
                  for i in (0, 12, 24) for a, b in ((25, 40), (100, 160), (240, 400))]
            print("  efficiency lambda (median) and eps (mean) at q' 25-40 / 100-160 / 240-400 for theta 0.24', 3.8', 61': "
                  + "; ".join(" / ".join(f"{l_:.4f}, {e_:+.2f}" for l_, e_ in lb[3 * k:3 * k + 3]) for k in range(3)))
            lam_all = np.concatenate([LE[i][0] for i in range(B.TH_K.size)]); eps_all = np.concatenate([LE[i][1] for i in range(B.TH_K.size)])
            print(f"  log-uniform clusters (q' 25-400): not credited {int(np.sum(~np.isfinite(np.concatenate([LE[i][3] for i in range(B.TH_K.size)]))))}; "
                  f"efficiency lambda per node (median): " + " ".join(f"{np.median(LE[i][0]):.3f}" for i in range(0, B.TH_K.size, 4))
                  + f" (nodes {np.round(B.TH_K[::4], 2).tolist()}); eps mean {eps_all.mean():+.3f}, std {eps_all.std():.3f}")
            # window and accuracy checks (b03)
            print(f"  window check (tiles 0-7, 2x window): q_obs identical for {X['wc_same']} of {X['wc_n']} clusters; changed across q = 5: "
                  f"{X.get('wc_changed_det5', 0)}; max |dq| {X['wc_maxdq']:.2e}")
            if len(X["acc"]):
                a = np.asarray(X["acc"])
                print(f"  accuracy of the virtual response (tile 0, 2 clusters per node, +-2W window): max |virtual - exact| / peak = "
                      f"{a[:, :, 1].max():.1e} (angle-averaged tables alone: {a[:, :, 0].max():.1e}; at theta <= 2.4': {a[:10, :, 0].max():.1e})")
            # half split
            ev = V["pid"] % 2 == 0
            QAe, Ke, *_ = build(V, ev)
            QAo, Ko, *_ = build(V, ~ev)
            KK = B.Kernel(B.TH_K, QA, K); KE = B.Kernel(B.TH_K, QAe, Ke); KO = B.Kernel(B.TH_K, QAo, Ko)
            Nf, zf = n_exp(KK, mode, variant); Ne, ze = n_exp(KE, mode, variant); No, zo = n_exp(KO, mode, variant)
            print(f"  half-split: N_exp(q > 5) over the realistic halos with the full / even-tile / odd-tile kernels: {Nf:.1f} / {Ne:.1f} / {No:.1f}; "
                  f"even/odd - 1 = {100 * (Ne / No - 1):+.3f}% (one realisation of a difference whose std is 2x the full kernel's error, see the bootstrap below); "
                  f"per coarse z bin (even/odd - 1): " + ", ".join(f"{100 * (a / b - 1):+.2f}%" for a, b in zip(ze, zo)))
            lo_ = V["pid"] < 162
            QA1, K1, *_ = build(V, lo_); QA2, K2, *_ = build(V, ~lo_)
            N1, _ = n_exp(B.Kernel(B.TH_K, QA1, K1), mode, variant); N2, _ = n_exp(B.Kernel(B.TH_K, QA2, K2), mode, variant)
            print(f"  second split (tiles 0-161 / 162-323): N_exp {N1:.1f} / {N2:.1f}, ratio - 1 = {100 * (N1 / N2 - 1):+.3f}%")
            rngk = np.random.default_rng(55 + B.MODES.index(mode))
            NB_ = []
            for b in range(NBK):
                wt = np.bincount(rngk.integers(0, 324, 324), minlength=324)
                ii = np.repeat(np.arange(V["pid"].size), wt[V["pid"]])
                Vb = {k: v[ii] for k, v in V.items()}
                QAb, Kb, *_ = build(Vb, np.ones(ii.size, bool))
                NB_.append(n_exp(B.Kernel(B.TH_K, QAb, Kb), mode, variant)[0])
            NB_ = np.array(NB_)
            print(f"  kernel bootstrap ({NBK} resamples of the 324 noise-only tiles): N_exp(q > 5) relative std {100 * NB_.std() / Nf:.3f}% "
                  f"(mean of resamples / full - 1 = {100 * (NB_.mean() / Nf - 1):+.3f}%)", flush=True)
            np.savez(os.path.join(B.OUT, f"kernel_{variant}_{mode}.npz"), th=B.TH_K, QA=QA, K=K, n=n, EDGES=E, Kraw=Kraw, nodes=nodes,
                     K_err=err, K_even=Ke, K_odd=Ko, lam_median=np.array([np.median(LE[i][0]) for i in range(B.TH_K.size)]),
                     eps_mean=np.array([LE[i][1].mean() for i in range(B.TH_K.size)]), eps_std=np.array([LE[i][1].std() for i in range(B.TH_K.size)]))
            print(f"  saved kernel_{variant}_{mode}.npz [{time.time() - t0:.0f} s]", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {B.peak_rss_gb():.2f} GB")
