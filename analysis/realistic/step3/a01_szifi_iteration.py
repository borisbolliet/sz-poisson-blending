"""Step 3 arbiter / a01: SZiFi's own masked-covariance iteration against the build's whitened one.

Question (szifi check, claims B2, B3, C1, C2, R2, R6): does the build's iteration (cluster mask applied to pre-whitened
maps, pseudo-spectra / f) realise what SZiFi's iteration realises with its defaults?  SZiFi (mmf.py find_clusters,
read, not run) reaches the covariance with the cluster mask only through inpainting: band-limited maps -> diffusive
inpainting of the holes (100 sweeps) -> circular cosine blend back towards the original map -> spectra, with the bins
above l_sep = 3000 divided by the unmasked fraction (spec.correct_for_inpainting).  Without inpainting the mask does
not enter at all.  szsim.finder.masked_cov(method='szifi') is the arbiter's own implementation of these steps.

Runs (all with the build's tiles and seeds):
  realistic tiles pid = 0, 11, ..., 319 (30 per noise level; includes the szifi check's 10 tiles 0, 33, ..., 297);
  noise-only tiles: the same pids without halos (as t02);
  injection tiles k = 0, 1, 30, 31, 60, 61, 90, 91, 120, 121, 180, 181 (two per size, t03's layouts and seeds).
Finder variants (iteration 1): 'whitened' (production), 'szifi' (lsep 3000, blend, 100 sweeps = SZiFi defaults),
'szifi_l12000' (lsep 12000 as in SZiFi's SO example script), 'szifi_noblend' (lsep 3000, inpainting only).
Outputs: data_real/step3/arbiter/a01_*.npz; this log.  Usage: python a01_szifi_iteration.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F

OUTD = os.path.join(S.OUT, "arbiter")
PIDS = list(range(0, 324, 11))
PIDS_V = list(range(0, 324, 33))                    # the szifi check's tiles
INJ_K = [0, 1, 30, 31, 60, 61, 90, 91, 120, 121, 180, 181]
TH_INJ = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
METHODS = (("whitened", "whitened", {}),
           ("szifi", "szifi", dict(lsep=3000.0, blend=True, n_inpaint=100)),
           ("szifi_l12000", "szifi", dict(lsep=12000.0, blend=True, n_inpaint=100)),
           ("szifi_noblend", "szifi", dict(lsep=3000.0, blend=False, n_inpaint=100)))
LR = ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))


def inj_layout(th, k):
    """t03's layout (same seeds and rules)."""
    sp = {0.3: 40.0, 0.6: 40.0, 1.2: 40.0, 2.4: 40.0, 4.8: 60.0, 9.6: 75.0}[th]
    rng = np.random.default_rng(700_000 + k)
    n = int(round(F.LBOX / sp))
    g = (np.arange(n) + 0.5) * sp
    yy, xx = np.meshgrid(g, g, indexing="ij")
    off = rng.uniform(0, sp, 2)
    y = (yy.ravel() + off[0] + rng.uniform(-3, 3, n * n)) % F.LBOX
    x = (xx.ravel() + off[1] + rng.uniform(-3, 3, n * n)) % F.LBOX
    q = rng.uniform(1.5, 9.5, n * n)
    hi = rng.random(n * n) < 0.15
    q[hi] = rng.choice(np.array([10.5, 12.0, 15.0, 20.0, 30.0]), hi.sum())
    return y, x, q


def inj_theta(k):
    edges = np.cumsum([30, 30, 30, 30, 60, 90])
    return float(TH_INJ[np.searchsorted(edges, k, side="right")])


def spec_ratio(C1, C0, G):
    lc = (np.arange(G.nbin) + 0.5) * F.DBIN
    out = []
    for lo, hi in LR:
        b = G.bins_used[(lc[G.bins_used] >= lo) & (lc[G.bins_used] < hi)]
        out.append(np.mean([np.mean(C1[b, i, i] / C0[b, i, i]) for i in range(6)]))
    return np.array(out)


def counts(q):
    return np.array([(q > t).sum() for t in (4.0, 4.5, 5.0)])


def credited(filt, det, cx_t, y, x):
    """t03's rule: a detection is credited to its nearest injected cluster if that cluster's filtered tSZ there is
    >= 2 sigma; per cluster the highest credited q (-inf if none)."""
    sig = filt.sigma_arcmin2
    nd = det["q"].size
    s_tsz = np.zeros(nd)
    for j in range(len(filt.thetas)):
        m = det["j"] == j
        if m.any():
            s_tsz[m] = (filt.yhat_map(cx_t, j) / F.AR ** 2)[det["iy"][m], det["ix"][m]]
    qb = np.full(y.size, -np.inf)
    if nd:
        dy = (det["y"][:, None] - y[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
        dx = (det["x"][:, None] - x[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
        near = np.hypot(dy, dx).argmin(1)
        for d in np.flatnonzero(s_tsz / sig[det["j"]] >= 2.0):
            qb[near[d]] = max(qb[near[d]], det["q"][d])
    return qb


def work(unit):
    kind, idx = unit
    fn = os.path.join(OUTD, f"a01_{kind}_{idx:03d}.npz")
    if os.path.exists(fn):
        return unit, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    res = dict(kind=kind, idx=idx)
    if kind == "real":
        d = S.truth(idx)
        Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], F.build_paint_tables(), G), G)
    elif kind == "inj":
        th = inj_theta(idx)
        y, x, q = inj_layout(th, idx)
        res.update(theta_inj=th, q_inj=q)
    for mode in S.MODES:
        if kind == "inj":
            Y = q * S.sigma_Y(mode, th)
            Xt = F.tsz_channels(F.paint_y(y, x, Y, np.full(y.size, th), F.build_paint_tables(), G), G)
            X = Xt + F.sim_cmb(S.SEED_INJ_CMB + idx, G) + F.sim_noise(S.SEED_INJ_NOISE + idx, mode, G)
        else:
            X = F.sim_cmb(S.SEED_CMB + idx, G) + F.sim_noise(S.SEED_NOISE + idx, mode, G)
            if kind == "real":
                X = X + Xt
        if kind == "noise":
            res[mode + "_sig_ana"] = F.Filter(G, mode=mode, analytic=True).sigma_arcmin2
        for name, meth, kw in METHODS:
            its = F.run_finder(X, G, mask_method=meth, mask_kw=kw)
            p = f"{mode}_{name}_"
            res[p + "sig0"] = its[0]["filt"].sigma_arcmin2
            res[p + "sig1"] = its[-1]["filt"].sigma_arcmin2
            res[p + "fsky"] = its[-1]["fsky"]
            res[p + "n0"] = counts(its[0]["det"]["q"])
            res[p + "n1"] = counts(its[-1]["det"]["q"])
            res[p + "spec"] = spec_ratio(its[-1]["Cbin"], its[0]["Cbin"], G)
            if kind == "inj":
                filt = its[-1]["filt"]
                res[p + "qbest"] = credited(filt, its[-1]["det"], filt.CX(Xt), y, x)
            if name == "szifi":
                d0 = its[0]["det"]
                m0 = d0["q"] > F.Q_TH_NOISE
                mask = F.disc_mask(d0["iy"][m0], d0["ix"][m0], F.MASK_RADIUS * d0["theta"][m0], F.N, F.PIX)
                w = F.szifi_blend_weights(mask)
                res[mode + "_w2_holes"] = float(np.mean(w[mask < 0.5] ** 2))
            del its
        del X
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return unit, res["t"], S.peak_rss_gb()


def ratio_line(lab, v, fmt="{:7.4f}"):
    return f"    {lab:44s}" + " ".join(fmt.format(a) for a in v)


if __name__ == "__main__":
    NPROC = int(sys.argv[1])
    os.makedirs(OUTD, exist_ok=True)
    t00 = time.time()
    units = [("real", p) for p in PIDS] + [("noise", p) for p in PIDS] + [("inj", k) for k in INJ_K]
    print(f"a01: {len(units)} units ({len(PIDS)} realistic, {len(PIDS)} noise-only, {len(INJ_K)} injection tiles) x 2 noise "
          f"levels x {len(METHODS)} finder variants, {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=6) as pool:
        for i, (u, tt, rss) in enumerate(pool.imap_unordered(work, units)):
            if i % 8 == 0:
                print(f"  {u}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"runs done in {time.time() - t00:.0f} s wall", flush=True)

    s2 = dict(np.load(f"{S.ROOT}/data_real/step2/inputs.npz"))
    th = F.THETAS15
    for mode in S.MODES:
        R = {p: dict(np.load(os.path.join(OUTD, f"a01_real_{p:03d}.npz"))) for p in PIDS}
        NO = {p: dict(np.load(os.path.join(OUTD, f"a01_noise_{p:03d}.npz"))) for p in PIDS}
        IJ = [dict(np.load(os.path.join(OUTD, f"a01_inj_{k:03d}.npz"))) for k in INJ_K]
        sY = S.sigma_Y(mode, th)
        ri = np.interp(np.log(th), np.log(s2[mode + "_th"]), s2[mode + "_r_i"])
        k3 = np.load(f"{S.ROOT}/data_real/step2/kernel_{mode}_effiii.npz")
        riii = np.interp(np.log(th), np.log(k3["th"]), k3["r"])
        print(f"\n==================== {mode}")
        print(f"(1) r(theta) = sigma_finder / sigma_Y on realistic tiles, mean over {len(PIDS)} tiles [and over the szifi check's 10 tiles]")
        print(ratio_line("theta [']", th, "{:7.2f}"))
        # regeneration check against the production outputs (t04)
        dmax = 0.0
        for p in PIDS:
            st = np.load(os.path.join(S.OUT, "tiles", f"tile_{p:03d}.npz"))[f"{mode}_it1_sig"]
            dmax = max(dmax, np.abs(R[p][f"{mode}_whitened_sig1"] / st - 1).max())
        print(f"    (regeneration: production finder rerun vs t04's stored sigma, max |ratio - 1| = {dmax:.1e})")
        rows = {}
        print(ratio_line("before the iteration (all variants)", np.mean([R[p][f'{mode}_whitened_sig0'] / sY for p in PIDS], 0)))
        for name, _, _ in METHODS:
            r1 = np.array([R[p][f"{mode}_{name}_sig1"] / sY for p in PIDS])
            rv = np.array([R[p][f"{mode}_{name}_sig1"] / sY for p in PIDS_V])
            rows[name] = r1.mean(0)
            print(ratio_line(f"after, {name}", r1.mean(0)))
            print(ratio_line("   tile std", r1.std(0)))
            print(ratio_line("   mean over the szifi check's 10 tiles", rv.mean(0)))
        print(ratio_line("step 2 (i)", ri))
        print(ratio_line("step 2 (iii)", riii))
        for name in ("whitened", "szifi", "szifi_l12000"):
            print(ratio_line(f"{name}: (r - 1)/(r_(i) - 1)", (rows[name] - 1) / (ri - 1), "{:7.2f}"))
            print(ratio_line(f"{name}: (r - 1)/(r_(iii) - 1)", (rows[name] - 1) / (riii - 1), "{:7.2f}"))
        print("(2) iteration-1 auto spectra / iteration-0 auto spectra (mean of the six channels and the bins), l ranges "
              + ", ".join(f"{a}-{b}" for a, b in LR))
        for name, _, _ in METHODS:
            print(ratio_line(name, np.mean([R[p][f"{mode}_{name}_spec"] for p in PIDS], 0)))
        print(f"    unmasked fraction f: {np.mean([R[p][f'{mode}_szifi_fsky'] for p in PIDS]):.3f}; SZiFi blend: mean w^2 over the "
              f"hole pixels {np.mean([R[p][f'{mode}_w2_holes'] for p in PIDS]):.3f}")
        print("(3) detections per tile q > 4 / 4.5 / 5, before -> after the iteration (realistic tiles)")
        n0 = np.mean([R[p][f"{mode}_whitened_n0"] for p in PIDS], 0)
        print(f"    before (all variants): {n0[0]:.1f} / {n0[1]:.1f} / {n0[2]:.1f}")
        for name, _, _ in METHODS:
            n1 = np.mean([R[p][f"{mode}_{name}_n1"] for p in PIDS], 0)
            print(f"    after, {name:14s}: {n1[0]:.1f} / {n1[1]:.1f} / {n1[2]:.1f};  q > 5 change {100 * (n1[2] / n0[2] - 1):+.1f}%")
        print(f"(4) noise-only tiles ({len(PIDS)}): sigma / analytic sigma, mean over tiles, at 0.5' / 1.04' / 4.45' / 15'; "
              "false detections per 100 deg^2 q > 4 / 4.5 / 5 (final)")
        jj = [0, 3, 9, 14]
        s0 = np.mean([NO[p][f"{mode}_whitened_sig0"] / NO[p][f"{mode}_sig_ana"] for p in PIDS], 0)
        print(f"    before the iteration        : " + " / ".join(f"{s0[j]:.4f}" for j in jj))
        area = 100.0 * len(PIDS)
        for name, _, _ in METHODS:
            s1 = np.mean([NO[p][f"{mode}_{name}_sig1"] / NO[p][f"{mode}_sig_ana"] for p in PIDS], 0)
            nf = np.sum([NO[p][f"{mode}_{name}_n1"] for p in PIDS], 0)
            print(f"    after, {name:14s}: " + " / ".join(f"{s1[j]:.4f}" for j in jj)
                  + f";  false {100 * nf[0] / area:.1f} / {100 * nf[1] / area:.2f} / {100 * nf[2] / area:.2f}"
                  + f"  (f = {np.mean([NO[p][f'{mode}_{name}_fsky'] for p in PIDS]):.3f})")
        print(f"(5) injection tiles ({len(INJ_K)}, two per size): completeness P(credited q_obs > 5) for q_inj 4.5-6, and the "
              "mean q_obs(variant) - q_obs(whitened) for clusters credited in both with q_inj 6-9.5")
        for thv in TH_INJ:
            tl = [r for r in IJ if float(r["theta_inj"]) == thv]
            q = np.concatenate([r["q_inj"] for r in tl])
            line = []
            for name, _, _ in METHODS:
                qb = np.concatenate([r[f"{mode}_{name}_qbest"] for r in tl])
                qw = np.concatenate([r[f"{mode}_whitened_qbest"] for r in tl])
                m = (q >= 4.5) & (q < 6)
                mm = (q >= 6) & (q < 9.5) & np.isfinite(qb) & np.isfinite(qw)
                line.append(f"{name} {np.mean(qb[m] > 5):.2f}, dq {np.mean(qb[mm] - qw[mm]):+.2f}")
            print(f"    theta_inj {thv:3.1f}': " + "; ".join(line) + f"  (N = {((q >= 4.5) & (q < 6)).sum()} at q_inj 4.5-6)")
        print(f"    finder sigma / sigma_Y at 0.5' on these tiles: " + ", ".join(
            f"{name} {np.mean([r[f'{mode}_{name}_sig1'][0] for r in IJ]) / sY[0]:.4f}" for name, _, _ in METHODS))
    print(f"\ndone [{time.time() - t00:.0f} s]")
