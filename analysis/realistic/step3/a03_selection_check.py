"""Step 3 arbiter / a03: how well is the count change N1 calibrated, and what does its split mean?

N1 = (halos credited with a real q > 5 detection) / N_exp - 1, N_exp = sum over halos of P_cred(q', theta500), with P_cred
the finder's own credited selection of isolated injected clusters.  The build split N1 into 'neighbours' (credited vs the
isolated search, a window maximum, on the same maps) and 'normalisation' (isolated search vs P_cred).  The two parts mix
two criteria.  This script measures, on the same injected clusters:
  * P_cred: credited detection with q_obs > 5 (t03 / a02 rule);
  * P_alone: the isolated search of the realistic-tile analysis (s3lib.alone_window: maximum over 13 x 13 pixels around
    the cluster's pixel and all 15 scales of [own signal (response tables) + CMB + noise] / sigma; no credit test).
Injection tiles: the a02 tiles k = 1000-1059 (0.3'), 1120-1179 (0.6'), 1240-1299 (1.2'), 1360-1419 (2.4'),
1480-1539 (4.8'), 1720-1779 (9.6'), regenerated from their seeds (the credited q is checked against a02's file).
Then, on the build's 324 realistic tiles (t04 outputs, truth tables):
  (a) N1 with every calibration: the build's t03 fit, the a02 fit alone, the pooled fit, the pooled table, each quarter
      of a02 (tiles k mod 4), all with P_cred = 0 below q_inj = 2 r_inj;
  (b) a criterion-matched split: normalisation = N_alone / sum P_alone(q') - 1 over the halos the isolated search
      visits (q_true > 2), with P_alone from these injection tiles; neighbours = (1 + N1) / (1 + normalisation) - 1
      computed on the same halos.
Outputs: data_real/step3/arbiter/a03_inj_MODE_K.npz; this log.  Usage: python a03_selection_check.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
import a02_injections_hi as A2                     # noqa: E402  (layouts, seeds, fit form)
F = S.F

OUTD = os.path.join(S.OUT, "arbiter")
KSEL = [k for k0 in (1000, 1120, 1240, 1360, 1480, 1720) for k in range(k0, k0 + 60)]
TH_INJ = A2.TH_INJ
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
QE = np.r_[np.arange(1.0, 8.0 + 1e-9, 0.25)]
_B = {}


def basis():
    if "b" not in _B:
        _B["b"] = F.ResponseBasis(F.grid(), F.build_paint_tables())
    return _B["b"]


def work(job):
    mode, k = job
    fn = os.path.join(OUTD, f"a03_inj_{mode}_{k:04d}.npz")
    if os.path.exists(fn):
        return job, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    bas = basis()
    th = A2.KTH[k]
    y, x, q = A2.layout(th, k)
    Y = q * S.sigma_Y(mode, th)
    Xt = F.tsz_channels(F.paint_y(y, x, Y, np.full(y.size, th), F.build_paint_tables(), G), G)
    Xcn = F.sim_cmb(S.SEED_INJ_CMB + k, G) + F.sim_noise(S.SEED_INJ_NOISE + k, mode, G)
    its = F.run_finder(Xt + Xcn, G)
    filt, det = its[-1]["filt"], its[-1]["det"]
    sig = filt.sigma_arcmin2
    # credited best q (a02's rule)
    cx_t = filt.CX(Xt)
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
    # isolated search, exactly as s3lib.decompose (window +-6 pixels, all scales, own signal from the response tables)
    Rn, Rf = F.response_tables(filt, bas)
    targs = (Rn, Rf, bas.lnth0, bas.dln, F.R_SWITCH, bas.rn[1] - bas.rn[0], F.R_FAR_MIN, bas.rf[1] - bas.rf[0], F.LBOX)
    cy = np.rint(y / F.PIX).astype(np.int64) % F.N
    cxp = np.rint(x / F.PIX).astype(np.int64) % F.N
    ar = np.arange(-S.WIN, S.WIN + 1)
    cx_cn = filt.CX(Xcn)
    nwin = np.zeros((y.size, len(filt.thetas), ar.size, ar.size))
    for j in range(len(filt.thetas)):
        mcn = filt.yhat_map(cx_cn, j) / F.AR ** 2
        nwin[:, j] = mcn[(cy[:, None, None] + ar[None, :, None]) % F.N, (cxp[:, None, None] + ar[None, None, :]) % F.N]
    qa = np.zeros(y.size); qj = np.zeros(y.size, np.int64); qy = np.zeros(y.size, np.int64); qx = np.zeros(y.size, np.int64)
    S.alone_window(cy, cxp, np.arange(y.size, dtype=np.int64), y, x, Y, np.full(y.size, np.log(th)), *targs, F.PIX, S.WIN,
                   nwin, sig, qa, qj, qy, qx)
    ref = np.load(os.path.join(A2.OUTD, f"inj_{mode}_{k:04d}.npz"))["qbest"]
    np.savez(fn, k=k, theta_inj=th, q_inj=q, q_cred=qb, q_alone=qa, regen=np.nanmax(np.abs(np.where(np.isfinite(qb), qb - ref, 0.0))),
             sig=sig, t=time.time() - t0)
    return job, time.time() - t0, S.peak_rss_gb()


def tab(q, d, edges):
    i = np.digitize(q, edges) - 1
    ok = (i >= 0) & (i < edges.size - 1)
    n = np.bincount(i[ok], minlength=edges.size - 1).astype(float)
    k = np.bincount(i[ok], weights=d[ok].astype(float), minlength=edges.size - 1)
    return k / np.maximum(n, 1), n


def p_tab(P, edges, rinj, q, th):
    """Per-size binned table (linear between bin centres; 1 above the last centre), linear in ln theta (clamped)."""
    qc = 0.5 * (edges[1:] + edges[:-1])
    Pq = np.array([np.interp(q, qc, Pi, left=0.0, right=1.0) for Pi in P])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * Pq[i, n] + w * Pq[i + 1, n]


def p_fits(fits, q, th, qmin):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return np.where(q < qmin, 0.0, (1 - w) * P[i, n] + w * P[i + 1, n])


if __name__ == "__main__":
    NPROC = int(sys.argv[1])
    os.makedirs(OUTD, exist_ok=True)
    t00 = time.time()
    jobs = [(m, k) for k in KSEL for m in S.MODES]
    print(f"a03: {len(jobs)} injection tile runs (a02 tiles, 60 per size, both levels), {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=60) as pool:
        for i, (job, tt, rss) in enumerate(pool.imap_unordered(work, jobs)):
            if i % 60 == 0:
                print(f"  run {i:3d} {job}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"runs done in {time.time() - t00:.0f} s wall", flush=True)

    SA = dict(np.load(os.path.join(S.OUT, "selection_arbiter.npz")))
    SB = dict(np.load(os.path.join(S.OUT, "selection.npz")))
    rng = np.random.default_rng(3)
    for mode in S.MODES:
        R = [dict(np.load(os.path.join(OUTD, f"a03_inj_{mode}_{k:04d}.npz"))) for k in KSEL]
        print(f"\n==================== {mode}: {len(R)} injection tiles, {sum(r['q_inj'].size for r in R)} clusters")
        print(f"  regeneration: max |q_cred - a02's stored q| = {max(float(r['regen']) for r in R):.1e}")
        rinj = SA[mode + "_rinj"]
        Palone, Pcred = [], []
        print("  (1) on the same clusters: P_alone (isolated search > 5) / P_cred (credited q_obs > 5) per q_inj bin")
        for th in TH_INJ:
            rr = [r for r in R if float(r["theta_inj"]) == th]
            q = np.concatenate([r["q_inj"] for r in rr]); qa = np.concatenate([r["q_alone"] for r in rr])
            qc = np.concatenate([r["q_cred"] for r in rr])
            pa, n = tab(q, qa > 5, QE); pc, _ = tab(q, qc > 5, QE)
            Palone.append(pa); Pcred.append(pc)
            line = []
            for lo, hi in ((2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0)):
                m = (q >= lo) & (q < hi)
                line.append(f"{lo}-{hi}: {np.mean(qa[m] > 5):.3f}/{np.mean(qc[m] > 5):.3f}")
            both = np.mean((qa > 5) & (qc > 5)); ao = np.mean((qa > 5) & ~(qc > 5)); co = np.mean(~(qa > 5) & (qc > 5))
            print(f"    {th:3.1f}': " + "  ".join(line) + f"   | alone only {100 * ao:.2f}%, credited only {100 * co:.2f}% of all")
        Palone = np.array(Palone); Pcred = np.array(Pcred)
        # ---------------- realistic tiles
        Ncred, Nalone, Ne = [], [], {k_: [] for k_ in ("t03", "a02", "pooled", "pooled_tab", "q0", "q1", "q2", "q3",
                                                        "cred_tab3", "alone_tab3")}
        # per-quarter and a02-only fits
        fits_q = {}
        for lab, sel in (("a02", lambda t: t >= 1000), ("q0", lambda t: (t >= 1000) & (t % 4 == 0)), ("q1", lambda t: (t >= 1000) & (t % 4 == 1)),
                         ("q2", lambda t: (t >= 1000) & (t % 4 == 2)), ("q3", lambda t: (t >= 1000) & (t % 4 == 3))):
            ff = []
            for it_, th in enumerate(TH_INJ):
                tl = SA[f"{mode}_{th}_tile"]; m = sel(tl)
                ff.append(A2.fit_model(SA[f"{mode}_{th}_q"][m], SA[f"{mode}_{th}_det"][m].astype(float), x0=SA[mode + "_fit"][it_]))
            fits_q[lab] = np.array(ff)
        rinj_b = np.mean([np.load(os.path.join(S.OUT, "injections", f))["sig"] for f in sorted(os.listdir(os.path.join(S.OUT, "injections")))
                          if f.startswith(f"inj_{mode}_")], 0) / S.sigma_Y(mode, F.THETAS15)
        kc, nc = SA[mode + "_tab_k"], SA[mode + "_tab_n"]
        for pid in range(324):
            r = np.load(os.path.join(S.OUT, "tiles", f"tile_{pid:03d}.npz"))
            t = np.load(os.path.join(S.OUT, "truth", f"truth_{pid:03d}.npz"))
            Yh, th5, z = t["Y500"].astype(float), t["theta500"].astype(float), t["z"].astype(float)
            sig1 = r[f"{mode}_it1_sig"]; qv, j = r[f"{mode}_it1_q"], r[f"{mode}_it1_j"]
            ex, ti = r[f"{mode}_d_top_ex"], r[f"{mode}_d_top_idx"]
            d5 = np.flatnonzero(qv > 5)
            real = ex[d5, 0] / sig1[j[d5]] >= 2.0
            cred = np.unique(ti[d5[real], 0])
            sigf = np.interp(np.log(th5), np.log(F.THETAS15), sig1 / S.sigma_Y(mode, F.THETAS15)) * S.sigma_Y(mode, th5)   # as t05
            fidx, fqa = r[f"{mode}_f_idx"], r[f"{mode}_f_q_alone"]
            q2 = t[f"q_{mode}"].astype(float) > 2.0                         # the halos the isolated search visits
            Ncred.append(np.histogram(z[cred], ZB)[0]); Nalone.append(np.histogram(z[fidx[fqa > 5]], ZB)[0])
            for lab in Ne:
                if lab == "t03":
                    ri = np.interp(np.log(th5), np.log(F.THETAS15), rinj_b); fits = SB[mode + "_fit"]
                else:
                    ri = np.interp(np.log(th5), np.log(F.THETAS15), rinj)
                    fits = SA[mode + "_fit"] if lab in ("pooled",) else fits_q.get(lab)
                qp = Yh / sigf * ri
                qmin = 2.0 * ri
                if lab == "pooled_tab":
                    w = p_tab(kc / np.maximum(nc, 1), SA["qedges"], rinj, qp, th5); w = np.where(qp < qmin, 0.0, w)
                elif lab == "cred_tab3":                                    # this script's P_cred table, visited halos only
                    w = np.where(q2 & (qp >= qmin), p_tab(Pcred, QE, rinj, qp, th5), 0.0)
                elif lab == "alone_tab3":                                   # P_alone table, visited halos only
                    w = np.where(q2, p_tab(Palone, QE, rinj, qp, th5), 0.0)
                else:
                    w = p_fits(fits, qp, th5, qmin)
                Ne[lab].append(np.histogram(z, ZB, weights=w)[0])
        Ncred = np.array(Ncred, float); Nalone = np.array(Nalone, float)
        Ne = {k_: np.array(v) for k_, v in Ne.items()}

        def boot(num, den, nb=2000):
            b = rng.integers(0, num.shape[0], (nb, num.shape[0]))
            return num.sum() / den.sum() - 1, (num[b].sum(1).sum(1) / den[b].sum(1).sum(1) - 1).std(), num.sum(0) / den.sum(0) - 1
        print(f"  (2) N1 = credited / N_exp - 1 on the build's 324 tiles (credited {Ncred.sum():.0f}), P_cred = 0 below q_inj = 2 r_inj; "
              "tile-bootstrap error; z bins " + ", ".join(f"{a}-{b}" for a, b in zip(ZB[:-1], ZB[1:])))
        for lab, txt in (("t03", "build's t03 fit (38,760 clusters)"), ("a02", "a02 fit alone (155,040)"), ("pooled", "pooled fit (nominal)"),
                         ("pooled_tab", "pooled binned table"), ("q0", "a02 quarter 0 (38,760)"), ("q1", "a02 quarter 1"),
                         ("q2", "a02 quarter 2"), ("q3", "a02 quarter 3"), ("cred_tab3", "this script's P_cred table, halos q_true > 2")):
            T_, e_, zb_ = boot(Ncred, Ne[lab])
            print(f"     {txt:42s}: N_exp {Ne[lab].sum():9.1f}; N1 = {100 * T_:+.2f} +- {100 * e_:.2f}%; z bins " + ", ".join(f"{100 * v:+.1f}" for v in zb_))
        qv_ = [100 * (Ncred.sum() / Ne[lab].sum() - 1) for lab in ("q0", "q1", "q2", "q3")]
        print(f"     scatter of N1 between the four a02 quarters: std {np.std(qv_, ddof=1):.2f} pp (each quarter has t03's size)")
        T_, e_, zb_ = boot(Nalone, Ne["alone_tab3"])
        Tc, ec, zc_ = boot(Ncred, Ne["cred_tab3"])
        print(f"  (3) criterion-matched split on the same halos (q_true > 2), P from this script's injection tiles:")
        print(f"     normalisation: isolated search on the realistic tiles vs P_alone of isolated clusters: {100 * T_:+.2f} +- {100 * e_:.2f}%; z bins "
              + ", ".join(f"{100 * v:+.1f}" for v in zb_))
        print(f"     N1 with this script's P_cred table: {100 * Tc:+.2f} +- {100 * ec:.2f}%")
        print(f"     neighbours and search at fixed noise: (1 + N1) / (1 + normalisation) - 1 = {100 * ((1 + Tc) / (1 + T_) - 1):+.2f}%"
              f"   [criterion ratio sum P_alone / sum P_cred - 1 = {100 * (Ne['alone_tab3'].sum() / Ne['cred_tab3'].sum() - 1):+.2f}%;"
              f" credited / isolated search - 1 = {100 * (Ncred.sum() / Nalone.sum() - 1):+.2f}%]")
    print(f"\ndone [{time.time() - t00:.0f} s]")
