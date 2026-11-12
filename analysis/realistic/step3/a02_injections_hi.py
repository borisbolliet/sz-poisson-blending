"""Step 3 arbiter / a02: a second, 4x larger injection set for the finder's own selection of isolated clusters.

Why: the count-change preview (claims N1-N3, N5, N7) divides by N_exp = sum over all halos of P_det(q', theta500).
The indep check found (i) a spurious floor of the fitted P_det far below the injected range and (ii) a 3-sigma spread
between injection sets.  This set tests the build's calibration (t03) with 4x the clusters, concentrated on the q range
that sets N_exp, and gives a pooled calibration.

Design: as t03 (same sizes, layout rules, painting, finder, crediting rule), with new seeds (tile k = 1000..2079:
layout 700000 + k, CMB 500000 + k, noise 600000 + k) and q_inj = Y500 / sigma_Y(theta_inj) uniform in [1.8, 8] for all
clusters.  Tiles per size: 120 (0.3', 0.6', 1.2', 2.4'), 240 (4.8'), 360 (9.6'): 1080 tiles per noise level.
Credited: a final detection is credited to its nearest injected cluster if that cluster's filtered tSZ there is
>= 2 sigma.  A credited detection therefore needs q_inj >= 2 r_inj (r_inj = sigma_finder / sigma_Y on the injection
tiles): P_det = 0 below, exactly.
Outputs: data_real/step3/injections_hi/inj_MODE_K.npz; data_real/step3/selection_arbiter.npz (pooled t03 + a02 fits,
binned table, r_inj); this log.  Usage: python a02_injections_hi.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F

OUTD = os.path.join(S.OUT, "injections_hi")
OLDD = os.path.join(S.OUT, "injections")
TH_INJ = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
SPACING = {0.3: 40.0, 0.6: 40.0, 1.2: 40.0, 2.4: 40.0, 4.8: 60.0, 9.6: 75.0}
NTILES = {0.3: 120, 0.6: 120, 1.2: 120, 2.4: 120, 4.8: 240, 9.6: 360}
K0 = 1000
JOBS, KTH = [], {}
_k = K0
for _th in TH_INJ:
    for _n in range(NTILES[_th]):
        KTH[_k] = float(_th)
        for _m in S.MODES:
            JOBS.append((_m, float(_th), _k))
        _k += 1
QE = np.r_[1.5, np.arange(2.0, 8.0 + 1e-9, 0.25), 8.5, 9.0, 9.5]      # table edges (q_inj)


def layout(th, k):
    rng = np.random.default_rng(700_000 + k)
    sp = SPACING[th]
    n = int(round(F.LBOX / sp))
    g = (np.arange(n) + 0.5) * sp
    yy, xx = np.meshgrid(g, g, indexing="ij")
    off = rng.uniform(0, sp, 2)
    y = (yy.ravel() + off[0] + rng.uniform(-3, 3, n * n)) % F.LBOX
    x = (xx.ravel() + off[1] + rng.uniform(-3, 3, n * n)) % F.LBOX
    q = rng.uniform(1.8, 8.0, n * n)
    return y, x, q


def work(job):
    mode, th, k = job
    fn = os.path.join(OUTD, f"inj_{mode}_{k:04d}.npz")
    if os.path.exists(fn):
        return job, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    y, x, q = layout(th, k)
    Y = q * S.sigma_Y(mode, th)
    Xt = F.tsz_channels(F.paint_y(y, x, Y, np.full(y.size, th), F.build_paint_tables(), G), G)
    X = Xt + F.sim_cmb(S.SEED_INJ_CMB + k, G) + F.sim_noise(S.SEED_INJ_NOISE + k, mode, G)
    its = F.run_finder(X, G)
    del X
    filt, det = its[-1]["filt"], its[-1]["det"]
    cx_t = filt.CX(Xt)
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
    np.savez(fn, mode=mode, theta_inj=th, k=k, q_inj=q, qbest=qb, sig=sig, t=time.time() - t0)
    return job, time.time() - t0, S.peak_rss_gb()


def fit_model(q, det, x0=(0.0, 0.0, 0.0)):
    """P(q_obs > 5 | q_inj) = Phi((sqrt(q^2 + b) - 5 - mu) / s), maximum likelihood (t03's form)."""
    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, list(x0), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000)).x


def p_fit(p, q):
    b, mu, ls = p
    return ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls))


def load(mode):
    """Per size: (q_inj, detected q_obs > 5, tile id) for the old (t03) and the new (a02) set; r_inj of each set."""
    old = {th: ([], [], []) for th in TH_INJ}
    new = {th: ([], [], []) for th in TH_INJ}
    rold, rnew = [], []
    for f in sorted(os.listdir(OLDD)):
        if f.startswith(f"inj_{mode}_"):
            r = np.load(os.path.join(OLDD, f))
            th = float(r["theta_inj"])
            qb = r["d_q"]; sig = r["sig"]
            cred = r["d_S_tsz"] / sig[r["d_j"]] >= 2.0
            best = np.full(r["q_inj"].size, -np.inf)
            for d in np.flatnonzero(cred):
                best[r["d_near"][d]] = max(best[r["d_near"][d]], qb[d])
            old[th][0].append(r["q_inj"]); old[th][1].append(best > 5); old[th][2].append(np.full(r["q_inj"].size, int(r["k"])))
            rold.append(sig / S.sigma_Y(mode, F.THETAS15))
    for f in sorted(os.listdir(OUTD)):
        if f.startswith(f"inj_{mode}_"):
            r = np.load(os.path.join(OUTD, f))
            th = float(r["theta_inj"])
            new[th][0].append(r["q_inj"]); new[th][1].append(r["qbest"] > 5); new[th][2].append(np.full(r["q_inj"].size, int(r["k"])))
            rnew.append(r["sig"] / S.sigma_Y(mode, F.THETAS15))
    cat = lambda D: {th: tuple(np.concatenate(v) for v in D[th]) for th in TH_INJ}
    return cat(old), cat(new), np.array(rold), np.array(rnew)


def comp_table(q, d, edges):
    i = np.digitize(q, edges) - 1
    ok = (i >= 0) & (i < edges.size - 1)
    n = np.bincount(i[ok], minlength=edges.size - 1).astype(float)
    k = np.bincount(i[ok], weights=d[ok].astype(float), minlength=edges.size - 1)
    return k, n


if __name__ == "__main__":
    NPROC = int(sys.argv[1])
    os.makedirs(OUTD, exist_ok=True)
    t00 = time.time()
    print(f"a02: {len(JOBS)} tile runs ({len(JOBS) // 2} tiles x 2 noise levels), {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=40) as pool:
        for i, (job, tt, rss) in enumerate(pool.imap_unordered(work, JOBS)):
            if i % 200 == 0:
                print(f"  run {i:4d} {job}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"runs done in {time.time() - t00:.0f} s wall", flush=True)

    rng = np.random.default_rng(5)
    out = dict(theta_inj=TH_INJ, qedges=QE)
    for mode in S.MODES:
        old, new, rold, rnew = load(mode)
        SB = dict(np.load(os.path.join(S.OUT, "selection.npz")))
        fits_b = SB[mode + "_fit"]
        print(f"\n==================== {mode}: old set (t03) {rold.shape[0]} tiles, "
              f"{sum(old[t][0].size for t in TH_INJ)} clusters; new set (a02) {rnew.shape[0]} tiles, {sum(new[t][0].size for t in TH_INJ)} clusters")
        print(f"  r_inj = sigma_finder / sigma_Y at 0.5' / 4.45' / 15': old {rold.mean(0)[[0, 9, 14]].round(4)}, new {rnew.mean(0)[[0, 9, 14]].round(4)}")
        print("(1) P(credited q_obs > 5 | q_inj 4-6) per size (as the indep check's x13), tile-bootstrap errors")
        dsum = []
        for th in TH_INJ:
            vals = []
            for D in (old, new):
                q, d, tl = D[th]
                m = (q >= 4) & (q < 6)
                ut = np.unique(tl[m])
                pb = []
                for _ in range(400):
                    pick = rng.choice(ut, ut.size)
                    mm = np.concatenate([np.flatnonzero(m & (tl == u)) for u in pick])
                    pb.append(d[mm].mean())
                vals.append((d[m].mean(), np.std(pb)))
            diff = (vals[0][0] - vals[1][0]) / np.hypot(vals[0][1], vals[1][1])
            dsum.append(diff)
            print(f"    theta {th:3.1f}': old {vals[0][0]:.4f} +- {vals[0][1]:.4f}   new {vals[1][0]:.4f} +- {vals[1][1]:.4f}   old - new {diff:+.1f} sigma")
        print(f"    chi2 (old - new) = {np.sum(np.square(dsum)):.1f} for 6")
        print("(2) the build's fit (t03) against the new set: detections observed / expected per q_inj range (all sizes)")
        for lo, hi in ((1.8, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0)):
            ob = ex = 0.0
            for th, fb in zip(TH_INJ, fits_b):
                q, d, _ = new[th]
                m = (q >= lo) & (q < hi)
                ob += d[m].sum(); ex += p_fit(fb, q[m]).sum()
            print(f"    q_inj {lo:3.1f}-{hi:3.1f}: observed {ob:6.0f}, build's fit {ex:8.1f}  ({(ob - ex) / np.sqrt(max(ex, 1)):+.1f} Poisson sigma)")
        print("(3) detections below the credit limit q_inj < 2 r_inj (old + new): "
              + ", ".join(f"{th}': {int(sum(D[th][1][D[th][0] < 2 * rnew.mean(0)[0]].sum() for D in (old, new)))} of "
                          f"{int(sum((D[th][0] < 2 * rnew.mean(0)[0]).sum() for D in (old, new)))}" for th in TH_INJ))
        # pooled fits and table
        fits, tab_k, tab_n = [], [], []
        print("(4) pooled fits (old + new), P = Phi((sqrt(q^2 + b) - 5 - mu) / s): b, mu, s  [build's t03 fit]")
        for th, fb in zip(TH_INJ, fits_b):
            q = np.concatenate([old[th][0], new[th][0]]); d = np.concatenate([old[th][1], new[th][1]])
            p = fit_model(q, d.astype(float), x0=fb)
            fits.append(p)
            k, n = comp_table(q, d, QE)
            tab_k.append(k); tab_n.append(n)
            print(f"    theta {th:3.1f}': b = {p[0]:+.2f}, mu = {p[1]:+.3f}, s = {np.exp(p[2]):.3f}   [{fb[0]:+.2f}, {fb[1]:+.3f}, {np.exp(fb[2]):.3f}]  (N = {q.size})")
        fits = np.array(fits); tab_k = np.array(tab_k); tab_n = np.array(tab_n)
        print("(5) pooled completeness table vs pooled fit, selected bins (all sizes): q_inj bin: observed / fit")
        qc = 0.5 * (QE[1:] + QE[:-1])
        for b_ in range(QE.size - 1):
            if QE[b_] in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
                ex = 0.0
                for i_, th in enumerate(TH_INJ):
                    q = np.concatenate([old[th][0], new[th][0]])
                    m = (q >= QE[b_]) & (q < QE[b_ + 1])
                    ex += p_fit(fits[i_], q[m]).sum()
                print(f"    {QE[b_]:4.2f}-{QE[b_ + 1]:4.2f}: {tab_k[:, b_].sum():6.0f} / {ex:8.1f}   per size P: "
                      + " ".join(f"{tab_k[i_, b_] / max(tab_n[i_, b_], 1):.3f}" for i_ in range(len(TH_INJ))))
        rinj = (rold.sum(0) + rnew.sum(0)) / (rold.shape[0] + rnew.shape[0])
        out[mode + "_fit"] = fits
        out[mode + "_tab_k"] = tab_k
        out[mode + "_tab_n"] = tab_n
        out[mode + "_rinj"] = rinj
        for th in TH_INJ:
            out[f"{mode}_{th}_q"] = np.concatenate([old[th][0], new[th][0]])
            out[f"{mode}_{th}_det"] = np.concatenate([old[th][1], new[th][1]])
            out[f"{mode}_{th}_tile"] = np.concatenate([old[th][2], new[th][2]])
    np.savez(os.path.join(S.OUT, "selection_arbiter.npz"), **out)
    print(f"\nsaved {os.path.join(S.OUT, 'selection_arbiter.npz')}\ndone [{time.time() - t00:.0f} s]")
