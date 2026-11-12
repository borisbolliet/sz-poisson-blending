"""Step 3 / t03: check (c), injections of isolated clusters into noise-only tiles: the finder's own selection.

Per tile: CMB + noise (fresh seeds 500000 + k, 600000 + k) plus isolated Arnaud clusters of one size theta_inj on a
square grid with random offset and +-3' jitter: spacing 40' for theta_inj <= 2.4' (225 clusters), 60' for 4.8' (100),
75' for 9.6' (64).  q_inj = Y500 / sigma_Y(theta_inj) (step 1's sigma_Y): uniform in [1.5, 9.5] for 85% of the
clusters, else one of 10.5, 12, 15, 20, 30.  Painted exactly as the halos of the realistic tiles.
The finder runs as in production.  A final detection (q > 4) is credited to its nearest injected cluster when that
cluster's filtered tSZ signal at the detection is >= 2 sigma (the same 'real cluster' criterion as for the realistic
tiles; here the tSZ value is the exact filtered tSZ map, the only halos being the injected ones).  Per cluster: q_obs =
highest credited q, its scale and position; also the forced value at the nearest pixel and nearest scale.
Outputs: data_real/step3/injections/inj_MODE_K.npz; summary tables in this log and data_real/step3/selection.npz.
Usage: python t03_injections.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F
NPROC = int(sys.argv[1])
OUTD = os.path.join(S.OUT, "injections")
TH_INJ = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
SPACING = {0.3: 40.0, 0.6: 40.0, 1.2: 40.0, 2.4: 40.0, 4.8: 60.0, 9.6: 75.0}
NTILES = {0.3: 30, 0.6: 30, 1.2: 30, 2.4: 30, 4.8: 60, 9.6: 90}
QHIGH = np.array([10.5, 12.0, 15.0, 20.0, 30.0])
JOBS = []
k = 0
for th in TH_INJ:
    for n in range(NTILES[th]):
        for mode in S.MODES:
            JOBS.append((mode, float(th), k))
        k += 1


def layout(th, k):
    rng = np.random.default_rng(700_000 + k)
    sp = SPACING[th]
    n = int(round(F.LBOX / sp))
    g = (np.arange(n) + 0.5) * sp
    yy, xx = np.meshgrid(g, g, indexing="ij")
    off = rng.uniform(0, sp, 2)
    y = (yy.ravel() + off[0] + rng.uniform(-3, 3, n * n)) % F.LBOX
    x = (xx.ravel() + off[1] + rng.uniform(-3, 3, n * n)) % F.LBOX
    q = rng.uniform(1.5, 9.5, n * n)
    hi = rng.random(n * n) < 0.15
    q[hi] = rng.choice(QHIGH, hi.sum())
    return y, x, q


def work(job):
    mode, th, k = job
    fn = os.path.join(OUTD, f"inj_{mode}_{k:03d}.npz")
    if os.path.exists(fn):
        return job, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    tables = F.build_paint_tables()
    y, x, q = layout(th, k)
    Y = q * S.sigma_Y(mode, th)
    Xt = F.tsz_channels(F.paint_y(y, x, Y, np.full(y.size, th), tables, G), G)
    Xcn = F.sim_cmb(S.SEED_INJ_CMB + k, G) + F.sim_noise(S.SEED_INJ_NOISE + k, mode, G)
    its = F.run_finder(Xt + Xcn, G)
    it = its[-1]
    filt, det = it["filt"], it["det"]
    cx_t, cx_n = filt.CX(Xt), filt.CX(Xcn)
    del Xt, Xcn
    nd = det["q"].size
    jn = np.abs(np.log(th) - np.log(filt.thetas)).argmin()
    fiy = np.rint(y / F.PIX).astype(np.int64) % F.N
    fix = np.rint(x / F.PIX).astype(np.int64) % F.N
    s_tsz = np.zeros(nd)
    f_tsz = np.zeros(y.size); f_n = np.zeros(y.size)
    for j in range(len(filt.thetas)):
        m = det["j"] == j
        mt = filt.yhat_map(cx_t, j) / F.AR ** 2
        s_tsz[m] = mt[det["iy"][m], det["ix"][m]]
        if j == jn:
            f_tsz = mt[fiy, fix]
            f_n = filt.yhat_map(cx_n, j)[fiy, fix] / F.AR ** 2
    dy = (det["y"][:, None] - y[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
    dx = (det["x"][:, None] - x[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
    dist = np.hypot(dy, dx)
    near = dist.argmin(1) if nd else np.zeros(0, np.int64)
    dnear = dist[np.arange(nd), near] if nd else np.zeros(0)
    sig = filt.sigma_arcmin2
    res = dict(mode=mode, theta_inj=th, k=k, y=y, x=x, q_inj=q, Y=Y, sig=sig, sig0=its[0]["filt"].sigma_arcmin2,
               d_j=det["j"], d_iy=det["iy"], d_ix=det["ix"], d_q=det["q"], d_S_tsz=s_tsz, d_near=near, d_dist=dnear,
               f_q=(f_tsz + f_n) / sig[jn], f_own=f_tsz / sig[jn], f_j=jn, n_it=len(its),
               fsky=it["fsky"], t=time.time() - t0)
    np.savez(fn, **res)
    return job, res["t"], S.peak_rss_gb()


def per_cluster(r):
    """Credited detections -> per injected cluster: best q_obs, its scale, offset; number of credited q > 5."""
    sig = r["sig"]
    cred = r["d_S_tsz"] / sig[r["d_j"]] >= 2.0
    n = r["q_inj"].size
    qbest = np.full(n, -np.inf); jbest = np.full(n, -1); dbest = np.full(n, np.nan); n5 = np.zeros(n, int)
    for d in np.flatnonzero(cred):
        h = r["d_near"][d]
        if r["d_q"][d] > 5:
            n5[h] += 1
        if r["d_q"][d] > qbest[h]:
            qbest[h] = r["d_q"][d]; jbest[h] = r["d_j"][d]; dbest[h] = r["d_dist"][d]
    spur = (~cred) & (r["d_q"] > 5)
    return qbest, jbest, dbest, n5, spur.sum()


def fit_model(q, det):
    """P(q_obs > 5 | q_inj) = Phi((sqrt(q^2 + b) - 5 - mu) / s): maximum likelihood (Bernoulli)."""
    from scipy.optimize import minimize

    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    r = minimize(nll, [3.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return r.x


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    print(f"t03: {len(JOBS)} tile runs ({len(JOBS) // 2} tiles x 2 noise levels), {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=20) as pool:
        for i, (job, tt, rss) in enumerate(pool.imap_unordered(work, JOBS)):
            if i % 40 == 0:
                print(f"  run {i:3d} {job}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t0:.0f} s]", flush=True)
    print(f"runs done in {time.time() - t0:.0f} s wall", flush=True)

    qedges = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])
    sel_out = {}
    for mode in S.MODES:
        rows = {th: dict(q=[], qobs=[], j=[], dist=[], n5=[], fq=[], tile=[]) for th in TH_INJ}
        nspur, ntile, siginj = 0, 0, []
        for (m, th, k) in JOBS:
            if m != mode:
                continue
            r = dict(np.load(os.path.join(OUTD, f"inj_{mode}_{k:03d}.npz")))
            qb, jb, db, n5, sp = per_cluster(r)
            R = rows[th]
            R["q"].append(r["q_inj"]); R["qobs"].append(qb); R["j"].append(jb); R["dist"].append(db); R["n5"].append(n5)
            R["fq"].append(r["f_q"]); R["tile"].append(np.full(r["q_inj"].size, k))
            nspur += sp; ntile += 1
            siginj.append(r["sig"] / S.sigma_Y(mode, F.THETAS15))
        for th in TH_INJ:
            rows[th] = {kk: np.concatenate(v) for kk, v in rows[th].items()}
        print(f"\n=========== {mode}: {ntile} injection tiles; credited = filtered tSZ of the nearest injected cluster >= 2 sigma")
        sr = np.array(siginj)
        print(f"  finder sigma (final) / sigma_Y on injection tiles: mean {sr.mean():.4f}, range of tile means "
              f"{sr.mean(1).min():.4f}-{sr.mean(1).max():.4f}")
        print(f"  uncredited detections q > 5 (noise peaks): {nspur} in {ntile * 100} deg^2 = {100 * nspur / (ntile * 100):.3f} per 100 deg^2")
        print("  completeness P(credited q_obs > 5) in q_inj bins [models: N(q,1): Phi(q-5); N(sqrt(q^2+3),1)]:")
        hdr = "  q_inj bin    " + " ".join(f"th={th:>4}'     " for th in TH_INJ) + "  Phi(q-5)  Phi(sqrt(q^2+3)-5)"
        print(hdr)
        comp = np.zeros((len(TH_INJ), qedges.size - 1)); comp_e = np.zeros_like(comp)
        for b in range(qedges.size - 1):
            qa, qb_ = qedges[b], qedges[b + 1]
            qm = 0.5 * (qa + qb_)
            line = []
            for it_, th in enumerate(TH_INJ):
                R = rows[th]
                m = (R["q"] >= qa) & (R["q"] < qb_)
                p = np.mean(R["qobs"][m] > 5) if m.any() else np.nan
                e = np.sqrt(p * (1 - p) / max(m.sum(), 1))
                comp[it_, b] = p; comp_e[it_, b] = e
                line.append(f"{p:.3f}+-{e:.3f}")
            print(f"  {qa:4.1f}-{qb_:4.1f}   " + "  ".join(line) +
                  f"   {np.mean(ndtr(np.linspace(qa, qb_, 50) - 5)):.3f}     {np.mean(ndtr(np.sqrt(np.linspace(qa, qb_, 50) ** 2 + 3) - 5)):.3f}")
        print("  q_inj >= 10 (discrete values): completeness, mean q_obs - q_inj [std]; mean q_obs - sqrt(q^2+3) ")
        for qv in QHIGH:
            line = []
            for th in TH_INJ:
                R = rows[th]
                m = np.isclose(R["q"], qv)
                qo = R["qobs"][m]
                line.append(f"{np.mean(qo > 5):.3f}, {np.mean(qo - qv):+.2f} [{np.std(qo):.2f}], {np.mean(qo - np.sqrt(qv ** 2 + 3)):+.2f}")
            print(f"  q = {qv:4.1f}: " + " | ".join(line))
        print("  fit P(q_obs > 5) = Phi((sqrt(q^2 + b) - 5 - mu) / s) per theta_inj (all injections, maximum likelihood):")
        fits = []
        for th in TH_INJ:
            R = rows[th]
            p = fit_model(R["q"], (R["qobs"] > 5).astype(float))
            fits.append(p)
            print(f"    theta_inj {th}': b = {p[0]:+.2f}, mu = {p[1]:+.3f}, s = {np.exp(p[2]):.3f}   (N = {R['q'].size})")
        print("  among credited detections (q_obs > 5), q_inj in [6, 9.5]: q_obs - q_inj mean [std]; vs N(sqrt(q^2+3),1): mean of q_obs - sqrt(q^2+3)")
        for th in TH_INJ:
            R = rows[th]
            m = (R["q"] >= 6) & (R["q"] < 9.5) & (R["qobs"] > 5)
            dq = R["qobs"][m] - R["q"][m]
            dq3 = R["qobs"][m] - np.sqrt(R["q"][m] ** 2 + 3)
            fm = (R["q"] >= 6) & (R["q"] < 9.5)
            fq = R["fq"][fm] - R["q"][fm]
            print(f"    theta_inj {th}': {dq.mean():+.3f} [{dq.std():.3f}]; {dq3.mean():+.3f} [{dq3.std():.3f}];  forced (true pixel, nearest scale): "
                  f"{fq.mean():+.3f} [{fq.std():.3f}]  (N = {m.sum()})")
        print("  theta and position recovery for credited detections q_obs > 5: theta_est / theta_inj median [16-84%]; offset median [84%] (arcmin)")
        for th in TH_INJ:
            R = rows[th]
            for lo, hi in ((5, 7), (7, 10), (10, 31)):
                m = (R["qobs"] > 5) & (R["q"] >= lo) & (R["q"] < hi)
                if not m.any():
                    continue
                rt = F.THETAS15[R["j"][m]] / th
                print(f"    theta_inj {th:4}', q_inj {lo}-{hi}: theta ratio {np.median(rt):.2f} [{np.percentile(rt, 16):.2f}-{np.percentile(rt, 84):.2f}], "
                      f"factor > 2 off: {np.mean((rt > 2) | (rt < 0.5)):.3f}; offset {np.median(R['dist'][m]):.2f}' [{np.percentile(R['dist'][m], 84):.2f}']; "
                      f"fragments (extra credited q > 5) per detected cluster {np.mean(R['n5'][m] - 1):.3f}  (N = {m.sum()})")
        sel_out[mode + "_comp"] = comp; sel_out[mode + "_comp_err"] = comp_e
        sel_out[mode + "_fit"] = np.array(fits)
        for th in TH_INJ:
            for kk in ("q", "qobs", "j", "tile"):
                sel_out[f"{mode}_{th}_{kk}"] = rows[th][kk]
    sel_out["qedges"] = qedges; sel_out["theta_inj"] = TH_INJ
    np.savez(os.path.join(S.OUT, "selection.npz"), **sel_out)
    print(f"\ndone [{time.time() - t0:.0f} s]")
