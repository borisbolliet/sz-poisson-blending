"""x03 (indep lens), check (c): own noise-only tiles and own injection tiles, own finder.

Noise-only (job 'noise', tile t = 0..NNOISE-1): own lensed CMB (seed 8100000 + t) + own white + atmospheric noise
(seed 8200000 + t; same unit draws for both levels).  Per level the own finder runs with the whitened masked covariance
(production rules) and with the literal binary f_sky estimate.  Stored: binned it0 covariance (spectra check), sigma
before / after, catalogues.

Injections (job 'inj', tile k): own CMB (seed 8300000 + k) + own noise (seed 8400000 + k) + isolated Arnaud clusters of one
size theta_inj, painted with the own NUFFT painter, on a square grid with random offset and +-3' jitter (layout seed
8500000 + k): spacing 40' (theta <= 2.4'), 60' (4.8'), 75' (9.6'); 30 / 30 / 30 / 30 / 60 / 90 tiles per size (the
build's design, fresh draws).  q_inj = Y500 / sigma_Y(theta_inj) (step 1's sigma_Y): uniform in [1.5, 9.5] for 85%,
else 10.5, 12, 15, 20 or 30.  A final detection is credited to its nearest injected cluster when the filtered tSZ at the
detection is >= 2 sigma.  Stored per cluster: best credited q, its scale and offset, number of credited q > 5 detections;
the value at the true pixel / nearest scale; per tile sigma_final / sigma_Y.
Outputs: out/noise/noise_TTT.npz, out/inj/inj_MODE_KKK.npz.  Usage: python x03_noise_inj.py NNOISE NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import vlib3 as V

ODN = os.path.join(V.OUT, "noise"); ODI = os.path.join(V.OUT, "inj")
TH_INJ = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
SPACING = {0.3: 40.0, 0.6: 40.0, 1.2: 40.0, 2.4: 40.0, 4.8: 60.0, 9.6: 75.0}
NTILES = {0.3: 30, 0.6: 30, 1.2: 30, 2.4: 30, 4.8: 60, 9.6: 90}
QHIGH = np.array([10.5, 12.0, 15.0, 20.0, 30.0])
INJ_JOBS = []
_k = 0
for _th in TH_INJ:
    for _n in range(NTILES[_th]):
        INJ_JOBS.append(("inj", float(_th), _k))
        _k += 1
_S = {}


def setup():
    if not _S:
        _S["G"] = V.grid()
        _S["P"] = V.Painter(_S["G"], eps=1e-9)
    return _S


def layout(th, k):
    rng = np.random.default_rng(8_500_000 + k)
    sp = SPACING[th]
    n = int(round(V.LBOX / sp))
    g = (np.arange(n) + 0.5) * sp
    yy, xx = np.meshgrid(g, g, indexing="ij")
    off = rng.uniform(0, sp, 2)
    y = (yy.ravel() + off[0] + rng.uniform(-3, 3, n * n)) % V.LBOX
    x = (xx.ravel() + off[1] + rng.uniform(-3, 3, n * n)) % V.LBOX
    q = rng.uniform(1.5, 9.5, n * n)
    hi = rng.random(n * n) < 0.15
    q[hi] = rng.choice(QHIGH, hi.sum())
    return y, x, q


def work_noise(t):
    fn = os.path.join(ODN, f"noise_{t:03d}.npz")
    if os.path.exists(fn):
        return ("noise", t), 0.0, V.peak_rss_gb()
    t0 = time.time()
    G = setup()["G"]
    res = dict(t=t)
    for mode in V.MODES:
        X = V.cmb_mine(np.random.default_rng(8_100_000 + t), G)
        X += V.noise_mine(np.random.default_rng(8_200_000 + t), mode, G)
        for meth in ("whitened", "binary"):
            if meth == "whitened":
                R = V.run_finder(X, G, mask_method="whitened")
                res[f"{mode}_C0"] = R["C0"]
                res[f"{mode}_sig0"] = R["F0"].sigma_am
                for k in ("j", "iy", "ix", "q"):
                    res[f"{mode}_it0_{k}"] = R["det0"][k]
                C0 = R["C0"]; mask = R["mask"]
            else:
                C1, f = V.masked_cov(X, mask, G, method="binary")
                F1 = V.MMF(G, C=C1)
                R = dict(F1=F1, det1=V.detect(F1, F1.combine(X)), fsky=f)
            res[f"{mode}_{meth}_sig1"] = R["F1"].sigma_am
            res[f"{mode}_{meth}_fsky"] = R["fsky"]
            for k in ("j", "iy", "ix", "q"):
                res[f"{mode}_{meth}_it1_{k}"] = R["det1"][k]
            del R
        del X
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return ("noise", t), res["t"], V.peak_rss_gb()


def work_inj(job):
    _, th, k = job
    t0 = time.time()
    S = setup()
    G, P = S["G"], S["P"]
    y, x, q = layout(th, k)
    out = {}
    for mode in V.MODES:
        fn = os.path.join(ODI, f"inj_{mode}_{k:03d}.npz")
        if os.path.exists(fn):
            continue
        Y = q * V.sigma_Y(mode, th)
        Xt = V.tsz_channels(P.T(y, x, Y, np.full(y.size, th)), G)
        X = V.cmb_mine(np.random.default_rng(8_300_000 + k), G)
        X += V.noise_mine(np.random.default_rng(8_400_000 + k), mode, G)
        X += Xt
        R = V.run_finder(X, G)
        F1, det = R["F1"], R["det1"]
        Dtot = F1.combine(X)
        del X
        Dt = F1.combine(Xt)
        del Xt
        nd = det["q"].size
        jn = int(np.abs(np.log(th) - np.log(V.THETAS)).argmin())
        fiy = np.rint(y / V.PIX).astype(np.int64) % V.N
        fix = np.rint(x / V.PIX).astype(np.int64) % V.N
        s_tsz = np.zeros(nd)
        for jj in range(15):
            m = det["j"] == jj
            if m.any():
                s_tsz[m] = F1.ymap(Dt, jj)[det["iy"][m], det["ix"][m]]
        dy = (det["iy"][:, None] * V.PIX - y[None, :] + V.LBOX / 2) % V.LBOX - V.LBOX / 2
        dx = (det["ix"][:, None] * V.PIX - x[None, :] + V.LBOX / 2) % V.LBOX - V.LBOX / 2
        dist = np.hypot(dy, dx)
        near = dist.argmin(1) if nd else np.zeros(0, np.int64)
        dnear = dist[np.arange(nd), near] if nd else np.zeros(0)
        sig = F1.sigma_am
        cred = s_tsz / sig[det["j"]] >= 2.0
        n = y.size
        qbest = np.full(n, -np.inf); jbest = np.full(n, -1); dbest = np.full(n, np.nan); n5 = np.zeros(n, np.int64)
        for d in np.flatnonzero(cred):
            h = near[d]
            if det["q"][d] > 5:
                n5[h] += 1
            if det["q"][d] > qbest[h]:
                qbest[h] = det["q"][d]; jbest[h] = det["j"][d]; dbest[h] = dnear[d]
        # forced value at the true pixel and nearest scale (total map = tSZ + CMB + noise)
        fq = F1.ymap(Dtot, jn)[fiy, fix] / sig[jn]
        np.savez(fn, mode=mode, theta_inj=th, k=k, q_inj=q, qbest=qbest, jbest=jbest, dbest=dbest, n5=n5, f_q=fq, sig=sig,
                 sig0=R["F0"].sigma_am, n_spur5=int(((~cred) & (det["q"] > 5)).sum()), d_q=det["q"], d_cred=cred)
        del R, F1, Dt, Dtot
    return job, time.time() - t0, V.peak_rss_gb()


def dispatch(job):
    return work_noise(job[1]) if job[0] == "noise" else work_inj(job)


if __name__ == "__main__":
    NNOISE, NPROC = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(ODN, exist_ok=True); os.makedirs(ODI, exist_ok=True)
    log = V.Tee(os.path.join(V.HERE, "x03_noise_inj_runs.log"))
    jobs = [("noise", t) for t in range(NNOISE)] + INJ_JOBS
    t0 = time.time()
    log(f"x03: {NNOISE} noise-only tiles (x 2 levels x 2 mask methods) + {len(INJ_JOBS)} injection tiles (x 2 levels), {NPROC} processes")
    with Pool(NPROC, maxtasksperchild=25) as pool:
        for i, (job, tt, rss) in enumerate(pool.imap_unordered(dispatch, jobs)):
            if i % 25 == 0:
                log(f"  {i:4d} {job}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t0:.0f} s]")
    log(f"x03 runs done in {time.time() - t0:.0f} s wall")
