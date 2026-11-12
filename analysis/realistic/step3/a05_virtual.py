"""Step 3 arbiter / a05: does the 'unit-noise' reference of the selection depend on the tiles it is measured on?

The indep check calibrated the isolated search with virtual clusters on noise-only tiles (its x10, own code) and found
a normalisation effect of -2.4% / -3.2%; the arbiter's a03 calibrated the same search on the a02 injection tiles and
finds -1.2% / -1.7%.  This script repeats the virtual calibration with the build's code: noise-only tiles pid = 0..59
(t02's seeds, the realistic tiles without halos), production finder, then 12,000 virtual clusters per tile and level at
random positions (2,000 per injected size of t03 / a02, q_inj uniform in [1.8, 8]), each evaluated alone on the tile's
CMB + noise with s3lib.alone_window (the isolated search of the realistic tiles).  It compares P_alone(q_inj, theta)
with a03's (same search, painted isolated clusters on injection tiles) and with the indep check's x10 table, and the
implied N_exp over the visited halos (q_true > 2) of the build's 324 tiles.
Outputs: data_real/step3/arbiter/a05_noise_PID.npz; this log.  Usage: python a05_virtual.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
import a03_selection_check as A3                   # noqa: E402
F = S.F

OUTD = os.path.join(S.OUT, "arbiter")
PIDS = list(range(60))
NV_SIZE = 2000
TH_INJ = A3.TH_INJ
QE = A3.QE
ZB = A3.ZB
VIRT = f"{S.ROOT}/verify_real/step3_indep/out/virt"


def work(pid):
    fn = os.path.join(OUTD, f"a05_noise_{pid:03d}.npz")
    if os.path.exists(fn):
        return pid, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    bas = A3.basis()
    rng = np.random.default_rng(12_000_000 + pid)
    th = np.repeat(TH_INJ, NV_SIZE)
    q = rng.uniform(1.8, 8.0, th.size)
    y = rng.uniform(0, F.LBOX, th.size); x = rng.uniform(0, F.LBOX, th.size)
    cy = np.rint(y / F.PIX).astype(np.int64) % F.N
    cx = np.rint(x / F.PIX).astype(np.int64) % F.N
    ar = np.arange(-S.WIN, S.WIN + 1)
    res = dict(pid=pid, theta=th, q_inj=q)
    for mode in S.MODES:
        X = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        its = F.run_finder(X, G)
        filt = its[-1]["filt"]
        sig = filt.sigma_arcmin2
        Rn, Rf = F.response_tables(filt, bas)
        targs = (Rn, Rf, bas.lnth0, bas.dln, F.R_SWITCH, bas.rn[1] - bas.rn[0], F.R_FAR_MIN, bas.rf[1] - bas.rf[0], F.LBOX)
        cxn = filt.CX(X)
        maps = [filt.yhat_map(cxn, j) / F.AR ** 2 for j in range(len(filt.thetas))]
        Y = q * S.sigma_Y(mode, th)
        qa = np.zeros(th.size)
        for a in range(0, th.size, 2000):
            sl = slice(a, min(a + 2000, th.size))
            n = sl.stop - sl.start
            nwin = np.stack([m[(cy[sl, None, None] + ar[None, :, None]) % F.N, (cx[sl, None, None] + ar[None, None, :]) % F.N] for m in maps], 1)
            ov = np.zeros(n); oj = np.zeros(n, np.int64); oy = np.zeros(n, np.int64); ox = np.zeros(n, np.int64)
            S.alone_window(cy[sl], cx[sl], np.arange(n, dtype=np.int64), y[sl], x[sl], Y[sl], np.log(th[sl]), *targs, F.PIX, S.WIN,
                           nwin, sig, ov, oj, oy, ox)
            qa[sl] = ov
        res[mode + "_q_alone"] = qa
        res[mode + "_r_noise"] = sig / S.sigma_Y(mode, F.THETAS15)
        del maps, X, its
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return pid, res["t"], S.peak_rss_gb()


if __name__ == "__main__":
    NPROC = int(sys.argv[1])
    os.makedirs(OUTD, exist_ok=True)
    t00 = time.time()
    print(f"a05: {len(PIDS)} noise-only tiles x 2 levels, {6 * NV_SIZE} virtual clusters each, {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=30) as pool:
        for i, (pid, tt, rss) in enumerate(pool.imap_unordered(work, PIDS)):
            if i % 10 == 0:
                print(f"  tile {pid:3d}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"runs done in {time.time() - t00:.0f} s wall", flush=True)
    R = [dict(np.load(os.path.join(OUTD, f"a05_noise_{p:03d}.npz"))) for p in PIDS]
    VF = [dict(np.load(os.path.join(VIRT, f))) for f in sorted(os.listdir(VIRT)) if f.endswith(".npz")] if os.path.isdir(VIRT) else []
    SA = dict(np.load(os.path.join(S.OUT, "selection_arbiter.npz")))
    for mode in S.MODES:
        th = np.concatenate([r["theta"] for r in R]); q = np.concatenate([r["q_inj"] for r in R])
        qa = np.concatenate([r[mode + "_q_alone"] for r in R])
        rnoise = np.mean([r[mode + "_r_noise"] for r in R], 0)
        I = [dict(np.load(os.path.join(OUTD, f"a03_inj_{mode}_{k:04d}.npz"))) for k in A3.KSEL]
        iq = np.concatenate([r["q_inj"] for r in I]); ith = np.concatenate([np.full(r["q_inj"].size, float(r["theta_inj"])) for r in I])
        iqa = np.concatenate([r["q_alone"] for r in I])
        rinj = SA[mode + "_rinj"]
        print(f"\n==================== {mode}: {len(R)} noise-only tiles, {q.size} virtual clusters; r = sigma_finder / sigma_Y at 0.5' / 15': "
              f"noise-only {rnoise[0]:.4f} / {rnoise[-1]:.4f}, a03 injection tiles {np.mean([r['sig'] for r in I], 0)[0] / S.sigma_Y(mode, 0.5):.4f} / "
              f"{np.mean([r['sig'] for r in I], 0)[-1] / S.sigma_Y(mode, 15.0):.4f}")
        if VF:
            vq = np.concatenate([f["q_inj"] for f in VF]).astype(float); vth = np.concatenate([f["theta"] for f in VF]).astype(float)
            vqa = np.concatenate([f[mode + "_q_alone"] for f in VF])
        print("  P_alone(q_inj) per size: noise-only virtual (this script) / injection tiles (a03) [/ indep x10, |ln theta ratio| < 0.2]")
        Pv, Pi = [], []
        for t in TH_INJ:
            m = th == t
            pv, _ = A3.tab(q[m], qa[m] > 5, QE); Pv.append(pv)
            mi = ith == t
            pi, _ = A3.tab(iq[mi], iqa[mi] > 5, QE); Pi.append(pi)
            line = []
            for lo, hi in ((2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 7.0)):
                a_ = np.mean(qa[m & (q >= lo) & (q < hi)] > 5); b_ = np.mean(iqa[mi & (iq >= lo) & (iq < hi)] > 5)
                s_ = f"{lo}-{hi}: {a_:.3f}/{b_:.3f}"
                if VF:
                    mv = (np.abs(np.log(vth / t)) < 0.2) & (vq >= lo) & (vq < hi)
                    s_ += f"/{np.mean(vqa[mv] > 5):.3f}"
                line.append(s_)
            print(f"    {t:3.1f}': " + "  ".join(line))
        Pv = np.array(Pv); Pi = np.array(Pi)
        # implied N_exp over the visited halos of the build's skies
        Nv = Ni = Na = 0.0
        for pid in range(324):
            r = np.load(os.path.join(S.OUT, "tiles", f"tile_{pid:03d}.npz"))
            t_ = np.load(os.path.join(S.OUT, "truth", f"truth_{pid:03d}.npz"))
            Yh, th5 = t_["Y500"].astype(float), t_["theta500"].astype(float)
            vis = t_[f"q_{mode}"].astype(float) > 2.0
            sig1 = r[f"{mode}_it1_sig"]
            sigf = np.interp(np.log(th5), np.log(F.THETAS15), sig1 / S.sigma_Y(mode, F.THETAS15)) * S.sigma_Y(mode, th5)
            qv = Yh / sigf * np.interp(np.log(th5), np.log(F.THETAS15), rnoise)
            qi = Yh / sigf * np.interp(np.log(th5), np.log(F.THETAS15), rinj)
            Nv += np.sum(A3.p_tab(Pv, QE, rnoise, qv[vis], th5[vis]))
            Ni += np.sum(A3.p_tab(Pi, QE, rinj, qi[vis], th5[vis]))
            Na += np.sum(r[f"{mode}_f_q_alone"] > 5)
        print(f"  visited halos of the build's 324 tiles: isolated search finds {Na:.0f}; expected from noise-only virtual clusters {Nv:.1f} "
              f"(normalisation {100 * (Na / Nv - 1):+.2f}%), from injection tiles (a03) {Ni:.1f} ({100 * (Na / Ni - 1):+.2f}%); "
              f"ratio of the two references {100 * (Nv / Ni - 1):+.2f}%")
    print(f"\ndone [{time.time() - t00:.0f} s]")
