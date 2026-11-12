"""Step 3 / t04: the finder on realistic tiles (all halos M500 > 1e13), SO baseline and goal, with the exact decomposition.

Per tile pid (stored Q2 patch): paint the halos once; six-channel maps = tSZ + CMB (seed 300000 + pid) + noise
(seed 400000 + pid; same unit draws for both noise levels).  Per noise level:
  * the SZiFi-like finder (szsim.finder.run_finder: estimated covariance, q > 4 masked with 3 theta discs, rerun once);
  * covariance diagnostics: sigma(theta) from (a) the CMB + noise part alone, unmasked; (b) the same with the
    iteration mask, whitened f_sky (the finder's method) and (c) literal binary f_sky; (d) the total map with the
    literal binary f_sky (what the iteration would give without whitening); (e) the analytic CMB + noise covariance
    plus the tile's own binned tSZ spectrum;
  * exact decomposition of every final detection (s3lib.decompose), the isolated search, forced photometry at the
    true positions of halos with q_true > 2.
Outputs: data_real/step3/<sub>/tile_PID.npz and data_real/step3/truth/truth_PID.npz.
Usage: python t04_tiles.py PID0 PID1 NPROC [NSCALES (15)] [MODES (baseline,goal)] [SUB (tiles)] [MASK (whitened)]
MASK = 'szifi' (added by the arbiter): the iteration uses SZiFi's own masked covariance with its defaults
(szsim.finder.masked_cov method 'szifi': diffusive inpainting, cosine blend, bins above l = 3000 divided by f).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F

A = sys.argv
P0, P1, NPROC = int(A[1]), int(A[2]), int(A[3])
NSC = int(A[4]) if len(A) > 4 else 15
MODES = tuple(A[5].split(",")) if len(A) > 5 else S.MODES
SUB = A[6] if len(A) > 6 else "tiles"
MASK = A[7] if len(A) > 7 else "whitened"
MASK_KW = dict(lsep=3000.0, blend=True, n_inpaint=100) if MASK == "szifi" else {}
THETAS = np.geomspace(0.5, 15.0, NSC)
OUTD = os.path.join(S.OUT, SUB)
TRD = os.path.join(S.OUT, "truth")
_B = {}


def basis():
    if "b" not in _B:
        G = F.grid()
        _B["tables"] = F.build_paint_tables()
        _B["b"] = F.ResponseBasis(G, _B["tables"])
    return _B["b"], _B["tables"]


def work(pid):
    fn = os.path.join(OUTD, f"tile_{pid:03d}.npz")
    if os.path.exists(fn):
        return pid, 0.0, S.peak_rss_gb(), None
    t0 = time.time()
    G = F.grid()
    bas, tables = basis()
    d = S.truth(pid)
    trf = os.path.join(TRD, f"truth_{pid:03d}.npz")
    if not os.path.exists(trf):
        np.savez(trf, x=d["x"], y=d["y"], z=d["z"].astype(np.float32), M500=d["M500"].astype(np.float32),
                 theta500=d["theta500"].astype(np.float32), Y500=d["Y500"].astype(np.float32),
                 q_baseline=d["q_baseline"].astype(np.float32), q_goal=d["q_goal"].astype(np.float32))
    t1 = time.time()
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
    t_paint = time.time() - t1
    Xc = F.sim_cmb(S.SEED_CMB + pid, G)
    Ctsz = F.binned_cov(Xt, G)
    res = dict(pid=pid, thetas=THETAS, seed_cmb=S.SEED_CMB + pid, seed_noise=S.SEED_NOISE + pid, t_paint=t_paint)
    tim = {}
    del Xc
    for mode in MODES:
        t1 = time.time()
        Xc = F.sim_cmb(S.SEED_CMB + pid, G)
        Xn = F.sim_noise(S.SEED_NOISE + pid, mode, G)
        X = Xt + Xc + Xn
        Xcn = Xc + Xn
        del Xc, Xn
        its = F.run_finder(X, G, thetas=THETAS, mask_method=MASK, mask_kw=MASK_KW)
        t_find = time.time() - t1
        res.update(S.finder_outputs(its, prefix=mode + "_"))
        # ---- covariance diagnostics
        t1 = time.time()
        det0 = its[0]["det"]
        m0 = det0["q"] > F.Q_TH_NOISE
        mask = F.disc_mask(det0["iy"][m0], det0["ix"][m0], F.MASK_RADIUS * det0["theta"][m0], F.N, F.PIX) if m0.any() else None
        Cb_lit = F.masked_cov(X, mask, G, method="binary")[0] if mask is not None else its[0]["Cbin"]
        res[mode + "_sig_lit_it1"] = F.Filter(G, Cbin=Cb_lit, thetas=THETAS).sigma_arcmin2
        del X
        Ccn = F.binned_cov(Xcn, G)
        res[mode + "_sig_cn_est"] = F.Filter(G, Cbin=Ccn, thetas=THETAS).sigma_arcmin2
        if mask is not None:
            res[mode + "_sig_cn_mask"] = F.Filter(G, Cbin=F.masked_cov(Xcn, mask, G, Cref=Ccn)[0], thetas=THETAS).sigma_arcmin2
            res[mode + "_sig_cn_mask_bin"] = F.Filter(G, Cbin=F.masked_cov(Xcn, mask, G, method="binary")[0], thetas=THETAS).sigma_arcmin2
        del Xcn
        res[mode + "_sig_ana_tsz"] = F.Filter(G, Cbin=F.analytic_bin_cov(mode, G) + Ctsz, thetas=THETAS).sigma_arcmin2
        res[mode + "_C_tsz"] = Ctsz
        t_diag = time.time() - t1
        # ---- decomposition (component CX's with the final filter; components regenerated from their seeds)
        t1 = time.time()
        filt = its[-1]["filt"]
        cxs = dict(tsz=filt.CX(Xt))
        Xc = F.sim_cmb(S.SEED_CMB + pid, G); cxs["cmb"] = filt.CX(Xc); del Xc
        Xn = F.sim_noise(S.SEED_NOISE + pid, mode, G); cxs["noise"] = filt.CX(Xn); del Xn
        dec = S.decompose(its[-1], cxs, d, bas, "q_" + mode)
        t_dec = time.time() - t1
        res.update({mode + "_" + k: v for k, v in dec.items()})
        tim[mode] = (t_find, t_diag, t_dec)
        res[mode + "_timing"] = np.array([t_find, t_diag, t_dec])
        del its, cxs
    res["t_total"] = time.time() - t0
    res["peak_rss_gb"] = S.peak_rss_gb()
    np.savez(fn, **res)
    return pid, res["t_total"], res["peak_rss_gb"], (t_paint, tim)


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(TRD, exist_ok=True)
    t0 = time.time()
    print(f"t04: tiles {P0}..{P1 - 1}, {NPROC} processes, {NSC} scales {np.round(THETAS, 3).tolist()}, modes {MODES}, mask {MASK} -> {OUTD}",
          flush=True)
    pids = list(range(P0, P1))
    with Pool(NPROC, maxtasksperchild=8) as pool:
        for pid, tt, rss, tim in pool.imap_unordered(work, pids):
            if tim is None:
                print(f"  tile {pid:3d}: exists", flush=True)
                continue
            tp, tm = tim
            print(f"  tile {pid:3d}: {tt:5.1f} s (paint {tp:.1f}; " + "; ".join(
                f"{m}: find {a:.1f} diag {b:.1f} decomp {c:.1f}" for m, (a, b, c) in tm.items()) +
                f"); worker peak RSS {rss:.2f} GB  [elapsed {time.time() - t0:.0f} s]", flush=True)
    print(f"t04 done: {len(pids)} tiles in {time.time() - t0:.0f} s wall", flush=True)
