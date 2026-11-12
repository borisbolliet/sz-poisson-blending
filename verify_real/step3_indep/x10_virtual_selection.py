"""x10 (indep lens): criterion-matched unit-noise calibration of the ISOLATED SEARCH ('virtual injections').

The build's split 2 ('normalisation convention') compares the isolated-search count on the realistic tiles (q_alone > 5)
with N_exp from a selection calibrated on a DIFFERENT criterion (the finder's credited detections on injection tiles).
Here the same isolated search is calibrated on noise-only tiles, with very many virtual clusters, so that the
normalisation effect can be measured without the finder-injection calibration:
  per noise-only tile t (the x03 noise tiles: CMB seed 8100000 + t, noise seed 8200000 + t) and level: own finder (it0,
  q > 4 discs, whitened masked covariance, it1); filtered CMB + noise maps of the final filter at the 15 scales; NV
  virtual clusters at uniform random positions, theta500 log-uniform in [0.25', 12'], q_inj = Y500 / sigma_Y(theta500)
  uniform in [1.5, 9] (not painted: each one is evaluated alone on the noise); q_alone = max over the 13 x 13 pixel
  window around its nearest pixel and the 15 scales of [Y R_j(|p - x|) + CN_j(p)] / sigma_j (own response tables of the
  tile's filter), exactly the isolated search of x04.  Also stored: sigma_final / sigma_Y (r_noise).
Outputs: out/virt/virt_TTT.npz.  Usage: python x10_virtual_selection.py T0 T1 NPROC [NV]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import vlib3 as V

OUTD = os.path.join(V.OUT, "virt")
NV = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
WIN = 6
_S = {}


def setup():
    if not _S:
        from scipy.special import j0
        G = V.grid()
        _S["G"] = G
        _S["HN"] = V.HankelNodes(G)
        _S["J0"] = j0(np.outer(_S["HN"].l, V.R_NEAR * V.AR))
    return _S


def work(t):
    fn = os.path.join(OUTD, f"virt_{t:03d}.npz")
    if os.path.exists(fn):
        return t, 0.0, V.peak_rss_gb()
    t0 = time.time()
    S = setup(); G = S["G"]
    rng = np.random.default_rng(9_900_000 + t)
    th = np.exp(rng.uniform(np.log(0.25), np.log(12.0), NV))
    qi = rng.uniform(1.5, 9.0, NV)
    y = rng.uniform(0, V.LBOX, NV); x = rng.uniform(0, V.LBOX, NV)
    lth = np.log(th)
    i0, _ = V.lagrange4(lth)
    nodes = np.arange(i0.min() - 2, i0.max() + 4)
    cy = np.rint(y / V.PIX).astype(np.int64) % V.N
    cx = np.rint(x / V.PIX).astype(np.int64) % V.N
    ar = np.arange(-WIN, WIN + 1)
    res = dict(t=t, theta=th.astype(np.float32), q_inj=qi.astype(np.float32))
    for mode in V.MODES:
        X = V.cmb_mine(np.random.default_rng(8_100_000 + t), G)
        X += V.noise_mine(np.random.default_rng(8_200_000 + t), mode, G)
        R = V.run_finder(X, G)
        F1 = R["F1"]
        D = F1.combine(X)
        del X, R
        Y = qi * V.sigma_Y(mode, th)
        RT = V.ResponseTables(F1, S["HN"], nodes, S["J0"])
        qa = np.zeros(NV); qj = np.zeros(NV, np.int64)
        maps = [F1.ymap(D, j) for j in range(15)]
        for a in range(0, NV, 2000):
            sl = slice(a, min(a + 2000, NV))
            nwin = np.stack([m[(cy[sl, None, None] + ar[None, :, None]) % V.N, (cx[sl, None, None] + ar[None, None, :]) % V.N]
                             for m in maps], 1)
            idx = np.arange(sl.stop - sl.start, dtype=np.int64)
            ov = np.zeros(idx.size); oj = np.zeros(idx.size, np.int64)
            V.alone_search(cy[sl], cx[sl], idx, y[sl], x[sl], Y[sl], lth[sl], RT.R, RT.n0, RT.dr, np.log(V.TH0_NODE), V.DNODE,
                           V.LBOX, V.PIX, WIN, nwin, F1.sigma_am, ov, oj)
            qa[sl] = ov; qj[sl] = oj
        del maps, D
        res[mode + "_q_alone"] = qa.astype(np.float32)
        res[mode + "_r_noise"] = F1.sigma_am / V.sigma_Y(mode, V.THETAS)
    res["t_run"] = time.time() - t0
    np.savez(fn, **res)
    return t, res["t_run"], V.peak_rss_gb()


if __name__ == "__main__":
    T0, T1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUTD, exist_ok=True)
    log = V.Tee(os.path.join(V.HERE, "x10_virtual_selection.log"))
    t00 = time.time()
    log(f"x10: virtual injections on noise-only tiles {T0}..{T1 - 1}, {NV} virtual clusters per tile, {NPROC} processes")
    with Pool(NPROC, maxtasksperchild=20) as pool:
        for i, (t, tt, rss) in enumerate(pool.imap_unordered(work, range(T0, T1))):
            if i % 10 == 0:
                log(f"  tile {t:3d}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]")
    log(f"x10 done in {time.time() - t00:.0f} s wall")
