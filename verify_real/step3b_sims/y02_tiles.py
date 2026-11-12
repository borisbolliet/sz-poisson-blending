"""y02 (sims lens): fresh Poisson tiles through the own pipeline, both noise levels.

Per tile t (halos: vlib3b.draw_tile_halos(9_310_000 + t) = szsim.sampler's HMF cells x 100 deg^2, its exact bilinear
in-cell draw, uniform positions on the periodic 10 x 10 deg tile, Y500 with the lognormal scatter, M500 > 1e13;
own NUFFT painter; own CMB (seed 9_320_000 + t) and noise (seed 9_330_000 + t; the same unit draws for both levels)):
  * halo-free tile (CMB + noise): finder (METHOD: 'whitened' = production, or 'szifi' = SZiFi defaults); r_free;
    noise peaks; final S/N cube; response tables of its final filter;
  * realistic tile (halos + CMB + noise): finder; r; decomposition of the q > 5 detections (primary, c_p / sigma);
    credited halos (c_p >= 2 sigma; each halo once, at its highest credited q) for merge none, closest 1', fof 10';
    'no real cluster' entries;
  * per halo with q' = Y500 / sigma_finder(theta500) >= 1.5 (sigma_finder = r(theta) sigma_Y(theta), r linear in
    ln theta between the 15 scales, constant beyond): its isolated copy on the halo-free tile (same position, same q'
    in the halo-free tile's units), evaluated as a virtual cluster;
  * NPLACE virtual clusters at random positions (theta cycling over the 25 kernel nodes), each evaluated on the whole
    q' grid; counts of (credited with q_obs > e) per (theta node, q', edge e).
Output: out/tiles_<METHOD>/t_TTTT.npz.   Usage: python y02_tiles.py T0 T1 NPROC METHOD [NPLACE] [WINCHECK_TILES]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib as L
import numpy as np
from multiprocessing import Pool
V = L.V

SEED_H, SEED_C, SEED_N, SEED_P = 9_310_000, 9_320_000, 9_330_000, 9_340_000
_S = {}
ARGS = {}


def setup():
    if not _S:
        G = V.grid()
        _S["G"] = G
        _S["P"] = V.Painter(G, eps=1e-9, scheme="lag4")
        _S["HN"] = V.HankelNodes(G)
    return _S


def work(task):
    t, meth, nplace, wct = task
    outd = os.path.join(L.OUT, f"tiles_{meth}")
    fn = os.path.join(outd, f"t_{t:04d}.npz")
    if os.path.exists(fn):
        return t, None
    S = setup()
    G, P, HN = S["G"], S["P"], S["HN"]
    t0 = time.time()
    h = V.draw_tile_halos(SEED_H + t)
    hy, hx, hY, hth, hz, hM = h["y"], h["x"], h["Y500"], h["theta500"], h["z"], h["M500"]
    nh = hy.size
    Xt = V.tsz_channels(P.T(hy, hx, hY, hth), G)
    res = dict(tile=t, nh=nh, lam=h["lam"], method=meth, nplace=nplace)
    tim = {}
    nodes_r = L.nodes_for(hth[hth > 0.2])
    for li, mode in enumerate(L.MODES):
        pre = mode + "_"
        t1 = time.time()
        Xcn = V.cmb_mine(np.random.default_rng(SEED_C + t), G)
        Xcn += V.noise_mine(np.random.default_rng(SEED_N + t), mode, G)
        sy = V.sigma_Y(mode, L.THETAS)
        # ---------------- halo-free tile
        Rf = L.run_finder(Xcn, G, method=meth, keep_cube=True)
        rf = Rf["sig1"] / sy
        F1f, cube, detf = Rf["F1"], Rf["cube"], Rf["det1"]
        res[pre + "rf"] = rf; res[pre + "rf0"] = Rf["sig0"] / sy
        res[pre + "noise_q"] = detf["q"][detf["q"] > 4.0]
        for m in ("closest1",):
            k = V.apply_merge(detf["iy"] * L.PIX, detf["ix"] * L.PIX, detf["q"], m, qcut=5.0)
            res[pre + f"noise_n5_{m}"] = int(k.size)
        sigf = F1f.sigma_am.copy()
        del Rf
        tim["free"] = time.time() - t1
        # ---------------- realistic tile
        t1 = time.time()
        Rr = L.run_finder(Xt + Xcn, G, method=meth)
        rr = Rr["sig1"] / sy
        res[pre + "r1"] = rr; res[pre + "r0"] = Rr["sig0"] / sy; res[pre + "fsky"] = Rr["fsky"]
        det = Rr["det1"]
        RTr = L.response_tables(Rr["F1"], HN, nodes_r, 30.0)
        dec = L.decompose(det, Rr["F1"], RTr, P, hy, hx, hY, hth)
        del Rr, RTr
        res[pre + "ndet5"] = int(np.sum(det["q"] > 5)); res[pre + "ndet4"] = int(np.sum(det["q"] > 4))
        res[pre + "nexact"] = dec["nexact"]; res[pre + "nrerank"] = dec["nchanged"]
        qo = {}
        for m in ("none", "closest1", "fof10"):
            qo[m], nreal, nnr, nrl = L.credit(det, dec, nh, m)
            res[pre + f"nreal_{m}"] = nreal; res[pre + f"nnr_{m}"] = nnr
            if m == "none":
                a = np.array(nrl).reshape(-1, 3)
                res[pre + "nr_q"] = a[:, 0]; res[pre + "nr_prim"] = a[:, 1].astype(np.int64); res[pre + "nr_cp"] = a[:, 2]
                pz = np.where(a[:, 1] >= 0, hz[np.clip(a[:, 1].astype(np.int64), 0, None)], -1.0)
                res[pre + "nr_z"] = pz
        res[pre + "det5_q"] = det["q"][dec["s5"]]; res[pre + "det5_prim"] = dec["prim"]; res[pre + "det5_cp"] = dec["cp"]
        res[pre + "det5_j"] = det["j"][dec["s5"]]
        tim["real"] = time.time() - t1
        # ---------------- halo rows, isolated copies
        t1 = time.time()
        qp = hY / L.sigma_finder(mode, hth, rr)
        rows = np.flatnonzero(qp >= 1.5)
        nodes_f = L.nodes_for(np.r_[L.TH_K, hth[rows]])
        RTf = L.response_tables(F1f, HN, nodes_f, 30.0)
        cp_ = rows[qp[rows] >= 1.9]
        ycopy = qp[cp_] * L.sigma_finder(mode, hth[cp_], rf)
        qiso = np.full(rows.size, -np.inf)
        if cp_.size:
            qi, _ = L.virt_many(cube, hy[cp_], hx[cp_], hth[cp_], ycopy, RTf, sigf, np.array([1.0]))
            qiso[np.searchsorted(rows, cp_)] = qi[:, 0]
        res[pre + "rows"] = rows; res[pre + "qp"] = qp[rows]; res[pre + "qiso"] = qiso
        res[pre + "z"] = hz[rows]; res[pre + "th"] = hth[rows]; res[pre + "M"] = hM[rows]; res[pre + "Y"] = hY[rows]
        for m in qo:
            res[pre + f"qobs_{m}"] = qo[m][rows]
            res[pre + f"ncred_{m}"] = int(np.isfinite(qo[m]).sum())
            res[pre + f"ncred_below_{m}"] = int(np.sum(np.isfinite(qo[m]) & (qp < 2.0)))
        tim["copies"] = time.time() - t1
        # ---------------- virtual placements (kernel)
        t1 = time.time()
        rng = np.random.default_rng(SEED_P + 2 * t + li)
        ys = rng.uniform(0, L.LBOX, nplace); xs = rng.uniform(0, L.LBOX, nplace)
        ti = np.arange(nplace) % L.TH_K.size
        ths = L.TH_K[ti]
        yus = L.sigma_finder(mode, ths, rf)
        qk, off = L.virt_many(cube, ys, xs, ths, yus, RTf, sigf, L.QG)
        kc = np.zeros((L.TH_K.size, L.QG.size, L.EDGES.size), np.uint16)
        for i in range(L.TH_K.size):
            s = qk[ti == i]
            kc[i] = (s[:, :, None] > L.EDGES[None, None, :]).sum(0)
        res[pre + "kc"] = kc; res[pre + "kn"] = np.bincount(ti, minlength=L.TH_K.size)
        res[pre + "k_offmax"] = np.array([np.max(np.where(qk[ti == i] > 4, off[ti == i], 0.0)) for i in range(L.TH_K.size)])
        res[pre + "k_below2"] = int(np.sum(np.isfinite(qk[:, L.QG < 2.0])))
        if t < wct:                                     # window doubled for the first wct tiles
            sub = np.arange(min(nplace, 500))
            RT45 = L.response_tables(F1f, HN, nodes_f, 45.0)
            qk2, _ = L.virt_many(cube, ys[sub], xs[sub], ths[sub], yus[sub], RT45, sigf, L.QG, wmult=2.0)
            res[pre + "wc_changed"] = int(np.sum(qk2 != qk[sub])); res[pre + "wc_n"] = int(qk2.size)
            del RT45
        tim["place"] = time.time() - t1
        res[pre + "timing"] = np.array([tim["free"], tim["real"], tim["copies"], tim["place"]])
        del F1f, cube, RTf, Xcn, qk, off
    res["t_total"] = time.time() - t0
    res["peak_rss_gb"] = L.peak_rss_gb()
    os.makedirs(outd, exist_ok=True)
    np.savez_compressed(fn, **res)
    return t, (res["t_total"], res["peak_rss_gb"], nh, {m: (res[m + "_ncred_none"], res[m + "_ndet5"]) for m in L.MODES},
               {m: res[m + "_timing"] for m in L.MODES})


if __name__ == "__main__":
    T0, T1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    meth = sys.argv[4]
    nplace = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    wct = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    ARGS.update(meth=meth, nplace=nplace, wct=wct)
    log = V.Tee(os.path.join(L.HERE, f"y02_tiles_{meth}_{T0}_{T1}.log"))
    t00 = time.time()
    log(f"y02: tiles {T0}..{T1 - 1}, method {meth}, {nplace} virtual placements per tile and level, window check on "
        f"tiles < {wct}; {NPROC} processes; threads {L.threads_info()}")
    tasks = [(t, meth, nplace, wct) for t in range(T0, T1)]
    if NPROC == 1:
        it = map(work, tasks)
    else:
        pool = Pool(NPROC, maxtasksperchild=6)
        it = pool.imap_unordered(work, tasks)
    for t, info in it:
        if info is None:
            log(f"  tile {t:4d}: exists"); continue
        tt, rss, nh, nc, tm = info
        log(f"  tile {t:4d}: {nh} halos, {tt:5.1f} s; credited/q>5 dets " + ", ".join(f"{m} {a}/{b}" for m, (a, b) in nc.items())
            + "; timing free/real/copies/place " + " | ".join(" ".join(f"{v:.1f}" for v in tm[m]) for m in tm)
            + f"; worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]")
    log(f"y02 done in {time.time() - t00:.0f} s wall")
