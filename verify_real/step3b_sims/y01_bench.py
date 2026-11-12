"""y01 (sims lens): timing, memory and accuracy of the own pipeline on one fresh tile (seed 9_310_000 + 9999).
(1) finder runs (production whitened, SZiFi defaults) on the realistic and the halo-free tile, both levels: r at
    0.5'/4.45'/15';
(2) accuracy of the virtual response (angle-averaged tables at the painter's nodes) against a painted cluster filtered
    exactly by FFT through the same final filter, for 6 sizes at random sub-pixel positions: max |diff| / peak over a
    window of half width 12' (all 15 scales);
(3) the same for the q_obs of virtual clusters: painted-through-filter + noise vs the virtual route (same noise);
(4) timing of the virtual placements and of the decomposition.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib as L
import numpy as np
V = L.V

log = V.Tee(os.path.join(L.HERE, "y01_bench.log"))
t00 = time.time()
G = V.grid()
P = V.Painter(G, eps=1e-9, scheme="lag4")
HN = V.HankelNodes(G)
log(f"threads: {L.threads_info()}")
t = 9999
h = V.draw_tile_halos(9_310_000 + t)
log(f"halos: {h['y'].size} (lambda {h['lam']:.1f})  [{time.time()-t00:.1f}s]")
t0 = time.time()
Xt = V.tsz_channels(P.T(h["y"], h["x"], h["Y500"], h["theta500"]), G)
log(f"paint: {time.time()-t0:.1f}s; rss {L.peak_rss_gb():.2f} GB")
rng = np.random.default_rng(1)
for mode in L.MODES:
    t0 = time.time()
    Xcn = V.cmb_mine(np.random.default_rng(9_320_000 + t), G) + V.noise_mine(np.random.default_rng(9_330_000 + t), mode, G)
    log(f"[{mode}] CMB+noise draw {time.time()-t0:.1f}s")
    for meth in ("whitened", "szifi"):
        t0 = time.time()
        Rr = L.run_finder(Xt + Xcn, G, method=meth)
        t1 = time.time() - t0
        t0 = time.time()
        Rf = L.run_finder(Xcn, G, method=meth, keep_cube=(meth == "whitened"))
        t2 = time.time() - t0
        sy = V.sigma_Y(mode, L.THETAS)
        rr = Rr["sig1"] / sy; rf = Rf["sig1"] / sy
        log(f"[{mode}] {meth:8s}: finder realistic {t1:.1f}s, halo-free {t2:.1f}s; r after (0.5'/4.45'/15') realistic "
            f"{rr[0]:.4f}/{rr[9]:.4f}/{rr[14]:.4f}, before {Rr['sig0'][0]/sy[0]:.4f}; halo-free {rf[0]:.4f}/{rf[9]:.4f}/{rf[14]:.4f}; "
            f"q>5 dets {np.sum(Rr['det1']['q'] > 5)} (it0 {np.sum(Rr['det0']['q'] > 5)}); noise q>5 {np.sum(Rf['det1']['q'] > 5)}; "
            f"fsky {Rr['fsky']:.3f}; rss {L.peak_rss_gb():.2f} GB")
        if meth == "whitened":
            Rr_w, Rf_w = Rr, Rf
        del Rr
    # ---------------- (2) response accuracy on the halo-free production filter
    F1 = Rf_w["F1"]; cube = Rf_w["cube"]
    nodes = L.nodes_for(np.r_[L.TH_K, h["theta500"][h["theta500"] > 0.2]])
    t0 = time.time()
    RT = L.response_tables(F1, HN, nodes, 45.0)
    log(f"[{mode}] response tables ({nodes.size} nodes, r <= 45'): {time.time()-t0:.1f}s; rss {L.peak_rss_gb():.2f} GB")
    for th in (0.3, 1.2, 4.8, 15.0, 30.0, 61.0):
        y0, x0 = rng.uniform(100, 500, 2)
        T = P.T(np.array([y0]), np.array([x0]), np.array([1.0]), np.array([th]))
        D = F1.combine(V.tsz_channels(T, G))
        w = 24
        cy, cx = int(np.floor(y0 / 0.5 + 0.5)), int(np.floor(x0 / 0.5 + 0.5))
        worst = 0.0; worst_all = 0.0
        for j in range(15):
            ym = F1.ymap(D, j)
            win = ym[cy - w:cy + w + 1, cx - w:cx + w + 1]
            ay = (np.arange(cy - w, cy + w + 1) * 0.5 - y0)[:, None]
            ax = (np.arange(cx - w, cx + w + 1) * 0.5 - x0)[None, :]
            rr_ = np.sqrt(ay ** 2 + ax ** 2)
            tab = np.array([[V.resp_eval(RT.R, j, RT.n0, np.log(th), rr_[a, b], RT.dr, L.LNTH0, L.DNODE)
                             for b in range(2 * w + 1)] for a in range(2 * w + 1)])
            pk = np.abs(win).max()
            worst = max(worst, np.abs(tab - win).max() / pk)
            worst_all = max(worst_all, np.abs(tab - win)[rr_ < 6].max() / pk)
        log(f"[{mode}]   theta {th:5.1f}': virtual table vs exact painted response, max |diff|/peak over 12' window "
            f"{worst:.2e} (within 6': {worst_all:.2e})")
    # ---------------- (3) q_obs: painted cluster through the filter vs virtual, 60 clusters at q' 3-8
    sig = F1.sigma_am.astype(float)
    rf = Rf_w["sig1"] / V.sigma_Y(mode, L.THETAS)
    nd = 0; nn = 0; dq = []
    for k in range(60):
        th = float(L.TH_K[rng.integers(0, 25)]); qp = rng.uniform(3, 8)
        y0, x0 = rng.uniform(0, 600, 2)
        yu = float(L.sigma_finder(mode, th, rf))
        T = P.T(np.array([y0]), np.array([x0]), np.array([qp * yu]), np.array([th]))
        D = F1.combine(V.tsz_channels(T, G))
        w = L.half_width_pix(th)
        cy, cx = int(np.floor(y0 / 0.5 + 0.5)), int(np.floor(x0 / 0.5 + 0.5))
        idy = np.arange(cy - w, cy + w + 1) % V.N; idx = np.arange(cx - w, cx + w + 1) % V.N
        Sx = np.stack([F1.qmap(D, j)[np.ix_(idy, idx)] for j in range(15)])
        Nw = cube[:, idy][:, :, idx].astype(float)
        Q = Nw + Sx
        best = -np.inf
        for j in range(15):
            for a in range(1, 2 * w):
                for b in range(1, 2 * w):
                    v = Q[j, a, b]
                    if Sx[j, a, b] < 2 or v <= 4 or v <= best:
                        continue
                    if not (v > Q[j, a - 1, b] and v > Q[j, a + 1, b] and v > Q[j, a, b - 1] and v > Q[j, a, b + 1]):
                        continue
                    ok = (v > 0 and v >= Q[1, a, b]) if j == 0 else (v > Q[13, a, b] if j == 14 else (v > Q[j - 1, a, b] and v >= Q[j + 1, a, b]))
                    if ok:
                        best = v
        qv, _ = L.virt_many(cube, [y0], [x0], [th], [yu], RT, sig, np.array([qp]))
        nn += 1
        if np.isfinite(best) != np.isfinite(qv[0, 0]):
            nd += 1
        elif np.isfinite(best):
            dq.append(best - qv[0, 0])
    dq = np.array(dq)
    log(f"[{mode}] (3) q_obs painted-through-filter vs virtual (same noise window): {nn} clusters, credited differently "
        f"{nd}; max |dq| {np.abs(dq).max():.2e}")
    # ---------------- (4) timing: 500 placements over the q' grid
    t0 = time.time()
    ths = L.TH_K[np.arange(500) % 25]
    ys, xs = rng.uniform(0, 600, 500), rng.uniform(0, 600, 500)
    yus = L.sigma_finder(mode, ths, rf)
    qo, off = L.virt_many(cube, ys, xs, ths, yus, RT, sig, L.QG)
    log(f"[{mode}] 500 placements x {L.QG.size} q' values: {time.time()-t0:.1f}s; credited at q'=5 (e=5): "
        f"{np.mean(qo[:, np.argmin(np.abs(L.QG-5))] > 5):.3f}; max peak offset (arcmin) by theta node: "
        + " ".join(f"{np.nanmax(np.where(qo[ths==th] > 4, off[ths==th], np.nan)):.1f}" for th in L.TH_K[::4]))
    t0 = time.time()
    qo2, _ = L.virt_many(cube, ys, xs, ths, yus, RT, sig, L.QG, wmult=2.0)
    log(f"[{mode}] window doubled: {time.time()-t0:.1f}s; entries changed {np.sum(qo2 != qo)} of {qo.size}")
    # ---------------- decomposition timing on the realistic tile
    nodes_r = L.nodes_for(h["theta500"][h["theta500"] > 0.2])
    t0 = time.time()
    RTr = L.response_tables(Rr_w["F1"], HN, nodes_r, 30.0)
    t1 = time.time() - t0
    t0 = time.time()
    dec = L.decompose(Rr_w["det1"], Rr_w["F1"], RTr, P, h["y"], h["x"], h["Y500"], h["theta500"])
    log(f"[{mode}] decomposition: tables {t1:.1f}s, ranking+exact {time.time()-t0:.1f}s ({dec['s5'].size} dets, "
        f"{dec['nexact']} exact, {dec['nchanged']} re-ranked); no-real {np.sum(dec['cp'] < 2)}")
    for m in ("none", "closest1", "fof10"):
        qo_, nreal, nnr, _ = L.credit(Rr_w["det1"], dec, h["y"].size, m)
        log(f"[{mode}]   merge {m}: credited halos {np.isfinite(qo_).sum()}, real entries {nreal}, no-real {nnr}")
    del Rr_w, Rf_w, F1, cube, RT, RTr
log(f"total {time.time()-t00:.0f}s; peak rss {L.peak_rss_gb():.2f} GB")
