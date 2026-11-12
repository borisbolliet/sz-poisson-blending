"""x04b (indep lens), claim S1: the x04 fresh-sky pipeline with NSC log-spaced filter scales 0.5'-15' (12 or 20) instead
of 15, on the same skies (same seeds).  Covariance diagnostics are skipped.  Outputs: out/fresh_sNSC/fresh_TTT.npz.
Usage: python x04b_scales.py T0 T1 NPROC NSC
--- original x04 description ---
x04 (indep lens), checks (b) and (e): fresh skies, own code end to end.

Per tile t: halos seed 7000000 + t (vlib3.draw_tile_halos: szsim.sampler's HMF cells x 100 deg^2, exact bilinear in-cell
draw, uniform positions on the periodic 10 x 10 deg tile, Y500 with the model's log-normal scatter, M500 > 1e13);
own NUFFT painter; own lensed CMB (seed 7100000 + t); own white + atmospheric noise (seed 7200000 + t, common-mode tube
decomposition; the same unit draws for baseline and goal).  Per noise level:
  * own SZiFi-like finder (vlib3.run_finder): binned covariance of the tile (bins of 72 from l = 0, 'nearest',
    l = 100-20000), 15 scales 0.5'-15' with step 1's template, (theta, y, x) maxima q > 4, q > 4 masked with 3 theta
    discs, whitened masked covariance, rerun once;
  * diagnostics: sigma of the CMB + noise part alone (estimated; masked), the literal binary f_sky iteration, analytic
    CMB + noise + the tile's own binned tSZ spectrum, the pixel std of the CMB + noise part of the final S/N maps;
  * decomposition of the final q > 5 detections: own continuous Hankel tables rank all halos within 60', the 6 largest
    are recomputed as exact lattice sums (template = the painter's), re-ranked; delta = filtered tSZ - c_p;
  * forced set (q_true > 2 and every primary): nearest pixel / nearest scale values, own table response, largest
    neighbour (table, 60'); isolated search q_alone (13 x 13 pixels, all scales; own response + CMB + noise);
  * step 1's analytic MMF at the true positions: largest single neighbour x_max (own rho table) for q_true > 3 halos
    and the two largest contributors of every q > 5 detection.
Outputs: out/fresh/fresh_TTT.npz.  Usage: python x04_fresh.py T0 T1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import vlib3 as V

SEED_H, SEED_C, SEED_N = 7_000_000, 7_100_000, 7_200_000
NSC = int(sys.argv[4]) if len(sys.argv) > 4 else 15
THS = np.geomspace(0.5, 15.0, NSC)
OUTD = os.path.join(V.OUT, f"fresh_s{NSC}")
WIN, K, KEEP, QT_FORCED, RCUT = 6, 6, 4, 2.0, 60.0
_S = {}


def setup():
    if not _S:
        from scipy.special import j0
        G = V.grid()
        _S["G"] = G
        _S["P"] = V.Painter(G, eps=1e-9, scheme="lag4")
        _S["HN"] = V.HankelNodes(G)
        _S["J0"] = j0(np.outer(_S["HN"].l, V.R_NEAR * V.AR))
        _S["rho1"] = {m: V.Rho1(m) for m in V.MODES}
        _S["Cana"] = {m: G.analytic_cov_binned(m) for m in V.MODES}
    return _S


def work(t):
    fn = os.path.join(OUTD, f"fresh_{t:03d}.npz")
    if os.path.exists(fn):
        return t, None
    S = setup()
    G, P = S["G"], S["P"]
    t0 = time.time()
    h = V.draw_tile_halos(SEED_H + t)
    hy, hx, hY, hth = h["y"], h["x"], h["Y500"], h["theta500"]
    lth = np.log(hth)
    Xt = V.tsz_channels(P.T(hy, hx, hY, hth), G)
    t_paint = time.time() - t0
    Ctsz = V.binned_cov(Xt, G)
    res = dict(tile=t, nh=hy.size, lam=h["lam"], Ysum=hY.sum(), t_paint=t_paint)
    i0, _ = V.lagrange4(lth)
    nodes = np.arange(i0.min() - 2, i0.max() + 4)
    targs_static = (np.log(V.TH0_NODE), V.DNODE)
    tim = []
    for mode in V.MODES:
        pre = mode + "_"
        t1 = time.time()
        Xcn = V.cmb_mine(np.random.default_rng(SEED_C + t), G)          # same CMB draw for both levels
        Xcn += V.noise_mine(np.random.default_rng(SEED_N + t), mode, G)
        X = Xt + Xcn
        R = V.run_finder(X, G, thetas=THS)
        F0, F1, det0, det1 = R["F0"], R["F1"], R["det0"], R["det1"]
        res[pre + "sig0"] = F0.sigma_am; res[pre + "sig1"] = F1.sigma_am; res[pre + "fsky1"] = R["fsky"]
        res[pre + "it0_q"] = det0["q"]
        R["F0"] = None; del F0
        t_find = time.time() - t1
        t1 = time.time()
        del X, R
        t_diag = 0.0
        # ---- pass 1: components at the final detections; pixel std of the CMB + noise S/N maps
        t1 = time.time()
        Dt = F1.combine(Xt); Dcn = F1.combine(Xcn)
        j, iy, ix, q = det1["j"], det1["iy"], det1["ix"], det1["q"]
        St = np.zeros(q.size); Scn = np.zeros(q.size); wcn = np.zeros(NSC)
        for jj in range(NSC):
            yt = F1.ymap(Dt, jj); ycn = F1.ymap(Dcn, jj)
            m = j == jj
            St[m] = yt[iy[m], ix[m]]; Scn[m] = ycn[iy[m], ix[m]]
            wcn[jj] = ycn.std() / F1.sigma_am[jj]
        del yt, ycn
        RT = V.ResponseTables(F1, S["HN"], nodes, S["J0"])
        targs = (RT.R, RT.n0, RT.dr) + targs_static
        # ---- exact decomposition of the q > 5 detections
        s5 = np.flatnonzero(q > 5)
        qy, qx, qj = iy[s5] * V.PIX, ix[s5] * V.PIX, j[s5]
        cidx = np.full((s5.size, K), -1, np.int64); cval = np.zeros((s5.size, K)); csum = np.zeros(s5.size)
        V.table_topk(qy, qx, qj, np.full(s5.size, -1, np.int64), hy, hx, hY, lth, *targs, RCUT, V.LBOX, K, cidx, cval, csum)
        EX = V.Exact(F1, P)
        ex = np.full((s5.size, K), np.nan)
        for a in range(s5.size):
            for k in range(K):
                hh = cidx[a, k]
                if hh < 0:
                    continue
                dy = (qy[a] - hy[hh] + 300) % 600 - 300
                dx = (qx[a] - hx[hh] + 300) % 600 - 300
                ex[a, k] = EX.contrib(int(qj[a]), hY[hh], hth[hh], dy, dx, template="nodes")
            o = np.argsort(-np.nan_to_num(ex[a], nan=-1e300))
            ex[a] = ex[a, o]; cidx[a] = cidx[a, o]; cval[a] = cval[a, o]
        t_dec = time.time() - t1
        # ---- forced set: q_true > 2 and every primary
        t1 = time.time()
        qt = hY / V.sigma_Y(mode, hth)
        prim = cidx[:, 0] if s5.size else np.zeros(0, np.int64)
        fsel = np.union1d(np.flatnonzero(qt > QT_FORCED), prim[prim >= 0]).astype(np.int64)
        fiy = np.rint(hy[fsel] / V.PIX).astype(np.int64) % V.N
        fix = np.rint(hx[fsel] / V.PIX).astype(np.int64) % V.N
        fj = np.abs(lth[fsel][:, None] - np.log(THS)[None, :]).argmin(1).astype(np.int64)
        fSt = np.zeros(fsel.size); fScn = np.zeros(fsel.size)
        nwin = np.zeros((fsel.size, NSC, 2 * WIN + 1, 2 * WIN + 1))
        ar = np.arange(-WIN, WIN + 1)
        for jj in range(NSC):
            yt = F1.ymap(Dt, jj); ycn = F1.ymap(Dcn, jj)
            m = fj == jj
            fSt[m] = yt[fiy[m], fix[m]]; fScn[m] = ycn[fiy[m], fix[m]]
            nwin[:, jj] = ycn[(fiy[:, None, None] + ar[None, :, None]) % V.N, (fix[:, None, None] + ar[None, None, :]) % V.N]
        del Dt, Dcn, yt, ycn
        qa = np.zeros(fsel.size); qaj = np.zeros(fsel.size, np.int64)
        V.alone_search(fiy, fix, fsel, hy, hx, hY, lth, *targs, V.LBOX, V.PIX, WIN, nwin, F1.sigma_am, qa, qaj)
        del nwin
        fown = np.zeros(fsel.size)
        V.own_resp(fiy * V.PIX, fix * V.PIX, fj, fsel, hy, hx, hY, lth, *targs, V.LBOX, fown)
        nbi = np.full((fsel.size, 2), -1, np.int64); nbv = np.zeros((fsel.size, 2)); nbs = np.zeros(fsel.size)
        V.table_topk(fiy * V.PIX, fix * V.PIX, fj, fsel, hy, hx, hY, lth, *targs, RCUT, V.LBOX, 2, nbi, nbv, nbs)
        # ---- step 1's analytic MMF at the true positions
        hq3 = np.flatnonzero(qt > 3)
        xi = np.unique(np.concatenate([hq3, cidx[:, :2].ravel()]))
        xi = xi[xi >= 0].astype(np.int64)
        rho = S["rho1"][mode]
        xm = np.zeros(xi.size); xa = np.zeros(xi.size, np.int64)
        V.xmax_true(xi, hy, hx, hY, lth, V.sigma_Y(mode, hth[xi]), rho.tab, np.log(V.Rho1.TH0), V.Rho1.D,
                    V.Rho1.S[1] - V.Rho1.S[0], RCUT, V.LBOX, xm, xa)
        t_forced = time.time() - t1
        res.update({pre + k: v for k, v in dict(
            width_cn=wcn, j=j, iy=iy, ix=ix, q=q, S_tsz=St, S_cn=Scn, d5=s5, top_idx=cidx[:, :KEEP], top_ex=ex[:, :KEEP],
            top_tab=cval[:, :KEEP], tab_sum=csum, f_idx=fsel, f_j=fj, f_iy=fiy, f_ix=fix, f_S_tsz=fSt, f_S_cn=fScn,
            f_own=fown, f_nb_idx=nbi, f_nb_val=nbv, f_q_alone=qa, f_q_alone_j=qaj, x_idx=xi, x_max=xm, x_arg=xa,
            timing=np.array([t_find, t_diag, t_dec, t_forced])).items()})
        tim.append((t_find, t_diag, t_dec, t_forced))
        del F1, RT, EX, Xcn
    res["t_total"] = time.time() - t0
    res["peak_rss_gb"] = V.peak_rss_gb()
    np.savez(fn, **res)
    return t, (res["t_total"], t_paint, tim, res["peak_rss_gb"], hy.size)


if __name__ == "__main__":
    T0, T1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    assert NSC != 15
    os.makedirs(OUTD, exist_ok=True)
    log = V.Tee(os.path.join(V.HERE, f"x04b_scales_s{NSC}_{T0}_{T1}.log"))
    t00 = time.time()
    for m in V.MODES:
        V.Rho1(m)                         # build / cache the step-1 rho tables before any worker starts
    log(f"x04: fresh tiles {T0}..{T1 - 1} with {NPROC} processes -> {OUTD}")
    if NPROC == 1:
        it = map(work, range(T0, T1))
    else:
        pool = Pool(NPROC, maxtasksperchild=10)
        it = pool.imap_unordered(work, range(T0, T1))
    for t, info in it:
        if info is None:
            log(f"  tile {t:3d}: exists"); continue
        tt, tp, tim, rss, nh = info
        log(f"  tile {t:3d}: {nh} halos, {tt:5.1f} s (paint {tp:.1f}; " + "; ".join(
            f"{m}: find {a:.1f} diag {b:.1f} dec {c:.1f} forced {d:.1f}" for m, (a, b, c, d) in zip(V.MODES, tim))
            + f"); worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]")
    log(f"x04 done in {time.time() - t00:.0f} s wall")
