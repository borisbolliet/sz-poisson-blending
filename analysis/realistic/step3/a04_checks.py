"""Step 3 arbiter / a04: targeted tests of claims D3, R5 and W1, and of SZiFi's 'nearest' tie rule (szifi check, bug 2).

  (D3) The per-halo table closure with the finder's binned covariance (median 8.6e-3 sigma) is said to come from the
       direction dependence of the binned filter's response, which no radial table can follow.  Test: for q > 5
       detections of three tiles, replace the table values of every halo within R of the detection by exact 2D
       lattice sums (szsim.finder.exact_contrib) and recompute the closure tSZ_map - sum over halos.  If the claim
       holds, the residual falls by orders of magnitude as R grows.
  (R5) The width of the CMB + noise part of the finder's S/N maps after the iteration: the build gives an analytic
       expectation (analytic CMB + noise covariance through each tile's filter); the indep check measured the pixel
       standard deviation.  Both, on 10 tiles per level, against 1 / r_after.
  (W1) The isolated search uses a +-3' window.  Test: rerun it with a +-6' window for every forced halo with
       q_alone > 4.5 on three tiles; how often does the maximum move, and how far from the halo is it.
  (tie) SZiFi evaluates the binned covariance at each mode with scipy 'nearest' interpolation over the bin centres,
       which puts the modes exactly on a bin edge (|k| = 72 m) into the lower bin; the build puts them into the upper
       bin.  sigma(theta) with both rules on the stored iteration-0 and iteration-1 covariances of 10 tiles.
All tiles are regenerated from their seeds with the production finder; the regenerated catalogue is checked against
the stored one.  Usage: python a04_checks.py NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F

OUTD = os.path.join(S.OUT, "arbiter")
P_DW = [0, 107, 213]                                # D3 and W1
P_R5 = list(range(0, 324, 33))                      # R5 and the tie rule (10 tiles)
R_EXACT = (5.0, 15.0, 30.0)
WIN2 = 12                                           # +-6' window
N_D3 = 40                                           # detections per tile and level for D3


def regen(pid, mode, G, tables):
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
    Xcn = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
    its = F.run_finder(Xt + Xcn, G)
    return d, Xt, Xcn, its


def work(unit):
    kind, pid = unit
    fn = os.path.join(OUTD, f"a04_{kind}_{pid:03d}.npz")
    if os.path.exists(fn):
        return unit, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    tables = F.build_paint_tables()
    st = dict(np.load(os.path.join(S.OUT, "tiles", f"tile_{pid:03d}.npz")))
    res = dict(pid=pid)
    if kind == "dw":
        bas = F.ResponseBasis(G, tables)
    for mode in S.MODES:
        d, Xt, Xcn, its = regen(pid, mode, G, tables)
        filt, det = its[-1]["filt"], its[-1]["det"]
        q_st = st[f"{mode}_it1_q"]
        res[mode + "_regen_dq"] = np.abs(det["q"] - q_st).max() if det["q"].size == q_st.size else np.inf
        sig = filt.sigma_arcmin2
        if kind == "r5":
            # realised pixel std of the CMB + noise S/N maps; analytic expectation as t05; 1 / r_after
            cx = filt.CX(Xcn)
            res[mode + "_std_pix"] = np.array([np.std(filt.yhat_map(cx, j) / filt.sigma[j]) for j in range(len(filt.thetas))])
            Ccn = F.analytic_bin_cov(mode, G)
            C1 = filt.Cbin
            Ci = np.zeros_like(C1); Ci[G.bins_used] = np.linalg.inv(C1[G.bins_used])
            M = np.zeros_like(C1); M[G.bins_used] = np.einsum("bij,bjk,bkl->bil", Ci[G.bins_used], Ccn[G.bins_used], Ci[G.bins_used])
            a_m = G.a_modes()
            aCa = np.einsum("mi,mij,mj->m", a_m, Ci[G.m_bin], a_m); aMa = np.einsum("mi,mij,mj->m", a_m, M[G.m_bin], a_m)
            res[mode + "_std_ana"] = np.array([np.sqrt(np.sum(G.m_w * F.gt_lin(G.m_L * t * F.AR) ** 2 * aMa)
                                                       / np.sum(G.m_w * F.gt_lin(G.m_L * t * F.AR) ** 2 * aCa)) for t in F.THETAS15])
            res[mode + "_r1"] = sig / S.sigma_Y(mode, F.THETAS15)
            # the tie rule, on the stored covariances (iterations 0 and 1)
            n2s = G.shell_n2[G.m_shell]
            isq = np.floor(np.sqrt(n2s.astype(float))).astype(np.int64)
            isq -= (isq * isq > n2s); isq += ((isq + 1) * (isq + 1) <= n2s)
            tie = (isq * isq == n2s) & (isq % 2 == 0) & np.isin(G.m_bin - 1, G.bins_used)
            alt = np.where(tie, G.m_bin - 1, G.m_bin)
            res["n_tie"] = int(tie.sum()); res["n_modes"] = int(tie.size)
            for it in (0, 1):
                Cb = st[f"{mode}_it{it}_C"]
                Ci = np.zeros_like(Cb); Ci[G.bins_used] = np.linalg.inv(Cb[G.bins_used])
                out = []
                for mb in (G.m_bin, alt):
                    aCa = np.einsum("mi,mij,mj->m", a_m, Ci[mb], a_m)
                    out.append([1.0 / np.sqrt(np.sum(G.m_w * (F.gt_lin(G.m_L * t * F.AR) / G.Om) ** 2 * aCa)) for t in F.THETAS15])
                res[f"{mode}_tie_it{it}"] = np.array(out[1]) / np.array(out[0])
        else:
            hy, hx, hY, hth = d["y"], d["x"], d["Y500"], d["theta500"]
            hl = np.log(hth)
            Rn, Rf = F.response_tables(filt, bas)
            targs = (Rn, Rf, bas.lnth0, bas.dln, F.R_SWITCH, bas.rn[1] - bas.rn[0], F.R_FAR_MIN, bas.rf[1] - bas.rf[0], F.LBOX)
            # ---- D3: closure with exact lattice sums for the halos within R
            s_tsz, s_tab = st[f"{mode}_d_S_tsz"], st[f"{mode}_d_tab_sum"]
            sd = sig[det["j"]]
            clo = (s_tsz - s_tab) / sd
            d5 = np.flatnonzero(det["q"] > 5)
            rng = np.random.default_rng(pid)
            pick = np.unique(np.r_[d5[np.argsort(-np.abs(clo[d5]))[:N_D3 // 2]], rng.choice(d5, N_D3 // 2, replace=False)])
            out = np.zeros((pick.size, len(R_EXACT)))
            nnear = np.zeros((pick.size, len(R_EXACT)), np.int64)
            for a, dd in enumerate(pick):
                dy = (hy - det["y"][dd] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
                dx = (hx - det["x"][dd] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
                rr = np.hypot(dy, dx)
                order = np.argsort(rr)
                ex_cum = 0.0; tb_cum = 0.0; k0 = 0
                for b, R in enumerate(R_EXACT):
                    near = order[k0:np.searchsorted(rr[order], R)]
                    for h in near:
                        ex_cum += F.exact_contrib(filt, bas, det["j"][dd], hY[h], hth[h], -dy[h], -dx[h])
                    if near.size:
                        tsum = np.zeros(1); ti = np.zeros((1, 1), np.int64); tv = np.zeros((1, 1)); nc = np.zeros(1, np.int64)
                        F.table_contributions(det["j"][dd:dd + 1], det["y"][dd:dd + 1], det["x"][dd:dd + 1], hy[near], hx[near],
                                              hY[near], hl[near], *targs, 1, tsum, ti, tv, nc)
                        tb_cum += tsum[0]
                    k0 += near.size
                    nnear[a, b] = k0
                    out[a, b] = (s_tsz[dd] - (s_tab[dd] - tb_cum + ex_cum)) / sd[dd]
            res[mode + "_d3_pick"] = pick; res[mode + "_d3_table"] = clo[pick]; res[mode + "_d3_hybrid"] = out
            res[mode + "_d3_nnear"] = nnear
            # ---- W1: isolated search with a +-6' window
            fidx, fqa = st[f"{mode}_f_idx"], st[f"{mode}_f_q_alone"]
            sel = np.flatnonzero(fqa > 4.5)
            h_ = fidx[sel]
            fiy = np.rint(hy[h_] / F.PIX).astype(np.int64) % F.N
            fix = np.rint(hx[h_] / F.PIX).astype(np.int64) % F.N
            cx = filt.CX(Xcn)
            ar = np.arange(-WIN2, WIN2 + 1)
            nwin = np.zeros((h_.size, len(filt.thetas), ar.size, ar.size))
            for j in range(len(filt.thetas)):
                mcn = filt.yhat_map(cx, j) / F.AR ** 2
                nwin[:, j] = mcn[(fiy[:, None, None] + ar[None, :, None]) % F.N, (fix[:, None, None] + ar[None, None, :]) % F.N]
            qa = np.zeros(h_.size); qj = np.zeros(h_.size, np.int64); qy = np.zeros(h_.size, np.int64); qx = np.zeros(h_.size, np.int64)
            S.alone_window(fiy, fix, h_.astype(np.int64), hy, hx, hY, hl, *targs, F.PIX, WIN2, nwin, sig, qa, qj, qy, qx)
            res[mode + "_w1_qa3"] = fqa[sel]; res[mode + "_w1_qa6"] = qa
            res[mode + "_w1_off6"] = F.PIX * np.hypot(qy, qx)
            res[mode + "_w1_off3"] = F.PIX * np.hypot(st[f"{mode}_f_q_alone_dy"][sel], st[f"{mode}_f_q_alone_dx"][sel])
            del nwin
        del its, Xt, Xcn
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return unit, res["t"], S.peak_rss_gb()


if __name__ == "__main__":
    NPROC = int(sys.argv[1])
    os.makedirs(OUTD, exist_ok=True)
    t00 = time.time()
    units = [("dw", p) for p in P_DW] + [("r5", p) for p in P_R5]
    print(f"a04: {len(units)} tile units, {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=4) as pool:
        for u, tt, rss in pool.imap_unordered(work, units):
            print(f"  {u}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    DW = [dict(np.load(os.path.join(OUTD, f"a04_dw_{p:03d}.npz"))) for p in P_DW]
    R5 = [dict(np.load(os.path.join(OUTD, f"a04_r5_{p:03d}.npz"))) for p in P_R5]
    print(f"regeneration: max |q - stored q| over all regenerated final catalogues = "
          f"{max(max(r[m + '_regen_dq'] for m in S.MODES) for r in DW + R5):.1e}")
    for mode in S.MODES:
        print(f"\n==================== {mode}")
        tb = np.concatenate([r[mode + "_d3_table"] for r in DW]); hb = np.concatenate([r[mode + "_d3_hybrid"] for r in DW])
        nn = np.concatenate([r[mode + "_d3_nnear"] for r in DW])
        print(f"(D3) closure tSZ_map - sum over halos, in sigma, {tb.size} q > 5 detections of tiles {P_DW} (half the worst, half random):")
        print(f"     radial tables for every halo (the build's d_tab_sum): median |.| {np.median(np.abs(tb)):.1e}, max {np.abs(tb).max():.1e}")
        for b, R in enumerate(R_EXACT):
            print(f"     exact lattice sums within {R:4.0f}' ({np.median(nn[:, b]):.0f} halos, median), tables beyond: median |.| "
                  f"{np.median(np.abs(hb[:, b])):.1e}, max {np.abs(hb[:, b]).max():.1e}")
        qa3 = np.concatenate([r[mode + "_w1_qa3"] for r in DW]); qa6 = np.concatenate([r[mode + "_w1_qa6"] for r in DW])
        o3 = np.concatenate([r[mode + "_w1_off3"] for r in DW]); o6 = np.concatenate([r[mode + "_w1_off6"] for r in DW])
        m5 = qa6 > 5
        print(f"(W1) isolated search, {qa3.size} halos with q_alone(+-3') > 4.5 on tiles {P_DW}:")
        print(f"     +-6' maximum higher than the +-3' one: {np.mean(qa6 > qa3 + 1e-9) * 100:.2f}% (max increase {np.max(qa6 - qa3):.3f}); "
              f"q_alone crosses 5 because of the larger window: {np.sum((qa6 > 5) & (qa3 <= 5))} of {np.sum(qa3 > 5)} above 5")
        print(f"     q_alone(+-6') > 5: maximum within 1' of the halo for {100 * np.mean(o6[m5] <= 1.0 + 1e-9):.2f}%, within 2' for "
              f"{100 * np.mean(o6[m5] <= 2.0 + 1e-9):.2f}% (+-3' window: {100 * np.mean(o3[qa3 > 5] <= 1.0 + 1e-9):.2f}% within 1')")
        th = F.THETAS15
        sp = np.array([r[mode + "_std_pix"] for r in R5]); sa = np.array([r[mode + "_std_ana"] for r in R5])
        r1 = np.array([r[mode + "_r1"] for r in R5])
        print(f"(R5) width of the CMB + noise part of the finder's S/N maps after the iteration, {len(R5)} tiles, at "
              + " / ".join(f"{t:.2f}'" for t in th[[0, 3, 6, 9, 12, 14]]))
        print("     pixel standard deviation (realised)      : " + " / ".join(f"{v:.4f}" for v in sp.mean(0)[[0, 3, 6, 9, 12, 14]])
              + f"   [range over theta {sp.mean(0).min():.4f}-{sp.mean(0).max():.4f}]")
        print("     analytic expectation (the build's R5)    : " + " / ".join(f"{v:.4f}" for v in sa.mean(0)[[0, 3, 6, 9, 12, 14]])
              + f"   [range {sa.mean(0).min():.4f}-{sa.mean(0).max():.4f}]")
        print("     1 / r_after                               : " + " / ".join(f"{v:.4f}" for v in (1 / r1).mean(0)[[0, 3, 6, 9, 12, 14]]))
        print(f"     realised / (1 / r_after) - 1: {np.abs(sp * r1 - 1).mean(0).max():.4f} (largest over theta of the tile mean |.|)")
        ti0 = np.array([r[mode + "_tie_it0"] for r in R5]); ti1 = np.array([r[mode + "_tie_it1"] for r in R5])
        print(f"(tie) modes exactly on a bin edge: {R5[0]['n_tie']} of {R5[0]['n_modes']} in-band rfft modes "
              f"({100 * R5[0]['n_tie'] / R5[0]['n_modes']:.2f}%); sigma(SZiFi's lower-bin rule) / sigma(build) - 1 over "
              f"theta and tiles: iteration 0 {100 * (ti0 - 1).min():+.3f}% to {100 * (ti0 - 1).max():+.3f}%, iteration 1 "
              f"{100 * (ti1 - 1).min():+.3f}% to {100 * (ti1 - 1).max():+.3f}%")
    print(f"\ndone [{time.time() - t00:.0f} s]")
