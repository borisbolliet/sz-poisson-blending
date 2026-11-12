"""f02 (step-3b verification, lens 'fits', task a): the finder's data vector recomputed from the build's per-halo arrays
with my own binning, my own interpolation of the build's kernel tables and my own estimator of the pooled correction.

Per setup (headline prod_none and prod_fof10 on 1,548 tiles; szifi3000_none and prod_none_q2 on the 324 Q2 tiles) and
noise level, q_cut 5 and 6, step 2's 16 z x 9 q_obs bins (z edges and geometric q edges written out here):
  N_det(z, q)  credited halos (finite stored q_obs of the merge rule), true z, binned by my code;
  N_exp(z, q)  sum over halos of K(e_lo) - K(e_hi) at each halo's (theta500, q'), K = the build's kernel table with
               (i) 'bilin' (the build's scheme: linear in q' and in ln theta) and (ii) 'alt' (PCHIP in q', cubic Lagrange
               in ln theta); zero below the credit limit q' = 2;
  N_iso(z, q)  the stored isolated-copy q_obs;
  R = N_det / N_exp, R_p = N_det / N_iso (1 where the denominator < 1, as the build);
  R_s (MY estimator): top-hat cells in (ln theta500, q') -- theta cell boundaries AT the 25 kernel nodes (26 cells, open
       ends), q' boundaries 1.75, 2, 2.25 .. 10 (0.25), .. 20 (0.5), .. 40 (1), .. 80 (2.5), .. 120 (5), 140 .. 400, inf
       (105 cells) -- dK_c(e) = sum_{h in c} [1(q_obs > e) - 1(q_iso > e)] / n_c (n_c = all halos of the table in c);
       R_s(z, b) = 1 + sum_c I_c(z) [dK_c(e_b) - dK_c(e_b+1)] / mu(z, b), with I_c(z) the full-sky count of model halos in
       cell c (my count model at the truth, lognormal scatter exact) and mu my model's counts with the kernel ('alt').
Checks: my sigma_Y against the stored q_true; q' recomputed from Y500, theta500 and the tile's sigma_finder against the
stored q'; credited halo counts; no credited halo below q' = 2; my model mu against the build's mu; N_exp of the tiles
against my model x area (K12-like).  Tile bootstrap: NB resamples (my own RNG).  Compared bin by bin and in total with
the build's datavec_*.npz.  Output: out/dv_SETUP_MODE.npz.
"""
import os, sys, time, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import f00lib as L
import numpy as np

NB = 2000
SETUPS = {"prod_none": ("prod", "all", "none"), "prod_fof10": ("prod", "all", "fof10"),
          "szifi3000_none": ("szifi3000", "q2", "none"), "prod_none_q2": ("prod", "q2", "none")}
ZC = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
QCELL = np.round(np.concatenate([[1.75], np.arange(2.0, 10.0, 0.25), np.arange(10.0, 20.0, 0.5), np.arange(20.0, 40.0, 1.0),
                                 np.arange(40.0, 80.0, 2.5), np.arange(80.0, 120.0, 5.0), [120.0, 140.0, 160.0, 200.0, 250.0, 300.0, 400.0]]), 6)
QCELL_E = np.concatenate([[0.0], QCELL, [np.inf]])          # 106 edges -> 105 q' cells (cell 0 = q' < 1.75)
AREA_SKY = 4 * np.pi * (180 / np.pi) ** 2


def zbin(z):
    return np.searchsorted(L.ZED, z, side="right") - 1


def qbin(q, qcut):
    return np.searchsorted(L.qbins(qcut), q, side="right") - 1


def hist_tiles(t, z, q, qcut, nt):
    iz, iq = zbin(z), qbin(q, qcut)
    ok = (iz >= 0) & (iz < 16) & (iq >= 0) & (iq < 9)
    return np.bincount((t[ok] * 16 + iz[ok]) * 9 + iq[ok], minlength=nt * 144).reshape(nt, 16, 9).astype(float)


def exp_tiles(t, z, Kh, ei, nt):
    """N_exp per tile and bin from K at the 19 edges (Kh: (n, 19)); ei = edge indices of the 9 lower edges."""
    Ke = Kh[:, ei]
    D = Ke - np.concatenate([Ke[:, 1:], np.zeros((Ke.shape[0], 1))], 1)
    iz = zbin(z); ok = (iz >= 0) & (iz < 16)
    out = np.zeros((nt * 16, 9))
    idx = t[ok] * 16 + iz[ok]
    for b in range(9):
        out[:, b] = np.bincount(idx, weights=D[ok, b], minlength=nt * 16)
    return out.reshape(nt, 16, 9)


def ratio(num, den):
    return np.where(den >= 1.0, num / np.where(den > 0, den, 1.0), 1.0)


def boot_weights(nt, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nt, (NB, nt))
    W = np.zeros((NB, nt), np.float32)
    for b in range(NB):
        W[b] = np.bincount(idx[b], minlength=nt)
    return W


def cells(th, qp, lth_edges, qe):
    ci = np.searchsorted(lth_edges, np.log(th), side="right")              # 0 .. len(edges)
    ck = np.clip(np.searchsorted(qe, qp, side="right") - 1, 0, qe.size - 2)
    return ci * (qe.size - 1) + ck


def main():
    t00 = time.time()
    summary = {}
    for mode in L.MODES:
        cacheH = {}
        for setup, (variant, tiles, merge) in SETUPS.items():
            t0 = time.time()
            key = (variant, "all" if variant == "prod" else "q2")
            if key not in cacheH:
                H = L.load_halos(mode, variant, key[1], merges=("none", "fof10"))
                # ---------------- checks of the stored units
                sY = L.sigma_Y(mode, H["th"])
                rq = H["Y"] / sY / H["qt"] - 1
                r15 = H["sig1"] / L.sigma_Y(mode, L.THETAS15)[None, :]
                rt = np.empty(H["t"].size)
                for tt in range(H["nt"]):                      # r(theta) of each halo's tile
                    sl = slice(np.searchsorted(H["t"], tt), np.searchsorted(H["t"], tt + 1))
                    rt[sl] = L.r_interp(H["th"][sl], r15[tt])
                assert np.all(np.diff(H["t"]) >= 0)
                rp = H["Y"] / (rt * sY) / H["qp"] - 1
                print(f"==================== {mode} / {variant} ({key[1]} tiles: {H['nt']}; {H['t'].size} halos with q_true >= 1.5)", flush=True)
                print(f"  my sigma_Y vs the stored q_true: max |Y/sigma_Y/q_true - 1| = {np.abs(rq).max():.2e}; q' from Y500, theta500 and the "
                      f"tile's sigma_finder vs the stored q': max |rel| = {np.abs(rp).max():.2e} (median {np.median(np.abs(rp)):.1e})", flush=True)
                for m in ("none", "fof10"):
                    cr = np.isfinite(H["qo_" + m])
                    q2 = H["t"] < 324
                    print(f"  credited halos (merge {m}): Q2 tiles {int((cr & q2).sum())}"
                          + (f", all tiles {int(cr.sum())}" if H["nt"] > 324 else "")
                          + f"; credited with q' < 2: {int((cr & (H['qp'] < 2)).sum())}; min q' of a credited halo {H['qp'][cr].min():.3f}", flush=True)
                print(f"  isolated copies credited with q' < 2: {int((np.isfinite(H['qiso']) & (H['qp'] < 2)).sum())}; halos with q' >= 1.9 "
                      f"without an isolated copy (Q2 tiles): {int(((~H['iso_seen']) & (H['qp'] >= 1.9)).sum())}", flush=True)
                # kernel at the halos (both interpolations), all 19 edges, q' >= 2 only -> N_exp per tile and bin
                Kt = L.KTable(os.path.join(L.D3B, f"kernel_{variant}_{mode}.npz"))
                m2 = H["qp"] >= L.CRED
                NeA = {}
                for how in ("bilin", "alt"):
                    tk = time.time()
                    Kh = np.zeros((H["t"].size, 19), np.float32)
                    Kh[m2] = Kt.at(H["th"][m2], H["qp"][m2], np.arange(19), how=how)
                    for qc in (5.0, 6.0):
                        NeA[(how, qc)] = exp_tiles(H["t"], H["z"], Kh.astype(float), L.edge_idx(qc), H["nt"])
                    del Kh
                    print(f"  kernel at the halos ({how}): {time.time() - tk:.0f} s", flush=True)
                for k_ in ("Y", "qt", "iso_seen"):
                    H.pop(k_, None)
                del sY, rq, rt, rp
                cacheH.clear()
                cacheH[key] = (H, Kt, NeA)
            H, Kt, NeA = cacheH[key]
            if tiles == "q2":
                sel = H["t"] < 324
            else:
                sel = np.ones(H["t"].size, bool)
            nt = 324 if tiles == "q2" else H["nt"]
            t_, z_, th_, qp_ = H["t"][sel], H["z"][sel], H["th"][sel], H["qp"][sel]
            qo_, qi_ = H["qo_" + merge][sel], H["qiso"][sel]
            print(f"  ---- setup {setup}: {nt} tiles, merge {merge}", flush=True)
            # ---------------- R_s cells (my top-hat cells)
            lthe = Kt.lth                                                     # boundaries at the kernel nodes
            nci, nqc = lthe.size + 1, QCELL_E.size - 1
            ncell = nci * nqc
            c_ = cells(th_, qp_, lthe, QCELL_E)
            dmat = (qo_[:, None] > L.EDGES[None, :]).astype(np.int8) - (qi_[:, None] > L.EDGES[None, :]).astype(np.int8)
            nzr = np.flatnonzero(np.any(dmat != 0, 1))
            S = np.zeros((nt * ncell, 19), np.float32)
            fl = t_[nzr] * ncell + c_[nzr]
            for e in range(19):
                S[:, e] = np.bincount(fl, weights=dmat[nzr, e].astype(float), minlength=nt * ncell)
            S = S.reshape(nt, ncell, 19)
            n_ = np.bincount(t_ * ncell + c_, minlength=nt * ncell).reshape(nt, ncell).astype(np.float32)
            Stot = S.sum(0, dtype=np.float64); ntot = n_.sum(0, dtype=np.float64)
            dK = np.where(ntot[:, None] > 0, Stot / np.maximum(ntot, 1)[:, None], 0.0)
            e5 = int(np.argmin(np.abs(L.EDGES - 5.0)))
            print(f"  halos whose blended and isolated outcome differ at e = 5: {int(np.abs(dmat[:, e5]).sum())} of "
                  f"{int(((qo_ > 5) | (qi_ > 5)).sum())} credited above 5 in either; cells {nci} x {nqc}", flush=True)
            # ---------------- bootstrap weights
            Wb = boot_weights(nt, 4242 + nt + (0 if mode == "baseline" else 1))
            res = dict(setup=setup, mode=mode, ntiles=nt)
            dv = np.load(os.path.join(L.D3B, f"datavec_{setup}_{mode}.npz"))
            r15bar = L.rbar(variant, mode)
            for qc in (5.0, 6.0):
                s = f"q{qc:.0f}"
                ei = L.edge_idx(qc)
                Nd = hist_tiles(t_, z_, qo_, qc, nt); Ni = hist_tiles(t_, z_, qi_, qc, nt)
                Ne = {how: NeA[(how, qc)][:nt] for how in ("bilin", "alt")}
                M = L.KModel(mode, qc, r15bar, Kt, how="alt")
                mu = M.counts(L.FID)
                Mb = L.KModel(mode, qc, r15bar, Kt, how="bilin")
                mub = Mb.counts(L.FID)
                I = M.cell_integrals(L.FID, lthe, QCELL_E)                       # (16, nci, nqc)
                I2 = I.reshape(16, ncell)

                def rs_of(dKx, mu_):
                    X = I2 @ dKx[..., ei] if dKx.ndim == 2 else np.einsum("zc,bce->bze", I2, dKx[..., ei])
                    D = X - np.concatenate([X[..., 1:], np.zeros(X.shape[:-1] + (1,))], -1)
                    return 1.0 + D / mu_
                Rs = rs_of(dK, mu)
                Rs_b = rs_of(dK, mub)                                            # the same dK with the bilinear model mu
                Rs_var = {}
                if setup in ("prod_none", "szifi3000_none"):
                    # cell-grid dependence of my estimator: theta and q' cells halved ('fine') or doubled ('coarse')
                    qfin = QCELL_E[1:-1]
                    for lab, lte, qe in (("fine", np.sort(np.concatenate([lthe, 0.5 * (lthe[1:] + lthe[:-1])])),
                                          np.concatenate([[0.0], np.sort(np.concatenate([qfin, 0.5 * (qfin[1:] + qfin[:-1])])), [np.inf]])),
                                         ("coarse", lthe[::2], np.concatenate([[0.0], qfin[::2], [np.inf]]))):
                        nq_ = qe.size - 1
                        cc = cells(th_, qp_, lte, qe)
                        Sv = np.zeros(((lte.size + 1) * nq_, 19))
                        for e in range(19):
                            Sv[:, e] = np.bincount(cc[nzr], weights=dmat[nzr, e].astype(float), minlength=Sv.shape[0])
                        nv = np.bincount(cc, minlength=Sv.shape[0]).astype(float)
                        dKv = np.where(nv[:, None] > 0, Sv / np.maximum(nv, 1)[:, None], 0.0)
                        Iv = M.cell_integrals(L.FID, lte, qe).reshape(16, -1)
                        Xv = Iv @ dKv[:, ei]
                        Dv = Xv - np.concatenate([Xv[:, 1:], np.zeros((16, 1))], 1)
                        Rs_var[lab] = 1.0 + Dv / mu
                Ndt, Net, Nit = Nd.sum(0), {h: v.sum(0) for h, v in Ne.items()}, Ni.sum(0)
                R = {h: ratio(Ndt, Net[h]) for h in Net}; Rp = ratio(Ndt, Nit)
                # bootstrap
                RB = {h: np.empty((NB, 16, 9), np.float32) for h in Ne}; RpB = np.empty((NB, 16, 9), np.float32)
                RsB = np.empty((NB, 16, 9), np.float32)
                Ndf, Nif = Nd.reshape(nt, -1), Ni.reshape(nt, -1)
                Nef = {h: v.reshape(nt, -1) for h, v in Ne.items()}
                bd, bi = Wb @ Ndf, Wb @ Nif
                for h in Ne:
                    RB[h][:] = ratio(bd, Wb @ Nef[h]).reshape(NB, 16, 9)
                RpB[:] = ratio(bd, bi).reshape(NB, 16, 9)
                Sf = S.reshape(nt, -1)
                for a in range(0, NB, 100):
                    w = Wb[a:a + 100]                                          # float32 (exact: small integers)
                    Sb = (w @ Sf).astype(np.float64).reshape(-1, ncell, 19); nb_ = (w @ n_).astype(np.float64)
                    dKb = np.where(nb_[..., None] > 0, Sb / np.maximum(nb_, 1)[..., None], 0.0)
                    RsB[a:a + 100] = rs_of(dKb, mu)
                tot = lambda num, den: (num.sum() / den.sum() - 1, ((Wb @ num.reshape(nt, -1).sum(1)) / (Wb @ den.reshape(nt, -1).sum(1)) - 1).std())
                wN = Net["alt"]
                tws = lambda Rx, w_: np.sum(w_ * (Rx - 1)) / w_.sum()
                for lab, v in Rs_var.items():
                    print(f"     R_s with {lab} cells: model-weighted q > {qc:.0f} total {100 * (np.sum(mu * v) / mu.sum() - 1):+.3f}% (standard cells "
                          f"{100 * (np.sum(mu * Rs) / mu.sum() - 1):+.3f}%); max |dR| over bins with N_exp >= 10: "
                          f"{np.abs(v - Rs)[Net['bilin'] >= 10].max():.4f}", flush=True)
                res[s] = dict(Ndet=Ndt, Nexp_bilin=Net["bilin"], Nexp_alt=Net["alt"], Niso=Nit, mu=mu, mu_bilin=mub,
                              **{f"Rs_{k}": v for k, v in Rs_var.items()},
                              R_bilin=R["bilin"], R_alt=R["alt"], Rp=Rp, Rs=Rs, Rs_mubilin=Rs_b, Rboot_bilin=RB["bilin"], Rboot_alt=RB["alt"],
                              Rpboot=RpB, Rsboot=RsB, dK=dK, I=I, Nexp_tile=Ne["alt"].sum((1, 2)), Ndet_tile=Nd.sum((1, 2)))
                # ---------------- comparisons with the build
                bd_ = {k: dv[f"{k}_{s}"] for k in ("Ndet", "Nexp", "Niso", "R", "Rp", "Rs", "mu", "R_err", "Rp_err", "Rs_err")}
                ok = bd_["Nexp"] >= 10
                print(f"  [{setup} q>{qc:.0f}] N_det {Ndt.sum():.0f} (build {bd_['Ndet'].sum():.0f}; max |bin diff| {np.abs(Ndt - bd_['Ndet']).max():.0f}); "
                      f"N_iso {Nit.sum():.0f} (build {bd_['Niso'].sum():.0f}; max |bin diff| {np.abs(Nit - bd_['Niso']).max():.0f}); N_exp bilin "
                      f"{Net['bilin'].sum():.1f} / alt {Net['alt'].sum():.1f} (build {bd_['Nexp'].sum():.1f}); max |rel bin diff| (N_exp >= 10): bilin "
                      f"{np.abs(Net['bilin'] / bd_['Nexp'] - 1)[ok].max():.2e}, alt {np.abs(Net['alt'] / bd_['Nexp'] - 1)[ok].max():.2e}", flush=True)
                print(f"     my model mu (alt) vs build mu: total {mu.sum() / bd_['mu'].sum() - 1:+.2e}, per bin with mu > 10: max |rel| "
                      f"{np.abs(mu / bd_['mu'] - 1)[bd_['mu'] > 10].max():.2e}; bilinear model: total {mub.sum() / bd_['mu'].sum() - 1:+.2e}, "
                      f"z > 0.1 max |rel| {np.abs(mub / bd_['mu'] - 1)[1:][bd_['mu'][1:] > 10].max():.2e}, z < 0.1 total {mub[0].sum() / bd_['mu'][0].sum() - 1:+.2e}", flush=True)
                # K12-like: tiles' N_exp vs model x area (tiles of 100 deg^2)
                area = nt * 100.0 / AREA_SKY
                pz = Net["alt"].sum(1); mz = mu.sum(1) * area
                chi = np.sum((pz - mz) ** 2 / mz)
                print(f"     N_exp(tiles) / (my model mu x area) - 1 = {pz.sum() / mz.sum() - 1:+.4%} ({(pz.sum() - mz.sum()) / np.sqrt(mz.sum()):+.2f} Poisson sigma); "
                      f"per z bin chi^2 {chi:.1f} for 16 bins", flush=True)
                for lab, A, Bv, eb in (("R (bilin)", R["bilin"], bd_["R"], bd_["R_err"]), ("R (alt)", R["alt"], bd_["R"], bd_["R_err"]),
                                       ("R_p", Rp, bd_["Rp"], bd_["Rp_err"]), ("R_s (my cells)", Rs, bd_["Rs"], bd_["Rs_err"])):
                    pull = ((A - Bv) / np.where(eb > 0, eb, 1))[ok]
                    print(f"     {lab:15s} vs build, {int(ok.sum())} bins with N_exp >= 10: max |diff| {np.abs(A - Bv)[ok].max():.4f}, "
                          f"max |diff|/build error {np.abs(pull).max():.2f}, rms {np.sqrt(np.mean(pull ** 2)):.3f}", flush=True)
                if qc == 5.0:
                    T = {}
                    for lab, num, den in (("R bilin", Nd, Ne["bilin"]), ("R alt", Nd, Ne["alt"]), ("R_p", Nd, Ni), ("N_iso/N_exp", Ni, Ne["bilin"])):
                        T[lab] = tot(num, den)
                    Ts = tws(Rs, wN); eTs = np.array([tws(RsB[b], wN) for b in range(NB)]).std()
                    Tm = tws(Rs, mu); eTm = np.array([tws(RsB[b], mu) for b in range(NB)]).std()
                    print(f"     q > 5 totals [%]: " + "; ".join(f"{k} {100 * v[0]:+.2f} +- {100 * v[1]:.2f}" for k, v in T.items())
                          + f"; R_s (N_exp weights) {100 * Ts:+.2f} +- {100 * eTs:.2f}; R_s (model weights) {100 * Tm:+.2f} +- {100 * eTm:.2f}", flush=True)
                    bT = tws(bd_["Rs"], bd_["mu"])
                    print(f"     build R_s (model weights) {100 * bT:+.2f}%; difference mine - build {100 * (Tm - bT):+.3f} pp", flush=True)
                    zr = []
                    for a, b in zip(ZC[:-1], ZC[1:]):
                        kz = (L.ZED[:-1] >= a - 1e-9) & (L.ZED[1:] <= b + 1e-9)
                        w_ = wN[kz]
                        v = tws(Rs[kz], w_); e_ = np.array([tws(RsB[bb][kz], w_) for bb in range(0, NB, 4)]).std()
                        vb = tws(bd_["Rs"][kz], bd_["Nexp"][kz])
                        zr.append(f"{100 * v:+.2f}+-{100 * e_:.2f} (build {100 * vb:+.2f})")
                    print("     R_s by coarse z (0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3) [%]: " + ", ".join(zr), flush=True)
                    qr = []
                    for k in range(9):
                        w_ = wN[:, k]
                        if w_.sum() <= 0:
                            continue
                        v = np.sum(w_ * (Rs[:, k] - 1)) / w_.sum(); e_ = np.array([np.sum(w_ * (RsB[bb][:, k] - 1)) / w_.sum() for bb in range(0, NB, 4)]).std()
                        vb = np.sum(bd_["Nexp"][:, k] * (bd_["Rs"][:, k] - 1)) / bd_["Nexp"][:, k].sum()
                        qr.append(f"{100 * v:+.2f}+-{100 * e_:.2f} (build {100 * vb:+.2f})")
                    print("     R_s by q bin [%]: " + ", ".join(qr), flush=True)
                    # bootstrap errors per bin: mine vs build
                    for lab, AB, eb in (("R", RB["bilin"], bd_["R_err"]), ("R_p", RpB, bd_["Rp_err"]), ("R_s", RsB, bd_["Rs_err"])):
                        r_ = (AB.std(0) / np.where(eb > 0, eb, 1))[ok]
                        print(f"     bootstrap error per bin, mine / build ({lab}): median {np.median(r_):.3f}, 5-95% {np.percentile(r_, 5):.3f}-{np.percentile(r_, 95):.3f}", flush=True)
                    # estimator agreement (K16-like): R_p vs R_s, R vs R_s with bootstrap errors of the difference (paired)
                    for lab, A, AB in (("R_p", Rp, RpB), ("R (bilin)", R["bilin"], RB["bilin"])):
                        dd = A - Rs; ddB = AB - RsB
                        sd = ddB.std(0); ok2 = ok & (sd > 0)
                        print(f"     {lab} vs my R_s: chi^2 (diagonal, bootstrap errors of the difference) {np.sum((dd[ok2] / sd[ok2]) ** 2):.1f} "
                              f"for {int(ok2.sum())} bins", flush=True)
                    summary[f"{setup}_{mode}"] = dict({k: [float(v[0]), float(v[1])] for k, v in T.items()}, Rs_N=[float(Ts), float(eTs)],
                                                      Rs_mu=[float(Tm), float(eTm)], Rs_build=float(bT))
            np.savez(os.path.join(L.OUT, f"dv_{setup}_{mode}.npz"),
                     **{f"{k}_{s}": v for s in ("q5", "q6") for k, v in res[s].items()}, ntiles=nt, QCELL_E=QCELL_E, lthe=lthe)
            print(f"     [{time.time() - t0:.0f} s; peak RSS {L.rss_gb():.2f} GB]", flush=True)
            del S, Sf, n_, dmat, Wb
        cacheH.clear()
    json.dump(summary, open(os.path.join(L.OUT, "f02_summary.json"), "w"), indent=1)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {L.rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
