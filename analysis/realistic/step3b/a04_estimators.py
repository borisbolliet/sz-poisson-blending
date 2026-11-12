"""Step 3b arbitration / a04: the pooled estimator R_s against the unsmoothed paired estimator R_p, and R_s's grid.

The fits verifier finds that R_s depends on its discretisation (top-hat cells vs the build's tents: +0.04 sigma in
config d at SO40) and that for SZiFi's defaults at baseline R_s and R_p disagree (paired +0.21 +- 0.08 sigma).  Here:
  (1) linearised S8 bias (configs a, d, e; SO40 and full sky; q_cut 5) of R_p and of the build's stored R_s, and the paired
      difference R_s - R_p over the build's own 2,000 tile-bootstrap resamples (b07), for every setup;
  (2) R_s rebuilt from the per-halo data on other tent grids (s3bcosmo's construction, generalised to any theta and q'
      nodes): the build's grid (must reproduce the stored R_s), q' nodes twice as fine, theta nodes twice as fine, both,
      and q' nodes twice as coarse; the bias shift against the build's grid and against R_p with a 20-group delete-one
      jackknife over tiles (paired: the same halos in every grid).
If R_s converges to R_p as the grid is refined, the difference is smoothing; if not, it is noise.
Usage: python a04_estimators.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np
import s3blib as B
import s3bcosmo as SC
C2, CL = SC.C2, SC.CL
NG = 20
QC = 5.0
SETUPS = [("prod_none", "prod", "all", "none"), ("szifi3000_none", "szifi3000", "q2", "none"),
          ("szifi12000_none", "szifi12000", "q2", "none"), ("prod_fof10", "prod", "all", "fof10")]


def fine(nodes, log=False):
    """Insert midpoints (geometric if log)."""
    m = np.sqrt(nodes[1:] * nodes[:-1]) if log else 0.5 * (nodes[1:] + nodes[:-1])
    return np.sort(np.concatenate([nodes, m]))


QF = np.round(np.concatenate([fine(SC.QDK[SC.QDK < 20.0]), SC.QDK[SC.QDK >= 20.0]]), 6)
QF = np.unique(QF)
GRIDS = {"build (25 theta x 108 q')": (B.TH_K, SC.QDK), "q' x2 finer": (B.TH_K, QF), "theta x2 finer": (fine(B.TH_K, True), SC.QDK),
         "both finer": (fine(B.TH_K, True), QF), "q' x2 coarser": (B.TH_K, SC.QDK[::2])}


def weights(theta, qp, thn, qn):
    nK, nq = thn.size, qn.size
    dl = np.log(thn[1] / thn[0])
    x = np.clip((np.log(theta) - np.log(thn[0])) / dl, 0, nK - 1.000001)
    i0 = x.astype(np.int64); wt = x - i0
    qq = np.clip(qp, qn[0], qn[-1])
    k0 = np.clip(np.searchsorted(qn, qq, side="right") - 1, 0, nq - 2)
    wq = np.clip((qq - qn[k0]) / (qn[k0 + 1] - qn[k0]), 0.0, 1.0)
    idx = np.stack([i0 * nq + k0, (i0 + 1) * nq + k0, i0 * nq + k0 + 1, (i0 + 1) * nq + k0 + 1], 1)
    w = np.stack([(1 - wt) * (1 - wq), wt * (1 - wq), (1 - wt) * wq, wt * wq], 1)
    return idx, w


def integrals(M, truth, thn, qn):
    """s3bcosmo.cell_integrals with theta nodes thn (log-uniform) instead of the kernel nodes."""
    lnN, th500, lnq0 = M.nodes(truth)
    Wn = np.exp(lnN) * M.wM[None, :]
    nK, nq = thn.size, qn.size
    fa = np.clip((np.log(th500) - np.log(thn[0])) / np.log(thn[1] / thn[0]), 0, nK - 1.000001)
    ia = fa.astype(np.int64); wa = fa - ia
    f = np.clip((lnq0 - M.lq[0]) / M.h, 0, len(M.lq) - 1.000001); i0 = f.astype(np.int64); w = f - i0
    q = np.exp(M.lq)
    out = np.zeros((M.nzb, nK, nq))
    eye = np.eye(nq)
    rows = np.arange(Wn.shape[0])[:, None].repeat(Wn.shape[1], 1)
    for k in range(nq):
        Ts = C2.smooth_lnq(np.interp(q, qn, eye[k])[None, :], truth[4], M.h)[0]
        v = Wn * (Ts[i0] * (1 - w) + Ts[i0 + 1] * w)
        Aa = np.zeros((v.shape[0], nK))
        np.add.at(Aa, (rows, ia), v * (1 - wa)); np.add.at(Aa, (rows, ia + 1), v * wa)
        out[:, :, k] = M.Wz @ Aa
    return out


def lin(Ls, R):
    return {k: float(SC.lin_bias(L, R)[0][0]) for k, L in Ls.items()}


if __name__ == "__main__":
    t00 = time.time()
    zed, qed = CL.default_bins(QC)
    sel = [int(np.argmin(np.abs(B.EDGES - e))) for e in qed[:-1]]
    for setup, variant, which, merge in SETUPS:
        for mode in ("baseline", "goal"):
            t0 = time.time()
            dv = np.load(os.path.join(A.OUT, f"datavec_{setup}_{mode}.npz"))
            M = SC.model(variant, mode, QC)
            mu = M.counts(SC.FID)
            Ls = {(sv, cn): SC.linear_parts(M, SC.FID, dv["Rs_q5"], cn, fs) for sv, fs in SC.SURVEYS.items() for cn in ("a", "d", "e")}
            # (1) stored estimators, paired bootstrap
            bR = lin(Ls, dv["R_q5"]); bRp = lin(Ls, dv["Rp_q5"]); bRs = lin(Ls, dv["Rs_q5"])
            print(f"==== {setup} {mode} ({int(dv['ntiles'])} tiles), q_cut 5: linearised S8 bias [sigma]; R_s - R_p paired over the build's "
                  f"2,000 tile-bootstrap resamples", flush=True)
            RsB, RpB = dv["Rsboot_q5"].astype(float), dv["Rpboot_q5"].astype(float)
            for (sv, cn), L in Ls.items():
                a = SC.lin_bias(L, RsB)[0]; b = SC.lin_bias(L, RpB)[0]
                d = a - b
                print(f"   {sv:4s} {cn}: R {bR[(sv, cn)]:+.3f}, R_p {bRp[(sv, cn)]:+.3f} +- {b.std():.3f}, R_s {bRs[(sv, cn)]:+.3f} +- {a.std():.3f}; "
                      f"R_s - R_p {bRs[(sv, cn)] - bRp[(sv, cn)]:+.3f} +- {d.std():.3f} ({(bRs[(sv, cn)] - bRp[(sv, cn)]) / d.std():+.1f} sigma)", flush=True)
            del RsB, RpB
            # (2) R_s on other grids, from the per-halo data, 20-group jackknife
            D = A.load_build(mode, which, variant)
            nt = D["ntiles"]
            grp = (np.arange(nt) * NG // nt)[D["t"]]
            dd = ((D["qo_" + merge][:, None] > B.EDGES[sel][None, :]).astype(np.float32)
                  - (D["qiso"][:, None] > B.EDGES[sel][None, :]).astype(np.float32))
            # R_p per group (bins of q_cut 5)
            iz = np.searchsorted(zed, D["z"], side="right") - 1
            okz = (iz >= 0) & (iz < zed.size - 1)

            def hist(q):
                iq = np.searchsorted(qed, q, side="right") - 1
                ok = okz & (iq >= 0) & (iq < qed.size - 1)
                idx = (grp[ok] * (zed.size - 1) + iz[ok]) * (qed.size - 1) + iq[ok]
                return np.bincount(idx, minlength=NG * (zed.size - 1) * (qed.size - 1)).reshape(NG, zed.size - 1, qed.size - 1).astype(float)
            Hd, Hi = hist(D["qo_" + merge]), hist(D["qiso"])
            ratio = lambda a, b: np.where(b >= 1, a / np.maximum(b, 1e-300), 1.0)
            Rp_full = ratio(Hd.sum(0), Hi.sum(0))
            Rp_jk = [ratio(Hd.sum(0) - Hd[g], Hi.sum(0) - Hi[g]) for g in range(NG)]
            res = {}
            for gname, (thn, qn) in GRIDS.items():
                idx, w = weights(D["th"], D["qp"], thn, qn)
                ncell = thn.size * qn.size
                flat = (grp[:, None] * ncell + idx).ravel()
                Wg = np.bincount(flat, weights=w.ravel(), minlength=NG * ncell).reshape(NG, ncell)
                nz = np.flatnonzero(np.any(dd != 0, 1))
                fl = (grp[nz, None] * ncell + idx[nz]).ravel()
                Sg = np.stack([np.bincount(fl, weights=(w[nz] * dd[nz, e][:, None]).ravel(), minlength=NG * ncell).reshape(NG, ncell)
                               for e in range(len(sel))], -1)
                I = integrals(M, SC.FID, thn, qn)

                def rs(S, W):
                    dK = np.where(W[:, None] > 0, S / np.where(W > 0, W, 1.0)[:, None], 0.0).reshape(thn.size, qn.size, len(sel))
                    X = np.einsum("zik,ike->ze", I, dK)
                    Dx = X - np.concatenate([X[:, 1:], np.zeros((X.shape[0], 1))], 1)
                    return 1.0 + Dx / mu
                St, Wt = Sg.sum(0), Wg.sum(0)
                res[gname] = (rs(St, Wt), [rs(St - Sg[g], Wt - Wg[g]) for g in range(NG)])
                del Sg, Wg, idx, w, flat
            b0 = res["build (25 theta x 108 q')"][0]
            print(f"   rebuilt R_s on the build's grid vs the stored R_s: max |diff| over bins with mu > 10: {np.abs(b0 - dv['Rs_q5'])[mu > 10].max():.2e}", flush=True)
            jk = lambda vals: np.sqrt((NG - 1) / NG * np.sum((np.array(vals) - np.mean(vals)) ** 2))
            for (sv, cn), L in Ls.items():
                if cn == "a" and sv == "full":
                    continue
                bp = float(SC.lin_bias(L, Rp_full)[0][0]); bpj = [float(SC.lin_bias(L, r)[0][0]) for r in Rp_jk]
                parts = []
                for gname, (rf, rj) in res.items():
                    b = float(SC.lin_bias(L, rf)[0][0]); bj = [float(SC.lin_bias(L, r)[0][0]) for r in rj]
                    b0_ = float(SC.lin_bias(L, res["build (25 theta x 108 q')"][0])[0][0])
                    b0j = [float(SC.lin_bias(L, r)[0][0]) for r in res["build (25 theta x 108 q')"][1]]
                    parts.append(f"{gname}: {b:+.3f} (- build grid {b - b0_:+.3f} +- {jk(np.array(bj) - np.array(b0j)):.3f}; - R_p {b - bp:+.3f} +- "
                                 f"{jk(np.array(bj) - np.array(bpj)):.3f})")
                print(f"   {sv:4s} {cn}: R_p {bp:+.3f} +- {jk(bpj):.3f} (jackknife) | " + " | ".join(parts), flush=True)
            del D, dd
            print(f"   [{time.time() - t0:.0f} s; peak RSS {A.peak_rss_gb():.2f} GB]", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
