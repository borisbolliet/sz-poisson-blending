"""f07 (step-3b verification, lens 'fits'): the build's pooled estimator R_s re-implemented with my code (bilinear tents in
ln theta500 on the 25 kernel nodes and in q' on the build's node list, written out here), next to my top-hat estimator
of f02, on the SAME tile-bootstrap resamples (same seeds as f02), so that their difference can be judged against its
own paired scatter.

Tents: theta weight linear in ln theta between the kernel nodes (clamped at the ends); q' weight linear between the
nodes QDK (clamped to [0, 300]).  dK_node(e) = sum_h w_h,node [1(q_obs > e) - 1(q_iso > e)] / sum_h w_h,node (all halos
of the table).  Model side (my count model at the truth): I_node(z) = full-sky count integral of tent_node(theta500,
q'_true), the q' tent convolved with the lognormal scatter on my fine u grid (exact hat-function weights).
R_s(z, b) = 1 + sum_node I_node(z) [dK_node(e_b) - dK_node(e_b+1)] / mu(z, b).
Printed: tent R_s (mine) against the build's stored R_s, bin by bin and in total; linearised S8 bias (my model) with
my top-hat R_s, my tent R_s and the build's R_s; paired bootstrap std of (top-hat - tent).
Usage: f07_tents.py   Output: log, out/f07_SETUP_MODE.npz
"""
import os, sys, time, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f00lib as L
from scipy.signal import fftconvolve
import f02_datavec as F2

QDK = np.round(np.concatenate([np.arange(0.0, 2.0, 0.25), np.arange(2.0, 10.0, 0.25), np.arange(10.0, 20.0, 0.5),
                               np.arange(20.0, 40.0, 1.0), np.arange(40.0, 80.0, 2.5), np.arange(80.0, 120.0, 5.0),
                               [120.0, 150.0, 200.0, 300.0]]), 6)
SETS = {"prod_none": ("prod", "all", "none"), "szifi3000_none": ("szifi3000", "q2", "none"), "prod_fof10": ("prod", "all", "fof10")}


def tent_w(theta, qp, lth):
    nK, nq = lth.size, QDK.size
    x = np.clip((np.log(theta) - lth[0]) / (lth[1] - lth[0]), 0, nK - 1.000001)
    i0 = x.astype(np.int64); wt = x - i0
    qq = np.clip(qp, QDK[0], QDK[-1])
    k0 = np.clip(np.searchsorted(QDK, qq, side="right") - 1, 0, nq - 2)
    wq = np.clip((qq - QDK[k0]) / (QDK[k0 + 1] - QDK[k0]), 0.0, 1.0)
    idx = np.stack([i0 * nq + k0, (i0 + 1) * nq + k0, i0 * nq + k0 + 1, (i0 + 1) * nq + k0 + 1], 1)
    w = np.stack([(1 - wt) * (1 - wq), wt * (1 - wq), (1 - wt) * wq, wt * wq], 1)
    return idx, w


def tent_integrals(M, th, lth):
    """I[z bin, node] for the tents (theta node i, q' node k), node = i * nq + k."""
    lnN, u, lt = M.nodes(th)
    s = th[4]
    A = np.exp(lnN) * M.WM[None, :] * M.ZW[:, None]
    nK, nq = lth.size, QDK.size
    x = np.clip((lt - lth[0]) / (lth[1] - lth[0]), 0, nK - 1.000001)
    ia = x.astype(np.int64); wa = x - ia
    q = np.exp(M.U)
    w, K = L.hat_gauss_weights(float(s), M.H)
    f = np.clip((u - M.U[0]) / M.H, 0, M.U.size - 1.000001); i0 = f.astype(np.int64); wf = f - i0
    out = np.zeros((16, nK, nq))
    eye = np.eye(nq)
    for k in range(nq):
        T = np.interp(q, QDK, eye[k])
        Tp = np.concatenate([np.repeat(T[:1], K), T, np.repeat(T[-1:], K)])
        Ts = fftconvolve(Tp, w, mode="valid")
        v = A * (Ts[i0] * (1 - wf) + Ts[i0 + 1] * wf)                        # (nZN, nM)
        for zb in range(16):
            m = M.ZB == zb
            out[zb, :, k] = np.bincount(ia[m].ravel(), weights=(v[m] * (1 - wa[m])).ravel(), minlength=nK) + \
                np.bincount((ia[m] + 1).ravel(), weights=(v[m] * wa[m]).ravel(), minlength=nK)[:nK]
    return out.reshape(16, nK * nq)


def main():
    t00 = time.time()
    res = {}
    for mode in L.MODES:
        for setup, (variant, tiles, merge) in SETS.items():
            t0 = time.time()
            H = L.load_halos(mode, variant, tiles, merges=(merge,))
            nt = H["nt"]
            Kt = L.KTable(os.path.join(L.D3B, f"kernel_{variant}_{mode}.npz"))
            lth = Kt.lth
            qo, qi = H["qo_" + merge], H["qiso"]
            dmat = (qo[:, None] > L.EDGES[None, :]).astype(np.int8) - (qi[:, None] > L.EDGES[None, :]).astype(np.int8)
            nzr = np.flatnonzero(np.any(dmat != 0, 1))
            nnode = lth.size * QDK.size
            idx, w = tent_w(H["th"], H["qp"], lth)
            Wn = np.zeros(nt * nnode, np.float64)
            fl = (H["t"][:, None] * nnode + idx).ravel()
            Wn = np.bincount(fl, weights=w.ravel(), minlength=nt * nnode).reshape(nt, nnode).astype(np.float32)
            del fl
            S = np.zeros((nt, nnode, 19), np.float32)
            fz = (H["t"][nzr][:, None] * nnode + idx[nzr]).ravel()
            for e in range(19):
                S[:, :, e] = np.bincount(fz, weights=(w[nzr] * dmat[nzr, e][:, None]).ravel(), minlength=nt * nnode).reshape(nt, nnode)
            del idx, w, dmat
            dK = np.where(Wn.sum(0)[:, None] > 0, S.sum(0, dtype=np.float64) / np.maximum(Wn.sum(0, dtype=np.float64), 1e-300)[:, None], 0.0)
            Wb = F2.boot_weights(nt, 4242 + nt + (0 if mode == "baseline" else 1))                # f02's resamples
            mine = np.load(os.path.join(L.OUT, f"dv_{setup}_{mode}.npz"))
            bld = np.load(os.path.join(L.D3B, f"datavec_{setup}_{mode}.npz"))
            out = {}
            for qc in (5.0, 6.0):
                s = f"q{qc:.0f}"
                ei = L.edge_idx(qc)
                M = L.KModel(mode, qc, L.rbar(variant, mode), Kt, how="alt")
                mu = M.counts(L.FID)
                I = tent_integrals(M, L.FID, lth)

                def rs_of(dKx):
                    X = I @ dKx[..., ei] if dKx.ndim == 2 else np.einsum("zc,bce->bze", I, dKx[..., ei])
                    D = X - np.concatenate([X[..., 1:], np.zeros(X.shape[:-1] + (1,))], -1)
                    return 1.0 + D / mu
                Rt = rs_of(dK)
                RtB = np.empty((F2.NB, 16, 9), np.float32)
                Sf = S.reshape(nt, -1)
                for a in range(0, F2.NB, 100):
                    wb = Wb[a:a + 100]
                    Sb = (wb @ Sf).astype(np.float64).reshape(-1, nnode, 19); nb_ = (wb @ Wn).astype(np.float64)
                    dKb = np.where(nb_[..., None] > 0, Sb / np.maximum(nb_, 1e-300)[..., None], 0.0)
                    RtB[a:a + 100] = rs_of(dKb)
                Rb = bld[f"Rs_{s}"]; ok = bld[f"Nexp_{s}"] >= 10
                print(f"  [{setup} {mode} q>{qc:.0f}] tent R_s (mine) vs the build's R_s: max |diff| over {int(ok.sum())} bins with N_exp >= 10: "
                      f"{np.abs(Rt - Rb)[ok].max():.5f}; model-weighted total {100 * (np.sum(mu * Rt) / mu.sum() - 1):+.3f}% vs build "
                      f"{100 * (np.sum(bld[f'mu_{s}'] * Rb) / bld[f'mu_{s}'].sum() - 1):+.3f}%; bootstrap error per bin, mine / build: median "
                      f"{np.median((RtB.std(0) / np.where(bld[f'Rs_err_{s}'] > 0, bld[f'Rs_err_{s}'], 1))[ok]):.3f}", flush=True)
                out[s] = (Rt, RtB)
                # linearised S8 bias, paired comparison with my top-hat R_s
                Rh, RhB = mine[f"Rs_{s}"], mine[f"Rsboot_{s}"].astype(float)
                for sv, fs in L.SURV.items():
                    for cfg in ("a", "d", "e"):
                        N0 = M.counts(L.FID)
                        Fo = L.Fit(M, N0 * fs, cfg, L.FID, fs)
                        F, J, _ = Fo.fisher(Fo.truth[Fo.free], L.STEPS[Fo.free])
                        sig, sS8, g, C = Fo.sigmas(F)
                        A_ = np.linalg.solve(F, J.T)
                        lb = lambda R: ((np.asarray(R) - 1).reshape(-1, 144) @ A_.T) @ g / sS8
                        bh, bt, bb = lb(Rh)[0], lb(Rt)[0], lb(Rb)[0]
                        dh, dt = lb(RhB), lb(RtB.astype(float))
                        res[f"{setup}|{mode}|{s}|{sv}|{cfg}"] = dict(tophat=float(bh), tent=float(bt), build=float(bb), e_tophat=float(dh.std()),
                                                                    e_tent=float(dt.std()), e_diff=float((dh - dt).std()), corr=float(np.corrcoef(dh, dt)[0, 1]))
                        if cfg != "a" or sv == "SO40":
                            print(f"     {sv:4s} {cfg}: linearised S8 bias: my top-hat R_s {bh:+.3f} +- {dh.std():.3f}; my tent R_s {bt:+.3f} +- {dt.std():.3f}; "
                                  f"build R_s {bb:+.3f} | top-hat - tent {bh - bt:+.3f} +- {(dh - dt).std():.3f} (paired; corr {np.corrcoef(dh, dt)[0, 1]:.2f})", flush=True)
            np.savez(os.path.join(L.OUT, f"f07_{setup}_{mode}.npz"), Rt_q5=out["q5"][0], Rtboot_q5=out["q5"][1], Rt_q6=out["q6"][0], Rtboot_q6=out["q6"][1])
            del S, Sf, Wn, H
            print(f"     [{time.time() - t0:.0f} s; peak RSS {L.rss_gb():.2f} GB]", flush=True)
    json.dump(res, open(os.path.join(L.OUT, "f07.json"), "w"), indent=1)
    print(f"done [{time.time() - t00:.0f} s]")


if __name__ == "__main__":
    main()
