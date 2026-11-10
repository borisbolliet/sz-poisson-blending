"""Independent map pipeline, one 10x10 deg periodic patch per call.  Halos drawn directly on the flat torus
(seed 7_000_000+i), noise seed 9_000_000+i.  Saves candidates, decomposition, halo-side quantities, variances."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
import indep_lib as I

GRID = I.hmf_grid()
PT, PR = I.paint_table()
TFT, SIG, TSUM = I.templates()
TFT = TFT.copy(); TFT[:, 0, 0] = 0.0          # remove l=0 (map mean)
RT, RR = I.resp_table(SIG)
SIGY = I.sigY_interp()
OUTD = f"{I.OUT}/patches"; os.makedirs(OUTD, exist_ok=True)
EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e4)
QCAND = 3.9


def paint_ft(d, sel):
    m = np.zeros((I.NPIX, I.NPIX))
    I.paint(m, d["x"][sel], d["y"][sel], d["Y"][sel], d["th"][sel], PT, PR)
    return sfft.rfft2(m, workers=2), m.mean()


class Tiers:
    def __init__(self, d, mask):
        idx_all = np.flatnonzero(mask)
        self.t = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            idx = idx_all[(d["th"][idx_all] >= lo) & (d["th"][idx_all] < hi)]
            if idx.size:
                self.t.append((idx, d["th"][idx].max(), cKDTree(np.stack([d["x"][idx], d["y"][idx]], 1), boxsize=I.LBOX)))
        self.d = d; self.lnth = np.log(d["th"])

    def contrib(self, x0, y0, k):
        I_, C_ = [], []
        for idx, tmax, tree in self.t:
            rq = 5 * tmax + 5 * I.THK[k] + 8 * I.SB
            loc = tree.query_ball_point([x0 % I.LBOX, y0 % I.LBOX], rq)
            if loc:
                ii = idx[np.asarray(loc, dtype=np.int64)]
                c = I.contrib(RT, RR, k, x0, y0, self.d["x"], self.d["y"], self.d["Y"], self.lnth, ii)
                nz = c != 0
                I_.append(ii[nz]); C_.append(c[nz])
        if not I_:
            return np.zeros(0, np.int64), np.zeros(0)
        return np.concatenate(I_), np.concatenate(C_)


def local_max(Q, thr):
    mf = maximum_filter(Q, size=3, mode="wrap")
    iy, ix = np.nonzero((Q >= mf) & (Q >= thr))
    return iy, ix


def run(i):
    t0 = time.time()
    d = I.draw_patch(7_000_000 + i, GRID)
    n = d["x"].size
    d["q"] = d["Y"] / SIGY(d["th"])
    mA = d["M"] < 1e14; mC = ~mA
    FA, yA = paint_ft(d, mA); FC, yC = paint_ft(d, mC)
    FQ2, _ = paint_ft(d, d["q"] < 2.0); FQ5, _ = paint_ft(d, d["q"] < 5.0)
    rng = np.random.default_rng(9_000_000 + i)
    FN = sfft.rfft2(rng.standard_normal((I.NPIX, I.NPIX)) * I.SIGPIX, workers=2)
    tpaint = time.time() - t0
    best = {v: [np.full((I.NPIX, I.NPIX), -np.inf, np.float32), np.zeros((I.NPIX, I.NPIX), np.int8),
                np.zeros((I.NPIX, I.NPIX), np.float32), np.zeros((I.NPIX, I.NPIX), np.float32)] for v in ("V13", "V14", "V0")}
    hs = np.flatnonzero(d["q"] >= 1.0)
    hix = np.floor(d["x"][hs] / I.PIX).astype(int) % I.NPIX; hiy = np.floor(d["y"][hs] / I.PIX).astype(int) % I.NPIX
    hv = np.zeros((hs.size, I.K, 3))
    var = np.zeros((I.K, 5))
    npk0 = None
    for k in range(I.K):
        T = TFT[k]
        A = sfft.irfft2(FA * T, s=(I.NPIX, I.NPIX), workers=2)
        C = sfft.irfft2(FC * T, s=(I.NPIX, I.NPIX), workers=2)
        N = sfft.irfft2(FN * T, s=(I.NPIX, I.NPIX), workers=2)
        var[k, 0] = N.var(); var[k, 1] = A.var()
        var[k, 2] = sfft.irfft2(FQ2 * T, s=(I.NPIX, I.NPIX), workers=2).var()
        var[k, 3] = sfft.irfft2(FQ5 * T, s=(I.NPIX, I.NPIX), workers=2).var()
        var[k, 4] = C.var()
        S13 = A + C
        hv[:, k, 0] = S13[hiy, hix]; hv[:, k, 1] = C[hiy, hix]; hv[:, k, 2] = N[hiy, hix]
        if k == 0:
            iy, ix = local_max(N / SIG[0], 3.0); npk0 = (N[iy, ix] / SIG[0]).astype(np.float32)
        for v, S in (("V13", S13), ("V14", C), ("V0", None)):
            Qb, kb, Sb, Nb = best[v]
            q = ((N if S is None else S + N) / SIG[k]).astype(np.float32)
            up = q > Qb
            Qb[up] = q[up]; kb[up] = k
            Sb[up] = 0.0 if S is None else S[up]; Nb[up] = N[up]
        del A, C, N, S13
    out = dict(seed=7_000_000 + i, n=n, lam=d["lam"], ybarA=yA, ybarC=yC, var=var, npk0=npk0, tpaint=tpaint)
    cands = {}
    for v in best:
        Qb, kb, Sb, Nb = best[v]
        iy, ix = local_max(Qb, QCAND)
        cands[v] = (iy, ix, kb[iy, ix].astype(int), Qb[iy, ix].astype(float), Sb[iy, ix].astype(float), Nb[iy, ix].astype(float))
        out[f"cand_{v}"] = np.stack(cands[v], 1)
    del best
    # ------------------ decomposition (V13 with all halos, V14 with M>=1e14 only)
    q4 = np.flatnonzero(d["q"] >= 4.0)
    tq4 = cKDTree(np.stack([d["x"][q4], d["y"][q4]], 1), boxsize=I.LBOX)
    for v, mask, ybar in (("V13", np.ones(n, bool), yA + yC), ("V14", mC, yC)):
        tr = Tiers(d, mask)
        iy, ix, kk, Q, S, Nn = cands[v]
        rec = np.zeros((iy.size, 9)); nearq = np.full((iy.size, 6, 2), np.nan)
        for j in range(iy.size):
            x0, y0 = (ix[j] + 0.5) * I.PIX, (iy[j] + 0.5) * I.PIX
            ii, cc = tr.contrib(x0, y0, kk[j])
            yk = ybar * SIG[kk[j]] ** 2 * I.RATIO / I.NW
            if cc.size == 0:
                rec[j] = [-1, 0, 0, -yk, 99, 0, 0, yk, 0]; continue
            o = np.argsort(-cc); ip = ii[o[0]]; cp = cc[o[0]]
            dx = (d["x"][ip] - x0 + I.LBOX / 2) % I.LBOX - I.LBOX / 2; dy = (d["y"][ip] - y0 + I.LBOX / 2) % I.LBOX - I.LBOX / 2
            c2 = cc[o[1]] if cc.size > 1 else 0.0
            rec[j] = [ip, cp, c2, cc.sum() - yk, np.hypot(dx, dy), (cc[o[1:]] > 0.1 * cp).sum(), d["q"][ip], yk, d["M"][ip]]
            if v == "V13":
                dd, jj = tq4.query([x0, y0], k=6, distance_upper_bound=3.0)
                ok = np.isfinite(dd)
                nearq[j, :ok.sum(), 0] = dd[ok]; nearq[j, :ok.sum(), 1] = d["q"][q4[jj[ok]]]
        out[f"dec_{v}"] = rec
        if v == "V13":
            out["near_V13"] = nearq
            # catalogue-level, unselected halos q_true>=4 at true position, own scale
            kown = np.abs(np.log(d["th"][q4])[:, None] - np.log(I.THK)[None, :]).argmin(1)
            cl = np.zeros((q4.size, 5))
            for t, h in enumerate(q4):
                ii, cc = tr.contrib(d["x"][h], d["y"][h], kown[t])
                me = ii == h
                oth = cc[~me]
                yk = ybar * SIG[kown[t]] ** 2 * I.RATIO / I.NW
                cl[t] = [kown[t], cc[me].sum(), oth.max() if oth.size else 0.0, oth.sum() - yk, d["q"][h]]
            out["catlev"] = cl
    # ------------------ halo side
    px = (np.floor(d["x"][hs] / I.PIX) + 0.5) * I.PIX; py = (np.floor(d["y"][hs] / I.PIX) + 0.5) * I.PIX
    dl = np.hypot(d["x"][hs] - px, d["y"][hs] - py)
    lnth = np.log(d["th"][hs])
    own = np.stack([d["Y"][hs] * np.array([I.tab_eval(RT[k], RR, a, b) for a, b in zip(lnth, dl)]) for k in range(I.K)], 1)
    out["h_idx"] = hs
    out["h_qiso"] = ((own + hv[:, :, 2]) / SIG[None, :]).max(1)
    out["h_qpos13"] = ((hv[:, :, 0] + hv[:, :, 2]) / SIG[None, :]).max(1)
    out["h_qpos14"] = ((hv[:, :, 1] + hv[:, :, 2]) / SIG[None, :]).max(1)
    out["h_q"] = d["q"][hs]; out["h_M"] = d["M"][hs]; out["h_x"] = d["x"][hs]; out["h_y"] = d["y"][hs]
    out["h_th"] = d["th"][hs]; out["h_Y"] = d["Y"][hs]
    # population summary for densities / pop-sum confusion
    out["all_q"] = d["q"].astype(np.float32)
    out["pop_th"] = d["th"].astype(np.float32); out["pop_Y"] = d["Y"].astype(np.float32); out["pop_M"] = d["M"].astype(np.float32)
    out["pop_z"] = d["z"].astype(np.float32)
    out["t_total"] = time.time() - t0
    np.savez(f"{OUTD}/p{i:03d}.npz", **out)
    return out["t_total"], tpaint


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for i in range(a, b):
        if os.path.exists(f"{OUTD}/p{i:03d}.npz"):
            continue
        tt, tp = run(i)
        print(f"patch {i}: {tt:.1f}s (paint {tp:.1f}s)", flush=True)
