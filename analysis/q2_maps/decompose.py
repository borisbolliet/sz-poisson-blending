"""Exact truth decomposition of every pre-merge candidate (Q>=4) and of every true halo with q_true>=1.

For candidate at pixel p and best scale k:   S_map(p) = sum_i Y_i R_k(|x_i-p|; theta_i) - ybar * kappa_k
(the second term is the l=0 mode removed by the filter).  Primary = brightest contributor (largest c_i).
For halo i (nearest pixel p_i, offset delta_i <= 0.18'):
   q_iso   = max_k [Y_i R_k(delta_i; theta_i) + N_k(p_i)] / sigma_k        (would-be SNR if alone)
   q_pos_v = max_k [S_v,k(p_i) + N_k(p_i)] / sigma_k                       (forced photometry, blended)
   own-scale k_i = scale nearest to theta500_i: X_i = S_V13,k_i(p_i) - Y_i R_k_i(delta_i)  (net contamination)
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import q2lib as L

TAB = L.build_tables()
HMASK = {"V13": None, "V3e13": ("M", 3e13), "V14": ("M", 1e14)}
COMP_ON = {"V13": [0, 1, 2], "V3e13": [1, 2], "V14": [2]}
MODE = os.environ.get("DETMODE", "maxQ")          # "maxQ" or "ps" (per-scale peaks)
SUF = "" if MODE == "maxQ" else "ps"
OUT = f"{L.DATA}/decomp" + ("" if MODE == "maxQ" else "_ps")


def work(pid):
    fn = f"{OUT}/decomp_{pid:03d}.npz"
    if os.path.exists(fn):
        return pid, 0
    o = dict(np.load(f"{L.DATA}/patch_out/patch_{pid:03d}.npz"))
    d = L.load_patch(pid)
    sig = o["sig"]; K = len(sig)
    res = {}
    q3 = np.flatnonzero(d["q"] >= 3.0)
    from scipy.spatial import cKDTree
    tq3 = cKDTree(np.stack([d["x"][q3], d["y"][q3]], 1) % L.LBOX, boxsize=L.LBOX)
    for v, comps in COMP_ON.items():
        mask = None if HMASK[v] is None else d["M500"] >= HMASK[v][1]
        tr = L.Truth(d, TAB, mask)
        ybar = o["means"][comps].sum()
        c, cv = o[f"cand{SUF}_{v}"], o[f"candv{SUF}_{v}"]
        sel = cv[:, 0] >= 4.0
        c, cv = c[sel], cv[sel]
        n = len(c)
        rec = np.zeros((n, 14)); near = np.full((n, 5, 2), np.nan)
        for j in range(n):
            iy, ix, k = c[j]
            x0, y0 = (ix + 0.5) * L.PIX, (iy + 0.5) * L.PIX
            I, C = tr.contributions(x0, y0, k)
            yk = ybar * L.kappa_mean(sig[k])
            if C.size == 0:          # pure-noise peak with no painted halo in range (possible in V14)
                rec[j] = [iy, ix, k, cv[j, 0], cv[j, 1], cv[j, 2], -1, 0.0, 0.0, -1, -yk, 0, 99.0, yk]
                continue
            o_ = np.argsort(-C)
            ip = I[o_[0]]; cp = C[o_[0]]
            c2 = C[o_[1]] if C.size > 1 else 0.0; ip2 = I[o_[1]] if C.size > 1 else -1
            dx = (d["x"][ip] - x0 + L.LBOX / 2) % L.LBOX - L.LBOX / 2
            dy = (d["y"][ip] - y0 + L.LBOX / 2) % L.LBOX - L.LBOX / 2
            others = C[o_[1:]]
            rec[j] = [iy, ix, k, cv[j, 0], cv[j, 1], cv[j, 2], ip, cp, c2, ip2, C.sum() - yk,
                      (others > 0.1 * cp).sum(), np.hypot(dx, dy), yk]
            dd, ii = tq3.query([x0 % L.LBOX, y0 % L.LBOX], k=5, distance_upper_bound=10.0)
            ok = np.isfinite(dd)
            near[j, :ok.sum(), 0] = dd[ok]; near[j, :ok.sum(), 1] = d["q"][q3[ii[ok]]]
        res[f"cand_{v}"] = rec; res[f"near_{v}"] = near
    # ---- halos
    hs = o["hsel"]; hv = o["hval"].astype(np.float64)        # (nh, K, 4): A, B, C, N
    px = (np.floor(d["x"][hs] / L.PIX) + 0.5) * L.PIX; py = (np.floor(d["y"][hs] / L.PIX) + 0.5) * L.PIX
    dlt = np.hypot(d["x"][hs] - px, d["y"][hs] - py)
    lnth = np.log(d["theta500"][hs])
    own = np.stack([d["Y500"][hs] * L.resp_eval(TAB["resp"][k], TAB["rR"], L.LNTH0, L.DLNTH, lnth, dlt)
                    for k in range(K)], 1)                       # (nh, K)
    N = hv[:, :, 3]
    q_iso = (own + N) / sig[None, :]
    res["h_idx"] = hs
    res["h_qiso"] = q_iso.max(1); res["h_kiso"] = q_iso.argmax(1)
    for v, comps in COMP_ON.items():
        Sv = hv[:, :, comps].sum(2)
        res[f"h_qpos_{v}"] = ((Sv + N) / sig[None, :]).max(1)
    kown = np.abs(np.log(d["theta500"][hs])[:, None] - np.log(L.THETA_K)[None, :]).argmin(1)
    r_ = np.arange(hs.size)
    S13 = hv[:, :, :3].sum(2)
    res["h_kown"] = kown
    res["h_own"] = own[r_, kown]; res["h_Xnet"] = S13[r_, kown] - own[r_, kown]
    res["h_Nown"] = N[r_, kown]
    res["h_Xnet14"] = hv[r_, kown, 2] - own[r_, kown] * (d["M500"][hs] >= 1e14)
    np.savez(fn, **res)
    return pid, sum(len(res[f"cand_{v}"]) for v in COMP_ON)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, b, npr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    import time; t0 = time.time()
    with Pool(npr) as p:
        for pid, nc in p.imap_unordered(work, range(a, b)):
            print(f"decomp patch {pid:3d}: {nc} candidates  elapsed {time.time()-t0:.0f}s", flush=True)
