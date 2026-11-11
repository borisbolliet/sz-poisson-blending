"""Extract map-level detections (324 patches x 100 deg^2, maxQ over 12 scales, greedy 2' merge = baseline R2,
all halos M>=1e13 painted, l=0 removed) with their primary's z, and the catalogue-level rows (filter at the
scale k_own nearest to theta500, true position, actual map noise and actual painted neighbours) with z.
Output: data_cosmo/map_detections.npz (read-only use of data/q2_maps)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/q2_maps")
import q2lib as L

PIDS = list(range(324))
det = []      # pid, Q, k, z_primary, q_true_primary, dup, cP/sigma_k, (cP+N)/sigma_k
for p in PIDS:
    dc = np.load(f"{L.DATA}/decomp/decomp_{p:03d}.npz")
    d = L.load_patch(p)
    o = np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz"); sig = o["sig"]
    rec = dc["cand_V13"]
    Q, k, ip, cp, N = rec[:, 3], rec[:, 2].astype(int), rec[:, 6].astype(int), rec[:, 7], rec[:, 5]
    x = (rec[:, 1] + 0.5) * L.PIX; y = (rec[:, 0] + 0.5) * L.PIX
    acc = L.merge_greedy(x.copy(), y.copy(), L.THETA_K[k].astype(np.float64), Q.copy(), 2.0, 0.0, L.LBOX)
    acc = acc[Q[acc] >= 4.0]
    acc = acc[np.argsort(-Q[acc])]
    seen = set(); dup = np.zeros(acc.size, bool)
    for t, j in enumerate(acc):
        if ip[j] in seen:
            dup[t] = True
        seen.add(ip[j])
    zp = np.where(ip[acc] >= 0, d["z"][np.maximum(ip[acc], 0)], -1.0)
    qp = np.where(ip[acc] >= 0, d["q"][np.maximum(ip[acc], 0)], -1.0)
    sk = sig[k[acc]]
    det.append(np.column_stack([np.full(acc.size, p), Q[acc], k[acc], zp, qp, dup, cp[acc] / sk, (cp[acc] + N[acc]) / sk]))
    if p % 50 == 0:
        print("patch", p, flush=True)
det = np.concatenate(det)
R = np.load(f"{L.DATA}/catalog_level_rows_ALL.npy")       # ip_, h, k, q_true, M, own, net, mx, N, sig
zr = np.empty(len(R))
for ip_ in np.unique(R[:, 0].astype(int)):
    s = R[:, 0].astype(int) == ip_
    zr[s] = L.load_patch(PIDS[ip_])["z"][R[s, 1].astype(int)]
np.savez(f"/Users/boris/GitHub/sz-poisson-blending/data_cosmo/map_detections.npz", det=det, rows=R, rows_z=zr,
         THETA_K=L.THETA_K, area_deg2=len(PIDS) * (L.LBOX / 60) ** 2)
print(f"detections Q>=4: {len(det)}, Q>=5: {(det[:,1]>=5).sum()} ({(det[:,1]>=5).sum()/(len(PIDS)*100):.4f}/deg^2), "
      f"duplicates among Q>=5: {det[det[:,1]>=5,5].mean():.4f}; catalogue rows {len(R)}")
