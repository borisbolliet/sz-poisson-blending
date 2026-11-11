"""Map-level (ground-truth pipeline) BLENDED counts in (z, q_obs) bins from the 324 stored 10x10 deg patches.

Detections: variant V13 (all halos M>=1e13 painted), maxQ peak finder, greedy 2' merge (rule R2), Q >= q_cut,
exactly as analysis/q2_maps/analysis_blend.py.  Cluster z = z of the PRIMARY halo (largest noise-free contribution
to the filtered signal at the peak), as an optical/X-ray counterpart redshift would give.
Reference ('isolated') counts on the SAME halos and SAME noise: halo i counted at (z_i, q_iso_i), with
q_iso = max over the 12 filter scales of (own signal + noise)/sigma_k at the halo pixel (decompose.py), i.e. the
pipeline's response to that halo alone.  Also the single-cluster catalogue expectation sum_i P(q_i + N(0,1) in bin).
Per-patch histograms are stored for a patch bootstrap:
   H_map     all accepted detections (incl. fragments and noise-dominated peaks)
   H_clean   minus duplicates (fragments) and noise-dominated detections (primary c_P/sigma < 2 and N > S - c_P)
   H_iso     isolated pipeline counts of the same halos
   H_single  single-cluster expectation (q_true + N(0,1)), catalogue convention
Output: data_cosmo/truth_maps.npz
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, f"{HERE}/../../q2_maps")
import cosmolib as CL
import q2lib as L

NPATCH = 324
QCS = (5.0, 6.0)
BINS = {qc: CL.default_bins(qc) for qc in QCS}


def hist(z, q, qc, w=None):
    zed, qed = BINS[qc]
    H, _, _ = np.histogram2d(z, q, bins=[zed, qed], weights=w)
    return H


def work(p):
    o = np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz"); sig = o["sig"]
    dc = np.load(f"{L.DATA}/decomp/decomp_{p:03d}.npz")
    d = L.load_patch(p)
    rec = dc["cand_V13"]
    iy, ix, k = rec[:, 0].astype(int), rec[:, 1].astype(int), rec[:, 2].astype(int)
    Q, S, N, ip, cp = rec[:, 3], rec[:, 4], rec[:, 5], rec[:, 6].astype(int), rec[:, 7]
    xy = np.stack([(ix + 0.5) * L.PIX, (iy + 0.5) * L.PIX], 1); th = L.THETA_K[k]; sk = sig[k]
    acc = L.merge_greedy(xy[:, 0].copy(), xy[:, 1].copy(), th.astype(np.float64), Q.copy(), 2.0, 0.0, L.LBOX)
    hs, qiso = dc["h_idx"], dc["h_qiso"]
    out = {}
    for qc in QCS:
        a = acc[Q[acc] >= qc]; a = a[np.argsort(-Q[a])]
        seen = set(); dup = np.zeros(a.size, bool)
        for t, j in enumerate(a):
            if ip[j] in seen:
                dup[t] = True
            seen.add(ip[j])
        nosig = (cp[a] / sk[a] < 2.0) & (N[a] > S[a] - cp[a])
        ok = ip[a] >= 0
        zdet = np.where(ok, d["z"][np.maximum(ip[a], 0)], -1.0)
        out[f"map{qc:.0f}"] = hist(zdet[ok], Q[a][ok], qc)
        cl = ok & ~dup & ~nosig
        out[f"clean{qc:.0f}"] = hist(zdet[cl], Q[a][cl], qc)
        out[f"iso{qc:.0f}"] = hist(d["z"][hs], qiso, qc)
        zed, qed = BINS[qc]
        P = ndtr(d["q"][:, None] - qed[None, :-1]); P = np.concatenate([P, np.zeros((P.shape[0], 1))], 1)
        Pb = P[:, :-1] - P[:, 1:]
        iz = np.searchsorted(zed, d["z"]) - 1
        Hs = np.zeros((len(zed) - 1, len(qed) - 1))
        okz = (iz >= 0) & (iz < len(zed) - 1)
        np.add.at(Hs, iz[okz], Pb[okz])
        out[f"single{qc:.0f}"] = Hs
        out[f"n_noprim{qc:.0f}"] = (~ok).sum(); out[f"n_dup{qc:.0f}"] = dup.sum(); out[f"n_nosig{qc:.0f}"] = (nosig & ~dup).sum()
    return p, out


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    with Pool(3) as pool:
        for p, o in pool.imap_unordered(work, range(NPATCH)):
            for kk, v in o.items():
                res.setdefault(kk, [None] * NPATCH)[p] = v
            if p % 40 == 0:
                print(f"patch {p} done [{time.time()-t0:.0f}s]", flush=True)
    res = {kk: np.array(v) for kk, v in res.items()}
    area = NPATCH * (L.LBOX / 60.0) ** 2
    full = 4 * np.pi * (180 / np.pi) ** 2
    np.savez(f"{CL.DC}/truth_maps.npz", area_deg2=area, fullsky_deg2=full, **res)
    for qc in QCS:
        s = f"{qc:.0f}"
        tot = {kk: res[kk + s].sum() for kk in ("map", "clean", "iso", "single")}
        print(f"\n=== q_cut={qc}: {area:.0f} deg^2 ===")
        print("  per deg^2: " + "  ".join(f"{kk} {v/area:.4f}" for kk, v in tot.items()))
        print(f"  no-primary {res['n_noprim'+s].sum()}, duplicates {res['n_dup'+s].sum()}, noise-dominated {res['n_nosig'+s].sum()}"
              f" (of {tot['map']:.0f})")
        print(f"  map/iso = {tot['map']/tot['iso']:.4f}, clean/iso = {tot['clean']/tot['iso']:.4f}, "
              f"iso/single = {tot['iso']/tot['single']:.4f}, map/single = {tot['map']/tot['single']:.4f}")
        for kk in ("map", "clean"):
            D = res[kk + s].sum(0) - res["iso" + s].sum(0); I = res["iso" + s].sum(0)
            print(f"  ({kk}-iso)/iso per q bin:", np.round(D.sum(0) / I.sum(0), 4))
            print(f"  ({kk}-iso)/iso per z bin:", np.round(D.sum(1) / np.maximum(I.sum(1), 1), 4))
        D = res["iso" + s].sum(0) - res["single" + s].sum(0)
        print("  (iso-single)/single per q bin:", np.round(D.sum(0) / res["single" + s].sum(0).sum(0), 4))
    print(f"total {time.time()-t0:.0f}s")
