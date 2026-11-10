"""Monte Carlo of aperture statistics on full-sky Poisson realisations (M500>1e13 floor, so the
detected samples include every halo that can scatter above q_cut).

Per sky (seed) it stores:
  cdf[combo]  : (KMAX, nfine) P(d_k <= theta), k=1..KMAX, over all centres of the combo, where d_k is the
                k-th neighbour distance from the centre to the neighbour population (the centre itself
                excluded when it belongs to that population).  P(N(theta) >= k) = cdf[k-1].
  dsub[combo] : (NSUB, KMAX) kNN distances [arcmin] of a random subsample of NSUB centres (for KS tests)
  ncen[combo] : number of centres
  ball_*      : random-centre ball counts (mean, var) at THETAS for some populations
  pairs_*     : survey-wide unordered pair counts with separation < theta (THETAS)
  tri_det5    : unordered triples of det5 with all three separations < theta (TRI_THETAS)
  nge_det5    : number of det5 clusters with >= n other det5 within theta (n=1..5, THETAS)
  npop        : realised population sizes
usage: python q1_mc_fullsky.py seed0 seed1   (seeds in [seed0, seed1))
"""
import os, sys, time, resource
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kstest
import scipy.sparse as sp
from szsim import sampler
from q1_common import *
import json

WORKERS = int(os.environ.get("Q1_WORKERS", 2))
NRAND = 100_000
NRAND_BALL = 20_000
NSUB = 1000
FINE = np.unique(np.concatenate([THETAS, np.geomspace(0.2, 240, 91)]))
TRI_THETAS = np.array([1, 2, 3, 5, 7.5, 10.0])
OUT = os.path.join(DATA, "fullsky")
os.makedirs(OUT, exist_ok=True)
NB = json.load(open(os.path.join(DATA, "nbar.json")))


def knn_excl(tree, popidx, cen_vec, cen_gidx, k):
    """k nearest neighbour angular distances [arcmin] excluding the centre itself (by global index)."""
    d, i = tree.query(cen_vec, k=k + 1, workers=WORKERS)
    if cen_gidx is not None:
        g = popidx[np.minimum(i, len(popidx) - 1)]
        d = np.where(g == cen_gidx[:, None], np.inf, d)
    d = np.sort(d, axis=1)[:, :k]
    return chord_to_arcmin(d)


def run_sky(seed):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, fields=("ra", "dec", "z", "M500", "q_obs"))
    M, z, qo = cat["M500"], cat["z"], cat["q_obs"]
    sel = {}
    sel["det4"] = np.flatnonzero(qo > 4); sel["det5"] = np.flatnonzero(qo > 5); sel["det6"] = np.flatnonzero(qo > 6)
    sel["M1e14"] = np.flatnonzero(M >= 1e14)
    sel["M3e13"] = np.flatnonzero(M >= 3e13)
    for (z1, z2) in ZSLICES:
        sel[f"det5_z{z1}-{z2}"] = sel["det5"][(z[sel["det5"]] >= z1) & (z[sel["det5"]] < z2)]
    i14 = sel["M1e14"]
    sel["undet5_M1e14"] = i14[qo[i14] <= 5]
    keep = np.unique(np.concatenate(list(sel.values())))
    # global -> vector lookup only for kept rows
    vec_keep = radec_to_vec(cat["ra"][keep], cat["dec"][keep])
    zc = z[keep]; qc = qo[keep]; Mk = M[keep]
    del cat, M, z, qo
    pos = {k: np.searchsorted(keep, v) for k, v in sel.items()}     # indices into keep (= global ids)
    trees = {k: cKDTree(vec_keep[v]) for k, v in pos.items()}
    rng = np.random.default_rng(10_000_000 + seed)
    rvec = random_vec(NRAND, rng)

    d5 = pos["det5"]
    cen = {"rand": (rvec, None)}
    for q in QCUTS:
        cen[f"det{q}"] = (vec_keep[pos[f"det{q}"]], pos[f"det{q}"])
    for lab, m in [("det5_zc<0.5", zc[d5] < 0.5), ("det5_zc>1", zc[d5] >= 1.0), ("det5_q>20", qc[d5] > 20)]:
        cen[lab] = (vec_keep[d5[m]], d5[m])
    combos = [("rand", p) for p in pos]
    for q in QCUTS:
        combos += [(f"det{q}", f"det{q}"), (f"det{q}", "M1e14"), (f"det{q}", "M3e13")]
    combos += [("det5", f"det5_z{a}-{b}") for a, b in ZSLICES]
    combos += [("det5", "undet5_M1e14"), ("det5", "det4"), ("det5", "det6")]
    for c in ["det5_zc<0.5", "det5_zc>1", "det5_q>20"]:
        combos += [(c, "det5"), (c, "M3e13")]
    res = {"seed": seed, "npop": {k: len(v) for k, v in pos.items()}, "ncen": {}, "cdf": {}, "dsub": {}}
    for (c, p) in combos:
        cv, cg = cen[c]
        d = knn_excl(trees[p], pos[p], cv, cg, KMAX)
        key = f"{c}|{p}"
        res["cdf"][key] = np.array([(d[:, k][:, None] <= FINE[None, :]).mean(0) for k in range(KMAX)])
        res["ncen"][key] = len(cv)
        # random subsample of NSUB centres: kNN distances kept for KS tests (a small subsample keeps the
        # centres nearly independent, so a per-sky KS is valid; the full set is not iid given the sky)
        res["dsub"][key] = d[rng.choice(len(d), size=min(NSUB, len(d)), replace=False)].astype(np.float32)
    # random-centre ball counts: mean & variance of N(theta)
    rb = rvec[:NRAND_BALL]
    for p in ["det5", "M1e14", "M3e13"]:
        mv = np.zeros((len(THETAS), 2))
        for j, t in enumerate(THETAS):
            n = trees[p].query_ball_point(rb, chord(t), return_length=True, workers=WORKERS)
            mv[j] = n.mean(), n.var(ddof=1)
        res[f"ball_{p}"] = mv
    # survey-wide pair counts
    for p in ["det4", "det5", "det6", "M1e14"]:
        c = trees[p].count_neighbors(trees[p], chord(THETAS))
        res[f"pairs_{p}"] = (c - len(pos[p])) / 2
    # triples and nge for det5
    T5 = trees["det5"]; n5 = len(d5)
    tri = []
    for t in TRI_THETAS:
        pr = T5.query_pairs(chord(t), output_type="ndarray")
        A = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n5, n5)).tocsr()
        A = A + A.T
        tri.append((A @ A).multiply(A).sum() / 6)
    res["tri_det5"] = np.array(tri)
    dd = knn_excl(T5, d5, vec_keep[d5], d5, 5)
    res["nge_det5"] = np.array([[(dd[:, n - 1] <= t).sum() for t in THETAS] for n in range(1, 6)])
    res["time"] = time.time() - t0
    res["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.save(os.path.join(OUT, f"sky_{seed}.npy"), res, allow_pickle=True)
    print(f"seed {seed}: N(det5)={res['npop']['det5']} N(M3e13)={res['npop']['M3e13']} "
          f"t={res['time']:.1f}s rss={res['maxrss_GB']:.2f}GB", flush=True)


if __name__ == "__main__":
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    for s in range(s0, s1):
        run_sky(s)
