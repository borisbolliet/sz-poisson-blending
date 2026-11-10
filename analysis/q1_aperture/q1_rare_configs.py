"""Rare survey-wide configurations: probability that AT LEAST ONE pair (triple) with all separations
< theta exists anywhere on the sky, for rare populations where this probability is O(0.05-0.95).
Populations (floor M500>3e14 skies, seeds 7001..): A = M>3e14 (all z), B = M>3e14 & z>1.
Stores per sky the pair counts (A: PA_THETAS, B: PB_THETAS) and triple-clique counts (A: TA_THETAS).
usage: python q1_rare_configs.py seed0 seed1"""
import os, sys, time, resource
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
from szsim import sampler
from q1_common import *

PA_THETAS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
PB_THETAS = np.array([1, 2, 3, 5, 7.5, 10, 15.0])
TA_THETAS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12.0])
OUT = os.path.join(DATA, "rare"); os.makedirs(OUT, exist_ok=True)


def triangles(tree, n, t):
    pr = tree.query_pairs(chord(t), output_type="ndarray")
    if len(pr) < 3:
        return 0
    A = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    return int(round((A @ A).multiply(A).sum() / 6))


def main(s0, s1):
    rows = []
    t0 = time.time()
    for s in range(s0, s1):
        cat = sampler.draw_catalog(s, Mmin=3e14, fields=("ra", "dec", "z"))
        v = radec_to_vec(cat["ra"], cat["dec"]); z = cat["z"]
        tA = cKDTree(v); nA = len(v)
        vb = v[z >= 1.0]; tB = cKDTree(vb); nB = len(vb)
        pa = (tA.count_neighbors(tA, chord(PA_THETAS)) - nA) / 2
        pb = (tB.count_neighbors(tB, chord(PB_THETAS)) - nB) / 2
        ta = [triangles(tA, nA, t) for t in TA_THETAS]
        rows.append(np.concatenate([[s, nA, nB], pa, pb, ta]))
    np.save(os.path.join(OUT, f"rare_{s0}_{s1}.npy"), np.array(rows))
    print(f"seeds {s0}-{s1}: {time.time()-t0:.1f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f}GB")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
