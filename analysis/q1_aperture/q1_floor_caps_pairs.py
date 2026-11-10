"""Floor-1e14 skies (M500>1e14, seeds 5001..): (A) mean / variance of counts in apertures from 1' to
180 deg, exact spherical cap vs flat-sky pi theta^2; (B) survey-wide pair counts and their
variance, Var(P) = E[P] (1 + 2 mu), for M>1e14 and det5 (q_obs>5 among M>1e14).
usage: python q1_floor_caps_pairs.py seed0 seed1"""
import os, sys, time, resource, json
import numpy as np
from scipy.spatial import cKDTree
from szsim import sampler
from q1_common import *

WORKERS = 2
CAP_DEG = np.array([1/60, 10/60, 1, 3, 5, 10, 20, 30, 45, 60, 90, 120, 150, 179.99])
NCEN_SMALL, NCEN_LARGE = 5000, 400
OUT = os.path.join(DATA, "floor14"); os.makedirs(OUT, exist_ok=True)


def run(seed):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, Mmin=1e14, fields=("ra", "dec", "q_obs"))
    v = radec_to_vec(cat["ra"], cat["dec"]); qo = cat["q_obs"]
    N = len(qo)
    rng = np.random.default_rng(20_000_000 + seed)
    tree = cKDTree(v)
    small = CAP_DEG <= 1
    cnt = {}
    rc = random_vec(NCEN_SMALL, rng)
    for t in CAP_DEG[small]:
        cnt[t] = tree.query_ball_point(rc, chord(t * 60), return_length=True, workers=WORKERS)
    rl = random_vec(NCEN_LARGE, rng)
    thr = np.cos(np.radians(CAP_DEG[~small]))[::-1]          # increasing cos thresholds
    big = np.zeros((NCEN_LARGE, len(thr)), np.int64)
    for a in range(0, NCEN_LARGE, 25):
        c = rl[a:a + 25] @ v.T                                 # (25, N)
        b = np.searchsorted(thr, c, side="left")               # #thresholds strictly below c... count cos>thr
        for j in range(len(thr)):
            big[a:a + 25, j] = (b > j).sum(1)
    for j, t in enumerate(CAP_DEG[~small][::-1]):
        cnt[t] = big[:, j]
    res = {"seed": seed, "N": N, "N5": int((qo > 5).sum()),
           "cap_mean": np.array([cnt[t].mean() for t in CAP_DEG]),
           "cap_m2": np.array([(cnt[t].astype(float) ** 2).mean() for t in CAP_DEG]),
           "cap_n": np.array([len(cnt[t]) for t in CAP_DEG])}
    res["pairs_M1e14"] = (tree.count_neighbors(tree, chord(THETAS)) - N) / 2
    t5 = cKDTree(v[qo > 5])
    res["pairs_det5f"] = (t5.count_neighbors(t5, chord(THETAS)) - res["N5"]) / 2
    res["time"] = time.time() - t0
    res["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.save(os.path.join(OUT, f"sky_{seed}.npy"), res, allow_pickle=True)
    if seed % 20 == 0:
        print(f"seed {seed} N={N} t={res['time']:.2f}s rss={res['rss']:.2f}GB", flush=True)


if __name__ == "__main__":
    for s in range(int(sys.argv[1]), int(sys.argv[2])):
        run(s)
