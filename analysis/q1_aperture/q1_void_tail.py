"""Repair phase: dedicated test of the deep low tail P(N<=1) of M>=3e13 neighbours at theta~10' (mu~9.7), for
detected centres vs random centres, on NEW seeds (the 120-sky set showed det-centred +4% (z=+3.5) and
random-centred -2.7% (z=-2.4)).  Floor-3e13 skies: population B = all M>=3e13 halos; detected = q_obs>q among them.
Per sky stores, for theta in TH: number of centres with N(<theta) = 0 and <= 1 (centre excluded from B), and N.
usage: python q1_void_tail.py seed0 seed1"""
import os, sys, time, resource
import numpy as np
from scipy.spatial import cKDTree
from szsim import sampler
from q1_common import *
WORKERS = int(os.environ.get("Q1_WORKERS", 1))
TH = np.array([5, 7.5, 10, 12.5])
NRAND = 100_000
OUT = os.path.join(DATA, "void_tail")
def run(seed):
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    cat = sampler.draw_catalog(seed, Mmin=3e13, fields=("ra", "dec", "q_obs"))
    v = radec_to_vec(cat["ra"], cat["dec"]); qo = cat["q_obs"]; del cat
    tree = cKDTree(v)
    ch = chord(TH)
    res = {"seed": seed, "N": len(qo)}
    rng = np.random.default_rng(40_000_000 + seed)
    d, _ = tree.query(random_vec(NRAND, rng), k=2, workers=WORKERS)
    res["rand_n0"] = (d[:, 0][:, None] > ch).sum(0); res["rand_le1"] = (d[:, 1][:, None] > ch).sum(0)
    for q in (4, 5, 6):
        idx = np.flatnonzero(qo > q)
        d, i = tree.query(v[idx], k=3, workers=WORKERS)
        assert np.all(i[:, 0] == idx) or np.mean(i[:, 0] == idx) > 0.999   # self is the first hit
        d = np.where(i == idx[:, None], np.inf, d); d = np.sort(d, axis=1)[:, :2]
        res[f"n{q}"] = len(idx)
        res[f"det{q}_n0"] = (d[:, 0][:, None] > ch).sum(0); res[f"det{q}_le1"] = (d[:, 1][:, None] > ch).sum(0)
    res["time"] = time.time() - t0; res["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.save(os.path.join(OUT, f"sky_{seed}.npy"), res, allow_pickle=True)
    print(f"seed {seed} N={res['N']} n5={res['n5']} t={res['time']:.1f}s rss={res['rss']:.2f}GB", flush=True)


if __name__ == "__main__":
    for seed in range(int(sys.argv[1]), int(sys.argv[2])):
        run(seed)
