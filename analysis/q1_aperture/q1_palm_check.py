"""Repair-phase check of the Palm (Slivnyak-Mecke) claim on NEW, independent seeds.

Verifiers found that in the 40 original full-sky skies, P(>=1 other detected within theta | detected centre)
at 10'-15' sits ~2 sigma low for det4/5/6, even conditional on the realised N.  This script draws many
cheap floor-1e14 skies (M500 >= 1e14, so 'detected' = q_obs > q_cut among M>=1e14 halos; its expected
count Nbar = det{q}_floor1e14 in nbar.json) on fresh seeds and stores, per sky:
  N[q]             realised number of detected (q=4,5,6), and N (all M>=1e14)
  ge1_det[q]       number of detected centres with >= 1 other detected within theta (THP)
  ge2_det[q]       number of detected centres with >= 2 others within theta
  ge1_rand[q]      number of NRAND random centres with >= 1 detected within theta
  pairs_det[q]     unordered detected pairs with separation < theta
  pairs_all        unordered M>=1e14 pairs with separation < theta
usage: python q1_palm_check.py seed0 seed1 outdir_name
"""
import os, sys, time, resource
import numpy as np
from scipy.spatial import cKDTree
from szsim import sampler
from q1_common import *

WORKERS = int(os.environ.get("Q1_WORKERS", 2))
NRAND = 100_000
THP = np.array([1, 2, 3, 5, 7.5, 10, 15, 20, 30.0])
QS = [4, 5, 6]


def run(seed, out):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, Mmin=1e14, fields=("ra", "dec", "q_obs"))
    v = radec_to_vec(cat["ra"], cat["dec"]); qo = cat["q_obs"]
    rng = np.random.default_rng(30_000_000 + seed)
    rv = random_vec(NRAND, rng)
    ch = chord(THP)
    res = {"seed": seed, "Nall": len(qo)}
    tall = cKDTree(v)
    res["pairs_all"] = (tall.count_neighbors(tall, ch) - len(qo)) / 2
    for q in QS:
        vd = v[qo > q]; n = len(vd)
        t = cKDTree(vd)
        d, _ = t.query(vd, k=3, workers=WORKERS)      # d[:,0] = 0 (self, positions are distinct)
        assert np.all(d[:, 0] == 0)
        res[f"N{q}"] = n
        res[f"ge1_det{q}"] = (d[:, 1][:, None] <= ch[None, :]).sum(0)
        res[f"ge2_det{q}"] = (d[:, 2][:, None] <= ch[None, :]).sum(0)
        dr, _ = t.query(rv, k=1, workers=WORKERS)
        res[f"ge1_rand{q}"] = (dr[:, None] <= ch[None, :]).sum(0)
        res[f"pairs_det{q}"] = (t.count_neighbors(t, ch) - n) / 2
    res["time"] = time.time() - t0
    res["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.save(os.path.join(out, f"sky_{seed}.npy"), res, allow_pickle=True)
    if seed % 50 == 0:
        print(f"seed {seed} N5={res['N5']} t={res['time']:.2f}s rss={res['rss']:.2f}GB", flush=True)


if __name__ == "__main__":
    out = os.path.join(DATA, sys.argv[3]); os.makedirs(out, exist_ok=True)
    for s in range(int(sys.argv[1]), int(sys.argv[2])):
        run(s, out)
