"""V5: the authors' sampler on NEW seeds (3001..), counting with my brute-force code: is the ~0.3-0.6% deficit of
detected-centred P(>=1 other detected) at 10'-15' in the 40 analysis skies reproducible (systematic) or a
fluctuation?  Also random-centred P for det5.  usage: v5 seed0 seed1"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_common import SortedPts, counts
from szsim import sampler

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "their_sampler_new"); os.makedirs(OUT, exist_ok=True)
TH = np.array([2, 5, 10, 15.0])
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, fields=("ra", "dec", "q_obs"))
    ra = np.radians(cat["ra"].astype(float)); dec = np.radians(cat["dec"].astype(float)); qo = cat["q_obs"]
    del cat
    sel = qo > 4
    ids = np.flatnonzero(sel); q4 = qo[sel]
    v = np.column_stack([np.cos(dec[sel]) * np.cos(ra[sel]), np.cos(dec[sel]) * np.sin(ra[sel]), np.sin(dec[sel])])
    del ra, dec, qo
    rng = np.random.default_rng(777_000 + seed)
    g = rng.standard_normal((100_000, 3)); rv = g / np.linalg.norm(g, axis=1)[:, None]
    res = {"seed": seed}
    for q in (4, 5, 6):
        s = q4 > q
        S = SortedPts(v[s], ids[s])
        Nc = counts(v[s], ids[s], S, TH)
        Nr = counts(rv, -np.ones(len(rv), np.int64), S, TH)
        res[f"N{q}"] = int(s.sum()); res[f"F{q}"] = (Nc >= 1).mean(0); res[f"R{q}"] = (Nr >= 1).mean(0)
        res[f"pairs{q}"] = Nc.sum(0) / 2
    np.save(os.path.join(OUT, f"sky_{seed}.npy"), res, allow_pickle=True)
    print(f"seed {seed} N5={res['N5']} t={time.time()-t0:.1f}s", flush=True)
