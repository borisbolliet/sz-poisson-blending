"""V2: re-derive, from the authors' own catalogues (same seeds 1001..), the per-sky statistics stored by
q1_mc_fullsky.py, using brute-force sorted-window counting instead of cKDTree. Any bug in self-exclusion,
k+1 querying, pair halving or triangle counting shows up as a mismatch.  usage: v2 seed0 seed1"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_common import SortedPts, counts, triangles
from szsim import sampler

DATA = "/Users/boris/GitHub/sz-poisson-blending/data/q1_aperture/fullsky"
THETAS = np.array([0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60.0])
TH10 = THETAS[THETAS <= 10]


def vec(ra, dec):
    ra = np.radians(ra.astype(np.float64)); dec = np.radians(dec.astype(np.float64))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def rand_vec_theirs(n, rng):     # replicates q1_common.random_vec to get the same random centres
    z = 2 * rng.random(n) - 1; phi = 2 * np.pi * rng.random(n); s = np.sqrt(1 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


def frac_ge(N, kmax=6):
    return np.array([(N >= k).mean(0) for k in range(1, kmax + 1)])   # (6, T)


for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    t0 = time.time()
    ref = np.load(os.path.join(DATA, f"sky_{seed}.npy"), allow_pickle=True).item()
    FINE = np.unique(np.concatenate([THETAS, np.geomspace(0.2, 240, 91)]))
    iT = np.array([np.argmin(abs(FINE - t)) for t in THETAS])
    cat = sampler.draw_catalog(seed, fields=("ra", "dec", "z", "M500", "q_obs"))
    v = vec(cat["ra"], cat["dec"]); M = cat["M500"]; qo = cat["q_obs"]; z = cat["z"]
    ids = np.arange(len(M))
    pops = {"det4": qo > 4, "det5": qo > 5, "det6": qo > 6, "M1e14": M >= 1e14, "M3e13": M >= 3e13}
    pops["undet5_M1e14"] = (M >= 1e14) & (qo <= 5)
    pops["det5_z1.0-3.0"] = (qo > 5) & (z >= 1.0) & (z < 3.0)
    npop = {k: int(s.sum()) for k, s in pops.items()}
    S = {k: SortedPts(v[s], ids[s]) for k, s in pops.items()}
    rvec = rand_vec_theirs(100_000, np.random.default_rng(10_000_000 + seed))
    out = []
    for c, p, th in [("rand", "det5", THETAS), ("rand", "M1e14", THETAS), ("rand", "M3e13", TH10),
                     ("det5", "det5", THETAS), ("det4", "det4", THETAS), ("det6", "det6", THETAS),
                     ("det5", "M1e14", THETAS), ("det5", "M3e13", TH10), ("det5", "undet5_M1e14", TH10),
                     ("det5", "det5_z1.0-3.0", THETAS), ("det5", "det4", TH10)]:
        if c == "rand":
            cv, cid = rvec, -np.ones(len(rvec), np.int64)
        else:
            cv, cid = v[pops[c]], ids[pops[c]]
        N = counts(cv, cid, S[p], th)
        mine = frac_ge(N)
        theirs = ref["cdf"][f"{c}|{p}"][:, iT[:len(th)]]
        nc = len(cv)
        diff_events = np.abs(mine - theirs) * nc
        out.append(f"{c}|{p}: ncen {nc} vs {ref['ncen'][f'{c}|{p}']}, max |diff| in #centres with N>=k over (k<=6, theta<={th.max():g}') = {diff_events.max():.3f}")
        if c == "det5" and p == "det5":
            pairs = N.sum(0) / 2
            out.append(f"   det5 pairs (sum N/2) mine {pairs[[0,2,4,6,8,11,13]]} | theirs {ref['pairs_det5'][[0,2,4,6,8,11,13]]}")
            nge = np.array([[(N[:, j] >= n).sum() for j in range(len(THETAS))] for n in range(1, 6)])
            out.append(f"   nge_det5 max |diff| = {np.abs(nge - ref['nge_det5']).max()}")
        if c == "det4" and p == "det4":
            out.append(f"   det4 pairs mine {(N.sum(0)/2)[[0,4,8,13]]} | theirs {ref['pairs_det4'][[0,4,8,13]]}")
    tri = [triangles(S["det5"], t) for t in [1, 2, 3, 5, 7.5, 10.0]]
    out.append(f"   det5 triangles mine {tri} | theirs {list(ref['tri_det5'])}")
    out.append(f"   npop mine {npop} | theirs { {k: ref['npop'][k] for k in npop} }")
    print(f"== seed {seed} ({time.time()-t0:.1f}s)\n" + "\n".join(out), flush=True)
