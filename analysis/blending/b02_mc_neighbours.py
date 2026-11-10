"""Monte Carlo neighbour sums on full-sky M>1e13 catalogues.

For every halo A with q_A > Q_PRIM (noise-free mean SNR) compute, over ALL other halos B (M>1e13):
   D_A[c]    = sum_B rho(thA, thB, s_AB) Y_B          (class c = M_B in [1e13,3e13), [3e13,1e14), >=1e14)
   xmax_A[c] = max_B rho Y_B
Also for random sky positions with fixed filter scale theta_A (Slivnyak: the neighbour field of a
halo is the same as that of a random point) -> direct test of the P(D) prediction.
Also: brute-force check of the zone search for 150 primaries (sky 1 only).
Usage: python b02_mc_neighbours.py seed [seed ...]   (seeds 1-3: stored catalogues; others: drawn)
"""
import os, sys, time, json, resource
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import sampler, signal as sg, cosmo_hmf as ch

Q_PRIM = 0.5
TIERS = [0.0, 2.0, 10.0, 1e9]
CAT = "/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed{}.npz"
FIELDS = ["ra", "dec", "M500", "Y500", "theta500", "q", "q_obs"]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


@nb.njit(parallel=True, cache=True)
def brute(qra, qdec, qth, qself, ra, dec, Y, th, cls, rho, lth0, dlth, R_BEAM, D):
    d2r = np.pi / 180.0
    for i in range(qra.shape[0]):
        cd0 = np.cos(qdec[i] * d2r)
        acc = np.zeros((nb.get_num_threads(), 3))
        n = ra.shape[0]
        nthr = nb.get_num_threads(); ch_ = (n + nthr - 1) // nthr
        for tid in nb.prange(nthr):
            for j in range(tid * ch_, min(n, (tid + 1) * ch_)):
                if j == qself[i]:
                    continue
                sdd = np.sin(0.5 * (np.float64(dec[j]) - qdec[i]) * d2r)
                sda = np.sin(0.5 * (np.float64(ra[j]) - qra[i]) * d2r)
                hv = sdd * sdd + cd0 * np.cos(np.float64(dec[j]) * d2r) * sda * sda
                s = 2.0 * np.arcsin(min(1.0, np.sqrt(hv))) / d2r * 60.0
                rr = bl.rho_at(rho, lth0, dlth, R_BEAM, qth[i], np.float64(th[j]), s)
                acc[tid, cls[j]] += rr * np.float64(Y[j])
        for tid in range(nthr):
            for c in range(3):
                D[i, c] += acc[tid, c]


def run_sky(seed, tab, out_dir, rand_thA, rand_n, do_brute=False):
    t0 = time.time()
    if seed <= 3:
        cat = sampler.load_catalog(CAT.format(seed), fields=FIELDS)
    else:
        cat = sampler.draw_catalog(seed, Mmin=1e13, fields=tuple(FIELDS))
    N = len(cat["ra"])
    print(f"[sky {seed}] N = {N}, load/draw {time.time()-t0:.1f}s, rss {rss_gb():.2f} GB", flush=True)
    cls = bl.mass_class(cat["M500"])
    ybar = np.array([cat["Y500"][cls == c].astype(np.float64).sum() for c in range(3)]) / bl.ARCMIN2_FULLSKY
    prim = np.nonzero(cat["q"] > Q_PRIM)[0]
    P = {k: cat[k][prim].copy() for k in FIELDS}
    P["idx"] = prim.astype(np.int64)
    for k in ("q", "q_obs", "M500"):
        del cat[k]
    orig = np.arange(N, dtype=np.int32)
    zsets = []
    th = cat["theta500"]
    for lo, hi in zip(TIERS[:-1], TIERS[1:]):
        s = (th > lo) & (th <= hi)
        zsets.append(bl.ZoneSet(cat["ra"][s], cat["dec"][s], cat["Y500"][s], th[s], cls[s], orig[s]))
        print(f"   tier ({lo},{hi}]: {zsets[-1].n} halos, thmax {zsets[-1].thmax:.2f}'", flush=True)
    keep_for_brute = None
    if do_brute:
        keep_for_brute = {k: cat[k] for k in ("ra", "dec", "Y500", "theta500")}
        keep_for_brute["cls"] = cls
    del cat, th
    print(f"   zones built {time.time()-t0:.1f}s, rss {rss_gb():.2f} GB", flush=True)

    t1 = time.time()
    D, xmax, npair = bl.neighbour_sums(zsets, tab, P["ra"], P["dec"], P["theta500"], P["idx"])
    print(f"   primaries: {len(prim)} (q>{Q_PRIM}), pairs {npair.sum():.3e}, {time.time()-t1:.1f}s", flush=True)

    if do_brute:
        rng = np.random.default_rng(seed + 999)
        # mix of typical and large-theta primaries
        pick = np.concatenate([rng.choice(len(prim), 120, replace=False),
                               np.argsort(P["theta500"])[-30:]])
        Db = np.zeros((len(pick), 3))
        tb = time.time()
        brute(P["ra"][pick].astype(np.float64), P["dec"][pick].astype(np.float64),
              P["theta500"][pick].astype(np.float64), P["idx"][pick],
              keep_for_brute["ra"], keep_for_brute["dec"], keep_for_brute["Y500"],
              keep_for_brute["theta500"], keep_for_brute["cls"], *tab.args(), Db)
        rel = np.abs(Db.sum(1) - D[pick].sum(1)) / np.abs(Db.sum(1))
        print(f"   BRUTE-FORCE check ({len(pick)} primaries, {time.time()-tb:.1f}s): "
              f"max |D_zone - D_brute|/D_brute = {rel.max():.2e}; per-class max abs rel "
              f"{np.max(np.abs(Db - D[pick]) / np.maximum(np.abs(Db), 1e-30)):.2e}", flush=True)
        del keep_for_brute

    # random points
    rng = np.random.default_rng(10_000 + seed)
    RD = {}
    t2 = time.time()
    for thA, n in zip(rand_thA, rand_n):
        ra = 360 * rng.random(n); dec = np.degrees(np.arcsin(2 * rng.random(n) - 1))
        Dr, xr, _ = bl.neighbour_sums(zsets, tab, ra, dec, np.full(n, thA), np.full(n, -1))
        RD[f"{thA:.6f}"] = (Dr, xr)
    print(f"   random points done {time.time()-t2:.1f}s", flush=True)
    np.savez(os.path.join(out_dir, f"mc_sky{seed}.npz"), ybar=ybar, N=N, Q_PRIM=Q_PRIM,
             D=D, xmax=xmax, npair=npair, **{f"p_{k}": v for k, v in P.items()},
             rand_thA=np.array(rand_thA), **{f"rand_D_{i}": RD[f"{t:.6f}"][0] for i, t in enumerate(rand_thA)},
             **{f"rand_xmax_{i}": RD[f"{t:.6f}"][1] for i, t in enumerate(rand_thA)})
    print(f"[sky {seed}] done {time.time()-t0:.1f}s, peak rss {rss_gb():.2f} GB", flush=True)


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [1]
    tab = bl.load_rho_table()
    out_dir = bl.DATA; os.makedirs(out_dir, exist_ok=True)
    # random-point filter scales: table nodes closest to 0.5, 1, 2, 4, 8 arcmin
    rand_thA = [float(tab.th[np.argmin(np.abs(np.log(tab.th / t)))]) for t in (0.5, 1, 2, 4, 8)]
    rand_n = [200_000, 200_000, 200_000, 100_000, 50_000]
    print("random-point theta_A nodes:", np.round(rand_thA, 4), flush=True)
    for s in seeds:
        run_sky(s, tab, out_dir, rand_thA, rand_n, do_brute=(s == 1))


if __name__ == "__main__":
    main()
