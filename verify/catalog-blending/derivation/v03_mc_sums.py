"""Verifier's own catalogue-level neighbour sums (independent of analysis/blending code).
Kernel: v01_kernel.npz (Fourier-slice template, own Hankel J0).  Neighbour search: HEALPix (ring)
pixel buckets per theta tier; primaries grouped by nside-256 pixel, one query_disc per group.
For each primary A (q_A>0.5) and each mass class c: D[c] = sum_{B!=A} rho(thA,thB,s) Y_B,
xm[c] = max_B rho Y_B.  Also random points at fixed theta_A.
Usage: python v03_mc_sums.py seed   (stored catalogue cat_M1e13_seed{seed}.npz)"""
import os, sys, time, resource
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, numba as nb, healpy as hp

HERE = os.path.dirname(os.path.abspath(__file__))
KD = np.load(os.path.join(HERE, "v01_kernel.npz"))
TH = KD["TH"]; RHO = np.ascontiguousarray(KD["RHO"]); RB = float(KD["RB"])
LT0 = float(np.log(TH[0])); DLT = float(np.log(TH[1] / TH[0]))
EDGES = np.array([1e13, 3e13, 1e14])
TIERS = [(0.0, 2.0, 256), (2.0, 10.0, 64), (10.0, 1e9, 16)]
NSQ = 256
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


@nb.njit(cache=True)
def rho_i(thA, thB, s):
    S = 5.0 * thA + 5.0 * thB + RB
    if s >= S:
        return 0.0
    NT = RHO.shape[0]; NU = RHO.shape[2]
    fa = (np.log(thA) - LT0) / DLT; fb = (np.log(thB) - LT0) / DLT; ft = np.sqrt(s / S) * (NU - 1)
    fa = min(max(fa, 0.0), NT - 1.000001); fb = min(max(fb, 0.0), NT - 1.000001); ft = min(ft, NU - 1.000001)
    ia = int(fa); ib = int(fb); it = int(ft); wa = fa - ia; wb = fb - ib; wt = ft - it
    r = 0.0
    for da in range(2):
        xa = wa if da == 1 else 1.0 - wa
        for db in range(2):
            xb = wb if db == 1 else 1.0 - wb
            r += xa * xb * ((1.0 - wt) * RHO[ia + da, ib + db, it] + wt * RHO[ia + da, ib + db, it + 1])
    return r


@nb.njit(parallel=True, cache=True)
def kernel(gstart, porder, px, py, pz, pth, pself, goff, gpix, pixstart, hx, hy, hz, hY, hth, hcls, hidx, D, XM):
    ng = gstart.shape[0] - 1
    for g in nb.prange(ng):
        for a in range(gstart[g], gstart[g + 1]):
            i = porder[a]
            ax = np.float64(px[i]); ay = np.float64(py[i]); az = np.float64(pz[i]); thA = pth[i]
            for kk in range(goff[g], goff[g + 1]):
                p = gpix[kk]
                for j in range(pixstart[p], pixstart[p + 1]):
                    if hidx[j] == pself[i]:
                        continue
                    dx = ax - hx[j]; dy = ay - hy[j]; dz = az - hz[j]
                    ch = np.sqrt(dx * dx + dy * dy + dz * dz)
                    s = 2.0 * np.arcsin(min(1.0, 0.5 * ch)) * 10800.0 / np.pi
                    if s >= 5.0 * thA + 5.0 * hth[j] + RB:
                        continue
                    x = rho_i(thA, np.float64(hth[j]), s) * np.float64(hY[j])
                    c = hcls[j]
                    D[i, c] += x
                    if x > XM[i, c]:
                        XM[i, c] = x


@nb.njit(parallel=True, cache=True)
def brute(ax, ay, az, thA, selfi, hx, hy, hz, hY, hth, hcls, out):
    n = hx.shape[0]
    nthr = nb.get_num_threads(); acc = np.zeros((nthr, 3)); ch_ = (n + nthr - 1) // nthr
    for t in nb.prange(nthr):
        for j in range(t * ch_, min(n, (t + 1) * ch_)):
            if j == selfi:
                continue
            dx = ax - hx[j]; dy = ay - hy[j]; dz = az - hz[j]
            s = 2.0 * np.arcsin(min(1.0, 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz))) * 10800.0 / np.pi
            acc[t, hcls[j]] += rho_i(thA, np.float64(hth[j]), s) * np.float64(hY[j])
    for t in range(nthr):
        for c in range(3):
            out[c] += acc[t, c]


def build_tier(x, y, z, Y, th, cls, idx, nside):
    pix = hp.vec2pix(nside, x, y, z)
    o = np.argsort(pix, kind="stable")
    ps = np.searchsorted(pix[o], np.arange(hp.nside2npix(nside) + 1)).astype(np.int64)
    return dict(nside=nside, ps=ps, x=x[o].astype(np.float64), y=y[o].astype(np.float64), z=z[o].astype(np.float64),
                Y=Y[o], th=th[o], cls=cls[o], idx=idx[o], thmax=float(th.max()) if len(th) else 0.0)


def sums(qx, qy, qz, qth, qself, tiers):
    n = len(qx)
    D = np.zeros((n, 3)); XM = np.zeros((n, 3))
    gp = hp.vec2pix(NSQ, qx, qy, qz)
    order = np.argsort(gp, kind="stable").astype(np.int64)
    gps = gp[order]
    ug, gstart = np.unique(gps, return_index=True)
    gstart = np.append(gstart, n).astype(np.int64)
    # max thA per group
    thmax_g = np.maximum.reduceat(qth[order], gstart[:-1])
    prad = hp.max_pixrad(NSQ)
    for T in tiers:
        if len(T["Y"]) == 0:
            continue
        offs = [0]; pl = []
        for g, pix in enumerate(ug):
            v = hp.pix2vec(NSQ, int(pix))
            r = np.radians((5 * thmax_g[g] + 5 * T["thmax"] + RB + 0.01) / 60.0) + prad
            lst = hp.query_disc(T["nside"], v, r, inclusive=True)
            pl.append(lst); offs.append(offs[-1] + len(lst))
        gpix = np.concatenate(pl).astype(np.int64); goff = np.array(offs, np.int64)
        kernel(gstart, order, qx, qy, qz, qth, qself, goff, gpix, T["ps"], T["x"], T["y"], T["z"], T["Y"], T["th"],
               T["cls"], T["idx"], D, XM)
    return D, XM


def main():
    seed = int(sys.argv[1])
    t0 = time.time()
    f = np.load(f"/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed{seed}.npz")
    ra = f["ra"].astype(np.float64); dec = f["dec"].astype(np.float64)
    x, y, z = hp.ang2vec(ra, dec, lonlat=True).T.astype(np.float32)
    del ra, dec
    Y = f["Y500"]; th = f["theta500"]; M = f["M500"]
    cls = (np.searchsorted(EDGES, M, side="right") - 1).astype(np.int8); del M
    q = f["q"]
    prim = np.nonzero(q > 0.5)[0]
    P = dict(q=q[prim], qo=f["q_obs"][prim], th=th[prim].astype(np.float64), x=x[prim].astype(np.float64),
             y=y[prim].astype(np.float64), z=z[prim].astype(np.float64), idx=prim.astype(np.int64))
    del q
    ybar = np.array([Y[cls == c].astype(np.float64).sum() for c in range(3)]) / (4 * np.pi / (np.pi / 10800) ** 2)
    N = len(Y)
    idx = np.arange(N, dtype=np.int64)
    tiers = []
    for lo, hi, ns in TIERS:
        s = (th > lo) & (th <= hi)
        tiers.append(build_tier(x[s], y[s], z[s], Y[s], th[s], cls[s], idx[s], ns))
        print(f"[sky {seed}] tier ({lo},{hi}]: {s.sum()} halos thmax {tiers[-1]['thmax']:.2f}", flush=True)
    if seed == 1:
        BR = dict(x=x.astype(np.float64), y=y.astype(np.float64), z=z.astype(np.float64), Y=Y, th=th, cls=cls)
    del x, y, z, Y, th, cls, idx
    print(f"[sky {seed}] N={N}, primaries {len(prim)}, setup {time.time()-t0:.0f}s rss {rss():.2f}GB", flush=True)
    t1 = time.time()
    D, XM = sums(P["x"], P["y"], P["z"], P["th"], P["idx"], tiers)
    print(f"[sky {seed}] primary sums {time.time()-t1:.0f}s rss {rss():.2f}GB", flush=True)
    if seed == 1:
        rb = np.random.default_rng(5)
        pick = np.concatenate([rb.choice(len(prim), 40, replace=False), np.argsort(P["th"])[-10:]])
        err = []
        for i in pick:
            o = np.zeros(3)
            brute(P["x"][i], P["y"][i], P["z"][i], P["th"][i], P["idx"][i], BR["x"], BR["y"], BR["z"], BR["Y"], BR["th"], BR["cls"], o)
            err.append(np.abs(o.sum() - D[i].sum()) / o.sum())
        print(f"[sky 1] brute-force check on {len(pick)} primaries: max rel diff {max(err):.2e}", flush=True)
        del BR
    rng = np.random.default_rng(777 + seed)
    R = {}
    for tA, n in ((0.524, 100000), (0.946, 100000), (1.983, 100000), (4.153, 60000), (7.505, 40000)):
        v = hp.ang2vec(360 * rng.random(n), np.degrees(np.arcsin(2 * rng.random(n) - 1)), lonlat=True)
        Dr, Xr = sums(v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy(), np.full(n, tA), np.full(n, -1, np.int64), tiers)
        R[f"rD_{tA}"] = Dr; R[f"rX_{tA}"] = Xr
        R[f"rv_{tA}"] = v.astype(np.float32)
    print(f"[sky {seed}] random points {time.time()-t1:.0f}s", flush=True)
    np.savez(os.path.join(HERE, f"v03_sky{seed}.npz"), ybar=ybar, N=N, D=D, XM=XM,
             **{f"p_{k}": v for k, v in P.items()}, **R)
    print(f"[sky {seed}] done {time.time()-t0:.0f}s peak rss {rss():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
