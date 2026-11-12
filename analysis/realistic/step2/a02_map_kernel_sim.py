"""Arbiter / a02 (simulation part): the blended detection kernel measured directly on painted maps at RANDOM points.

Question (C5, 'partially' from the maps verifier): do painted, filtered maps give a different blended detection
probability from the build's P(D) kernel at the theta_A nodes that carry high-z, small clusters (theta500 ~ 0.3-5')?
In a Poisson sky the neighbour field around a halo has the same distribution as at a random point (Slivnyak), so the
kernel K(theta_j, q_A; e) = P(A dominant and q_A + n + delta > e) can be measured at random points, with many more
independent samples per node than the halos themselves give.

Per fresh periodic 10 x 10 deg patch (the maps verifier's machinery, imported read-only from verify_real/step2_maps/vlib2
and verify_real/step1_maps/vlib: Poisson sampler M500 > 1e13, Arnaud painter, the build's MMF y-response filter,
type-2 NUFFT read-out, its Hankel rho tables and pair search for the brightest single neighbour):
  * NPTS random points (the same points for every node);
  * at nodes j = 7..25 (theta 0.29-4.69') of the baseline and goal MMF: delta_j(x) = Y_hat_j(x) / sigma_j (all halos are
    neighbours of a random point) and x_max,j(x) = brightest single-neighbour contribution (only neighbours with a peak
    contribution >= X0 = 0.5 are searched, so x_max is exact wherever the dominance threshold fdom q_A >= 0.5);
  * stored compactly: histogram of delta (bin 0.005, -4..16) for points with x_max < X0 (dominant for every q_A >= 0.5),
    and (delta, x_max) per point for the others.
Seeds: [2026, 11, 13, 5, p] (the maps verifier used [2026, 11, 12, 2, p]).
Usage: a02_map_kernel_sim.py P0 NPATCH   (VNTHREADS threads).  Output: data_real/step2/a02_mapkernel/patch_<p>.npz
"""
import os, sys, time, resource
os.environ.setdefault("VNTHREADS", "3")
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/verify_real/step2_maps")
import vlib2 as W                                      # noqa: E402  (read-only use of the maps verifier's code)
from vlib2 import V, np                                 # noqa: E402

P0, NPATCH = int(sys.argv[1]), int(sys.argv[2])
NPTS = int(os.environ.get("NPTS", "40000"))
NODES = list(range(7, 26))
MODES = ("baseline", "goal")
X0 = 0.5
DLO, DHI, DD = -4.0, 16.0, 0.005
NBIN = int(round((DHI - DLO) / DD))
OUT = f"{ROOT}/data_real/step2/a02_mapkernel"
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
log = W.Tee(os.path.join(HERE, f"a02_map_kernel_sim_{P0:03d}.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
SMP = V.PatchSampler(G.L)
blk = W.Block(G)
MF = {m: W.ModeFilter(G, P, m, blk, keep_g=False) for m in MODES}
TAB, ENV = {}, {}
for m in MODES:
    d = np.load(f"{ROOT}/verify_real/step2_maps/out/rho_{m}.npz")
    tab_, env_ = d["tab"], d["E"]
    TAB[m] = {j: np.ascontiguousarray(tab_[j]) for j in NODES}
    ENV[m] = {j: np.ascontiguousarray(env_[j]) for j in NODES}
    del tab_, env_
LTB = W.LTB0 + W.DLTB * np.arange(W.NTB)
log(f"[a02] patches {P0}..{P0+NPATCH-1}; {NPTS} random points per patch; nodes {NODES[0]}..{NODES[-1]} "
    f"({MF['goal'].th[NODES[0]]:.3f}-{MF['goal'].th[NODES[-1]]:.2f}'); modes {MODES}; X0 = {X0}; threads {W.NTHREADS}; "
    f"set-up {time.time()-t0:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
aid = -np.ones(NPTS, np.int64)
for p in range(P0, P0 + NPATCH):
    tp = time.time()
    rng = np.random.default_rng([2026, 11, 13, 5, p])
    h = SMP.draw(rng)
    keep = h["th"] <= 60.0
    x, y, Y, th = (np.ascontiguousarray(h[k][keep]) for k in ("x", "y", "Y", "th"))
    ids = np.arange(x.size, dtype=np.int64)
    lth = np.log(th)
    mp = P.paint(G, x, y, Y, th)
    yk = G.rfft(mp)
    del mp
    ybp = blk.take(yk) * blk.phase
    del yk
    xr = rng.random(NPTS) * G.L
    yr = rng.random(NPTS) * G.L
    out = dict(area=G.L ** 2, nhalo=x.size, nskip=int((~keep).sum()), npts=NPTS)
    tnu = tdom = 0.0
    nbr = 0
    for m in MODES:
        mf = MF[m]
        mom = np.zeros((len(NODES), 4))
        for k, jn in enumerate(NODES):
            ta = time.time()
            v = blk.evaluate(mf.K32[jn] * ybp, xr, yr)
            tnu += time.time() - ta
            dl = v / mf.sig[jn]
            ta = time.time()
            amp = Y / mf.sig[jn]
            xpk = np.interp(lth, LTB, ENV[m][jn][:, 0]) * amp
            br = np.nonzero(xpk >= X0)[0]
            Rb = W.search_radius(ENV[m][jn], lth[br], X0 / amp[br])
            xm = np.zeros(NPTS)
            W.xmax_pairs(xr, yr, aid, x[br], y[br], ids[br], lth[br], amp[br], Rb, TAB[m][jn], G.L, xm)
            tdom += time.time() - ta
            nbr += br.size
            bulk = (xm < X0) & (dl >= DLO) & (dl < DHI)
            ib = ((dl[bulk] - DLO) / DD).astype(np.int64)
            out[f"{m}_{jn}_h"] = np.bincount(np.minimum(ib, NBIN - 1), minlength=NBIN).astype(np.int32)
            out[f"{m}_{jn}_td"] = dl[~bulk].astype(np.float32)
            out[f"{m}_{jn}_tx"] = xm[~bulk].astype(np.float32)
            mom[k] = (dl.sum(), (dl * dl).sum(), (dl > 3).sum(), (xm >= X0).sum())
        out[f"{m}_mom"] = mom
    np.savez(os.path.join(OUT, f"patch_{p:04d}.npz"), **out)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    k1 = NODES.index(15)
    log(f"   patch {p}: {x.size:,} halos; goal node 1.007': mean delta {out['goal_mom'][k1,0]/NPTS:+.4f}, var "
        f"{out['goal_mom'][k1,1]/NPTS - (out['goal_mom'][k1,0]/NPTS)**2:.4f}, P(x_max >= {X0}) {out['goal_mom'][k1,3]/NPTS:.4f}; "
        f"bright {nbr:,}; nufft {tnu:.1f}s dom {tdom:.1f}s total {time.time()-tp:.1f}s [{time.time()-t0:.0f}s]; "
        f"peak rss {rss:.2f} GB")
log(f"[a02] done {time.time()-t0:.0f}s")
