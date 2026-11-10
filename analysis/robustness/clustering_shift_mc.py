"""Propagate the (linear-bias) clustering correction of the mean neighbour boost into the catalogue
blending fractions, on baseline Poisson skies (mean-field approximation):
   delta_A -> delta_A + W(theta_A) mu_A,   W from clustering_limber.py
i.e. the extra correlated neighbours add their mean signal; the change of the fluctuations is ignored.
[revised after verification] W variants (all from clustering_limber.py):
   lin_06 : linear bias, no exclusion, single neighbour size theta_B = 0.6'   (original analysis)
   lin_yw : linear bias, no exclusion, y-weighted theta_B distribution of M>1e13 neighbours
   ex_yw  : as lin_yw, with step halo exclusion (xi = -1 for r < <R500com>_det + <R500com>_y)
   ex_06  : exclusion, theta_B = 0.6'
Paired comparison on the same skies (seeds as given); per-seed ratios give the scatter.
Usage: python clustering_shift_mc.py SEED [SEED ...]
"""
import os, sys, time, resource
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl
import configs as C
from szsim import sampler, signal as sg, cosmo_hmf as ch

L = np.load(os.path.join(ch.DATA_DIR, "robustness", "limber.npz"))
VARS = ["lin_06", "lin_yw", "ex_yw", "ex_06"]
model = sg.SZModel(); rt = bl.RhoTable(1.4)
QC = (4.0, 5.0, 6.0)
per_seed = []   # list of dict (qc, c, tag) -> [Ndet_b, Nsingle, B2(0.1) count, B3 count, B2(0.2) count]
for seed in map(int, sys.argv[1:]):
    t0 = time.time()
    cat = sampler.draw_catalog(seed, Mmin=1e13, model=model)
    bl.add_distances(cat, model)
    N = len(cat["ra"])
    ybar = sg.RATIO_5R500 * cat["Y500"].astype(np.float64).sum() * bl.ARCMIN ** 2 / (4 * np.pi)
    prim = np.nonzero(cat["q"] > C.QPRIM)[0]
    cand = {k: cat[k][prim].copy() for k in ("ra", "dec", "theta500", "q", "q_obs", "chi", "Rcom")}
    for k in ("z", "M500", "q", "q_obs"):
        del cat[k]
    NG = bl.NeighbourGrids(cat, np.arange(N, dtype=np.int32), np.zeros(N, np.int8), 1.4,
                           thA_typ=float(np.median(cand["theta500"])))
    del cat
    S, X, O = NG.run(cand, prim.astype(np.int32), rt, 1)
    del NG
    th = cand["theta500"].astype(np.float64)
    sA = model.sigma_Y(th)
    q = cand["q"].astype(np.float64); n = (cand["q_obs"] - cand["q"]).astype(np.float64)
    D = S[:, 0] / sA; Xm = X[:, 0] / sA
    mu = sA * sg.RATIO_5R500 * ybar / model.noise_y_arcmin ** 2
    dom = Xm < q
    Wv = {v: np.interp(np.log(th), np.log(L["thA_grid"]), L[f"W_{v}"]) for v in VARS}
    acc = {}
    for qc in QC:
        single = q + n > qc
        for c in (0, 1):
            for tag, extra in [("poisson", 0.0)] + [(v, Wv[v] * mu) for v in VARS]:
                dp = D - c * mu + extra
                det = (q + n + dp > qc) & dom
                acc[(qc, c, tag)] = np.array([det.sum(), single.sum(), (det & (dp / q > 0.1)).sum(),
                                              (det & (q + n < qc)).sum(), (det & (dp / q > 0.2)).sum()], float)
    s5 = q + n > 5
    acc["W"] = {v: (np.mean(Wv[v][s5]), np.mean((Wv[v] * mu)[s5])) for v in VARS}
    per_seed.append(acc)
    print(f"seed {seed}: done {time.time()-t0:.0f}s, rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB; "
          + "; ".join(f"{v}: <W>={acc['W'][v][0]:.4f}, <W mu>={acc['W'][v][1]:.4f}" for v in VARS), flush=True)

ns = len(per_seed)
print(f"\n{ns} paired skies. Ratios clustered/Poisson with error = std of per-seed ratios / sqrt(nsky)")
for v in VARS:
    print(f"   {v}: mean W over q>5 single detections {np.mean([a['W'][v][0] for a in per_seed]):.4f}, "
          f"mean shift W*mu {np.mean([a['W'][v][1] for a in per_seed]):.4f}")


def fr(a):
    return np.array([a[0] / a[1], a[2] / a[0], a[3] / a[0], a[4] / a[0]])   # ratio, B2(0.1), B3, B2(0.2)


for qc in QC:
    for c in (0, 1):
        tot = {t: sum(a[(qc, c, t)] for a in per_seed) for t in ["poisson"] + VARS}
        p = fr(tot["poisson"])
        print(f"q_cut={qc:g} {'raw map     ' if c == 0 else 'mean removed'}: Poisson N_b/N_s {p[0]:.4f}, B2(0.1) {p[1]:.4f}, B3 {p[2]:.4f}")
        for v in VARS:
            cl = fr(tot[v])
            rs = np.array([fr(a[(qc, c, v)]) / fr(a[(qc, c, 'poisson')]) for a in per_seed])
            e = rs.std(0, ddof=1) / np.sqrt(ns) if ns > 1 else np.full(4, np.nan)
            print(f"      {v:6s}: N_det_b x{cl[0]/p[0]:.4f}±{e[0]:.4f}, B2(0.1) x{cl[1]/p[1]:.3f}±{e[1]:.3f}, "
                  f"B3 x{cl[2]/p[2]:.3f}±{e[2]:.3f}, B2(0.2) x{cl[3]/p[3]:.3f}±{e[3]:.3f}")
