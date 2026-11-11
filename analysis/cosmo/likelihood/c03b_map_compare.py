"""Blended convolution model vs the map-level simulations (32,400 deg^2).
 (1) Kernel test on the catalogue-level rows of the maps (true position, filter at scale k_own, the map's own
     noise and painted neighbours, patch-mean removal): predicted vs measured counts of q_cat = (own+net+N)/sigma_k
     for dominant halos (own >= brightest neighbour), conditional on each row's own SNR q_own = own/sigma_k.
 (2) Map detections (maxQ peaks over 12 scales, 2' merge) in (z_primary, Q) bins vs the naive and blended
     population models (no free parameter), then with a 1-parameter pipeline term q -> sqrt(q^2 + d)
     ('optimisation bias' of maximising over position and scale, as used by SPT with d = 3).
Errors: Poisson on the model counts, and a patch bootstrap (1000 resamples) of the measured counts."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import chi2 as chi2d
from scipy.optimize import minimize_scalar
import cclib as cc

D = np.load(f"{cc.OUT}/map_detections.npz")
det, R, zr, TK, AREA = D["det"], D["rows"], D["rows_z"], D["THETA_K"], float(D["area_deg2"])
G = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
nz, nq = len(ZE) - 1, len(QE) - 1
rng = np.random.default_rng(7)
NP = 324
BOOT = rng.integers(0, NP, (1000, NP))


def hist_patch(pid, z, q, sel):
    H = np.zeros((NP, nz, nq))
    zi = np.searchsorted(ZE, z, side="right") - 1; qi = np.searchsorted(QE, q, side="right") - 1
    ok = sel & (zi >= 0) & (zi < nz) & (qi >= 0) & (qi < nq)
    np.add.at(H, (pid[ok], zi[ok], qi[ok]), 1)
    return H


def report(lab, Hp, M, extra=""):
    """Hp: per-patch counts (NP, nz, nq); M: model counts (nz, nq) for the same area."""
    Hm = Hp.sum(0)
    bs = Hp[BOOT].sum(1)
    var = np.maximum(M, 1e-9)                       # Poisson variance of the model
    good = M > 5
    z_ = (Hm - M) / np.sqrt(var)
    c2 = np.sum(z_[good] ** 2); nd = good.sum()
    bvr = np.median((bs.var(0, ddof=1) / np.maximum(Hm, 1))[good])
    qm, qmm = Hm.sum(0), M.sum(0); zm, zmm = Hm.sum(1), M.sum(1)
    print(f"  {lab}: total meas {Hm.sum():.0f} model {M.sum():.1f} ({100*(Hm.sum()/M.sum()-1):+.2f}%) | (z,q) chi2 = {c2:.1f}/{nd}"
          f" (p = {chi2d.sf(c2, nd):.2g}), max|pull| {np.abs(z_[good]).max():.1f}; bootstrap var/count (median) {bvr:.2f} {extra}")
    print(f"       q marginal meas/model-1 [%]: {np.array2string(100*(qm/qmm-1), precision=1)}")
    print(f"       z marginal meas/model-1 [%]: {np.array2string(100*(zm/zmm-1), precision=1)}")
    return c2, nd


# ------------------------------------------------------------------ (1) kernel test on map catalogue rows
print("=== (1) kernel test: map catalogue-level rows (true position, own scale), conditional on q_own ===")
pid, k, qt, own, net, mx, N, sk = (R[:, 0].astype(int), R[:, 2].astype(int), R[:, 3], R[:, 5], R[:, 6], R[:, 7],
                                     R[:, 8], R[:, 9])
q_own = own / sk; q_cat = (own + net + N) / sk; dom = own >= mx
zb = np.searchsorted(ZE, zr, side="right") - 1; zb[(zr < ZE[0]) | (zr >= ZE[-1])] = -1
rowpop = dict(lth=np.log(TK[k]), q=q_own, w=np.ones(len(R)), zb=zb.astype(np.int64))
Mb = cc.counts_blended(G, rowpop, QE, nz)
Mn = cc.counts_gauss(rowpop, QE, nz)
report("blended kernel vs q_cat (dominant)", hist_patch(pid, zr, q_cat, dom), Mb)
report("naive kernel vs (own+N)/sigma     ", hist_patch(pid, zr, (own + N) / sk, np.ones(len(R), bool)), Mn)
report("naive kernel vs q_cat (dominant)  ", hist_patch(pid, zr, q_cat, dom), Mn, "(= ignoring blending)")
# realised patch mean vs ensemble mean: the map subtracts each patch's own ybar
print(f"  patch-level P_dom check: measured fraction of rows with q_own in [5,10) that are dominant "
      f"{dom[(q_own>=5)&(q_own<10)].mean():.4f}")

# ------------------------------------------------------------------ (2) map detections vs population model
print("\n=== (2) map detections (maxQ, merge 2') vs population model scaled to 32,400 deg^2 ===")
scale = AREA / cc.FULLSKY_DEG2
pop = cc.population(cc.FID, zedges=ZE)
Nb = cc.counts_blended(G, pop, QE, nz) * scale
Nn = cc.counts_gauss(pop, QE, nz) * scale
dpid = det[:, 0].astype(int); Q = det[:, 1]; zp = det[:, 3]; dup = det[:, 5] > 0
H_all = hist_patch(dpid, zp, Q, np.ones(len(det), bool)); H_nd = hist_patch(dpid, zp, Q, ~dup)
for lab, H in (("all detections", H_all), ("non-duplicates", H_nd)):
    report(f"naive  vs {lab}", H, Nn)
    report(f"blended vs {lab}", H, Nb)


def model_d(d, kind="b"):
    p2 = dict(pop); p2["q"] = np.sqrt(pop["q"] ** 2 + d)
    return (cc.counts_blended(G, p2, QE, nz) if kind == "b" else cc.counts_gauss(p2, QE, nz)) * scale


print("\n  with the pipeline term q_A -> sqrt(q_A^2 + d):")
for kind in ("n", "b"):
    for lab, H in (("all detections", H_all), ("non-duplicates", H_nd)):
        Hm = H.sum(0)
        f = lambda d: cc.cash(Hm, model_d(d, kind))
        r = minimize_scalar(f, bounds=(0, 15), method="bounded")
        print(f"   {'blended' if kind=='b' else 'naive  '} vs {lab}: best d = {r.x:.3f}")
        report(f"   {'blended' if kind=='b' else 'naive  '}+d vs {lab}", H, model_d(r.x, kind), f"[1 free parameter d={r.x:.2f}]")
        if kind == "b" and lab == "non-duplicates":
            report(f"   blended+d=3 (SPT value) vs {lab}", H, model_d(3.0, kind), "[no free parameter]")
np.savez(f"{cc.OUT}/map_compare.npz", H_all=H_all, H_nd=H_nd, Nb=Nb, Nn=Nn, ZE=ZE, QE=QE)
