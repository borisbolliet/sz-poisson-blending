"""Follow-up of c03b.
 (1) Proper conditional test of the blended kernel on the map catalogue rows: per-patch residuals
     (measured - predicted for that patch's own rows), patch-bootstrap covariance (diagonal) of the residual.
 (2) Pipeline term with a theta dependence, q_A -> sqrt(q_A^2 + d0 + d1 ln(theta500/2')), 2 free parameters,
     for the naive and the blended model vs the non-duplicate map detections."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import chi2 as chi2d
from scipy.optimize import minimize
import cclib as cc

D = np.load(f"{cc.OUT}/map_detections.npz")
det, R, zr, TK, AREA = D["det"], D["rows"], D["rows_z"], D["THETA_K"], float(D["area_deg2"])
G = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
nz, nq = len(ZE) - 1, len(QE) - 1
NP = 324
BOOT = np.random.default_rng(11).integers(0, NP, (2000, NP))

pid, k, own, net, mx, N, sk = (R[:, 0].astype(int), R[:, 2].astype(int), R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9])
q_own = own / sk; q_cat = (own + net + N) / sk; dom = own >= mx
zb = np.searchsorted(ZE, zr, side="right") - 1; ok = (zb >= 0) & (zb < nz)
comp = np.where(ok, pid * nz + zb, -1).astype(np.int64)
rowpop = dict(lth=np.log(TK[k]), q=q_own, w=np.ones(len(R)), zb=comp)
Mb = cc.counts_blended(G, rowpop, QE, NP * nz).reshape(NP, nz, nq)
Mn = cc.counts_gauss(dict(rowpop, zb=np.where(ok, comp, NP * nz)), QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)


def hist(z, q, sel):
    H = np.zeros((NP, nz, nq))
    zi = np.searchsorted(ZE, z, side="right") - 1; qi = np.searchsorted(QE, q, side="right") - 1
    s = sel & (zi >= 0) & (zi < nz) & (qi >= 0) & (qi < nq)
    np.add.at(H, (pid[s], zi[s], qi[s]), 1)
    return H


def cond_test(lab, H, M):
    res = H - M
    tot = res.sum(0)
    bs = res[BOOT].sum(1)
    var = bs.var(0, ddof=1)
    good = M.sum(0) > 5
    z_ = tot / np.sqrt(var)
    c2 = np.sum(z_[good] ** 2)
    # q-marginal and total
    tq = res.sum((0, 1)); vq = res[BOOT].sum((1, 2)).var(0, ddof=1)
    tt = res.sum(); vt = res[BOOT].sum((1, 2, 3)).var(ddof=1)
    print(f"  {lab}: (z,q) chi2 = {c2:.1f}/{good.sum()} (p = {chi2d.sf(c2, good.sum()):.3g}), max|pull| {np.abs(z_[good]).max():.2f};"
          f" total residual {tt:+.1f} +- {np.sqrt(vt):.1f} ({tt/np.sqrt(vt):+.2f} sigma) of {M.sum():.0f}")
    print(f"     q-marginal pulls: {np.array2string(tq/np.sqrt(vq), precision=2)}")


print("=== (1) conditional kernel test on map catalogue rows (patch-bootstrap of per-patch residuals) ===")
cond_test("blended kernel vs q_cat (dominant)      ", hist(zr, q_cat, dom), Mb)
cond_test("naive kernel vs (own+N)/sigma (no blend)", hist(zr, (own + N) / sk, np.ones(len(R), bool)), Mn)
cond_test("naive kernel vs q_cat (ignores blending)", hist(zr, q_cat, dom), Mn)

print("\n=== (2) map detections, non-duplicates: pipeline term q_A -> sqrt(q_A^2 + d0 + d1 ln(theta/2')) ===")
scale = AREA / cc.FULLSKY_DEG2
pop = cc.population(cc.FID, zedges=ZE)
dpid = det[:, 0].astype(int); Q = det[:, 1]; zp = det[:, 3]; dup = det[:, 5] > 0
Hm = np.zeros((nz, nq))
zi = np.searchsorted(ZE, zp, side="right") - 1; qi = np.searchsorted(QE, Q, side="right") - 1
s = ~dup & (zi >= 0) & (zi < nz) & (qi >= 0) & (qi < nq)
np.add.at(Hm, (zi[s], qi[s]), 1)


def model(p, kind):
    d = p[0] + p[1] * np.log(pop["th"] / 2.0)
    p2 = dict(pop); p2["q"] = np.sqrt(np.maximum(pop["q"] ** 2 + d, 1e-6))
    return (cc.counts_blended(G, p2, QE, nz) if kind == "b" else cc.counts_gauss(p2, QE, nz)) * scale


for kind in ("n", "b"):
    r = minimize(lambda p: cc.cash(Hm, model(p, kind)), [1.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3))
    M = model(r.x, kind)
    good = M > 5
    z_ = (Hm - M) / np.sqrt(M)
    c2 = np.sum(z_[good] ** 2)
    print(f"  {'blended' if kind=='b' else 'naive  '}: d0 = {r.x[0]:.3f}, d1 = {r.x[1]:.3f}; chi2 = {c2:.1f}/{good.sum()-2} (p = {chi2d.sf(c2, good.sum()-2):.3g}),"
          f" Cash = {r.fun:.1f}, max|pull| {np.abs(z_[good]).max():.2f}")
    print(f"       z marginal meas/model-1 [%]: {np.array2string(100*(Hm.sum(1)/M.sum(1)-1), precision=1)}")
    print(f"       q marginal meas/model-1 [%]: {np.array2string(100*(Hm.sum(0)/M.sum(0)-1), precision=1)}")
