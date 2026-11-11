"""Repair of the c03c conditional map test (verifier dispute on C3): paired test of the blending effect.
c03c compared (i) the blended kernel with q_cat = (own + net + N)/sigma_k for dominant rows and (ii) the naive kernel
with the no-blending control q_ctrl = (own + N)/sigma_k.  Both use the SAME map noise N, so the blending effect
proper is tested by the paired residual  [H_b - H_ctrl] - [M_b - M_n]  (common noise cancels), patch bootstrap.
Decomposition of the blending effect (same rows, same bins, q_obs > 5, 12 z x 10 q):
   dominance loss : measured  -H(q_ctrl | not dominant)            model  -sum (1 - P_dom) Phi-naive
   shift/scatter  : measured  H(q_cat|dom) - H(q_ctrl|dom)          model  M_b - sum P_dom Phi-naive
P_dom per row is interpolated from the kernel table (ln theta, q nodes).  Split by filter scale k as well."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cclib as cc

D = np.load(f"{cc.OUT}/map_detections.npz")
R, zr, TK = D["rows"], D["rows_z"], D["THETA_K"]
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
G, Pdom, th_n = K["G"], K["Pdom"], K["th"]
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
nz, nq, NP = len(ZE) - 1, len(QE) - 1, 324
pid, k, own, net, mx, N, sk = (R[:, 0].astype(int), R[:, 2].astype(int), R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9])
q_own = own / sk; q_cat = (own + net + N) / sk; q_ctrl = (own + N) / sk; dom = own >= mx
zb = np.searchsorted(ZE, zr, side="right") - 1; ok = (zb >= 0) & (zb < nz)
comp = np.where(ok, pid * nz + zb, NP * nz).astype(np.int64)
ip = RegularGridInterpolator((np.log(th_n), cc.QNODES), Pdom, bounds_error=False, fill_value=None)
pdr = np.clip(ip(np.column_stack([np.clip(np.log(TK[k]), np.log(th_n[0]), np.log(th_n[-1])),
                                  np.clip(q_own, cc.QNODES[0], cc.QNODES[-1])])), 0, 1)
BOOT = np.random.default_rng(2024).integers(0, NP, (4000, NP))


def models(sel):
    rp = dict(lth=np.log(TK[k][sel]), q=q_own[sel], w=np.ones(sel.sum()), zb=comp[sel])
    Mb = cc.counts_blended(G, rp, QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)
    Mn = cc.counts_gauss(rp, QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)
    Mnd = cc.counts_gauss(dict(rp, w=pdr[sel]), QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)
    return Mb, Mn, Mnd


def hist(q, sel):
    H = np.zeros((NP, nz, nq))
    qi = np.searchsorted(QE, q, side="right") - 1
    s = sel & ok & (qi >= 0) & (qi < nq)
    np.add.at(H, (pid[s], zb[s], qi[s]), 1)
    return H


def report(lab, meas, mod):
    res = meas - mod
    tot = res.sum(); bt = res[BOOT].sum(tuple(range(1, res.ndim + 1))); s = bt.std(ddof=1)
    tq = res.sum((0, 1)); sq = res[BOOT].sum((1, 2)).std(0, ddof=1)
    print(f"  {lab}: measured {meas.sum():+.1f}, model {mod.sum():+.1f}, residual {tot:+.1f} +- {s:.1f} ({tot/s:+.2f} sigma),"
          f" meas/model {meas.sum()/mod.sum():.3f} +- {s/abs(mod.sum()):.3f}")
    print(f"     q-marginal residual: {np.array2string(tq, precision=0)}; pulls {np.array2string(tq/sq, precision=2)}")
    return tot, s


allr = np.ones(len(R), bool)
Mb, Mn, Mnd = models(allr)
Hb = hist(q_cat, dom); Hc = hist(q_ctrl, allr); Hcd = hist(q_ctrl, dom)
print("=== c03c reproduction (unpaired) ===")
report("blended kernel vs q_cat (dominant)", Hb, Mb)
report("no-blending control: naive kernel vs q_ctrl (all rows)", Hc, Mn)
print("=== paired blending effect (common noise cancels) ===")
report("total blending effect  [Hb - Hc] vs [Mb - Mn]", Hb - Hc, Mb - Mn)
report("dominance loss  -H(q_ctrl|not dom) vs -(Mn - Mnd)", -(Hc - Hcd), -(Mn - Mnd))
report("shift/scatter   H(q_cat|dom) - H(q_ctrl|dom) vs Mb - Mnd", Hb - Hcd, Mb - Mnd)
print(f"  fraction of rows dominant: map {dom.mean():.5f}, model mean P_dom {pdr.mean():.5f};"
      f" among q_own>4: map {dom[q_own>4].mean():.5f}, model {pdr[q_own>4].mean():.5f}")
print("=== paired total blending effect by filter scale k ===")
for kk in np.unique(k):
    sel = k == kk
    if (q_own[sel] > 4).sum() < 200:
        continue
    Mb_, Mn_, Mnd_ = models(sel)
    hb = hist(q_cat, dom & sel); hc = hist(q_ctrl, sel); hcd = hist(q_ctrl, dom & sel)
    res = (hb - hc) - (Mb_ - Mn_); s = res[BOOT].sum((1, 2, 3)).std(ddof=1)
    rd = -(hc - hcd) + (Mn_ - Mnd_); sd = rd[BOOT].sum((1, 2, 3)).std(ddof=1)
    rs = (hb - hcd) - (Mb_ - Mnd_); ss = rs[BOOT].sum((1, 2, 3)).std(ddof=1)
    print(f"  k={kk} (theta {TK[kk]:.2f}'): effect meas {(hb-hc).sum():+.0f} model {(Mb_-Mn_).sum():+.1f} resid {res.sum():+.0f} +- {s:.0f} "
          f"({res.sum()/s:+.2f}) | dominance resid {rd.sum():+.0f} +- {sd:.0f} | shift resid {rs.sum():+.0f} +- {ss:.0f}")
