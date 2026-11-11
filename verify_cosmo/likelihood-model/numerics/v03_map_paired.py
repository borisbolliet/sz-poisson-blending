"""Paired version of the c03c conditional map test.  c03c compares the blended kernel with q_cat=(own+net+N)/sig for
dominant rows (+1.0 +- 145.3) and the naive kernel with the no-blending control (own+N)/sig (-255.3 +- 111.4).
Both share the SAME map noise N, so the blending effect proper is tested by the paired difference
    measured  [H(q_cat | dominant) - H((own+N)/sig)]   vs   model [M_blended - M_naive]
with a patch bootstrap of the paired residual (common noise cancels).  Same rows, bins and kernel as c03c."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import chi2 as chi2d
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc

D = np.load(f"{cc.OUT}/map_detections.npz")
R, zr, TK = D["rows"], D["rows_z"], D["THETA_K"]
G = np.load(f"{cc.OUT}/G_fid_b03rates.npz")["G"]
ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
nz, nq, NP = len(ZE) - 1, len(QE) - 1, 324
pid, k, own, net, mx, N, sk = (R[:, 0].astype(int), R[:, 2].astype(int), R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9])
q_own = own / sk; q_cat = (own + net + N) / sk; dom = own >= mx
zb = np.searchsorted(ZE, zr, side="right") - 1; ok = (zb >= 0) & (zb < nz)
comp = np.where(ok, pid * nz + zb, NP * nz).astype(np.int64)
rowpop = dict(lth=np.log(TK[k]), q=q_own, w=np.ones(len(R)), zb=comp)
Mb = cc.counts_blended(G, rowpop, QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)
Mn = cc.counts_gauss(rowpop, QE, NP * nz + 1)[:-1].reshape(NP, nz, nq)


def hist(q, sel):
    H = np.zeros((NP, nz, nq))
    zi = np.searchsorted(ZE, zr, side="right") - 1; qi = np.searchsorted(QE, q, side="right") - 1
    s = sel & (zi >= 0) & (zi < nz) & (qi >= 0) & (qi < nq)
    np.add.at(H, (pid[s], zi[s], qi[s]), 1)
    return H


Hb = hist(q_cat, dom); Hc = hist((own + N) / sk, np.ones(len(R), bool))
BOOT = np.random.default_rng(2024).integers(0, NP, (4000, NP))
for lab, res in (("blended (c03c repro)", Hb - Mb), ("control (c03c repro)", Hc - Mn), ("PAIRED blending effect", (Hb - Hc) - (Mb - Mn))):
    tot = res.sum(); vt = res[BOOT].sum((1, 2, 3)).var(ddof=1)
    tq = res.sum((0, 1)); vq = res[BOOT].sum((1, 2)).var(0, ddof=1)
    print(f"{lab}: total residual {tot:+.1f} +- {np.sqrt(vt):.1f} ({tot/np.sqrt(vt):+.2f} sigma); q-marginal pulls {np.array2string(tq/np.sqrt(vq), precision=2)}")
print(f"measured blending effect (Hb - Hc) total {(Hb-Hc).sum():+.0f}; model (Mb - Mn) {(Mb-Mn).sum():+.1f}; ratio meas/model {(Hb-Hc).sum()/(Mb-Mn).sum():.3f}")
print(f"  q 5-6 bin: measured {(Hb-Hc).sum((0,1))[0]:+.0f}, model {(Mb-Mn).sum((0,1))[0]:+.1f}")
# effect of the control residual: is the noise normalisation off?  per-row (N/sig) moments for |q_own| region
zN = N / sk
print(f"map noise N/sigma_k over all rows: mean {zN.mean():+.4f}, sd {zN.std():.4f}, n={zN.size}")
for kk in np.unique(k):
    s = k == kk
    if s.sum() > 2000:
        print(f"   scale k={kk} (theta {TK[kk]:.2f}'): n={s.sum()}, mean {zN[s].mean():+.4f} +- {zN[s].std()/np.sqrt(s.sum()):.4f}, sd {zN[s].std():.4f}")
