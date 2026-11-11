"""Numerical convergence of the config-b (and d) Fisher in MY model vs mass-grid / GH / z resolution.
nM=121 reproduces the mass spacing of the cosmolib HMF grid (dlnM = 0.0576)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmodel as V
fsky = 0.4
g5 = lambda n: np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1]] + [0] * (n - 2))
def setgrid(nM, ngh):
    V.LNM = np.linspace(np.log(1e13), np.log(1e16), nM)
    V.WM = np.full(nM, V.LNM[1] - V.LNM[0]); V.WM[[0, -1]] *= 0.5
    x, w = np.polynomial.hermite_e.hermegauss(ngh); V.GHX, V.GHW = x, w / w.sum()
    V._cosmo.cache_clear()
def fis(free, pri={}):
    mu = V.counts(V.FID, 5.0, fsky).ravel(); J = V.jac(V.FID.copy(), free, 5.0, fsky)
    F = J.T @ (J / mu[:, None])
    for i, s in pri.items(): k = free.index(i); F[k, k] += 1 / s ** 2
    C = np.linalg.inv(F); g = g5(len(free))
    return np.sqrt(np.diag(C)), np.sqrt(g @ C @ g), mu
ref = None
for nM, ngh in [(121, 40), (241, 40), (241, 60), (481, 60), (961, 100)]:
    setgrid(nM, ngh)
    sb, sSb, mu = fis([0, 1, 2]); sd, sSd, _ = fis([0, 1, 2, 3, 4], {2: 0.032})
    if ref is None: ref = mu
    print(f"nM={nM:4d} (dlnM={V.LNM[1]-V.LNM[0]:.4f}) nGH={ngh}: total {mu.sum():,.2f}; high-q bins (q>47) total {mu.reshape(16,9)[:,6:].sum():.2f} | "
          f"config b sig(Om,s8,omb)={np.round(sb,5)} S8 {sSb:.5f} | config d sig(Om,s8,omb,alpha,slnY)={np.round(sd,5)} S8 {sSd:.5f}", flush=True)
