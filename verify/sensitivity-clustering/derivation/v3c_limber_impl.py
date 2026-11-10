"""V3c: compare harmonic-space Limber (ell sum) with real-space Limber (k integral of P J0(k chi theta)) in own code,
to locate the 0.3-0.9% difference between v3 and the analysis' w(theta)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import j0
from colossus.lss import bias as cb
import own_model as om
cos = om.cos; h = om.h; ARC = om.ARC
z = np.linspace(0.01, 3.0, 300); lnM = np.linspace(np.log(1e13), np.log(1e16), 300); M = np.exp(lnM)
wz = om.trapw(z); wM = om.trapw(lnM)
d2N = np.array([om.dndlnM(M, zz) for zz in z]) * (4 * np.pi * om.dVdzdO(z))[:, None]
Pd = om.pdet_grid(z, M, 1.4, 4e-6, 0.173, 5.0)
bM = np.array([cb.haloBias(M * h, zz, "500c", model="tinker10") for zz in z])
W = d2N * Pd * wM; dNdz = W.sum(1); p = dNdz / (wz @ dNdz); b = (W * bM).sum(1) / dNdz
Hc = om.Ez(z) * cos.H0 / om.CK; chi = om.chi(z); D = cos.growthFactor(z)
Pk = lambda k: cos.matterPowerSpectrum(k / h, 0.0) / h ** 3
lk = np.linspace(np.log(1e-5), np.log(1e3), 2 ** 17); k = np.exp(lk); f = k * k * Pk(k) / (2 * np.pi)
for t in [1, 5, 30, 60]:
    I = np.array([np.trapezoid(f * j0(k * c * t * ARC), lk) for c in chi])
    wr = np.sum(wz * p * p * Hc * b * b * D ** 2 * I)
    # harmonic with and without +1/2, ell grid fine
    for shift in (0.5, 0.0):
        lg = np.geomspace(0.1, 3e6, 3000); C = np.zeros(len(lg))
        for i in range(len(z)):
            C += wz[i] * Hc[i] * p[i] ** 2 * b[i] ** 2 * D[i] ** 2 * Pk((lg + shift) / chi[i]) / chi[i] ** 2
        ll = np.exp(np.linspace(np.log(0.1), np.log(3e6), 400000))
        Ci = np.exp(np.interp(np.log(ll), np.log(lg), np.log(C)))
        wh = np.trapezoid(ll * ll * Ci * j0(ll * t * ARC), np.log(ll)) / (2 * np.pi)
        print(f"theta={t}': real-space Limber {wr:.5f}; harmonic (l+{shift}) {wh:.5f}; ratio {wh/wr:.4f}")
