"""V3d: exact (non-Limber) w(theta) = int dchi1 dchi2 n1 n2 b1 b2 D1 D2 xi0(r12) with colossus xi, vs real-space Limber."""
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
# n(chi) = p(z) H/c ; work on a fine chi grid
chif = np.linspace(chi[0], chi[-1], 60001); dchi = chif[1] - chif[0]
zf = np.interp(chif, chi, z)
nf = np.interp(zf, z, p * Hc); Af = nf * np.interp(zf, z, b * D)
rr = np.geomspace(2e-3, 700, 3000); xi = cos.correlationFunction(rr * h, 0.0)
xif = lambda r: np.where(r > 699, 0.0, np.interp(np.log(np.maximum(r, 2e-3)), np.log(rr), xi))
Pk = lambda k: cos.matterPowerSpectrum(k / h, 0.0) / h ** 3
lk = np.linspace(np.log(1e-5), np.log(1e3), 2 ** 17); k = np.exp(lk); f = k * k * Pk(k) / (2 * np.pi)
for t in [1, 5, 30]:
    th = t * ARC
    acc = 0.0
    step = 20
    for i in range(0, len(chif), step):
        c1 = chif[i]
        # inner integral over chi2 near c1: use offsets on a log grid out to 800 Mpc
        off = np.concatenate([-np.geomspace(700, 1e-4, 4000), [0], np.geomspace(1e-4, 700, 4000)])
        c2 = c1 + off; ok = (c2 > chif[0]) & (c2 < chif[-1])
        r12 = np.sqrt(np.maximum(c1 * c1 + c2[ok] ** 2 - 2 * c1 * c2[ok] * np.cos(th), 0))
        inner = np.trapezoid(np.interp(c2[ok], chif, Af) * xif(r12), off[ok])
        acc += Af[i] * inner * dchi * step
    I = np.array([np.trapezoid(f * j0(k * c * th), lk) for c in chi])
    wl = np.sum(wz * p * p * Hc * b * b * D ** 2 * I)
    print(f"theta={t}': exact (colossus xi) {acc:.5f}; real-space Limber {wl:.5f}; Limber/exact {wl/acc:.4f}")
