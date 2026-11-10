"""V5: does halo exclusion change the clustering boost W of the mean neighbour signal (claim C9 uses linear, no exclusion)?
W(thA, thB) = int rho(s) w_dy(s) d2s / int rho d2s ; w_dy = cross-corr of q>5 detections with the Y-weighted M>1e13 halo field.
Exclusion: xi -> -1 for r < rex(z) = <R500com>_det(z) + <R500com>_y(z)  (same step model as C7's bracket)."""
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
Rcom = np.array([om.R500(M, zz) * (1 + zz) for zz in z])
W = d2N * Pd * wM; dNdz = W.sum(1); p = dNdz / (wz @ dNdz); b = (W * bM).sum(1) / dNdz
Ymean = np.array([om.Ymed(M, zz) for zz in z]) * np.exp(0.5 * 0.173 ** 2)
Wy = d2N * wM * Ymean; dydz = Wy.sum(1); py = dydz / (wz @ dydz); by = (Wy * bM).sum(1) / dydz
Rdet = (W * Rcom).sum(1) / dNdz; Ry = (Wy * Rcom).sum(1) / dydz
rex = Rdet + Ry
print(f"rex(z=.1,.5,1,2) = {np.interp([.1,.5,1,2], z, rex).round(2)} Mpc (det {np.interp([.1,.5,1], z, Rdet).round(2)}, y-wtd nbr {np.interp([.1,.5,1], z, Ry).round(2)})")
Hc = om.Ez(z) * cos.H0 / om.CK; chi = om.chi(z); D = cos.growthFactor(z)
Pk = lambda k: cos.matterPowerSpectrum(k / h, 0.0) / h ** 3
lg = np.geomspace(1, 3e6, 1200)
C = np.zeros(len(lg))
for i in range(len(z)):
    C += wz[i] * Hc[i] * p[i] * py[i] * b[i] * by[i] * D[i] ** 2 * Pk((lg + .5) / chi[i]) / chi[i] ** 2
l = np.arange(1, 3_000_000, 2.0); Cli = np.exp(np.interp(np.log(l), np.log(lg), np.log(C)))
tg = np.concatenate([[0.0], np.geomspace(0.003, 400, 260)])
wlin = np.array([np.sum(l * Cli * j0(l * t * ARC)) * 2 / (2 * np.pi) for t in tg])
rr = np.geomspace(2e-3, 50, 400); xi0 = cos.correlationFunction(rr * h, 0.0)
def dex(t):
    acc = 0.0
    for i in range(len(z)):
        R = chi[i] * t * ARC
        if R >= rex[i]: continue
        Lm = np.sqrt(rex[i] ** 2 - R ** 2); Dl = np.linspace(0, Lm, 300); r = np.sqrt(R * R + Dl * Dl)
        xi = np.interp(np.log(np.maximum(r, 2e-3)), np.log(rr), xi0)
        acc += wz[i] * p[i] * py[i] * Hc[i] * 2 * np.trapezoid(-1 - b[i] * by[i] * D[i] ** 2 * xi, Dl)
    return acc
wex = wlin + np.array([dex(t) for t in tg])
print(f"w_dy lin(0, 1', 5') = {np.interp([0,1,5], tg, wlin).round(4)}; with exclusion = {np.interp([0,1,5], tg, wex).round(4)}")
sb = 1.4 / np.sqrt(8 * np.log(2)); ll = np.linspace(0, 12 / sb, 20000); B2 = np.exp(-(ll * sb) ** 2)
s = np.linspace(0, 400, 16001)
for tA in [0.2, 1, 3, 10, 60]:
    tB = 0.6
    CAA = np.trapezoid(ll * B2 * om.g_of(ll * tA) ** 2, ll)
    smax = 5 * (tA + tB) + 6; ss = s[s <= smax]
    r = np.array([np.trapezoid(ll * B2 * om.g_of(ll * tA) * om.g_of(ll * tB) * j0(ll * x), ll) for x in ss]) / CAA
    den = np.trapezoid(r * ss, ss)
    Wl = np.trapezoid(r * np.interp(ss, tg, wlin) * ss, ss) / den
    We = np.trapezoid(r * np.interp(ss, tg, wex) * ss, ss) / den
    print(f"theta_A={tA:5.1f}': W_lin = {Wl:.4f}, W_excl = {We:.4f}, ratio {We/Wl:.3f}")
