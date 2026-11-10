"""Variance of the neighbour boost (Parseval, untruncated) per neighbour mass class, including classes
below the model floor (1e12-1e13, extrapolated Tinker08) to judge convergence of monopole-convention
quantities that depend on the Gaussian core of delta (B2neg, lost)."""
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import v01_kernel as K
cos = cosmology.setCosmology("planck18"); h = cos.h; ARC = np.pi / 10800
def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n); return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w
lz, wlz = gl(np.log(0.01), np.log(3.0), 300); Z = np.exp(lz); WZ = wlz * Z
EDG = [1e12, 3e12, 1e13, 3e13, 1e14, 1e16]
for tA in (1.10, 4.153):
    sA = K.sigY(tA)[0]
    k = np.linspace(0, 6.5 / K.SB, 60000); dk = k[1] - k[0]
    base = k * np.exp(-2 * (k * K.SB) ** 2) * K.g(k * tA) ** 2
    tg = np.geomspace(0.02, 200, 500)
    J = np.array([np.sum(base * K.g(k * t) ** 2) * dk for t in tg]) / (2 * np.pi) / K.NW ** 2
    out = []
    for c in range(len(EDG) - 1):
        lm, wm = gl(np.log(EDG[c]), np.log(EDG[c + 1]), 60); M = np.exp(lm); v = 0; mu = 0
        for zi, wz in zip(Z, WZ):
            dn = mass_function.massFunction(M * h, zi, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
            chi = cos.comovingDistance(0.0, zi) / h; DA = chi / (1 + zi)
            w = dn * 299792.458 / cos.H0 / cos.Ez(zi) * chi ** 2 * wm * wz * ARC ** 2
            Y = 1e-4 * 10 ** -0.19 * cos.Ez(zi) ** (2 / 3) * (0.8 * M / 6e14) ** 1.79 / DA ** 2 / ARC ** 2
            R = (3 * M / (4 * np.pi * 500 * cos.rho_c(zi) * 1e9 * h ** 2)) ** (1 / 3); th = R / DA / ARC
            v += np.sum(w * Y ** 2 * np.exp(2 * 0.173 ** 2) * np.interp(np.log(th), np.log(tg), J))
            mu += np.sum(w * Y * np.exp(0.173 ** 2 / 2))
        out.append((EDG[c], EDG[c + 1], v * sA ** 2, mu * sA * K.RATIO ** 2 / K.NW))
    print(f"thA={tA}':")
    for lo, hi, v, mu in out:
        print(f"   M500 in [{lo:.0e},{hi:.0e}): var {v:.5f}  (sd {np.sqrt(v):.4f})   mean shift {mu:.4f}")
