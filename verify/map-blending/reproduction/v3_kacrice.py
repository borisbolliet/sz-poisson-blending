"""V3 (C5): deterministic Kac-Rice density of local maxima above nu for the unit-variance filtered noise field,
via an exact 1D reduction (trace t, traceless radius rho ~ Rayleigh), using the independent template.
  n(nu) = phi(nu)/(2 pi lam) * int_{t<0} N(t; -lam nu, s2/4 - lam^2) [t^2 - 2 su2 + 2 su2 exp(-t^2/(2 su2))] dt,  su2 = s2/8
Compared with the claimed Kac-Rice numbers and with my own maps' nearest-pixel peak counts (scale 0.5')."""
import glob
import numpy as np
from scipy.special import ndtr
import indep_lib as I

l = np.linspace(0, 12 / I.SB, 400001)
def moments(th):
    P = (I.g_of(l * th) * np.exp(-0.5 * (l * I.SB) ** 2)) ** 2
    s = [np.trapezoid(l ** (1 + 2 * j) * P, l) for j in range(3)]
    return s[1] / (2 * s[0]), s[2] / s[0]

def dens(nu, lam, s2):
    su2 = s2 / 8; vt = s2 / 4 - lam ** 2
    t = np.linspace(-lam * nu - 12 * np.sqrt(vt), 0, 20001)
    g = np.exp(-(t + lam * nu) ** 2 / (2 * vt)) / np.sqrt(2 * np.pi * vt)
    inner = t ** 2 - 2 * su2 + 2 * su2 * np.exp(-t ** 2 / (2 * su2))
    return np.exp(-nu ** 2 / 2) / np.sqrt(2 * np.pi) / (2 * np.pi * lam) * np.trapezoid(g * inner, t)

def above(nus, th):
    lam, s2 = moments(th)
    ng = np.linspace(min(nus), 10, 1401); dn = np.array([dens(v, lam, s2) for v in ng])
    return np.array([np.trapezoid(dn[ng >= v], ng[ng >= v]) for v in nus])

AREA = I.LBOX ** 2
nus = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
claimed = {0: [809.588, 184.475, 32.239, 4.328, 0.447], 6: [272.048, 61.223, 10.626, 1.421, 0.147], 11: [43.957, 9.579, 1.625, 0.214, 0.022]}
for k in (0, 6, 11):
    kr = above(nus, I.THK[k]) * AREA
    print(f"theta_k={I.THK[k]:.3f}': deterministic Kac-Rice per 100 deg^2: " + "  ".join(f"nu>{v}: {a:8.3f} (claimed {c:8.3f}, rel {a/c-1:+.4f})" for v, a, c in zip(nus, kr, claimed[k])))
kr0 = above(nus, I.THK[0]) * AREA
pk = [np.load(f)["npk0"] for f in sorted(glob.glob(f"{I.OUT}/patches/p*.npz"))]
c = np.array([[(p >= v).sum() for v in nus] for p in pk]); NPp = len(pk)
m, e = c.mean(0), np.sqrt(c.sum(0)) / NPp
print(f"my maps ({NPp} patches), 0.5' nearest-pixel peak heights:")
for i, v in enumerate(nus):
    print(f"   nu>{v}: {m[i]:8.3f} +- {e[i]:.3f}   Kac-Rice {kr0[i]:8.3f}  ratio {m[i]/kr0[i]:.4f} ({(m[i]-kr0[i])/e[i]:+.2f} sigma)")
# multi-scale sum of single-scale rates (C6 bounds)
k4 = np.array([above(np.array([4.0, 5.0, 6.0]), t) * 3600 for t in I.THK])
print(f"single-scale 0.5' rate /deg^2 q>4,5,6: {k4[0]}  ; sum over 12 scales: {k4.sum(0)}  (claimed 0.322/4.47e-3/2.19e-5 and 1.71/2.37e-2/1.16e-4)")
