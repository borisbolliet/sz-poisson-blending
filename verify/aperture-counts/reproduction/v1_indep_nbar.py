"""V1: independent analytic check of every Nbar / Sigma / mu number used by the aperture-counts claims.

Independent of szsim's HMFTable / expected_detections: direct colossus calls, own Gauss-Legendre quadrature in
(z, lnM), own scaling relation, own lognormal x Gaussian selection integral (Gauss-Hermite-free: plain
trapezoid in the lnY deviate).  The only szsim ingredient reused is sigma_Y(theta500) (matched-filter noise,
validated in the build phase, outside the scope of these claims) - and it is cross-checked at the
point-source limit against the closed form sqrt(4 pi sigma_b^2) * noise / RATIO.
"""
import json, time
import numpy as np
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function

t0 = time.time()
cosmo = cosmology.setCosmology("planck18")
h = cosmo.h
C = 299792.458
AM = np.pi / 180 / 60
NB = json.load(open("/Users/boris/GitHub/sz-poisson-blending/data/q1_aperture/nbar.json"))


def dVdzdOm(z):          # Mpc^3/sr
    chi = cosmo.comovingDistance(0.0, z) / h
    return C / (100 * h) / cosmo.Ez(z) * chi ** 2


def dndlnM(M, z):        # Mpc^-3, M in Msun
    return mass_function.massFunction(M * h, z, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3


# Gauss-Legendre nodes, split in sub-intervals for accuracy
def gl(a, b, n, nsub):
    x, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(a, b, nsub + 1)
    X = np.concatenate([(e1 - e0) / 2 * x + (e1 + e0) / 2 for e0, e1 in zip(edges[:-1], edges[1:])])
    W = np.concatenate([(e1 - e0) / 2 * w for e0, e1 in zip(edges[:-1], edges[1:])])
    return X, W


zq, wz = gl(0.01, 3.0, 20, 30)            # 600 z nodes
lq, wl = gl(np.log(1e13), np.log(1e16), 20, 30)   # 600 lnM nodes
Mq = np.exp(lq)
D = np.array([dndlnM(Mq, z) * dVdzdOm(z) * 4 * np.pi for z in zq])   # (nz, nM) full-sky d2N/dz/dlnM
print(f"grid built in {time.time()-t0:.1f}s")


def N_above(Mmin, zmin=0.01, zmax=3.0):
    # integrate with exact clipping: redo quadrature on clipped interval
    zq2, wz2 = gl(zmin, zmax, 20, 30)
    lq2, wl2 = gl(np.log(Mmin), np.log(1e16), 20, 30)
    D2 = np.array([dndlnM(np.exp(lq2), z) * dVdzdOm(z) * 4 * np.pi for z in zq2])
    return float(wz2 @ D2 @ wl2)


res = {}
for lab, M in [("M1e13", 1e13), ("M3e13", 3e13), ("M1e14", 1e14), ("M3e14", 3e14)]:
    res[lab] = N_above(M)
res["M3e14_z1"] = N_above(3e14, zmin=1.0)

# ---- own scaling relation
Ez = cosmo.Ez(zq)
DA = cosmo.comovingDistance(0.0, zq) / h / (1 + zq)
rhoc = cosmo.rho_c(zq) * 1e9 * h ** 2           # Msun/Mpc^3
Y_Mpc2 = 1e-4 * 10 ** (-0.19) * Ez[:, None] ** (2 / 3) * (0.8 * Mq[None, :] / 6e14) ** 1.79
Ymed = Y_Mpc2 / DA[:, None] ** 2 / AM ** 2      # arcmin^2
R500 = (3 * Mq[None, :] / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3)
th500 = R500 / DA[:, None] / AM                 # arcmin

from szsim import signal as sg
m = sg.SZModel()
# cross-check szsim's Y500 / theta500 against my implementation on a few points
zz = np.array([0.1, 0.5, 1.0, 2.0]); MM = np.array([1e14, 3e14, 6e14, 2e14])
for z_, M_ in zip(zz, MM):
    i = None
    Ez_ = cosmo.Ez(z_); DA_ = cosmo.comovingDistance(0, z_) / h / (1 + z_)
    Ym = 1e-4 * 10 ** -0.19 * Ez_ ** (2 / 3) * (0.8 * M_ / 6e14) ** 1.79 / DA_ ** 2 / AM ** 2
    th = (3 * M_ / (4 * np.pi * 500 * cosmo.rho_c(z_) * 1e9 * h ** 2)) ** (1 / 3) / DA_ / AM
    print(f"z={z_} M={M_:.0e}: Y500 mine {Ym:.6e} szsim {float(m.Y500_med(M_, z_)):.6e} ratio-1={Ym/float(m.Y500_med(M_, z_))-1:+.1e};"
          f" theta500 mine {th:.5f} szsim {float(m.theta500(M_, z_)):.5f}")
# point-source limit closed form
sb = 1.4 / np.sqrt(8 * np.log(2))
from szsim.signal import RATIO_5R500
print(f"sigma_Y point closed form = {np.sqrt(4*np.pi*sb**2)*4e-6/RATIO_5R500:.4e}; szsim sigma_Y(1e-3') = {float(m.sigma_Y(1e-3)):.4e}")

sig = m.sigma_Y(th500.ravel()).reshape(th500.shape)
qmed = Ymed / sig
x = np.linspace(-8, 8, 801); px = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi); dx = x[1] - x[0]


def pdet(qcut):
    out = np.zeros_like(qmed)
    for xi, pi in zip(x, px):
        out += pi * dx * ndtr(qmed * np.exp(0.173 * xi) - qcut)
    return out


for q in [4, 5, 6, 7]:
    P = pdet(q)
    res[f"det{q}"] = float(wz @ (D * P) @ wl)
    if q == 5:
        P5 = P
        res["det5_floor1e14"] = float(wz @ (D * P * (Mq >= 1e14)[None, :]) @ wl)  # coarse (node mask) - info only
# z slices for det5: weight by z masks using separate quadrature for exactness
for (a, b) in [(0.01, 0.5), (0.5, 1.0), (1.0, 3.0)]:
    zq2, wz2 = gl(a, b, 20, 30)
    Ez2 = cosmo.Ez(zq2); DA2 = cosmo.comovingDistance(0.0, zq2) / h / (1 + zq2); rc2 = cosmo.rho_c(zq2) * 1e9 * h ** 2
    D2 = np.array([dndlnM(Mq, z) * dVdzdOm(z) * 4 * np.pi for z in zq2])
    Ym2 = 1e-4 * 10 ** (-0.19) * Ez2[:, None] ** (2 / 3) * (0.8 * Mq[None, :] / 6e14) ** 1.79 / DA2[:, None] ** 2 / AM ** 2
    th2 = (3 * Mq[None, :] / (4 * np.pi * 500 * rc2[:, None])) ** (1 / 3) / DA2[:, None] / AM
    q2 = Ym2 / m.sigma_Y(th2.ravel()).reshape(th2.shape)
    P2 = sum(pi * dx * ndtr(q2 * np.exp(0.173 * xi) - 5) for xi, pi in zip(x, px))
    res[f"det5_z{a}-{b}"] = float(wz2 @ (D2 * P2) @ wl)

SKY = 4 * np.pi * (180 / np.pi) ** 2
print("\n== Nbar: independent vs claimed (nbar.json) ==")
for k, v in res.items():
    ref = NB.get(k, NB.get("M3e14") if k == "M3e14_z1" else None)
    if k == "M3e14_z1":
        ref = 1514.07
    print(f"{k:16s} mine={v:14.2f}  claimed={ref:14.2f}  rel diff={v/ref-1:+.2e}  Sigma={v/SKY:.4f}/deg^2")
print("det share from M<1e14 at q>5 (node-mask, coarse):", 1 - res["det5_floor1e14"] / res["det5"])

# ---- analytic numbers quoted in the claims (recomputed with my Nbar)
def mu(Nb, th_arcmin):
    return Nb * np.sin(th_arcmin * AM / 2) ** 2      # = Nb (1-cos)/2


print("\n== quoted analytic numbers, recomputed with MY Nbar ==")
d5, m14, m3, m314 = res["det5"], res["M1e14"], res["M3e13"], res["M3e14"]
for t in [1, 2, 5, 10]:
    print(f"det5 P(>=1 other det5 within {t}') = {1-np.exp(-mu(d5,t)):.4e}; P(>=2) = {1-np.exp(-mu(d5,t))*(1+mu(d5,t)):.4e}"
          f" | M1e14 P>=1 {1-np.exp(-mu(m14,t)):.4e} | M3e13 P>=1 {1-np.exp(-mu(m3,t)):.4e}")
for t in [0.5, 1, 2, 5, 10]:
    print(f"det5 full-sky pairs < {t}': E = {d5*mu(d5,t)/2:.2f}")
p3 = 1 - 3 * np.sqrt(3) / (4 * np.pi)
for t in [1, 2, 3, 5, 10]:
    print(f"det5 triples < {t}': E = {d5*mu(d5,t)**2*p3/6:.3f}")
print(f"P(exists M>3e14 pair <0.5') = {1-np.exp(-m314*mu(m314,0.5)/2):.4f}")
# flat vs cap
for tdeg in [1 / 60, 1, 6.3, 19.8, 61]:
    t = np.radians(tdeg)
    print(f"flat/cap-1 at {tdeg:.3f} deg = {np.pi*t**2/(2*np.pi*(1-np.cos(t)))-1:.4e} (theta^2/12 = {t**2/12:.4e})")
json.dump(res, open("/Users/boris/GitHub/sz-poisson-blending/verify/aperture-counts/reproduction/nbar_indep.json", "w"), indent=1)
print(f"time {time.time()-t0:.1f}s")
