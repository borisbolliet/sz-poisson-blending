"""V3: independent angular clustering of q>5 detections (harmonic-space Limber, own selection), exclusion bracket,
P_clust/P_Poisson, y-weighted cross-correlation W factor, and analytic 3D-overlap counts.
Does not import szsim or analysis code."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import j0, j1
from colossus.lss import bias as cb
import own_model as om

cos = om.cos; h = om.h; ARC = om.ARC
z = np.linspace(0.01, 3.0, 300); lnM = np.linspace(np.log(1e13), np.log(1e16), 300); M = np.exp(lnM)
wz = om.trapw(z); wM = om.trapw(lnM)
d2N = np.array([om.dndlnM(M, zz) for zz in z]) * (4 * np.pi * om.dVdzdO(z))[:, None]
Pd = om.pdet_grid(z, M, 1.4, 4e-6, 0.173, 5.0)
bM = np.array([cb.haloBias(M * h, zz, "500c", model="tinker10") for zz in z])
W = d2N * Pd * wM
dNdz = W.sum(1); Ndet = wz @ dNdz
p = dNdz / Ndet; b = (W * bM).sum(1) / dNdz
print(f"N_det = {Ndet:.1f}; Sigma = {Ndet/(4*np.pi)*ARC**2*3600:.4f}/deg2; b_eff(z=.1,.5,1,2) = {np.interp([.1,.5,1,2], z, b).round(3)}")
Hc = om.Ez(z) * cos.H0 / om.CK; chi = om.chi(z); D = cos.growthFactor(z)

def Pk(k):  # Mpc^3, k in 1/Mpc, z=0 linear
    return cos.matterPowerSpectrum(k / h, 0.0) / h ** 3

# harmonic Limber C_l = int dz (H/c) p1 p2 b1 b2 D^2 P((l+1/2)/chi) / chi^2
lg = np.geomspace(1, 3e6, 1200)

def Cl(p1, b1, p2, b2):
    out = np.zeros(len(lg))
    for i in range(len(z)):
        out += wz[i] * Hc[i] * p1[i] * p2[i] * b1[i] * b2[i] * D[i] ** 2 * Pk((lg + 0.5) / chi[i]) / chi[i] ** 2
    return out

def w_of(theta_arcmin, C, avg=False):
    l = np.arange(1, 3_000_000, 2.0)
    Cli = np.exp(np.interp(np.log(l), np.log(lg), np.log(C)))
    out = []
    for t in np.atleast_1d(theta_arcmin):
        x = l * t * ARC
        ker = 2 * j1(x) / x if avg else j0(x)
        out.append(np.sum(l * Cli * ker) * 2.0 / (2 * np.pi))
    return np.array(out)

C = Cl(p, b, p, b)
TH = np.array([0.01, 1, 2, 5, 10, 30, 60])
wv = w_of(TH, C); wb = w_of(TH[1:6], C, avg=True)
# convergence check: lmax 1.5e6
print("w(theta):", dict(zip(TH, wv.round(5))))
Sig = Ndet / (4 * np.pi)
print(" theta | 1+wbar | P_Poisson(>=1) | P_clust/P_Poisson")
for t, wbb in zip(TH[1:6], wb):
    A = 2 * np.pi * (1 - np.cos(t * ARC)); lam = Sig * A
    Pp = 1 - np.exp(-lam); Pc = 1 - np.exp(-lam * (1 + wbb))
    print(f" {t:5.1f}' | {1+wbb:.4f} | {Pp:.4e} | {Pc/Pp:.4f}")

# exclusion bracket: xi -> -1 for r < rex(z) = 2 <R500,com>_det(z)
Rcom = np.array([om.R500(M, zz) * (1 + zz) for zz in z])
rex = 2 * (W * Rcom).sum(1) / dNdz
print(f"r_ex(z=.1,.5,1,2) = {np.interp([.1,.5,1,2], z, rex).round(2)} Mpc")
rr = np.geomspace(2e-3, 50, 400)
xi0 = cos.correlationFunction(rr * h, 0.0)  # colossus xi (dimensionless), r in Mpc/h
def corr_ex(t):
    """w_ex - w_lin at angle t: int dz p^2 H/c int_{|D|<sqrt(rex^2-R^2)} [-1 - b^2 D^2 xi(r)] dD"""
    acc = 0.0
    for i in range(len(z)):
        R = chi[i] * t * ARC
        if R >= rex[i]: continue
        L = np.sqrt(rex[i] ** 2 - R ** 2); Dl = np.linspace(0, L, 400)
        r = np.sqrt(R ** 2 + Dl ** 2)
        xi = np.interp(np.log(np.maximum(r, 2e-3)), np.log(rr), xi0)
        acc += wz[i] * p[i] ** 2 * Hc[i] * 2 * np.trapezoid(-1 - b[i] ** 2 * D[i] ** 2 * xi, Dl)
    return acc
tt = np.concatenate([[0.0], np.geomspace(0.01, 30, 200)])
dex = np.array([corr_ex(t) for t in tt])
for t in [1, 2, 5, 10, 30]:
    m = tt <= t
    x = np.concatenate([tt[m], [t]]); y = np.interp(x, tt, dex)
    dbar = 2 * np.trapezoid(y * x, x) / t ** 2
    wbl = w_of([t], C, avg=True)[0]
    A = 2 * np.pi * (1 - np.cos(t * ARC)); lam = Sig * A
    print(f"exclusion: theta={t}': 1+wbar_lin {1+wbl:.4f} -> with exclusion {1+wbl+dbar:.4f}; P ratio {(1-np.exp(-lam*(1+wbl+dbar)))/(1-np.exp(-lam)):.4f}")

# y-weighted cross-correlation detections x (Y-weighted halos M>1e13)
Ymean = np.array([om.Ymed(M, zz) for zz in z]) * np.exp(0.5 * 0.173 ** 2)
Wy = d2N * wM * Ymean; dydz = Wy.sum(1)
py = dydz / (wz @ dydz); by = (Wy * bM).sum(1) / dydz
ybar = om.RATIO * (wz @ dydz) / (4 * np.pi) * ARC ** 2
print(f"own ybar(M>1e13) = {ybar:.4e}; b_y(z=.1,.5,1,2) = {np.interp([.1,.5,1,2], z, by).round(3)}")
Cy = Cl(p, b, py, by)
print(f"w_dy(0.01', 1', 5') = {w_of([0.01, 1, 5], Cy).round(4)}")
# W factor needs rho(s) for theta_A, theta_B: own Hankel
def rho_s(tA, tB, beam, s):
    sb = beam / np.sqrt(8 * np.log(2)); l = np.linspace(0, 12 / sb, 20000)
    w = l * np.exp(-(l * sb) ** 2) * om.g_of(l * tA) * om.g_of(l * tB)
    CAA = np.trapezoid(l * np.exp(-(l * sb) ** 2) * om.g_of(l * tA) ** 2, l)
    return np.array([np.trapezoid(w * j0(l * ss), l) for ss in s]) / CAA
sgrid = np.linspace(0, 400, 4001)
tg = np.geomspace(0.003, 500, 300); wdy_g = w_of(tg, Cy)
wdy_s = np.interp(sgrid, tg, wdy_g)
for tA in [0.2, 1, 3, 10, 60]:
    r = rho_s(tA, 0.6, 1.4, sgrid)
    print(f"W(theta_A={tA}', theta_B=0.6') = {np.trapezoid(r*wdy_s*sgrid, sgrid)/np.trapezoid(r*sgrid, sgrid):.4f}; "
          f"int rho d2s / (RATIO^2 sigma^2/N_w) = {2*np.pi*np.trapezoid(r*sgrid, sgrid) / (om.RATIO**2 * om.sigmaY(tA,1.4,4e-6)[0]**2/(4e-6)**2):.4f}")
print(f"spot: rho(3',0.5',0) = {rho_s(3,0.5,1.4,[0.0])[0]:.4f}; rho(2',2',2') = {rho_s(2,2,1.4,[2.0])[0]:.4f}")

# analytic overlap (3D comoving sep < R_A,com + R_B,com) counts
dV = 4 * np.pi * om.dVdzdO(z)
ncomdet = (d2N * Pd * wM) / dV[:, None]
ncomall = (d2N * wM) / dV[:, None]
E = 0.0
for i in range(len(z)):
    Rs = Rcom[i][:, None] + Rcom[i][None, :]
    E += wz[i] * (W[i][:, None] * ncomall[i][None, :] * 4 * np.pi / 3 * Rs ** 3).sum()
E /= Ndet
print(f"E[# M>1e13 halos overlapping a q>5 detection] = {E:.4e}; P(>=1) = {1-np.exp(-E):.4e}")
for t in [1, 2, 5, 10, 30]:
    acc = 0.0
    for i in range(len(z)):
        Rs = Rcom[i][:, None] + Rcom[i][None, :]; a = chi[i] * t * ARC
        V = 4 * np.pi / 3 * (Rs ** 3 - np.clip(Rs ** 2 - a * a, 0, None) ** 1.5)
        acc += wz[i] * (W[i][:, None] * ncomdet[i][None, :] * V).sum()
    Nov = 0.5 * acc; Np = 0.5 * Ndet * Sig * 2 * np.pi * (1 - np.cos(t * ARC))
    print(f"pairs < {t}': N_pairs {Np:.1f}, overlapping {Nov:.3f}, fraction {Nov/Np:.3e}")
