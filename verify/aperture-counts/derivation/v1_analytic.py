"""Independent analytic checks for aperture-counts claims (C1, C5, C8, C9, C10 formulas).
Uses colossus directly for HMF / distances; uses szsim ONLY for sigma_Y(theta500) of the matched filter."""
import numpy as np
from scipy import integrate, special, stats
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmo = cosmology.setCosmology('planck18'); h = cosmo.h
ARC = np.pi/180/60
FULL_SR = 4*np.pi
DEG2_FULL = 4*np.pi*(180/np.pi)**2
print("full sky deg^2 =", DEG2_FULL)

# ---------- HMF counts directly from colossus ----------
def d2N(lnM, z):  # full sky per dz per dlnM, M in Msun
    M = np.exp(lnM)
    dndlnM = mass_function.massFunction(M*h, z, mdef='500c', model='tinker08', q_out='dndlnM')*h**3  # (Mpc)^-3
    chi = cosmo.comovingDistance(0, z)/h  # Mpc
    dVdz = 299792.458/(100*h*cosmo.Ez(z)) * chi**2 * FULL_SR  # Mpc^3 per dz full sky
    return dndlnM*dVdz
zg, zw = np.polynomial.legendre.leggauss(400)
def N_above(Mmin, zmin=0.01, zmax=3.0, Mmax=1e16, weightfun=None):
    zz = 0.5*(zmax-zmin)*zg + 0.5*(zmax+zmin); wz = 0.5*(zmax-zmin)*zw
    a, b = np.log(Mmin), np.log(Mmax)
    lm = 0.5*(b-a)*zg + 0.5*(b+a); wm = 0.5*(b-a)*zw
    tot = 0.0
    for zi, wzi in zip(zz, wz):
        f = d2N(lm, zi)
        if weightfun is not None: f = f*weightfun(np.exp(lm), zi)
        tot += wzi*np.sum(wm*f)
    return tot
NB = {}
for Mmin in [1e13, 3e13, 1e14, 3e14]:
    NB[f"M{Mmin:.0e}"] = N_above(Mmin)
    print(f"N(M500>{Mmin:.0e} Msun) = {NB[f'M{Mmin:.0e}']:.1f}  Sigma={NB[f'M{Mmin:.0e}']/DEG2_FULL:.4f}/deg2")

# ---------- own selection integral (own Y-M; szsim sigma_Y only) ----------
from szsim.signal import SZModel
mod = SZModel()
def Y500_arcmin2(M, z):
    DA = cosmo.angularDiameterDistance(z)/h  # Mpc
    Y_Mpc2 = 1e-4*10**(-0.19)*(0.8*M/6e14)**1.79 * cosmo.Ez(z)**(2/3)/DA**2
    return Y_Mpc2/ARC**2  # Y_Mpc2 already divided by DA^2 -> sr
def theta500(M, z):
    rhoc = cosmo.rho_c(z)*h**2*1e9  # Msun/Mpc^3 (colossus rho_c in Msun h^2/kpc^3)
    R = (3*M/(4*np.pi*500*rhoc))**(1/3)
    return R/(cosmo.angularDiameterDistance(z)/h)/ARC
# spot check vs library
for M, z in [(2e14, .1), (3e14, .5), (1e14, 2.)]:
    print(f"spot M={M:.0e} z={z}: Y own {Y500_arcmin2(M,z):.5e} lib {mod.Y500_med(M,z):.5e}; th own {theta500(M,z):.5f} lib {mod.theta500(M,z):.5f}")
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60); gh_w = gh_w/np.sqrt(2*np.pi)
sig = 0.173
def pdet_fun(qcut):
    def f(M, z):
        q0 = Y500_arcmin2(M, z)/mod.sigma_Y(theta500(M, z))
        q = q0[:, None]*np.exp(sig*gh_x[None, :])
        return (stats.norm.sf(qcut - q)*gh_w[None, :]).sum(1)
    return f
for qc in [4, 5, 6, 7]:
    NB[f"det{qc}"] = N_above(1e13, weightfun=pdet_fun(qc))
    print(f"det{qc}: N={NB[f'det{qc}']:.1f} Sigma={NB[f'det{qc}']/DEG2_FULL:.4f}/deg2")
for lo, hi in [(0.01, .5), (.5, 1.), (1., 3.)]:
    print(f"det5 z[{lo},{hi}] = {N_above(1e13, lo, hi, weightfun=pdet_fun(5)):.1f}")
d5_14 = N_above(1e14, weightfun=pdet_fun(5))
print(f"det5 M>1e14 = {d5_14:.1f}; undet5 M>1e14 = {NB['M1e+14']-d5_14:.1f}; share M<1e14 of det5 = {1-d5_14/NB['det5']:.4f}")
for qc in [4, 6]:
    print(f"share M<1e14 of det{qc} = {1-N_above(1e14, weightfun=pdet_fun(qc))/NB[f'det{qc}']:.4f}")

# ---------- mu, tables ----------
def mu(N, th_arcmin): return N*(1-np.cos(th_arcmin*ARC))/2
print("\nmu table")
for k in ['det4', 'det5', 'det6', 'M1e+14', 'M3e+13']:
    print(k, ["%.4g" % mu(NB[k], t) for t in [1, 5, 10, 60]])
print("\nNeighbour probability table (analytic, own Nbar)")
for qc in [4, 5, 6]:
    for t in [1, 2, 5, 10]:
        for lab, key in [('a', f'det{qc}'), ('b', 'M1e+14'), ('c', 'M3e+13')]:
            m = mu(NB[key], t)
            print(f"q{qc} {t}' ({lab}) P>=1={-np.expm1(-m):.4e} P>=2={1-np.exp(-m)*(1+m):.4e}")
m10 = mu(NB['det5'], 10); print("det5 10': mu", m10, "P0", np.exp(-m10), "P1", m10*np.exp(-m10), "P2", m10**2/2*np.exp(-m10))

# ---------- flat vs cap ----------
print("\nflat/cap - 1 ; theta^2/12")
for tdeg in [1/60, 1, 5, 6.3, 10, 19.8, 30, 61, 90, 179.99, 180]:
    t = np.radians(tdeg); print(f"{tdeg:8.3f} deg: exact {np.pi*t**2/(4*np.pi*np.sin(t/2)**2)-1:.4e}  t^2/12 {t**2/12:.4e}")
from scipy.optimize import brentq
for target in [1e-3, 1e-2, 0.1]:
    r = brentq(lambda t: t**2/(2*(1-np.cos(t)))-1-target, 1e-3, 3)
    print(f"exact rel err {target}: theta={np.degrees(r):.2f} deg (theta^2/12 gives {np.degrees(np.sqrt(12*target)):.2f})")
print("flat prediction at 179.99deg, M>1e14:", NB['M1e+14']*np.radians(179.99)**2/4)

# ---------- p3 ----------
# P(|x2-x3|<1) for x2,x3 uniform in unit disk: = int_0^1 2r * A(r)/pi dr, A = lens area of unit circles at distance r
A = lambda r: 2*np.arccos(r/2) - (r/2)*np.sqrt(4-r*r)
p3num = integrate.quad(lambda r: 2*r*A(r)/np.pi, 0, 1, epsabs=1e-14)[0]
print("\np3 numeric", p3num, " closed", 1-3*np.sqrt(3)/(4*np.pi))
# pairs / triples det5
for t in [0.5, 1, 2, 5, 10]:
    m = mu(NB['det5'], t)
    print(f"det5 {t}': E[pairs]={NB['det5']*m/2:.2f}  Var={NB['det5']*m/2*(1+2*m):.2f} E[triples]={NB['det5']*m*m*p3num/6:.4f}")
m = mu(NB['M3e+14'], 0.5); E = NB['M3e+14']*m/2; print("M3e14 pairs 0.5': E", E, "1-exp(-E)", -np.expm1(-E))
