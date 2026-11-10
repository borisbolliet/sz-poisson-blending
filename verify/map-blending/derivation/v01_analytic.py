"""Independent analytic checks: template/RATIO, sigma_Y, HMF counts and patch expectation, ybar and l=0 bias,
Sigma(q_true>q_cut), Slivnyak pair formula, Poisson confusion variance, Kac-Rice peak counts.
Also spot-checks the szsim library against this independent implementation."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["NUMBA_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.special import ndtr
import own_model as om

t0 = time.time()
print(f"RATIO = Ysph(5)/Ysph(1) = {om.RATIO:.5f}   (claim 1.79644)")
u = np.array([0.0, 1.0, 3.0, 10.0])
print("g(u) at", u, "=", om.gfun(u))
f = np.load(os.path.join(os.path.dirname(__file__), "own_g_table.npz"))
print(f"projected template integral (trapz) = {float(f['norm']):.5f}  g(0) = {f['g'][0]:.5f}")

TK = np.geomspace(0.5, 10.0, 12)
sk = om.sigmaY(TK)
print("theta_k:", np.round(TK, 3))
print("sigma_Y(theta_k) [arcmin^2]:", " ".join(f"{s:.4e}" for s in sk))
print(f"claim: sigma(0.5')=5.63e-6, sigma(1.95')=1.062e-5, sigma(10')=4.308e-5 ; ours {sk[0]:.4e} {sk[5]:.4e} {sk[11]:.4e}")
print(f"sigma_Y point (theta=1e-4') = {om.sigmaY(1e-4)[0]:.4e} (claim 4.693e-6); 1',3',10' = {om.sigmaY([1,3,10])}")

# ---- library spot checks
from szsim import signal as sg, cosmo_hmf as ch
m = sg.SZModel()
print("library sigma_Y_exact / ours - 1 at theta_k:", np.round(m.sigma_Y_exact(TK) / sk - 1, 6))
for (M, z) in [(1e14, 0.1), (3e14, 0.5), (1e15, 1.0), (2e14, 2.0)]:
    print(f"M={M:.0e} z={z}: Y500 lib/ours-1 = {m.Y500_med(M, z)/om.Y500_med(M, z)-1:+.2e}, theta500 lib/ours-1 = "
          f"{m.theta500(M, z)/om.theta500(M, z)-1:+.2e}; ours Y={om.Y500_med(M,z):.3e} th={om.theta500(M,z):.3f}")
for (a, b, r) in [(3.0, 0.5, 0.0), (2.0, 2.0, 2.0), (0.5, 4.0, 3.0), (10.0, 1.0, 5.0)]:
    print(f"cross_response({a},{b},{r}): lib {float(np.ravel(m.cross_response(a, b, r))[0]):.5f} ours {om.response(a, b, r):.5f}")

# ---- HMF table (independent midpoint integration)
zc, lc, d2, dz, dl = om.hmf_table(nz=900, nM=600)
cell = d2 * dz * dl                     # expected per arcmin^2 per cell
area_patch = 600.0 ** 2
M = np.exp(lc)
for Mmin in (1e13, 3e13, 1e14, 3e14):
    n = cell[:, M >= Mmin].sum()
    # partial cell correction negligible: Mmin on grid edge? report
    print(f"N(>{Mmin:.0e}): full sky {n*om.DEG2_FULLSKY*3600:,.0f}  per 100 deg^2 {n*area_patch:.1f}")
T = ch.default_table()
print("library N_above:", [f"{T.N_above(x):,.1f}" for x in (1e13, 3e13, 1e14, 3e14)])
print(f"patch expectation (library N x 100/41252.96): {T.N_above(1e13)*100/om.DEG2_FULLSKY:.2f}")

# ---- per-cell signal quantities
Zg, Mg = np.meshgrid(zc, M, indexing="ij")
Ym = om.Y500_med(Mg, Zg)
th = om.theta500(Mg, Zg)
thg = np.geomspace(1e-3, 200, 400)
sY = np.exp(np.interp(np.log(th), np.log(thg), np.log(om.sigmaY(thg))))
s = om.SIG_LNY
ybar_cells = cell * Ym * np.exp(s * s / 2) * om.RATIO          # y = sum Y_tot / area
for lo, hi in [(1e13, 3e13), (3e13, 1e14), (1e14, 1e17), (1e13, 1e17)]:
    mm = (M >= lo) & (M < hi)
    print(f"ybar [{lo:.0e},{hi:.0e}) = {ybar_cells[:, mm].sum():.4e}")
yb = ybar_cells.sum()
kap = sk ** 2 * om.RATIO / om.NW
print("analytic l=0 bias ybar*kappa/sigma:", np.round(yb * kap / sk, 3), "(claim 0.383 .. 2.931 with patch-0 ybar=6.06e-7)")
print("with ybar=6.06e-7:", np.round(6.06e-7 * kap / sk, 3))

# ---- Sigma(q_true > q_cut) and Slivnyak values
for qc in (4, 5, 6, 7):
    a = np.log(qc * sY / Ym) / s
    Sig = (cell * (1 - ndtr(a))).sum() * 3600
    r = np.array([1, 2, 3]) / 60.0
    print(f"Sigma(q_true>{qc}) = {Sig:.4f}/deg^2  Poisson 1-exp(-Sig pi r^2) r=1,2,3': " +
          " ".join(f"{100*(1-np.exp(-Sig*np.pi*rr**2)):.3f}%" for rr in r))

# ---- confusion variance (Poisson one-halo)
LG, BL = om.LGRID, om.BL
gth = np.array([om.gfun(LG * t) for t in thg])               # (nth, nl)
W = np.empty((12, thg.size))
for k in range(12):
    gk = om.gfun(LG * TK[k])
    W[k] = (sk[k] ** 2 / om.NW) ** 2 * np.trapezoid(LG[None] * BL[None] ** 4 * gk[None] ** 2 * gth ** 2, LG, axis=1) / (2 * np.pi)
Wc = np.array([np.exp(np.interp(np.log(th), np.log(thg), np.log(W[k]))) for k in (0, 5, 11)])   # (3, nz, nM)


def confvar(qthr, mlo=0, mhi=1e30, cellv=cell, Ymv=Ym, sYv=sY, Wv=Wc, Mv=M):
    if qthr is None:
        fr = 1.0
    else:
        a = np.log(qthr * sYv / Ymv) / s
        fr = ndtr(a - 2 * s)
    E2 = Ymv ** 2 * np.exp(2 * s * s) * fr
    mm = (Mv >= mlo) & (Mv < mhi)
    return np.array([(cellv[:, mm] * E2[:, mm] * Wv[j][:, mm]).sum() for j in range(3)]) / sk[[0, 5, 11]] ** 2


for qthr in (2, 5):
    v = confvar(qthr)
    print(f"Var/sigma^2 (q_true<{qthr}) at 0.5',1.95',10' = {np.round(v, 4)}; sigma_eff/sigma = {np.round(np.sqrt(1+v), 4)}")
print(f"Var/sigma^2 [1e13,3e13) all q = {np.round(confvar(None, 1e13, 3e13), 4)}")
print(f"Var/sigma^2 [1e13,1e14) all q = {np.round(confvar(None, 1e13, 1e14), 4)}")
print(f"Var/sigma^2 [1e13,1e14) q<2 = {np.round(confvar(2, 1e13, 1e14), 4)}")

# lower mass slices with a separate low-mass table
for lo, hi in [(1e12, 1e13), (1e11, 1e12)]:
    zc2, lc2, d22, dz2, dl2 = om.hmf_table(lnMmin=np.log(lo), lnMmax=np.log(hi), nz=900, nM=60)
    c2 = d22 * dz2 * dl2
    Z2, M2 = np.meshgrid(zc2, np.exp(lc2), indexing="ij")
    Ym2 = om.Y500_med(M2, Z2); th2 = om.theta500(M2, Z2)
    sY2 = np.exp(np.interp(np.log(th2), np.log(thg), np.log(om.sigmaY(thg))))
    W2 = np.array([np.exp(np.interp(np.log(th2), np.log(thg), np.log(W[k]))) for k in (0, 5, 11)])
    print(f"Var/sigma^2 [{lo:.0e},{hi:.0e}) = {np.round(confvar(None, 0, 1e30, c2, Ym2, sY2, W2, np.exp(lc2)), 5)}")

# ---- z<0.05 share of the [1e13,1e14) variance at 10'
mm = (M >= 1e13) & (M < 1e14)
E2 = Ym ** 2 * np.exp(2 * s * s)
v10 = cell[:, mm] * E2[:, mm] * Wc[2][:, mm]
print(f"share of [1e13,1e14) 10' variance from z<0.05: {v10[zc < 0.05].sum()/v10.sum():.3f}")

# ---- Kac-Rice peak counts for filtered white noise (continuum moments)
def peak_counts(k, nus, area=600.0 ** 2):
    gk = om.gfun(LG * TK[k])
    P = sk[k] ** 4 * BL ** 2 * gk ** 2 / om.NW
    mom = [np.trapezoid(LG ** (2 * j + 1) * P, LG) / (2 * np.pi) for j in range(3)]
    s0, s1, s2 = np.sqrt(mom)
    gam = s1 ** 2 / (s0 * s2)
    mpar = s2 ** 2 / 4
    nug = np.linspace(-1, 10, 4401)
    xg = np.linspace(0, 1, 1)  # placeholder
    dens = np.empty(nug.size)
    for i, nu in enumerate(nug):
        mu, sd = gam * s2 * nu, s2 * np.sqrt(1 - gam ** 2)
        x = np.linspace(max(0.0, mu - 9 * sd), mu + 9 * sd, 3001)
        h = x * x / 4 - mpar + mpar * np.exp(-x * x / (4 * mpar))
        px = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)
        dens[i] = np.exp(-0.5 * nu * nu) / np.sqrt(2 * np.pi) / (np.pi * s1 ** 2) * np.trapezoid(h * px, x)
    out = []
    for nu in nus:
        mm_ = nug >= nu
        out.append(np.trapezoid(dens[mm_], nug[mm_]) * area)
    return np.array(out), (s0, s1, s2, gam)


nus = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
tot = np.zeros(len(nus))
for k in range(12):
    n_, (s0, s1, s2, gam) = peak_counts(k, nus)
    tot += n_
    print(f"Kac-Rice scale {TK[k]:.3f}': s0/sigma_k={s0/sk[k]:.5f} gamma={gam:.4f}  n(>nu) per 100deg2: " +
          " ".join(f"{nu}:{v:.3f}" for nu, v in zip(nus, n_)))
print("sum over 12 scales per deg^2: " + " ".join(f"{nu}:{v/100:.4e}" for nu, v in zip(nus, tot)))
print(f"elapsed {time.time()-t0:.0f}s")
