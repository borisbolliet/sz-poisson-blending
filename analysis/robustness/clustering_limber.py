"""Beyond Poisson: angular clustering of the detected cluster sample (q_cut = 5, baseline model).

w(theta) by Limber with the colossus linear matter power spectrum (planck18, Eisenstein-Hu 98) and
Tinker10 halo bias (mdef 500c), selection-weighted:
   p(z)   = dN_det/dz / N_det,   dN_det/dz = int dlnM d2N/dz/dlnM * P_det(M, z | q_cut)
   b(z)   = int dlnM d2N P_det b(M, z) / int dlnM d2N P_det
   w(th)  = int dz p(z)^2 (H/c) b(z)^2 D(z)^2 I(chi th),  I(R) = int dk k/(2pi) P0(k) J0(kR)
Checks:
   (a) k-integral convergence (grid size, kmax);
   (b) I(R) (Fourier) = int dDelta xi0(sqrt(R^2+Delta^2)) (real-space projection of colossus xi);
   (c) Limber vs the exact (non-Limber) double integral
       w = int dz1 dz2 p1 p2 b1 b2 D1 D2 xi0(r12),  r12^2 = chi1^2 + chi2^2 - 2 chi1 chi2 cos(theta);
   (d) z-grid convergence.
Pair probability with clustering: <number of other detections within theta> = Sigma A (1 + wbar),
wbar(theta) = area average of w inside theta; P(>=1) = 1 - exp(-Sigma A (1+wbar)).
Halo exclusion: halos cannot overlap, xi -> -1 for r < r_ex = (R500A + R500B) (comoving).
   w_ex(theta) = Limber real-space form with xi_ex(r) = b^2 xi_lin(r) for r > r_ex, -1 for r < r_ex.
   Analytic overlap fraction of Poisson pairs + MC check in aggregate.py (pairs from run_pass.py).
Cross-correlation detected x all halos (M>1e13), relevant for the neighbours that blend.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import j0
from scipy.interpolate import CubicSpline
from colossus.lss import bias as cbias

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from szsim import cosmo_hmf as ch, signal as sg

t0 = time.time()
cosmo = ch.get_cosmology(); h = cosmo.h
model = sg.SZModel()
T = ch.default_table()
QCUT = 5.0
C_KMS = ch.C_KMS


def selection(zsub=4, msub=5, qcut=QCUT, Mmin=None):
    z = T.z[::zsub]; lnM = T.lnM[::msub]; M = np.exp(lnM)
    d2N = T.d2N[::zsub, ::msub]
    wM = np.full(len(lnM), lnM[1] - lnM[0]); wM[0] *= .5; wM[-1] *= .5
    if Mmin is not None:
        wM = wM * (M >= Mmin)
    Pd = np.array([model.p_det(M, zz, qcut) for zz in z]) if qcut is not None else np.ones_like(d2N)
    b = np.array([cbias.haloBias(M * h, zz, "500c", model="tinker10") for zz in z])
    Rcom = np.array([model.R500_Mpc(M, zz) * (1 + zz) for zz in z])
    W = d2N * Pd * wM[None, :]
    dNdz = W.sum(1)
    beff = (W * b).sum(1) / dNdz
    Rm = (W * Rcom).sum(1) / dNdz
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= .5; wz[-1] *= .5
    Ntot = (dNdz * wz).sum()
    return dict(z=z, M=M, p=dNdz / Ntot, b=beff, N=Ntot, W=W, Pd=Pd, bM=b, Rcom=Rcom, Rmean=Rm, wz=wz)


S = selection()
print(f"selection q_cut={QCUT}: N_det (this grid) = {S['N']:.1f} vs model.expected_detections = "
      f"{model.expected_detections(QCUT):.1f}; b_eff(z=0.1,0.5,1,2) = "
      f"{np.interp([0.1, 0.5, 1, 2], S['z'], S['b']).round(3)}; mean b over p(z) = {np.sum(S['p']*S['b']*S['wz']):.3f}")
H = S  # halos M>1e13 (no selection) for the cross-correlation
Hn = selection(qcut=None)
print(f"all halos M>1e13: N = {Hn['N']:.4g}, b_eff(z=0.1,0.5,1,2) = {np.interp([0.1,0.5,1,2], Hn['z'], Hn['b']).round(3)}")

# ---------------- I(R) = int dk k/(2pi) P0(k) J0(k R), R in comoving Mpc
def P0(k_Mpc):
    return cosmo.matterPowerSpectrum(k_Mpc / h, 0.0) / h ** 3


def I_fourier(R, nk=2 ** 16, kmin=1e-5, kmax=1e3):
    lk = np.linspace(np.log(kmin), np.log(kmax), nk); k = np.exp(lk)
    f = k * k * P0(k) / (2 * np.pi)
    out = np.empty(len(R))
    for i, r in enumerate(R):
        out[i] = np.trapezoid(f * j0(k * r), lk)
    return out


Rg = np.geomspace(1e-3, 300.0, 300)
I_a = I_fourier(Rg)
I_b = I_fourier(Rg, nk=2 ** 18)
I_c = I_fourier(Rg, kmax=1e2)
sel = Rg < 100
print(f"(a) I(R) k-convergence, R in [1e-3,100] Mpc: max rel diff nk 2^16 vs 2^18 = {np.max(np.abs(I_b[sel]/I_a[sel]-1)):.2e}; "
      f"kmax 1e3 vs 1e2 = {np.max(np.abs(I_c[sel]/I_a[sel]-1)):.2e} (R>0.1: {np.max(np.abs(I_c[(Rg>0.1)&sel]/I_a[(Rg>0.1)&sel]-1)):.2e})")
Ispl = CubicSpline(np.log(Rg), I_a)

# (b) real-space projection of colossus xi
rr = np.geomspace(2e-3, 5000.0, 700)


def xi_own(r, nk=2 ** 18, kmin=1e-5, kmax=1e3):
    """xi0(r) = int dk k^2/(2pi^2) P0(k) sin(kr)/(kr) (log-k trapezoid, chunked)."""
    lk = np.linspace(np.log(kmin), np.log(kmax), nk); k = np.exp(lk)
    f = k ** 3 * P0(k) / (2 * np.pi ** 2)
    out = np.empty(len(r))
    for a in range(0, len(r), 20):
        rc = r[a:a + 20]
        out[a:a + 20] = np.trapezoid(f[None, :] * np.sinc(np.outer(rc, k) / np.pi), lk, axis=1)
    return out


xi0 = xi_own(rr)
inr = (rr * h > 1.001e-3) & (rr * h < 499.0)
xcol = cosmo.correlationFunction(rr[inr] * h, 0.0)
big = np.abs(xcol) > 1e-3
jw = np.argmax(np.abs(xi0[inr][big]/xcol[big]-1))
print(f"   worst point r = {rr[inr][big][jw]:.3g} Mpc (xi = {xcol[big][jw]:.3g}); for r in [0.01, 300] Mpc and |xi|>1e-3: max rel diff = "
      f"{np.max(np.abs(xi0[inr][big & (rr[inr]>0.01) & (rr[inr]<300)]/xcol[big & (rr[inr]>0.01) & (rr[inr]<300)]-1)):.2e}")
print(f"xi0 own vs colossus correlationFunction (r in [{rr[inr][0]:.3g},{rr[inr][-1]:.3g}] Mpc, |xi|>1e-3): "
      f"max rel diff = {np.max(np.abs(xi0[inr][big]/xcol[big]-1)):.2e}; max abs diff (all) = {np.max(np.abs(xi0[inr]-xcol)):.2e}")
xispl = CubicSpline(np.log(rr), xi0)


def xi_of(r):
    r = np.clip(r, rr[0], rr[-1])
    return np.where(r >= rr[-1] * 0.999, 0.0, xispl(np.log(r)))


def I_real(R):
    D = np.concatenate([[0], np.geomspace(1e-4, 2500.0, 6000)])
    return np.array([2 * np.trapezoid(xi_of(np.sqrt(r * r + D * D)), D) for r in R])


Rt = np.array([0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
Ir = I_real(Rt); If = np.exp(0) * Ispl(np.log(Rt))
print("(b) I(R) Fourier vs real-space projection of xi0 (own sine transform, checked against colossus above):")
for r, a, b in zip(Rt, If, Ir):
    print(f"     R = {r:6.2f} Mpc: Fourier {a:10.4f} Mpc, projected xi {b:10.4f} Mpc, ratio {b/a:.4f}")

# ---------------- Limber w(theta)
ARC = ch.ARCMIN


def limber_w(theta_arcmin, S1, S2=None, zsub=1, xi_mod=None):
    S2 = S1 if S2 is None else S2
    z = S1["z"][::zsub]; p1 = S1["p"][::zsub]; p2 = S2["p"][::zsub]; b1 = S1["b"][::zsub]; b2 = S2["b"][::zsub]
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= .5; wz[-1] *= .5
    Hc = ch.Ez(z) * cosmo.H0 / C_KMS                 # 1/Mpc
    chi = ch.chi_Mpc(z); D = cosmo.growthFactor(z)
    out = []
    for th in np.atleast_1d(theta_arcmin):
        R = chi * th * ARC
        if xi_mod is None:
            I = Ispl(np.log(np.clip(R, Rg[0], Rg[-1])))
            out.append(np.sum(wz * p1 * p2 * Hc * b1 * b2 * D ** 2 * I))
        else:
            out.append(np.sum(wz * p1 * p2 * Hc * xi_mod(R, z, b1 * b2 * D ** 2)))
    return np.array(out)


TH = np.geomspace(0.01, 1500.0, 460)   # extended to 1500' after verification (rho table reaches 1200')
w = limber_w(TH, S)
w2 = limber_w(TH, S, zsub=2)
print(f"(d) z-grid convergence (301 vs 151 nodes): max rel diff in w over 0.01'-120' = {np.max(np.abs(w2/w-1)[TH <= 120]):.2e}")


def wbar(theta, TH, w):
    """area average of w inside theta (flat sky)."""
    out = []
    for t in np.atleast_1d(theta):
        x = np.concatenate([[0.0], TH[TH < t], [t]]); y = np.interp(x, TH, w, left=w[0])
        out.append(2 * np.trapezoid(y * x, x) / t ** 2)
    return np.array(out)


# (c) exact non-Limber double integral
def w_exact(theta_arcmin, S1, nD=1500):
    z = S1["z"]; p = S1["p"]; b = S1["b"]
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= .5; wz[-1] *= .5
    chi = ch.chi_Mpc(z); D = cosmo.growthFactor(z)
    zf = np.linspace(0, 3.5, 7001); chif = ch.chi_Mpc(zf)
    z_of_chi = lambda c: np.interp(c, chif, zf)
    Hc = ch.Ez(zf) * cosmo.H0 / C_KMS
    pf = lambda zz: np.interp(zz, z, p, left=0.0, right=0.0)
    bf = lambda zz: np.interp(zz, z, b)
    Df = lambda zz: np.interp(zz, zf, cosmo.growthFactor(zf))
    Dl = np.concatenate([-np.geomspace(3000, 1e-4, nD), [0.0], np.geomspace(1e-4, 3000, nD)])
    out = []
    for th in np.atleast_1d(theta_arcmin):
        t = th * ARC
        acc = 0.0
        for i in range(len(z)):
            if p[i] == 0:
                continue
            c2 = chi[i] + Dl
            ok = c2 > 0
            z2 = z_of_chi(c2[ok])
            # dz2 = dchi2 * H(z2)/c
            H2 = np.interp(z2, zf, Hc)
            r12 = np.sqrt(chi[i] ** 2 + c2[ok] ** 2 - 2 * chi[i] * c2[ok] * np.cos(t))
            integ = pf(z2) * bf(z2) * Df(z2) * H2 * xi_of(r12)
            acc += wz[i] * p[i] * b[i] * D[i] * np.trapezoid(integ, Dl[ok])
        out.append(acc)
    return np.array(out)


TCHK = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 60.0])
we = w_exact(TCHK, S)
wl = np.interp(TCHK, TH, w)
print("(c) Limber vs exact non-Limber double integral (linear xi):")
for t, a, b in zip(TCHK, wl, we):
    print(f"     theta = {t:5.1f}': Limber {a:.5f}, exact {b:.5f}, ratio {a/b:.4f}")

# ---------------- exclusion: xi = -1 for r < r_ex(z) = 2 <R500,com>_det(z) (typical pair)
rex = 2 * S["Rmean"]


def xi_ex_factory(rex_z):
    D = np.concatenate([[0], np.geomspace(1e-4, 2500.0, 4000)])

    def f(R, z, amp):
        out = np.empty(len(z))
        for i in range(len(z)):
            r = np.sqrt(R[i] ** 2 + D ** 2)
            xi = np.where(r < rex_z[i], -1.0, amp[i] * xi_of(r))
            out[i] = 2 * np.trapezoid(xi, D)
        return out
    return f


TEX = np.geomspace(0.05, 120.0, 120)
wex = limber_w(TEX, S, xi_mod=xi_ex_factory(rex))
wlin_chk = limber_w(TEX, S, xi_mod=xi_ex_factory(np.zeros_like(rex)))
print(f"real-space Limber (no exclusion) vs Fourier Limber on 0.05'-120': max rel diff = "
      f"{np.max(np.abs(wlin_chk/np.interp(TEX, TH, w)-1)):.2e}")
print(f"r_ex(z) = 2<R500,com>_det: z=0.1: {np.interp(0.1, S['z'], rex):.2f} Mpc, z=0.5: {np.interp(0.5, S['z'], rex):.2f}, "
      f"z=1: {np.interp(1.0, S['z'], rex):.2f}, z=2: {np.interp(2.0, S['z'], rex):.2f}")

# ---------------- cross: detected x all halos M>1e13 (neighbours that blend)
wx = limber_w(TH, S, Hn)

Sigma = S["N"] / (4 * np.pi) * ARC ** 2  # per arcmin^2
print(f"\nSigma_det(q>5) = {Sigma:.4e} /arcmin^2 = {S['N']/(4*np.pi*(180/np.pi)**2):.4f} /deg^2")
print(" theta | w(theta)  | 1+wbar (lin) | 1+wbar (lin+excl) | 1+wbar_x (det x M>1e13) | P_Poisson(>=1) | P_clustered(>=1) | ratio")
rows = []
for t in [1.0, 2.0, 5.0, 10.0, 30.0]:
    A = 2 * np.pi * (1 - np.cos(t * ARC)) / ARC ** 2
    wb = wbar(t, TH, w)[0]; wbe = wbar(t, TEX, wex)[0]; wbx = wbar(t, TH, wx)[0]
    Pp = 1 - np.exp(-Sigma * A); Pc = 1 - np.exp(-Sigma * A * (1 + wb)); Pce = 1 - np.exp(-Sigma * A * (1 + wbe))
    rows.append(dict(theta=t, w=float(np.interp(t, TH, w)), one_plus_wbar=1 + wb, one_plus_wbar_ex=1 + wbe,
                     one_plus_wbar_x=1 + wbx, P_poisson=Pp, P_clust=Pc, P_clust_ex=Pce))
    print(f" {t:5.1f}' | {np.interp(t, TH, w):.5f} | {1+wb:.5f} | {1+wbe:.5f} | {1+wbx:.5f} | {Pp:.5e} | {Pc:.5e} | {Pc/Pp:.5f}"
          f" (with excl. {Pce/Pp:.5f})")
print(f"w(theta->0) (linear, theta=0.01') = {w[0]:.5f}; w_x(0.01') = {wx[0]:.5f}; w_ex(0.05') = {wex[0]:.4f}")

# ---------------- Poisson overlap (exclusion-violating) pairs, analytic with full mass dependence
# unordered detected pairs (full sky) with angular sep < theta AND 3D comoving sep < R_A + R_B:
#   N_ovl(<theta) = 1/2 int dz dlnM_A dlnM_B d2N_det(A) n_det,com(M_B, z) V(theta),
#   V = 4pi/3 [Rs^3 - (Rs^2 - a^2)_+^(3/2)],  a = chi theta, Rs = R_A,com + R_B,com
dVdz = 4 * np.pi * ch.dV_dz_dOmega_Mpc3(S["z"])                 # Mpc^3 per unit z, full sky
ncom = S["W"] / dVdz[:, None]                                    # per Mpc^3 (x dlnM weights)
chiz = ch.chi_Mpc(S["z"])
TOV = np.array(list(np.atleast_1d([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])) + [150.0])
Novl = []
for t in TOV:
    acc = 0.0
    for i in range(len(S["z"])):
        Rs = S["Rcom"][i][:, None] + S["Rcom"][i][None, :]
        a = chiz[i] * t * ARC
        V = 4 * np.pi / 3 * (Rs ** 3 - np.clip(Rs ** 2 - a * a, 0, None) ** 1.5)
        acc += S["wz"][i] * (S["W"][i][:, None] * ncom[i][None, :] * V).sum()
    Novl.append(0.5 * acc)
Novl = np.array(Novl)
Npairs = np.array([0.5 * S["N"] * S["N"] / (4 * np.pi) * 2 * np.pi * (1 - np.cos(t * ARC)) for t in TOV])
print("\nPoisson pairs of detections (q>5) that overlap in 3D (would violate halo exclusion), analytic:")
for t, a, b in zip(TOV, Novl, Npairs):
    print(f"   sep < {t:6.1f}': N_pairs = {b:12.1f}, N_overlap = {a:8.3f}, fraction = {a/b:.3e}")
# all-halo neighbours (M>1e13): expected number of physically overlapping halos per detected cluster
ncom_all = Hn["W"] / dVdz[:, None]
Eov = 0.0
for i in range(len(S["z"])):
    Rs = S["Rcom"][i][:, None] + Hn["Rcom"][i][None, :]
    Eov += S["wz"][i] * (S["W"][i][:, None] * ncom_all[i][None, :] * 4 * np.pi / 3 * Rs ** 3).sum()
Eov /= S["N"]
print(f"expected number of M>1e13 halos overlapping (3D sep < R_A+R_B) a detected q>5 cluster: {Eov:.4e} "
      f"-> P(>=1) = {1-np.exp(-Eov):.4e}")

# ---------------- clustering correction to the mean blending boost
# delta_A = sum_B rho_AB Y_B / sigma_A: its mean around a detected cluster is
#   E[delta_A] = mu_A [1 + W_A],  W_A = int d^2s rho(s) w_dy(s) / int d^2s rho(s),
# with w_dy the cross-correlation of detections with the y-weighted halo field (M>1e13):
#   p_y(z) = (dybar/dz)/ybar, b_y(z) = Y-weighted Tinker10 bias.
Ymean = np.array([model.Y500_med(Hn["M"], zz) for zz in Hn["z"]]) * np.exp(0.5 * model.sigma_lnY ** 2)
Wy = Hn["W"] * Ymean
dydz = Wy.sum(1)
Sy = dict(z=Hn["z"], p=dydz / (dydz * Hn["wz"]).sum(), b=(Wy * Hn["bM"]).sum(1) / dydz)
print(f"\ny-weighted halo field (M>1e13): b_y(z=0.1,0.5,1,2) = {np.interp([0.1,0.5,1,2], Sy['z'], Sy['b']).round(3)}; "
      f"median z of dybar/dz = {np.interp(0.5, np.cumsum(Sy['p']*Hn['wz']), Sy['z']):.3f}")
wdy = limber_w(TH, S, Sy)
import blendlib as bl
rt = bl.RhoTable(1.4)
Wfac = {}
for thA in [0.5, 1.0, 2.0, 5.0]:
    for thB in [0.5, 1.0]:
        sg_ = rt.sep
        rh = rt(np.full(sg_.shape, thA), np.full(sg_.shape, thB), sg_)
        wv = np.interp(sg_, TH, wdy, left=wdy[0])
        Wfac[(thA, thB)] = np.trapezoid(rh * wv * sg_, sg_) / np.trapezoid(rh * sg_, sg_)
print("clustering correction to the mean neighbour boost, W = <w_dy>_rho (linear bias, no exclusion):")
for (a, b), v in Wfac.items():
    print(f"   theta_A = {a:3.1f}', theta_B = {b:3.1f}': W = {v:.4f}")
thA_grid = np.geomspace(0.2, 60, 40)
W_grid = []
for thA in thA_grid:
    rh = rt(np.full(rt.sep.shape, thA), np.full(rt.sep.shape, 0.6), rt.sep)
    W_grid.append(np.trapezoid(rh * np.interp(rt.sep, TH, wdy, left=wdy[0]) * rt.sep, rt.sep) / np.trapezoid(rh * rt.sep, rt.sep))
W_grid = np.array(W_grid)
print(f"   W(theta_A; theta_B=0.6') at 0.2', 1', 3', 10', 60': {np.interp([0.2,1,3,10,60], thA_grid, W_grid).round(4)}")
print(f"   w_dy(1') = {np.interp(1.0, TH, wdy):.4f}, w_dy(5') = {np.interp(5.0, TH, wdy):.4f}, w_dy(0.01') = {wdy[0]:.4f}")

# ---------------- [added after verification] (i) y-weighted theta_B distribution instead of theta_B = 0.6';
# (ii) halo-exclusion bracket for w_dy (xi = -1 for r < <R500com>_det(z) + <R500com>_y-wtd(z)).
#   E[delta_A] = mu_A (1 + W_A),  W_A = int d2s w_dy(s) R_A(s) / int d2s R_A(s),
#   R_A(s) = sum_B Y_B rho(theta_A, theta_B, s) / sum_B Y_B   (y-weighted over the M>1e13 neighbour population)
thB_MB = np.array([model.theta500(Hn["M"], zz) for zz in Hn["z"]])          # (nz, nM) arcmin
wyB = Wy * Hn["wz"][:, None]
lo, hi = np.log(rt.th[0]), np.log(rt.th[-1])
edges = np.linspace(np.log(0.03), np.log(200.0), 81)
hB, _ = np.histogram(np.log(thB_MB).ravel(), bins=edges, weights=wyB.ravel())
cB = np.exp(0.5 * (edges[1:] + edges[:-1])); pB = hB / hB.sum()
cdf = np.cumsum(pB)
print(f"y-weighted neighbour theta500 (M>1e13): median {np.exp(np.interp(0.5, cdf, 0.5*(edges[1:]+edges[:-1]))):.3f}', "
      f"16-84%: {np.exp(np.interp(0.16, cdf, 0.5*(edges[1:]+edges[:-1]))):.3f}'-{np.exp(np.interp(0.84, cdf, 0.5*(edges[1:]+edges[:-1]))):.3f}'; "
      f"fraction of ybar with theta500 < {rt.th[0]}' (clipped to table edge): {pB[cB < rt.th[0]].sum():.2e}")
cBc = np.clip(cB, rt.th[0], rt.th[-1])
Rex_y = (Wy * Hn["Rcom"]).sum(1) / dydz
rex_dy = S["Rmean"] + Rex_y
print(f"exclusion radius det x y-wtd: r_ex(z=0.1,0.5,1,2) = {np.interp([0.1,0.5,1,2], S['z'], rex_dy).round(3)} Mpc")
TEXd = np.geomspace(0.05, 150.0, 110)
wdy_ex_s = limber_w(TEXd, S, Sy, xi_mod=xi_ex_factory(rex_dy))
wdy_chk = limber_w(TEXd, S, Sy, xi_mod=xi_ex_factory(np.zeros_like(rex_dy)))
print(f"real-space vs Fourier w_dy (no exclusion) on 0.05'-150': max rel diff {np.max(np.abs(wdy_chk/np.interp(TEXd, TH, wdy)-1)):.2e}; "
      f"w_dy_ex/w_dy at 150' = {wdy_ex_s[-1]/np.interp(150.0, TH, wdy):.4f}")
# exclusion only matters at small theta; beyond 150' use the linear w_dy
wdy_ex = np.where(TH <= 150.0, np.interp(TH, TEXd, wdy_ex_s, left=wdy_ex_s[0]), wdy)
print("w_dy linear vs with exclusion at 0.1, 0.5, 1, 2, 5, 10': "
      + ", ".join(f"{t}': {np.interp(t, TH, wdy):.4f}/{np.interp(t, TH, wdy_ex):.4f}" for t in (0.1, 0.5, 1, 2, 5, 10)))


def W_of(thA, wcurve, yweighted):
    sep = rt.sep
    wv = np.interp(sep, TH, wcurve)
    if yweighted:
        rh = sum(p * rt(np.full(sep.shape, thA), np.full(sep.shape, tb), sep) for p, tb in zip(pB, cBc) if p > 0)
    else:
        rh = rt(np.full(sep.shape, thA), np.full(sep.shape, 0.6), sep)
    return np.trapezoid(rh * wv * sep, sep) / np.trapezoid(rh * sep, sep)


WV = {}
for name, wc, yw in (("lin_06", wdy, False), ("lin_yw", wdy, True), ("ex_06", wdy_ex, False), ("ex_yw", wdy_ex, True)):
    WV[name] = np.array([W_of(t, wc, yw) for t in thA_grid])
    print(f"   W(theta_A) [{name:6s}] at 0.2', 1', 3', 10', 60': {np.interp([0.2,1,3,10,60], thA_grid, WV[name]).round(4)}")
print(f"   check: lin_06 recomputed vs W_grid (old loop, now with w_dy to 1500'): max abs diff {np.max(np.abs(WV['lin_06']-W_grid)):.1e}")
w120 = np.where(TH <= 120.0, wdy, np.interp(120.0, TH, wdy))
W120 = np.array([W_of(t, w120, False) for t in thA_grid])
print(f"   effect of the old flat extrapolation beyond 120': W(60') {np.interp(60, thA_grid, W120):.4f} (old) vs {np.interp(60, thA_grid, WV['lin_06']):.4f} (w_dy to 1500')")
for name in ("ex_06", "lin_yw", "ex_yw"):
    r = np.interp([0.2, 1, 3, 10, 60], thA_grid, WV[name]) / np.interp([0.2, 1, 3, 10, 60], thA_grid, WV["lin_06"])
    print(f"   ratio W[{name}]/W[lin_06] at 0.2,1,3,10,60': {r.round(3)}")

np.savez(os.path.join(ch.DATA_DIR, "robustness", "limber.npz"), wdy=wdy, wdy_ex=wdy_ex, thA_grid=thA_grid, W_grid=W_grid,
         W_lin_06=WV['lin_06'], W_lin_yw=WV['lin_yw'], W_ex_06=WV['ex_06'], W_ex_yw=WV['ex_yw'], thB_c=cB, thB_p=pB, TOV=TOV, Novl=Novl, Npairs=Npairs, Eov=Eov, TH=TH, w=w, wx=wx, TEX=TEX, wex=wex,
         z=S["z"], p=S["p"], b=S["b"], pn=Hn["p"], bn=Hn["b"], rex=rex, Sigma=Sigma, TCHK=TCHK, we=we,
         rows=json.dumps(rows))
print(f"done {time.time()-t0:.0f}s")
