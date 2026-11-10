"""V4: independent clustering numbers (C7, C8, C9 W-factors, C10 analytic overlap).
Different implementation: own selection (indep_model), I(R) by FFTLog (scipy.fft.fht) instead of trapezoid,
exclusion via colossus xi directly, wbar by quadrature; W from own response table, w_dy computed out to 2000'
(no clamping), and a Y-weighted theta_B average instead of a single theta_B = 0.6'.
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.fft import fht, fhtoffset
from scipy.interpolate import CubicSpline
from colossus.lss import bias as cbias
import indep_model as im

t0 = time.time()
cosmo = im.cosmo; h = im.h; ARC = im.ARC
inst = im.Instrument(1.4, 4e-6)

# ---------------- selection on a GL grid
z, wz, lnM, wm = im.zM_grid(nzp=50, nmp=40, k=6)
M = np.exp(lnM)
d2N = np.array([im.dndlnM(M, zz) * im.dVdzdO(zz) * 4 * np.pi for zz in z])
bM = np.array([cbias.haloBias(M * h, zz, "500c", model="tinker10") for zz in z])
qm = np.array([im.Ymed(M, zz) / inst.sigY(im.theta500(M, zz)) for zz in z])
Pd = im.p_det(qm, 0.173, 5.0)
W = d2N * Pd * wm[None, :]
dNdz = W.sum(1); Ndet = np.sum(wz * dNdz)
pz = dNdz / Ndet; bz = (W * bM).sum(1) / dNdz
print(f"N_det(q>5) = {Ndet:.1f}; b_eff(z=0.1,0.5,1,2) = {np.interp([0.1,0.5,1,2], z, bz).round(3)} (claimed 2.82, 4.99, 7.53, 13.45)")
Rcom = np.array([im.R500(M, zz) * (1 + zz) for zz in z])
Rm = (W * Rcom).sum(1) / dNdz
# y-weighted halos M>1e13
Wn = d2N * wm[None, :]
Ymean = np.array([im.Ymed(M, zz) for zz in z]) * np.exp(0.173 ** 2 / 2)
Wy = Wn * Ymean
dydz = Wy.sum(1); py = dydz / np.sum(wz * dydz); by = (Wy * bM).sum(1) / dydz
print(f"b_y(z=0.1,0.5,1,2) = {np.interp([0.1,0.5,1,2], z, by).round(3)}")

# ---------------- I(R) by FFTLog
n = 8192; dln = np.log(1e7) / n * 2.5
off = fhtoffset(dln, mu=0, initial=0.0)
j = np.arange(n); jc = (n - 1) / 2
k = 1.0 * np.exp((j - jc) * dln)  # 1/Mpc, centred at 1
rr = np.exp(off) / 1.0 * np.exp((j - jc) * dln)
Pk = cosmo.matterPowerSpectrum(np.clip(k / h, 1e-12, 1e8), 0.0) / h ** 3
Pk[(k < 1e-6) | (k > 1e4)] = 0.0
IR = fht(Pk / (2 * np.pi) * k, dln, mu=0, offset=off) / rr
ok = (rr > 1e-3) & (rr < 3000)
Ispl = CubicSpline(np.log(rr[ok]), IR[ok])
print(f"I(R) FFTLog at R=0.05,1,10,30 Mpc: {Ispl(np.log([0.05,1,10,30])).round(4)} (analysis: 79.7944, 59.3968, 20.8442, 5.8304)")

Hc = im.Ez(z) * cosmo.H0 / im.C_KMS
chi = im.chi(z); D = cosmo.growthFactor(z)


def limber(theta, p1, b1, p2, b2):
    th = np.atleast_1d(theta)
    out = np.empty(len(th))
    for i, t in enumerate(th):
        R = chi * t * ARC
        out[i] = np.sum(wz * p1 * p2 * Hc * b1 * b2 * D ** 2 * Ispl(np.log(R)))
    return out


TH = np.geomspace(0.003, 2000, 600)
w = limber(TH, pz, bz, pz, bz)
wsp = CubicSpline(np.log(TH), w)


def wbar(t, wf):
    x, wx = im._gl(0, t, 200)
    return 2 * np.sum(wx * x * wf(x)) / t ** 2


wf = lambda x: wsp(np.log(np.clip(x, TH[0], None)))

# ---------------- exclusion: xi -> -1 for r < rex(z) = 2 <R500,com>_det
rex = 2 * Rm
rg = np.geomspace(1.001e-3, 499.0, 800)  # Mpc/h
xi_col = CubicSpline(np.log(rg / h), cosmo.correlationFunction(rg, 0.0))


def I_ex(R, rx, amp):
    """projection of xi_ex = amp*xi for r>rx, -1 inside (amp = b^2 D^2)."""
    base = amp * Ispl(np.log(R))
    if R >= rx:
        return base
    L = np.sqrt(rx * rx - R * R)
    x, wx = im._gl(0, 1, 200)
    d = L * x ** 2; jac = 2 * L * x
    rr_ = np.sqrt(R * R + d * d)
    inner = 2 * np.sum(wx * jac * xi_col(np.log(np.clip(rr_, 1.001e-3 / h, None))))
    return base - amp * inner - 2 * L


def limber_ex(theta):
    out = []
    for t in np.atleast_1d(theta):
        R = chi * t * ARC
        out.append(np.sum([wz[i] * pz[i] ** 2 * Hc[i] * I_ex(R[i], rex[i], bz[i] ** 2 * D[i] ** 2) for i in range(len(z))]))
    return np.array(out)


TEX = np.geomspace(0.003, 200, 160)
wex = limber_ex(TEX)
wexsp = CubicSpline(np.log(TEX), wex)
wexf = lambda x: wexsp(np.log(np.clip(x, TEX[0], None)))

Sig = Ndet / (4 * np.pi) * ARC ** 2
claimed = {1: (1.267, 1.131, 1.816e-3, 1.267), 2: (1.242, 1.133, 7.243e-3, 1.241), 5: (1.196, 1.145, 4.441e-2, 1.190),
           10: (1.153, 1.137, 0.1662, 1.138), 30: (1.082, 1.080, 0.8051, 1.030)}
print(f"Sigma_det = {Ndet/(4*np.pi)/(180/np.pi)**2:.4f}/deg2; w(1') = {wf(1.0):.4f} (claimed 0.2515), w(0.01') = {wf(0.01):.4f} (claimed 0.315)")
print(" theta | 1+wbar lin (claimed) | 1+wbar excl (claimed) | P_Poisson (claimed) | P_clust/P_Poisson (claimed) | excl ratio")
for t, c in claimed.items():
    A = 2 * np.pi * (1 - np.cos(t * ARC)) / ARC ** 2
    wb = wbar(t, wf); wbe = wbar(t, wexf)
    Pp = 1 - np.exp(-Sig * A); Pc = 1 - np.exp(-Sig * A * (1 + wb)); Pe = 1 - np.exp(-Sig * A * (1 + wbe))
    print(f" {t:4d}' | {1+wb:.4f} ({c[0]}) | {1+wbe:.4f} ({c[1]}) | {Pp:.4e} ({c[2]}) | {Pc/Pp:.4f} ({c[3]}) | {Pe/Pp:.4f}")

# ---------------- C10 analytic overlap pairs (own)
dV = 4 * np.pi * im.dVdzdO(z)
ncom = W / dV[:, None]
print("analytic overlapping detected pairs (q>5):")
for t in (1.0, 2.0, 5.0, 10.0, 30.0):
    acc = 0.0
    for i in range(len(z)):
        Rs = Rcom[i][:, None] + Rcom[i][None, :]
        a = chi[i] * t * ARC
        V = 4 * np.pi / 3 * (Rs ** 3 - np.clip(Rs ** 2 - a * a, 0, None) ** 1.5)
        acc += wz[i] * (W[i][:, None] * ncom[i][None, :] * V).sum()
    Np = 0.5 * Ndet ** 2 / (4 * np.pi) * 2 * np.pi * (1 - np.cos(t * ARC))
    print(f"   sep<{t:4.0f}': N_pairs {Np:10.1f}, N_ovl {0.5*acc:.3f}, fraction {0.5*acc/Np:.3e}")
ncom_all = Wn / dV[:, None]
E = sum(wz[i] * (W[i][:, None] * ncom_all[i][None, :] * 4 * np.pi / 3 * (Rcom[i][:, None] + Rcom[i][None, :]) ** 3).sum() for i in range(len(z))) / Ndet
print(f"   E[# M>1e13 halos overlapping a detection] = {E:.4e} -> P(>=1) = {1-np.exp(-E):.4e} (claimed 2.167e-3)")

# ---------------- C9: W factor with own rho table and w_dy
wdy = limber(TH, pz, bz, py, by)
wdysp = CubicSpline(np.log(TH), wdy)
print(f"w_dy(1') = {wdysp(0.0):.4f} (claimed 0.1522), w_dy(5') = {wdysp(np.log(5)):.4f} (claimed 0.1026)")
thn = np.geomspace(0.04, 150, 130)
sepn = np.concatenate([np.linspace(0, 20, 401), np.geomspace(20, 700, 181)[1:]])
rho = inst.rho_table(thn, sepn)
wds = wdysp(np.log(np.clip(sepn, TH[0], None)))
wds_clamp = np.interp(sepn, TH[TH <= 120], wdy[TH <= 120])  # analysis' clamping beyond 120'


def Wfac(ia, ib, wv=wds):
    r = rho[ia, ib].astype(float)
    return np.trapezoid(r * wv * sepn, sepn) / np.trapezoid(r * sepn, sepn)


def idx(t):
    return int(np.argmin(np.abs(np.log(thn / t))))


print("W(theta_A; theta_B) own [node values]:")
for ta in (0.2, 1, 3, 10, 60):
    ia = idx(ta)
    print(f"   theta_A={thn[ia]:.3f}': W(thB~0.6)={Wfac(ia, idx(0.6)):.4f} (claimed at {ta}': {dict(zip([0.2,1,3,10,60],[0.149,0.143,0.126,0.091,0.033]))[ta]}), "
          f"clamped-w version {Wfac(ia, idx(0.6), wds_clamp):.4f}; W(thB~2)={Wfac(ia, idx(2.0)):.4f}, W(thB~5)={Wfac(ia, idx(5.0)):.4f}")
# Y-weighted theta_B distribution of neighbours (M>1e13), per detection-weighted theta_A
lthB = np.log(np.array([im.theta500(M, zz) for zz in z]))
yw = (Wy * wz[:, None]).ravel(); lthB = lthB.ravel()
hist, edges = np.histogram(lthB, bins=np.log(np.concatenate([[1e-3], np.sqrt(thn[1:] * thn[:-1]), [1e4]])), weights=yw)
hist /= hist.sum()
print(f"Y-weighted theta_B: median {np.exp(np.interp(0.5, np.cumsum(hist), np.log(thn))):.2f}', "
      f"16-84%: {np.exp(np.interp(0.16, np.cumsum(hist), np.log(thn))):.2f}'-{np.exp(np.interp(0.84, np.cumsum(hist), np.log(thn))):.2f}'")
# Exact Y-weighted W: W_A = sum_B y_B int rho_AB w / sum_B y_B int rho_AB ; int rho_AB d2s is independent of theta_B
Wgrid_y, Wgrid_06 = [], []
for ia in range(len(thn)):
    num = sum(hist[ib] * np.trapezoid(rho[ia, ib] * wds * sepn, sepn) for ib in range(len(thn)) if hist[ib] > 0)
    den = sum(hist[ib] * np.trapezoid(rho[ia, ib] * sepn, sepn) for ib in range(len(thn)) if hist[ib] > 0)
    Wgrid_y.append(num / den); Wgrid_06.append(Wfac(ia, idx(0.6)))
Wgrid_y = np.array(Wgrid_y); Wgrid_06 = np.array(Wgrid_06)
print("W, Y-weighted theta_B at theta_A = 0.2,1,3,10,60':", np.interp(np.log([0.2, 1, 3, 10, 60]), np.log(thn), Wgrid_y).round(4))
# mean over detections, and mean shift W*mu (mu with own ybar)
yb = im.ybar()
thd = np.array([im.theta500(M, zz) for zz in z])
wdet = W * wz[:, None]
for lab, Wg in (("theta_B=0.6'", Wgrid_06), ("Y-weighted theta_B", Wgrid_y)):
    Wd = np.interp(np.log(thd), np.log(thn), Wg)
    mu = inst.mu_coef(thd) * yb
    print(f"   mean over q>5 detections ({lab}): <W> = {np.sum(wdet*Wd)/np.sum(wdet):.4f} (claimed 0.129), <W mu> = {np.sum(wdet*Wd*mu)/np.sum(wdet):.4f} (claimed 0.125), <mu> = {np.sum(wdet*mu)/np.sum(wdet):.4f}")
np.savez(os.path.join(im.CACHE, "W_own.npz"), thn=thn, W06=Wgrid_06, Wy=Wgrid_y)
print(f"done {time.time()-t0:.0f}s")
