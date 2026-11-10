"""Independent population integrals straight from colossus (no szsim):
 ybar per mass class (incl. lognormal mean factor), mu(thA) per class, Parseval variance per class,
 Lambda(x>1), Sigma_det(q_cut) and the B1 Poisson numbers.  Masses Msun (h-free); colossus gets M*h."""
import os, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import v01_kernel as K

cos = cosmology.setCosmology("planck18"); h = cos.h
ARC = np.pi / 180 / 60
d = np.load("v01_kernel.npz")
TH, SIG = d["TH"], d["SIG"]
sigY = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))  # verifier sigma_Y (log-log interp)
# exact sigma for nodes of interest computed directly when needed
CK = 299792.458
SL = 0.173
# z grid: Gauss-Legendre in ln z on [0.01,3]; lnM GL per class
def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n); return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w
lz, wlz = gl(np.log(0.01), np.log(3.0), 400); Z = np.exp(lz); WZ = wlz * Z
EDGES = [1e13, 3e13, 1e14, 1e16]
gx, gw = np.polynomial.hermite_e.hermegauss(30); gw = gw / gw.sum()

def pop():
    rows = []  # z, M, dN/dOmega[per arcmin^2] weight, class
    for c in range(3):
        lm, wm = gl(np.log(EDGES[c]), np.log(EDGES[c + 1]), 60 if c < 2 else 120)
        M = np.exp(lm)
        for zi, wz in zip(Z, WZ):
            dn = mass_function.massFunction(M * h, zi, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
            chi = cos.comovingDistance(0.0, zi) / h
            dV = CK / cos.H0 / cos.Ez(zi) * chi ** 2  # Mpc^3/sr
            w = dn * dV * wm * wz * ARC ** 2          # per arcmin^2
            rows.append(np.column_stack([np.full(len(M), zi), M, w, np.full(len(M), c)]))
    return np.concatenate(rows)

def Ymed_th(M, z):
    Ez = cos.Ez(z); DA = cos.comovingDistance(0.0, z) / h / (1 + z)
    Y = 1e-4 * 10 ** -0.19 * Ez ** (2 / 3) * (0.8 * M / 6e14) ** 1.79 / DA ** 2 / ARC ** 2
    rhoc = cos.rho_c(z) * 1e9 * h ** 2
    R = (3 * M / (4 * np.pi * 500 * rhoc)) ** (1 / 3)
    return Y, R / DA / ARC

t0 = time.time()
Pp = pop()
z, M, w, c = Pp.T; c = c.astype(int)
Ym, thB = np.zeros(len(z)), np.zeros(len(z))
for zi in np.unique(z):
    s = z == zi; Ym[s], thB[s] = Ymed_th(M[s], zi)
print(f"population built {time.time()-t0:.0f}s")
NFS = 4 * np.pi / ARC ** 2
for k in range(3):
    print(f"class {k}: N full sky = {w[c==k].sum()*NFS:,.1f}")
print(f"N(M>1e14) full sky = {w[c==2].sum()*NFS:,.1f} (library 396,387)")
mf = np.exp(SL ** 2 / 2)
ybar = np.array([(w * Ym)[c == k].sum() * mf for k in range(3)])
print(f"ybar per class [arcmin^-2 * arcmin^2] = {ybar}, total {ybar.sum():.4e} (claim 3.59e-7)")
for tA in (0.524, 1.10, 2.30, 4.153, 8.70, 18.23):
    sA = K.sigY(tA)[0]
    mu = ybar * sA * K.RATIO ** 2 / K.NW
    print(f"mu(thA={tA}') per class = {np.round(mu, 4)}  total {mu.sum():.4f}")
# Parseval variance per class: var = sum n Y^2 <e^{2 sigma eps}> sigma_A^2 (1/2pi) int k B^4 gA^2 gB^2 dk / Nw^2
for tA in (1.10, 4.153):
    sA = K.sigY(tA)[0]
    k = np.linspace(0, 6.5 / K.SB, 60000); dk = k[1] - k[0]
    base = k * np.exp(-2 * (k * K.SB) ** 2) * K.g(k * tA) ** 2
    # tabulate J(thB) on a log grid
    tg = np.geomspace(thB.min() * 0.99, thB.max() * 1.01, 400)
    J = np.array([np.sum(base * K.g(k * t) ** 2) * dk for t in tg]) / (2 * np.pi) / K.NW ** 2
    Jp = np.interp(np.log(thB), np.log(tg), J)
    var = np.array([(w * Ym ** 2 * np.exp(2 * SL ** 2) * Jp)[c == kk].sum() for kk in range(3)]) * sA ** 2
    print(f"Parseval variance of delta at thA={tA}': per class {np.round(var, 4)}, total {var.sum():.4f}")
    # Lambda(x>1): area where rho(thA,thB,s) Y/sigma_A > 1
    tb = np.geomspace(max(thB.min(), 0.05), thB.max() * 1.01, 90)
    lnYg = np.linspace(np.log(1e-8), np.log(10.0), 400)
    AREA = np.zeros((len(tb), len(lnYg)))
    for i, t in enumerate(tb):
        S = 5 * tA + 5 * t + K.RB if hasattr(K, "RB") else 5 * tA + 5 * t + 5 * np.sqrt(2) * K.SB
        s = np.linspace(0, S, 4001)
        r = K.rho_exact(tA, t, s, sA)
        rmax = np.maximum.accumulate(r[::-1])[::-1]  # area where r > thr counted as disc to last crossing
        for j, ly in enumerate(lnYg):
            thr = sA / np.exp(ly)
            idx = np.nonzero(r > thr)[0]
            if len(idx):
                ss = s[idx]
                AREA[i, j] = np.sum(2 * np.pi * s * (r > thr)) * (s[1] - s[0])
    from scipy.interpolate import RegularGridInterpolator
    Ai = RegularGridInterpolator((np.log(tb), lnYg), AREA, bounds_error=False, fill_value=None)
    lam = np.zeros(3)
    for e, ww in zip(gx, gw):
        Yv = np.log(Ym) + SL * e
        a = Ai(np.column_stack([np.log(np.clip(thB, tb[0], tb[-1])), np.clip(Yv, lnYg[0], lnYg[-1])]))
        a = np.where(Yv > lnYg[-1], np.nan, np.clip(a, 0, None))
        if np.isnan(a).any():
            print("  warning: Y beyond grid", np.isnan(a).sum())
            a = np.nan_to_num(a)
        lam += np.array([(w * a * ww)[c == kk].sum() for kk in range(3)])
    print(f"Lambda(x>1) at thA={tA}': per class {np.round(lam, 4)}, total {lam.sum():.4f}")

# ---- Sigma_det and B1
for qc in (4, 5, 6, 7):
    q0 = Ym / sigY(thB)
    pdet = sum(ww * ndtr(q0 * np.exp(SL * e) - qc) for e, ww in zip(gx, gw))
    Nd = (w * pdet).sum() * NFS
    Sig = Nd / (4 * np.pi * (180 / np.pi) ** 2)
    line = f"q_cut={qc}: N_det={Nd:,.1f}, Sigma_det={Sig:.4f}/deg^2; f>=1: "
    for tb_ in (1, 2, 3, 5):
        mu = Nd / (4 * np.pi) * 2 * np.pi * (1 - np.cos(tb_ * ARC))
        line += f"{tb_}': {1-np.exp(-mu):.4e} "
        if qc == 5 and tb_ >= 2:
            line += f"(>=2: {1-np.exp(-mu)*(1+mu):.3e}) "
    print(line)
print(f"done {time.time()-t0:.0f}s")
