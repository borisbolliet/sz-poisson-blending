"""Verifier: independent re-implementation of the likelihood part (Fisher, fixed-correction residuals, linear and
nonlinear biases, emulator) from the stored c02 arrays, plus an independent naive-count model (v01 population:
astropy background) to (a) test cosmology propagation in cosmolib.naive counts and (b) redo nonlinear fits with
scipy.optimize (Poisson -lnL), checking convergence.
"""
import os, sys, itertools, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v01_mc_blend import population, ZE, QE
from szsim import signal as sg

D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz")); E = dict(np.load(f"{D}/c02_extra.npz"))
FID = np.array([0.81, 0.31, 0.8, 0.17]); STEP = np.array([0.01, 0.01, 0.02, 0.03]); SP = (0.006, 0.0073)
NODES = [[0.75, 0.81, 0.87], [0.27, 0.31, 0.35], [0.7, 0.8, 0.9], [0.1, 0.17, 0.3]]
m0 = sg.SZModel()
XG, WG = np.polynomial.hermite_e.hermegauss(40); WG = WG / WG.sum()


def find(S, p):
    d = np.abs(S["params"] - np.asarray(p)[None]).max(1)
    j = int(d.argmin()); assert d[j] < 1e-6, (p, d[j]); return j


def naive_mine(th):
    P = population(th[0], th[1], th[2])
    q = (P["Y"] / m0.sigma_Y(P["th"]))[:, None] * np.exp(th[3] * XG)[None]
    zb = np.clip(np.searchsorted(ZE, P["z"], side="right") - 1, 0, 7)
    out = np.zeros((8, 10))
    Ph = [ndtr(q - e) @ WG if np.isfinite(e) else np.zeros(len(q)) for e in QE]
    for e in range(10):
        out[:, e] = np.bincount(zb, (Ph[e] - Ph[e + 1]) * P["n"], minlength=8)
    return out


jf = find(G, FID)
Nb0 = G["Nzq_b"][jf]; Nn0 = G["Nzq_n"][jf]; Cfid = Nb0 / Nn0
print("[0] totals at fiducial: N_b %.1f N_n %.1f ratio %.5f" % (Nb0.sum(), Nn0.sum(), Nb0.sum() / Nn0.sum()))

# ---- [1] cosmology propagation in naive counts: mine vs stored, at many points
print("\n[1] naive counts: independent (astropy background) / stored cosmolib, per point: total ratio, max |bin ratio - fid bin ratio| (bins N>1)")
mine_f = naive_mine(FID); Rf = Nn0 / mine_f
for S, nm in ((E, "extra"), (G, "grid")):
    for j, p in enumerate(S["params"]):
        if nm == "grid" and (np.abs(p - FID) > 1e-9).sum() > 1:
            continue
        mi = naive_mine(p); st = S["Nzq_n"][j]; s = st > 1
        r = st / mi
        print(f"  {nm:5s} {np.round(p,4)}: tot {st.sum()/mi.sum():.5f}  max|r-r_fid| {np.abs(r-Rf)[s].max():.2e}")

# ---- [2] Fisher
def dN(S, key):
    out = []
    for a in range(4):
        pm = FID.copy(); pm[a] -= STEP[a]; pp = FID.copy(); pp[a] += STEP[a]
        out.append((S[key][find(S, pp)] - S[key][find(S, pm)]) / (2 * STEP[a]))
    return np.array(out)


dNb = dN(E, "Nzq_b"); dNn = dN(E, "Nzq_n")
CASES = {"A": ([0, 1], {}), "B": ([0, 1, 2, 3], {2: 0.04, 3: 0.05}), "C": ([0, 1, 2, 3], {})}
mask = Nb0 > 1e-3


def fisher(der, N, free, pri, fs):
    F = fs * (der[free][:, mask] / N[mask]) @ der[free][:, mask].T
    for i, a in enumerate(free):
        if a in pri: F[i, i] += pri[a] ** -2
    return F


print("\n[2] Fisher 1-sigma (independent code); also without the N>1e-3 mask and with naive-model derivatives")
SIG = {}
for c, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        Ci = np.linalg.inv(fisher(dNb, Nb0, free, pri, fs)); sg_ = np.sqrt(np.diag(Ci)); SIG[(c, fs)] = sg_
        Cn = np.linalg.inv(fisher(Cfid * dNn, Nb0, free, pri, fs))
        print(f"  {c} fs={fs}: " + " ".join(f"{x:.4f}" for x in sg_) + f"  r={Ci[0,1]/sg_[0]/sg_[1]:+.2f}"
              f" | model-deriv C_fid*dNn: " + " ".join(f"{x:.4f}" for x in np.sqrt(np.diag(Cn))))

# ---- [3] fixed-correction residuals, linear bias (two derivative choices), f_sky=0.4 scaling
print("\n[3] fixed correction: dN/N, dN/sqrtN, chi2(fs1), chi2(fs.4); linear bias/sigma A and B with dNb (as claimed) "
      "and with the model derivative C_fid dNn; and case B at fs=0.4 vs 0.63 x fs1")
pts = []
for a in range(2):
    for k in (-3, -1, 1, 3):
        p = FID.copy(); p[a] += k * SP[a]; pts.append((f"p{a} {k:+d}s", E, p))
for a in range(4):
    for v in (NODES[a][0], NODES[a][2]):
        p = FID.copy(); p[a] = v; pts.append((f"grid p{a}={v}", G, p))


def linbias(res, der, free, pri, fs):
    F = fisher(der, Nb0, free, pri, fs)
    return np.linalg.solve(F, fs * (der[free][:, mask] / Nb0[mask]) @ res[mask])


for name, S, p in pts:
    j = find(S, p); Nt = S["Nzq_b"][j]; Nm = Cfid * S["Nzq_n"][j]
    dn = (Nt.sum() - Nm.sum()); chi = ((Nt - Nm) ** 2 / Nm)[Nm > 0].sum()
    res = Nt - Nm
    out = []
    for c in ("A", "B"):
        free, pri = CASES[c]
        b1 = linbias(res, dNb, free, pri, 1.0) / SIG[(c, 1.0)]
        b2 = linbias(res, Cfid * dNn, free, pri, 1.0) / np.sqrt(np.diag(np.linalg.inv(fisher(Cfid * dNn, Nb0, free, pri, 1.0))))
        out.append(f"{c}: " + " ".join(f"{x:+.3f}" for x in b1) + " [" + " ".join(f"{x:+.3f}" for x in b2) + "]")
    b4 = linbias(res, dNb, *CASES["B"], 0.4) / SIG[("B", 0.4)]
    b1 = linbias(res, dNb, *CASES["B"], 1.0) / SIG[("B", 1.0)]
    print(f"  {name:14s} {dn/Nm.sum():+.2e} {dn/np.sqrt(Nm.sum()):+6.2f} {chi:7.2f} {0.4*chi:6.2f} | " + " | ".join(out)
          + f" | B fs.4: " + " ".join(f"{x:+.3f}" for x in b4) + " vs 0.63x: " + " ".join(f"{x:+.3f}" for x in 0.632 * b1))
chis = []
for j in range(len(G["params"])):
    Nt = G["Nzq_b"][j]; Nm = Cfid * G["Nzq_n"][j]; chis.append((((Nt - Nm) ** 2 / Nm)[Nm > 0].sum(), (Nt.sum() - Nm.sum()) / Nm.sum()))
chis = np.array(chis); k = chis[:, 0].argmax()
print(f"  max over 81 grid points: chi2 {chis[:,0].max():.1f} at {G['params'][k]}; max |dN/N| {np.abs(chis[:,1]).max():.4f}")
print(f"  C=1 at fiducial: dN/N {(Nb0.sum()-Nn0.sum())/Nn0.sum():+.4f}, dN/sqrtN {(Nb0.sum()-Nn0.sum())/np.sqrt(Nn0.sum()):+.2f}, "
      f"chi2 {((Nb0-Nn0)**2/Nn0)[Nn0>0].sum():.1f}")

# ---- [4] emulator (independent implementation)
lnC = np.log(np.where(G["Nzq_n"] > 0, G["Nzq_b"] / np.where(G["Nzq_n"] > 0, G["Nzq_n"], 1), 1.0))
T = np.zeros((3, 3, 3, 3, 8, 10))
for j, p in enumerate(G["params"]):
    T[tuple(int(np.argmin(np.abs(np.array(NODES[a]) - p[a]))) for a in range(4))] = lnC[j]


def emu(p):
    L = []
    for a in range(4):
        x0, x1, x2 = NODES[a]; x = p[a]
        L.append(np.array([(x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2)), (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2)),
                           (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))]))
    return np.exp(np.einsum("i,j,k,l,ijklzq->zq", *L, T))


print("\n[4] emulator: max |dN/N|, max chi2 fs1 over 28 extra points; max |bias/sigma| A,B at the 10 random points")
e_dn, e_chi, bA, bB = [], [], [], []
for j, p in enumerate(E["params"]):
    Nt = E["Nzq_b"][j]; Nm = emu(p) * E["Nzq_n"][j]
    e_dn.append((Nt.sum() - Nm.sum()) / Nm.sum()); e_chi.append((((Nt - Nm) ** 2 / Nm)[Nm > 0]).sum())
    if j >= 18:
        bA.append(np.abs(linbias(Nt - Nm, dNb, *CASES["A"], 1.0) / SIG[("A", 1.0)]).max())
        bB.append(np.abs(linbias(Nt - Nm, dNb, *CASES["B"], 1.0) / SIG[("B", 1.0)]).max())
print(f"  dN/N range {min(e_dn):+.2e} .. {max(e_dn):+.2e}; chi2 max {max(e_chi):.4f}; bias A max {max(bA):.3f}, B max {max(bB):.3f}")
# is the emulator test mostly on 1-D lines? count off-axis points
offax = [(np.abs(p - FID) > 1e-9).sum() for p in E["params"]]
print(f"  number of varied params per extra point: {np.bincount(offax)}")

# ---- [5] nonlinear fits with independent naive model, rescaled per bin to the stored naive at the fiducial
Rbin = np.where(mine_f > 0, Nn0 / np.where(mine_f > 0, mine_f, 1), 1.0)
cache = {}


def model_naive(th):
    key = tuple(np.round(th, 10))
    if key not in cache:
        cache[key] = Rbin * naive_mine(th)
    return cache[key]


def fitP(model, data, th0, free, pri, fs):
    def nll(v):
        th = th0.copy(); th[free] = v; M = model(th); s = M > 1e-3
        L = fs * (M[s] - data[s] * np.log(M[s])).sum()
        for i, a in enumerate(free):
            if a in pri: L += 0.5 * ((th[a] - th0[a]) / pri[a]) ** 2
        return L
    r = minimize(nll, th0[free], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000, initial_simplex=None))
    th = th0.copy(); th[free] = r.x
    # numerical gradient check
    g = []
    for i in range(len(free)):
        e = np.zeros(len(free)); e[i] = 1e-4
        g.append((nll(r.x + e) - nll(r.x - e)) / 2e-4)
    return th, r, np.array(g)


if __name__ == "__main__":
    print("\n[5] nonlinear Poisson fits (Nelder-Mead on -lnL), independent naive model")
    t0 = time.time()
    for c, (free, pri) in CASES.items():
        for fs in (1.0, 0.4):
            th, r, g = fitP(model_naive, Nb0, FID.copy(), free, pri, fs)
            print(f"  ignore blending {c} fs={fs}: " + " ".join(f"d{a}={th[a]-FID[a]:+.4f} ({(th[a]-FID[a])/SIG[(c,fs)][i]:+.2f}s)"
                  for i, a in enumerate(free)) + f"  nit {r.nit} grad {np.abs(g).max():.1e}", flush=True)
    for name, S, p in pts[:8] + [x for x in pts if x[0] == "grid p0=0.87"]:
        if name not in ("p0 +3s", "p1 +3s", "grid p0=0.87"):
            continue
        j = find(S, p)
        for c in ("A", "B"):
            free, pri = CASES[c]
            th, r, g = fitP(lambda t: Cfid * model_naive(t), S["Nzq_b"][j], p.copy(), free, pri, 1.0)
            print(f"  fixed corr {name} {c}: bias/sig " + " ".join(f"{(th[a]-p[a])/SIG[(c,1.0)][i]:+.3f}" for i, a in enumerate(free))
                  + f"  grad {np.abs(g).max():.1e}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)")
