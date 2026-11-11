"""Independent likelihood checks.  Naive counts: my v_lib (exact z-bin integration).  Blending correction
C(z,q) = Nzq_b/Nzq_n taken from the stored direct runs (c02_grid/c02_extra) or my own quadratic interpolation
of ln C on the 3^4 grid.  Poisson Asimov likelihood minimised with scipy (Powell + polish), NOT Gauss-Newton."""
import sys, time, numpy as np
from scipy.optimize import minimize
from v_lib import counts
D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz")); E = dict(np.load(f"{D}/c02_extra.npz"))
FID = np.array([0.81, 0.31, 0.8, 0.17]); NODES = [[.75, .81, .87], [.27, .31, .35], [.7, .8, .9], [.1, .17, .3]]
PN = ["s8", "Om", "1-b", "slnY"]; SP = (0.006, 0.0073)
def gi(S, p): 
    d = np.abs(S["params"] - np.array(p)).sum(1); j = int(np.argmin(d)); assert d[j] < 1e-6, p; return j
def Cdir(S, j):
    n = S["Nzq_n"][j]; return np.where(n > 0, S["Nzq_b"][j] / np.where(n > 0, n, 1), 1.0)
lnCg = np.zeros((3, 3, 3, 3, 8, 10))
for j, p in enumerate(G["params"]):
    lnCg[tuple(NODES[a].index(round(float(p[a]), 6)) for a in range(4))] = np.log(Cdir(G, j))
def lag(x, n):
    x0, x1, x2 = n
    return np.array([(x-x1)*(x-x2)/((x0-x1)*(x0-x2)), (x-x0)*(x-x2)/((x1-x0)*(x1-x2)), (x-x0)*(x-x1)/((x2-x0)*(x2-x1))])
def Cemu(p):
    L = [lag(p[a], NODES[a]) for a in range(4)]
    return np.exp(np.einsum("a,b,c,d,abcdzq->zq", *L, lnCg))
Cfid = Cdir(G, gi(G, FID))
_cache = {}
def Nn(p):
    k = tuple(np.round(p, 10))
    if k not in _cache: _cache[k] = counts(*p)
    return _cache[k]

CASES = {"A": ([0, 1], {}), "B": ([0, 1, 2, 3], {2: 0.04, 3: 0.05}), "C": ([0, 1, 2, 3], {})}
def fisher(model, p0, free, pri, fs, h=(0.005, 0.005, 0.01, 0.015)):
    M = model(p0); dM = []
    for a in free:
        pp, pm = p0.copy(), p0.copy(); pp[a] += h[a]; pm[a] -= h[a]; dM.append((model(pp) - model(pm)) / (2 * h[a]))
    dM = np.array(dM); s = M > 1e-6
    F = fs * np.einsum("ak,bk->ab", dM[:, s], dM[:, s] / M[s])
    for i, a in enumerate(free):
        if a in pri: F[i, i] += 1 / pri[a] ** 2
    return F, dM
def m2lnL(model, data, p, free, pri, fs, p0):
    M = model(p); s = data > 0
    v = 2 * fs * (M.sum() - data.sum() + (data[s] * np.log(data[s] / M[s])).sum())
    return v + sum(((p[a] - p0[a]) / pri[a]) ** 2 for a in free if a in pri)
def fit(model, data, start, free, pri, fs, p0):
    def f(x):
        p = np.array(start, float); p[free] = x; return m2lnL(model, data, p, free, pri, fs, p0)
    r = minimize(f, np.array(start)[free], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-6, maxiter=4000, maxfev=6000))
    r = minimize(f, r.x, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000))
    p = np.array(start, float); p[free] = r.x; return p, r.fun

blend = lambda p: Cemu(p) * Nn(p)
SIG = {}
print("[F] Fisher errors, blended model = Cemu x Nn_mine (and with dC=0 for comparison)")
for c, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        F, _ = fisher(blend, FID.copy(), free, pri, fs); Ci = np.linalg.inv(F); s = np.sqrt(np.diag(Ci))
        F0, _ = fisher(lambda p: Cfid * Nn(p), FID.copy(), free, pri, fs); s0 = np.sqrt(np.diag(np.linalg.inv(F0)))
        SIG[(c, fs)] = dict(zip(free, s))
        print(f"  {c} fs={fs}: " + " ".join(f"s({PN[a]})={x:.4f}" for a, x in zip(free, s)) + f"  r12={Ci[0,1]/s[0]/s[1]:+.2f}"
              + "  | dC=0: " + " ".join(f"{x:.4f}" for x in s0))

print("\n[I] ignore blending: data = Cfid x Nn(fid); model = Nn(theta).  Nonlinear Nelder-Mead fit.")
data0 = Cfid * Nn(FID)
for c, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        p, chi = fit(Nn, data0, FID, free, pri, fs, FID)
        sg = SIG[(c, fs)]
        print(f"  {c} fs={fs}: " + " ".join(f"d{PN[a]}={p[a]-FID[a]:+.4f}({(p[a]-FID[a])/sg[a]:+.2f}s)" for a in free) + f"  chi2min={chi:.1f}")
    # linear Fisher bias with naive-model derivatives
    F, dM = fisher(Nn, FID.copy(), free, pri, 1.0); M = Nn(FID); s = M > 1e-6
    db = np.linalg.solve(F, np.einsum("ak,k->a", dM[:, s], (data0 - M)[s] / M[s]))
    print(f"     linear Fisher fs=1: " + " ".join(f"{db[i]:+.4f}" for i in range(len(free))))

print("\n[P] profile likelihood of Om in case C (blended model fitting its own Asimov data): Delta chi2 at Fisher +-1,+-2 sigma")
dB = blend(FID); free, pri = CASES["C"]
for k in (-2, -1, 1, 2):
    om = FID[1] + k * SIG[("C", 1.0)][1]
    st = FID.copy(); st[1] = om
    def model_fixOm(p): q = p.copy(); q[1] = om; return blend(q)
    p, chi = fit(model_fixOm, dB, st, [0, 2, 3], {}, 1.0, FID)
    print(f"  Om = fid {k:+d} sigma_F: Delta chi2 = {chi:.3f} (Gaussian {k*k})  profiled s8,1-b,slnY = {p[0]:.4f},{p[2]:.4f},{p[3]:.4f}")
for k in (-1, 1):
    s8 = FID[0] + k * SIG[("C", 1.0)][0]; st = FID.copy(); st[0] = s8
    def model_fixs8(p): q = p.copy(); q[0] = s8; return blend(q)
    p, chi = fit(model_fixs8, dB, st, [1, 2, 3], {}, 1.0, FID)
    print(f"  s8 = fid {k:+d} sigma_F: Delta chi2 = {chi:.3f} (Gaussian 1)")
sys.stdout.flush()

print("\n[X] fixed fiducial correction: data = C_direct(truth) x Nn(truth); model = Cfid x Nn(theta). Nonlinear fits, full sky")
pts = []
for a, k in ((0, 3), (1, 3), (1, 1), (0, -3), (1, -3)):
    p = FID.copy(); p[a] += k * SP[a]; pts.append((f"{PN[a]} {k:+d}sP18", E, gi(E, p), p))
for a, v in ((0, 0.87), (1, 0.35)):
    p = FID.copy(); p[a] = v; pts.append((f"grid {PN[a]}={v}", G, gi(G, p), p))
fixed = lambda p: Cfid * Nn(p)
for name, S, j, pt in pts:
    data = Cdir(S, j) * Nn(pt)
    out = []
    for c, (free, pri) in CASES.items():
        p, chi = fit(fixed, data, pt, free, pri, 1.0, pt)
        sg = SIG[(c, 1.0)]
        out.append(c + ": " + " ".join(f"{(p[a]-pt[a])/sg[a]:+.2f}" for a in free))
    print(f"  {name:16s} " + " | ".join(out) + f"   dNtot/N={(data.sum()-fixed(pt).sum())/fixed(pt).sum():+.2e}", flush=True)

print("\n[E] my quadratic emulator vs direct runs at the 10 random off-grid points (dN/N, chi2 fs1) and nonlinear-fit bias (A, B)")
for j in range(18, 28):
    pt = E["params"][j].copy(); data = Cdir(E, j) * Nn(pt); mod = lambda p: Cemu(p) * Nn(p)
    m = mod(pt); chi2 = ((data - m) ** 2 / m)[m > 1e-6].sum()
    out = []
    for c in ("A", "B"):
        free, pri = CASES[c]
        F, dM = fisher(mod, pt, free, pri, 1.0); s = m > 1e-6
        db = np.linalg.solve(F, np.einsum("ak,k->a", dM[:, s], (data - m)[s] / m[s]))
        out.append(c + ": " + " ".join(f"{db[i]/SIG[(c,1.0)][a]:+.3f}" for i, a in enumerate(free)))
    print(f"  {np.round(pt,4)} dN/N={(data.sum()-m.sum())/m.sum():+.2e} chi2={chi2:.4f}  " + " | ".join(out))
