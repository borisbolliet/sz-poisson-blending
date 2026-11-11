"""Verifier: nonlinear Asimov fits with the independent naive model (rescaled per bin to the stored naive at the
fiducial).  Fisher scoring with own FD steps (half of the claimed STEP), iterated to convergence in sigma units,
then cross-checked by L-BFGS-B on parameters scaled by sigma.  Reports final gradient of -lnL in sigma units."""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v02_likelihood import (naive_mine, find, G, E, FID, STEP, SP, CASES, SIG, Nb0, Nn0, Cfid, mine_f)  # noqa

Rbin = np.where(mine_f > 0, Nn0 / np.where(mine_f > 0, mine_f, 1), 1.0)
cache = {}


def mnaive(th):
    k = tuple(np.round(th, 10))
    if k not in cache:
        cache[k] = Rbin * naive_mine(th)
    return cache[k]


def nll(model, data, th, th0, free, pri, fs):
    M = model(th); s = M > 1e-3
    L = fs * (M[s] - data[s] * np.log(M[s])).sum()
    for a in free:
        if a in pri: L += 0.5 * ((th[a] - th0[a]) / pri[a]) ** 2
    return L


def gn(model, data, th0, free, pri, fs, h=0.5):
    th = th0.copy()
    for it in range(30):
        M = model(th); s = M > 1e-3
        dM = []
        for a in free:
            tp, tm = th.copy(), th.copy(); tp[a] += h * STEP[a]; tm[a] -= h * STEP[a]
            dM.append((model(tp) - model(tm)) / (2 * h * STEP[a]))
        dM = np.array(dM)
        F = fs * (dM[:, s] / M[s]) @ dM[:, s].T
        g = fs * dM[:, s] @ ((data[s] - M[s]) / M[s])
        for i, a in enumerate(free):
            if a in pri: F[i, i] += pri[a] ** -2; g[i] -= (th[a] - th0[a]) / pri[a] ** 2
        st = np.linalg.solve(F, g)
        th[free] += st
        if np.max(np.abs(st) * np.sqrt(np.diag(F))) < 1e-4:
            break
    return th, it + 1, g * np.sqrt(np.diag(np.linalg.inv(F)))


def lbfgs(model, data, th0, start, free, pri, fs, sig):
    f = lambda v: nll(model, data, _set(th0, start, free, sig, v), th0, free, pri, fs)
    r = minimize(f, np.zeros(len(free)), method="L-BFGS-B", options=dict(ftol=1e-13, gtol=1e-7, eps=1e-3))
    return _set(th0, start, free, sig, r.x), r


def _set(th0, start, free, sig, v):
    th = start.copy(); th[free] = start[free] + v * sig; return th


t0 = time.time()
print("[1] ignore blending: data = stored blended counts at fiducial, model = naive (independent)")
for c, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        sig = SIG[(c, fs)]
        th, nit, gs = gn(mnaive, Nb0, FID.copy(), free, pri, fs)
        th2, r = lbfgs(mnaive, Nb0, FID.copy(), FID.copy(), free, pri, fs, sig)
        print(f"  {c} fs={fs}: GN  " + " ".join(f"{(th[a]-FID[a])/sig[i]:+.3f}" for i, a in enumerate(free))
              + f"  (d1-b={th[2]-FID[2]:+.4f}) it={nit} |grad*sig|={np.abs(gs).max():.1e}"
              + "  | LBFGS " + " ".join(f"{(th2[a]-FID[a])/sig[i]:+.3f}" for i, a in enumerate(free))
              + f" dlnL(GN-LBFGS)={nll(mnaive,Nb0,th,FID,free,pri,fs)-nll(mnaive,Nb0,th2,FID,free,pri,fs):+.2e}", flush=True)
print("\n[2] fixed correction nonlinear fits (model = C_fid x naive)")
for nm, S, a0, v in (("s8 +3s_P18", E, 0, FID[0] + 3 * SP[0]), ("Om +3s_P18", E, 1, FID[1] + 3 * SP[1]),
                      ("grid s8=0.87", G, 0, 0.87)):
    p = FID.copy(); p[a0] = v; j = find(S, p)
    for c in ("A", "B"):
        free, pri = CASES[c]; sig = SIG[(c, 1.0)]
        th, nit, gs = gn(lambda t: Cfid * mnaive(t), S["Nzq_b"][j], p.copy(), free, pri, 1.0)
        print(f"  {nm:13s} {c}: " + " ".join(f"{(th[a]-p[a])/sig[i]:+.3f}" for i, a in enumerate(free))
              + f"  it={nit} |grad*sig|={np.abs(gs).max():.1e}", flush=True)
print(f"({time.time()-t0:.0f}s)")
