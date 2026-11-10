"""C12 (unselected part): for halos with q_true>qc at their true position and own scale k (theta_k nearest theta500),
P(at least one other halo with c_b = Y_b R_k(r_b; theta_b) > f c_P), c_P = Y_P R_k(0; theta_P),
measured on our own Poisson patches vs the single-neighbour Poisson formula 1-exp(-lambda),
lambda = n * <pi r*^2> evaluated by Monte Carlo over an independent reference population (other patches).
Both sides are capped at r <= RC = 20' so the comparison is exact mathematics (Slivnyak + Campbell)."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.special import j0
from scipy.spatial import cKDTree
import own_model as om
import v03_mappipe as P

t0 = time.time()
RC = 20.0
TG = np.exp(np.linspace(np.log(0.01), np.log(150.0), 120)); LTG = np.log(TG)
RG = np.linspace(0, RC, 401)
_, sig = P.filters()
RT = np.zeros((12, TG.size, RG.size))
for k in range(12):
    for i, t in enumerate(TG):
        ig = P.LH[None, :] * P.BH2[None, :] * P.GK[k][None, :] * om.gfun(P.LH * t)[None, :] * j0(P.LH[None, :] * RG[:, None])
        RT[k, i] = np.trapezoid(ig, P.LH, axis=1) / (2 * np.pi) * sig[k] ** 2 / P.NW
print(f"response tables done {time.time()-t0:.0f}s; monotone in r: {(np.diff(RT, axis=2) <= 1e-12).mean():.5f}")


def Rint(k, th, r):
    u = np.clip((np.log(th) - LTG[0]) / (LTG[1] - LTG[0]), 0, TG.size - 1.000001); i = u.astype(int); w = u - i
    v = np.clip(r / (RG[1] - RG[0]), 0, RG.size - 1.000001); j = v.astype(int); f = v - j
    a = RT[k, i, j] * (1 - f) + RT[k, i, j + 1] * f
    b = RT[k, i + 1, j] * (1 - f) + RT[k, i + 1, j + 1] * f
    return np.where(r <= RC, a * (1 - w) + b * w, 0.0)


NT, NR = 40, 30
targets = []   # (k, cP, qtrue, hit0.1, hit0.2)
for pid in range(NT):
    h = P.draw(1000 + pid)
    tree = cKDTree(np.stack([h["x"], h["y"]], 1), boxsize=P.L)
    sel = np.flatnonzero(h["q"] > 4)
    kown = np.abs(np.log(h["th"][sel])[:, None] - np.log(P.TK)[None, :]).argmin(1)
    for t_, i in enumerate(sel):
        k = kown[t_]
        cP = h["Y"][i] * Rint(k, h["th"][i], 0.0)
        nb = np.array(tree.query_ball_point([h["x"][i], h["y"][i]], RC), int); nb = nb[nb != i]
        r = P.pdist(h["x"][nb], h["y"][nb], h["x"][i], h["y"][i])
        c = h["Y"][nb] * Rint(k, h["th"][nb], r)
        targets.append([k, cP, h["q"][i], (c > 0.1 * cP).any(), (c > 0.2 * cP).any()])
targets = np.array(targets)
print(f"{len(targets)} targets from {NT} patches  {time.time()-t0:.0f}s", flush=True)

# reference population: independent patches
ref = [P.draw(1000 + pid) for pid in range(200, 200 + NR)]
th = np.concatenate([r["th"] for r in ref]); Yr = np.concatenate([r["Y"] for r in ref]); del ref
Aref = NR * P.L ** 2
TT = np.exp(np.linspace(np.log(0.3), np.log(5e3), 70))       # thresholds in units of sigma_k
lam = np.zeros((12, TT.size))
for k in range(12):
    c0 = Yr * Rint(k, th, np.zeros_like(th)) / sig[k]
    keep = np.flatnonzero(c0 > TT[0])
    for a in range(0, keep.size, 40000):
        ii = keep[a:a + 40000]
        prof = np.stack([Yr[ii] * Rint(k, th[ii], np.full(ii.size, rr)) for rr in RG], 1) / sig[k]   # (n, nr)
        for it, T in enumerate(TT):
            above = prof > T
            nab = above.sum(1)                       # monotone: r* between RG[nab-1] and RG[nab]
            m = nab > 0
            j = nab[m] - 1
            rs = RG[j].copy()
            inner = j < RG.size - 1
            p0 = prof[m][np.arange(m.sum()), j]
            p1 = prof[m][np.arange(m.sum()), np.minimum(j + 1, RG.size - 1)]
            fr = np.where(inner & (p0 > p1), (p0 - T) / np.maximum(p0 - p1, 1e-300), 0.0)
            rs = rs + fr * (RG[1] - RG[0])
            lam[k, it] += (np.pi * rs ** 2).sum() / Aref
print(f"lambda tables done {time.time()-t0:.0f}s", flush=True)

k = targets[:, 0].astype(int); cPs = targets[:, 1] / sig[k]
for f, col in ((0.1, 3), (0.2, 4)):
    lt = np.array([np.exp(np.interp(np.log(f * c), np.log(TT), np.log(np.maximum(lam[kk], 1e-30)))) for kk, c in zip(k, cPs)])
    for qc in (4, 5, 6, 7):
        s = targets[:, 2] > qc
        meas = targets[s, col].mean(); err = np.sqrt(meas * (1 - meas) / s.sum())
        form = (1 - np.exp(-lt[s])).mean()
        print(f"f={f} q>{qc}: N={s.sum()} measured {100*meas:.2f}+-{100*err:.2f}%  Poisson formula {100*form:.2f}%  ({(meas-form)/err:+.2f} sig)")
print(f"elapsed {time.time()-t0:.0f}s")
