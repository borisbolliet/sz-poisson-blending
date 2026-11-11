"""Convergence check of the ignore-blending fits (cases B, C): trust-region least_squares on Poisson deviance
residuals (my naive model), and chi2 of the claimed fit points and of my Nelder-Mead points under my model."""
import numpy as np
from scipy.optimize import least_squares
from v_lib import counts
D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz"))
FID = np.array([0.81, 0.31, 0.8, 0.17])
j = int(np.argmin(np.abs(G["params"] - FID).sum(1)))
Cfid = np.where(G["Nzq_n"][j] > 0, G["Nzq_b"][j] / np.where(G["Nzq_n"][j] > 0, G["Nzq_n"][j], 1), 1.0)
data = Cfid * counts(*FID); s = data > 1e-6
def res(p, pri, fs):
    M = counts(*p)[s]; d = data[s]
    r = np.sign(d - M) * np.sqrt(np.clip(2 * fs * (M - d + d * np.log(d / M)), 0, None))
    return np.concatenate([r, [(p[a] - FID[a]) / pri[a] for a in pri]])
def chi2(p, pri, fs): return (res(np.asarray(p, float), pri, fs) ** 2).sum()
CASES = {"A": ({}, [0, 1]), "B": ({2: 0.04, 3: 0.05}, [0, 1, 2, 3]), "C": ({}, [0, 1, 2, 3])}
claimed = {"A": FID + [0.0050, -0.0065, 0, 0], "B": FID + [0.0018, -0.0085, 0.0124, 0.0064], "C": FID + [0.0003, -0.0096, 0.0190, 0.0066]}
mineNM = {"A": FID + [0.0051, -0.0066, 0, 0], "B": FID + [0.0030, -0.0071, 0.0053, 0.0113], "C": FID + [0.0017, -0.0079, 0.0104, 0.0140]}
SIG = {"A": [0.0011, 0.0012], "B": [0.0052, 0.0049, 0.0243, 0.0371], "C": [0.0063, 0.0070, 0.0338, 0.0593]}
for c, (pri, free) in CASES.items():
    def f(x):
        p = FID.copy(); p[free] = x; return res(p, pri, 1.0)
    best = None
    for st in (claimed[c], mineNM[c], FID):
        r = least_squares(f, st[free], x_scale=np.array([0.005, 0.005, 0.02, 0.03])[free], diff_step=1e-4, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        p = FID.copy(); p[free] = r.x
        print(f"  {c} start {np.round(st[free]-FID[free],4)} -> d={np.round(p[free]-FID[free],4)} chi2={2*r.cost:.3f}")
        if best is None or r.cost < best[1]: best = (p, r.cost)
    p = best[0]
    print(f"{c}: best d={np.round(p-FID,4)} bias/sig=" + " ".join(f"{(p[a]-FID[a])/SIG[c][i]:+.2f}" for i, a in enumerate(free))
          + f"  chi2 best {2*best[1]:.3f} | claimed point chi2 {chi2(claimed[c], pri, 1.0):.3f} | my NM point chi2 {chi2(mineNM[c], pri, 1.0):.3f}", flush=True)
