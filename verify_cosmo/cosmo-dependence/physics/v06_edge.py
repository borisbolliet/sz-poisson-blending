"""Fixed-correction bias at the grid edge sigma8=0.87 (case B, C) with trust-region least_squares from several starts,
and chi2 of the claimed linear-Fisher shifted point under the same model.  Data = C_direct(truth) x Nn_mine(truth)."""
import numpy as np
from scipy.optimize import least_squares
from v_lib import counts
D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz"))
def C(p):
    j = int(np.argmin(np.abs(G["params"] - np.array(p)).sum(1))); n = G["Nzq_n"][j]
    return np.where(n > 0, G["Nzq_b"][j] / np.where(n > 0, n, 1), 1.0)
FID = np.array([0.81, 0.31, 0.8, 0.17]); Cfid = C(FID)
SIG = {"B": np.array([0.0052, 0.0049, 0.0243, 0.0371]), "C": np.array([0.0063, 0.0070, 0.0338, 0.0593])}
for truth in ([0.87, 0.31, 0.8, 0.17], [0.81, 0.35, 0.8, 0.17]):
    T = np.array(truth); data = C(T) * counts(*T); s = data > 1e-6
    for c, pri in (("B", {2: 0.04, 3: 0.05}), ("C", {})):
        def f(x):
            M = (Cfid * counts(*x))[s]; d = data[s]
            r = np.sign(d - M) * np.sqrt(np.clip(2 * (M - d + d * np.log(d / M)), 0, None))
            return np.concatenate([r, [(x[a] - T[a]) / pri[a] for a in pri]])
        claimed = {(0.87, "B"): [-2.004, -2.637, 1.817, 1.154], (0.87, "C"): [-2.411, -2.105, 1.684, 1.231]}.get((truth[0], c))
        starts = [T.copy()] + ([T + np.array(claimed) * SIG[c]] if claimed else [])
        for st in starts:
            r = least_squares(f, st, x_scale=[0.005, 0.005, 0.02, 0.03], diff_step=1e-4, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            print(f"truth {truth} case {c} start {np.round((st-T)/SIG[c],2)}: bias/sig {np.round((r.x-T)/SIG[c],2)} chi2 {2*r.cost:.3f}", flush=True)
        if claimed:
            print(f"   chi2 at claimed linear-Fisher point: {(f(T + np.array(claimed)*SIG[c])**2).sum():.3f}; at truth {(f(T)**2).sum():.3f}", flush=True)
