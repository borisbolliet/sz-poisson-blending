"""w04b: goodness of agreement between Delta_map and the build's Delta (same primaries q_A >= 0.5), q_cut 5, full sky.
(w04's per-bin chi2 overflowed: bins with ~0 counts have ~0 bootstrap variance.)
  (1) per-bin chi2 with the diagonal bootstrap variance, bins with N_single > 50 (full sky) and bootstrap sigma > 0.5
      counts only (bins with a handful of objects, e.g. q > 100: ~8 objects, have a near-zero bootstrap variance that
      is not a meaningful error);
  (2) full-covariance chi2 on aggregated cells: 6 z groups x 3 q_obs groups (5-10.6, 10.6-32.5, > 32.5);
  (3) same for the redshift profile alone (6 z groups, all q) and the q profile alone (9 q bins, all z).
"""
import os
import numpy as np
from scipy.stats import chi2 as C2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
d = np.load(os.path.join(HERE, "out", "w04_delta.npz"))
f = open(os.path.join(HERE, "w04b_chi2.log"), "w")


def log(s=""):
    print(s); f.write(s + "\n")


ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
ZG = [np.nonzero((ZED[:-1] >= a - 1e-9) & (ZED[1:] <= b + 1e-9))[0]
      for a, b in ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))]
QG = [np.arange(0, 2), np.arange(2, 5), np.arange(5, 9)]
BUILD = {"baseline_f1": "baseline_f1", "baseline_f0.5": "baseline_f0.5", "baseline_f2": "baseline_f2", "goal_f1": "goal_f1",
         "goal_f0.5": "goal_f0.5", "goal_f2": "goal_f2", "white_f1": "white", "baseline_effi": "baseline_effi",
         "baseline_effii": "baseline_effii", "goal_effi": "goal_effi", "goal_effii": "goal_effii", "lowP_f1": "lowP_f1",
         "lowP_effi": "lowP_effi"}


def agg(D, groups_z, groups_q):
    return np.array([[D[..., gz, :][..., gq].sum((-1, -2)) for gq in groups_q] for gz in groups_z])


log("[w04b] Delta_map vs build Delta (q_A >= 0.5), q_cut 5, full sky; bootstrap covariance (2000 resamplings of 500 patches)")
log("   variant | per-bin chi2 / bins p | 6z x 3q full-cov chi2 / cells p | z profile chi2 / 6 p | q profile chi2 / bins p")
for nm, b in BUILD.items():
    D = d[f"{nm}_q5_D"]; Db = d[f"{nm}_q5_Dboot"].astype(float); Ns = d[f"{nm}_q5_Ns"]
    B = np.load(f"{ROOT}/data_real/step2/truth_{b}.npz")["D05_q5"]
    var = Db.var(0)
    sel = (Ns > 50) & (var > 0.25)
    c1 = float(np.sum((D - B)[sel] ** 2 / var[sel])); n1 = int(sel.sum())
    A = agg(D, ZG, QG).ravel(); AB = agg(B, ZG, QG).ravel()
    Ab = np.stack([agg(x, ZG, QG).ravel() for x in Db])
    C = np.cov(Ab.T)
    ok = (agg(Ns, ZG, QG).ravel() > 50) & (np.diag(C) > 0.25)
    dv = (A - AB)[ok]
    c2 = float(dv @ np.linalg.solve(C[np.ix_(ok, ok)], dv)); n2 = int(ok.sum())
    zp = np.array([D[g].sum() for g in ZG]); zb = np.array([B[g].sum() for g in ZG]); zbb = np.stack([[x[g].sum() for g in ZG] for x in Db])
    c3 = float((zp - zb) @ np.linalg.solve(np.cov(zbb.T), zp - zb))
    qp = D.sum(0); qb = B.sum(0); qbb = Db.sum(1)
    okq = (qbb.var(0) > 0.25) & (Ns.sum(0) > 50)
    Cq = np.cov(qbb[:, okq].T)
    c4 = float((qp - qb)[okq] @ np.linalg.solve(Cq, (qp - qb)[okq]))
    log(f"   {nm:15s} | {c1:6.1f} / {n1} p={C2.sf(c1, n1):.2f} | {c2:5.1f} / {n2} p={C2.sf(c2, n2):.2f} | {c3:5.1f} / 6 p={C2.sf(c3, 6):.2f} | "
        f"{c4:5.1f} / {okq.sum()} p={C2.sf(c4, okq.sum()):.2f}")
f.close()
