"""Repair phase: calibrated joint test of the Palm law on 1000 floor-1e14 skies (palm_new: seeds 40001-41000;
Q1_PALM_DIR=palm_rep: independent replication, seeds 41001-42000).
Per sky residuals r_s = count_s - N_s * P_cond(N_s) for detected-centred >=1 and >=2 neighbour counts at the 9
radii of q1_palm_check.THP (conditional binomial law, exact given N), and the same for random centres.
Hotelling T^2 (F-calibrated) on the per-sky residual vectors, per q_cut and jointly."""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_palm_check import THP, NRAND
NB = json.load(open(os.path.join(DATA, "nbar.json")))
P = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(DATA, os.environ.get("Q1_PALM_DIR", "palm_new"), "sky_*.npy")))]
R = len(P)
f = np.sin(THP * ARCMIN / 2) ** 2

def hot(D):
    Rr, d = D.shape; m = D.mean(0); S = np.cov(D, rowvar=False)
    T2 = Rr * m @ np.linalg.solve(S, m); F = (Rr - d) / (d * (Rr - 1)) * T2
    return T2, d, stats.f.sf(F, d, Rr - d)

blocks = {}
for q in QCUTS:
    N = np.array([s[f"N{q}"] for s in P], float)[:, None]
    g1 = np.array([s[f"ge1_det{q}"] for s in P], float); g2 = np.array([s[f"ge2_det{q}"] for s in P], float)
    gr = np.array([s[f"ge1_rand{q}"] for s in P], float)
    p1 = 1 - (1 - f) ** (N - 1); p2 = p1 - (N - 1) * f * (1 - f) ** (N - 2)
    r1 = (g1 - N * p1) / np.sqrt(N * p1 * (1 - p1)); r2 = (g2 - N * p2) / np.sqrt(N * p2 * (1 - p2))
    rr = (gr - NRAND * (1 - (1 - f) ** N)) / np.sqrt(NRAND * (1 - (1 - f) ** N) * (1 - f) ** N)
    blocks[q] = (r1, r2, rr)
    for lab, D in [(">=1 det-centred", r1), (">=2 det-centred", r2), (">=1 random-centred", rr),
                   ("det-centred >=1 and >=2", np.hstack([r1, r2]))]:
        T2, d, p = hot(D)
        print(f"det{q} {lab:26s}: Hotelling T2={T2:7.2f}, d={d:2d}, R={R}, p={p:.3f}")
D = np.hstack([np.hstack([blocks[q][0], blocks[q][1]]) for q in QCUTS])
T2, d, p = hot(D); print(f"ALL q, det-centred >=1 and >=2: T2={T2:.2f}, d={d}, p={p:.3f}")
D = np.hstack([blocks[q][2] for q in QCUTS])
T2, d, p = hot(D); print(f"ALL q, random-centred >=1: T2={T2:.2f}, d={d}, p={p:.3f}")
# simple sign check: det6 >=1 residual summed over the 9 radii (correlated) -> z from sky scatter
for q in QCUTS:
    s = blocks[q][0].sum(1); print(f"det{q}: sum over radii of standardized >=1 residual: mean {s.mean():+.4f} +- {s.std(ddof=1)/np.sqrt(R):.4f}"
                                   f" (z={s.mean()/(s.std(ddof=1)/np.sqrt(R)):+.2f})")
