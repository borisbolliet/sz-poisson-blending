"""Repair of c03: catalogue (compound-Poisson P(D)) blending change Delta_cat computed with the converged model.
Uses the stored blended kernel data_cosmo/kernel_catalogue.npz (K_b(theta_A, q_A; edge), unchanged).  For each theta
node, K_b and the naive kernel Phi(q_A - e) (tabulated on the SAME q_A grid, so interpolation errors cancel) are
interpolated onto the fine ln q grid of cosmolib2, differenced into q_obs bins, and convolved with the lognormal
Y scatter (Gaussian in ln q, slnY = 0.173).  Counts integrate these tables over the refined (z, lnM) nodes.
   Delta_cat = N_b - N_n (full sky);   data_cat = mu_naive_smooth(fid) + Delta_cat.
Output: data_cosmo/truth_catalogue_v2.npz
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "3")
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL, cosmolib2 as C2

t0 = time.time()
k = np.load(f"{CL.DC}/kernel_catalogue.npz")
Kb, th, QA, E = k["Kb"], k["th"], k["QA"], k["EDGES"]
lth = np.log(th)
assert np.allclose(np.diff(lth), lth[1] - lth[0])
Kn = ndtr(QA[:, None] - E[None, :])
old = np.load(f"{CL.DC}/truth_catalogue.npz")
res = {}
for qc in (5.0, 6.0):
    s = f"{qc:.0f}"
    M = C2.SmoothModel(qc)
    sel = np.array([int(np.argmin(np.abs(E - e))) for e in M.qed[:-1]])
    assert np.allclose(E[sel], M.qed[:-1])
    q = np.clip(np.exp(M.lq), QA[0], QA[-1])

    def table(Ktab_theta):          # (nQA, ne) -> P(bin | ln q_true) on the fine grid, scatter-convolved, (nlq, nqb)
        C = np.stack([np.interp(q, QA, Ktab_theta[:, j]) for j in sel], 0)       # P(q_obs > e), (ne, nlq)
        C = np.concatenate([C, np.zeros((1, C.shape[1]))], 0)
        return C2.smooth_lnq(C[:-1] - C[1:], CL.FID[4], M.h).T

    Pb = np.stack([table(Kb[i]) for i in range(len(th))])
    Pn = np.stack([table(Kn) for i in range(len(th))])
    Nb = M.counts_table(CL.FID, Pb, lth); Nn = M.counts_table(CL.FID, Pn, lth); N0 = M.counts(CL.FID)
    D = Nb - Nn
    print(f"=== q_cut={qc}: full sky ===")
    print(f" naive smooth {N0.sum():,.1f} | naive via kernel grid {Nn.sum():,.1f} (rel {Nn.sum()/N0.sum()-1:+.2e}; per-bin max "
          f"|rel| (mu>10) {np.abs(Nn/N0-1)[N0>10].max():.2e}) | blended {Nb.sum():,.1f}")
    print(f" blended/naive total = {Nb.sum()/Nn.sum():.4f} (old c03: {(old['Nb_q'+s].sum()/old['Nn_q'+s].sum()):.4f}; "
          f"b03_analytic.log {'84,494/85,898 = 0.9837' if qc == 5 else '61,118/61,630 = 0.9917'})")
    print(" Delta/naive per z bin:", np.round(D.sum(1) / Nn.sum(1), 4))
    print(" old c03            :", np.round(old['D_q' + s].sum(1) / old['Nn_q' + s].sum(1), 4))
    print(" Delta/naive per q bin:", np.round(D.sum(0) / Nn.sum(0), 4))
    print(" old c03            :", np.round(old['D_q' + s].sum(0) / old['Nn_q' + s].sum(0), 4))
    Do = old["D_q" + s]
    print(f" per-bin |D_new - D_old| / sqrt(0.4 N0): max {np.max(np.abs(D-Do)/np.sqrt(0.4*N0+1e-9)*0.4):.3f}, "
          f"rms {np.sqrt(np.mean((0.4*(D-Do))**2/(0.4*N0+1e-9))):.3f}  (in Poisson sigma at 40% sky)")
    res[f"D_q{s}"] = D; res[f"Nb_q{s}"] = Nb; res[f"Nn_q{s}"] = Nn; res[f"N0_q{s}"] = N0
np.savez(f"{CL.DC}/truth_catalogue_v2.npz", **res)
print(f"total {time.time()-t0:.0f}s")
