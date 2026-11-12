"""vc04b: where are the largest per-bin pulls (mine - build) / MC error?  Lists the top bins, and the pull pattern
summed over z and over q (Delta/N with errors), for the fiducial MMF variants and white, q_cut 5."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np
from scipy.stats import chi2

dm = np.load(f"{HERE}/out/delta_mine.npz")
for key, bset in (("baseline_f1", "baseline_f1"), ("goal_f1", "goal_f1"), ("baseline_f2", "baseline_f2"), ("white_f1", "white")):
    D, E, Ns = dm[f"{key}_q5_D"], dm[f"{key}_q5_err"], dm[f"{key}_q5_Ns"]
    B = np.load(f"{V.ROOT}/data_real/step2/truth_{bset}.npz")
    DB, NnB = B["D_q5"], B["Nn_q5"]
    pull = np.where(E > 0, (D - DB) / np.where(E > 0, E, 1), 0)
    print(f"[{key}] top |pull| bins (z bin, q bin: mine +- err | build | N_naive | |diff| in SO40 Poisson sigma):")
    order = np.argsort(-np.abs(pull).ravel())[:6]
    for o in order:
        iz, iq = np.unravel_index(o, pull.shape)
        print(f"   z {V.ZED[iz]:.2f}-{V.ZED[iz+1]:.2f}, q {V.QED5[iq]:.1f}-{(V.QED5[iq+1] if iq < 8 else np.inf):.1f}: "
              f"{D[iz, iq]:+8.2f} +- {E[iz, iq]:.2f} | {DB[iz, iq]:+8.2f} | {NnB[iz, iq]:9.1f} | pull {pull[iz, iq]:+.2f} | "
              f"{0.4*abs(D[iz,iq]-DB[iz,iq])/np.sqrt(0.4*NnB[iz,iq]):.3f}")
    # pull distribution: fraction beyond 2, 3 sigma vs Gaussian
    g = NnB > 1
    p = pull[g]
    print(f"   {g.sum()} bins: |pull| > 2: {np.mean(np.abs(p) > 2):.3f} (Gauss 0.046), > 3: {np.mean(np.abs(p) > 3):.3f} (0.003); "
          f"chi2 without the 3 largest: {np.sum(np.sort(p**2)[:-3]):.1f} / {g.sum()-3} (p = {chi2.sf(np.sum(np.sort(p**2)[:-3]), g.sum()-3):.3f})")
