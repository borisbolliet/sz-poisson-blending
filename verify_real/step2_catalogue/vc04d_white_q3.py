"""vc04d: the build's white-filter Delta (truth_white, truth_white_prefix) vs Q3's truth_catalogue_v2, bin by bin
(data only; claim C20), in Poisson sigma at f_sky = 0.4 (0.4 |dDelta| / sqrt(0.4 N0)); and my direct-route white Delta."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import numpy as np
import vclib as V
D2 = f"{V.ROOT}/data_real/step2"
Q = np.load(f"{V.ROOT}/data_cosmo/truth_catalogue_v2.npz")
dm = np.load(f"{HERE}/out/delta_mine.npz")
for qc in ("5", "6"):
    for name in ("white", "white_prefix"):
        B = np.load(f"{D2}/truth_{name}.npz")
        D, N0 = B[f"D_q{qc}"], B[f"N0_q{qc}"]
        s = 0.4 * np.abs(D - Q[f"D_q{qc}"]) / np.sqrt(0.4 * N0 + 1e-9)
        print(f"q>{qc} [{name}] total {D.sum():+.1f} vs Q3 {Q[f'D_q{qc}'].sum():+.1f}; per bin max {s.max():.4f} Poisson sigma (SO40)")
    Dm, Em = dm[f"white_f1_q{qc}_D"], dm[f"white_f1_q{qc}_err"]
    tot_err = np.sqrt(len(dm[f"white_f1_q{qc}_blocks"]) * dm[f"white_f1_q{qc}_blocks"].astype(float).sum((1, 2)).var(ddof=1)) / len(dm["seeds"])
    print(f"q>{qc} [mine, direct] total {Dm.sum():+.1f} +- {tot_err:.1f}")
