"""r01 (check for PROGRESS.md, written by the main session): step 3b's final headline (arbiter a13, pooled tiles) with the FITS VERIFIER's independent count model
and fitter (verify_real/step3b_fits/f00lib.py, f04_fits.ml_fit): smooth Tinker08 HMF grid, PCHIP kernel interpolation,
Nelder-Mead + Newton.  Data = mu_verifier(truth) x R_s(pooled, arbiter a10).  Read-only on the repo."""
import os, sys, time, json
sys.dont_write_bytecode = True
os.environ["VF_THREADS"] = "1"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "verify_real", "step3b_fits"))
sys.argv = ["f04_fits.py", "1"]
import numpy as np
import f00lib as L
import f04_fits as F4
ARB = os.path.join(REPO, "data_real", "step3b", "arbiter")
claims = {  # a13_pooled_fits.log: ML (lin) S8 bias, q_cut 5
    ("baseline", "SO40"): {"a": -2.33, "d": +0.32, "e": +0.38},
    ("baseline", "full"): {"a": -3.68, "d": +0.50, "e": +0.60},
    ("goal", "SO40"): {"a": -4.23, "d": +0.68, "e": +0.72},
    ("goal", "full"): {"a": -6.69, "d": +1.06, "e": +1.15},
}
t0 = time.time()
for mode in ("baseline", "goal"):
    dv = np.load(os.path.join(ARB, f"datavec_pooled_{mode}.npz"))
    Rs = dv["Rs_q5"]; Rp = dv["Rp_q5"]; RB = dv["Rsboot_q5"].astype(float)
    M = L.KModel(mode, 5.0, L.rbar("prod", mode), L.KTable(os.path.join(L.D3B, f"kernel_prod_{mode}.npz")), how="alt")
    N0 = M.counts(L.FID)
    cc = float((N0 * (Rs - 1)).sum() / N0.sum()); ccp = float((N0 * (Rp - 1)).sum() / N0.sum())
    ccb = ((N0[None] * (RB - 1)).reshape(RB.shape[0], -1).sum(1) / N0.sum()).std()
    print(f"== {mode}: tiles {int(dv['ntiles'])}; count change at q>5 with the verifier's model weights: "
          f"R_s {100*cc:+.2f} +- {100*ccb:.2f}%  R_p {100*ccp:+.2f}%   (claim -1.46 / -2.10)", flush=True)
    for sv in ("SO40", "full"):
        for cfg in ("a", "d", "e"):
            r = F4.ml_fit(M, N0 * Rs, cfg, L.SURV[sv])
            rp = F4.ml_fit(M, N0 * Rp, cfg, L.SURV[sv]) if cfg == "d" else None
            print(f"   {sv:4s} config {cfg}: S8 ML {r['ml']['S8']:+.2f} (lin {r['lin']['S8']:+.2f})  Om {r['ml']['Om']:+.2f} s8 {r['ml']['s8']:+.2f}"
                  f"  power {r['power']:.2f}  at_bound {r['at_bound']}   claim {claims[(mode, sv)][cfg]:+.2f}"
                  + (f"   | R_p: ML {rp['ml']['S8']:+.2f} lin {rp['lin']['S8']:+.2f}" if rp else "")
                  + f"   [{time.time()-t0:.0f} s]", flush=True)
print(f"done {time.time()-t0:.0f} s; peak RSS {L.rss_gb():.2f} GB")
