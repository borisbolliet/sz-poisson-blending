"""Evaluate my -2lnL (v04 objective: Cash + Gaussian priors) at the c05 best-fit points (data_cosmo/cosmo_bias_fsky1.0.json),
to decide which optimiser found the lower minimum where v04 and c05 differ."""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
for setup, kind, fs in (("S0", "eff", "1.0"), ("S1", "naive", "1.0"), ("S2", "naive", "1.0"), ("S2", "eff", "1.0"), ("S1", "naive", "0.4")):
    sys.argv = ["x", setup, kind, fs]
    src = open("v04_fits_independent.py").read().split("t0 = time.time()")[0]
    g = {}; exec(src, g)
    J = json.load(open(f"/Users/boris/GitHub/sz-poisson-blending/data_cosmo/cosmo_bias_fsky{1.0 if fs=='1.0' else 0.4}.json"))
    key = [k for k in J if k.startswith(f"fsky{float(fs)}|{setup}:") and k.endswith("|" + kind)][0]
    p = np.array(J[key]["best"])
    u = (p - g["p_fid"]) / g["st"]
    print(f"{setup} {kind} fsky={fs}: -2lnL at c05 best fit = {g['m2lnL'](u):.3f} (c05 Cash-only {J[key]['chi2_min']:.3f}); c05 best {np.round(p,4)}", flush=True)
