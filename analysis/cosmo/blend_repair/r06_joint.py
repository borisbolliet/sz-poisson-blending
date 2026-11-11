"""Ignore-blending fits: how far is the truth from the best fit jointly (not just marginally)?
Delta chi2 = chi2(truth) - chi2_min under the naive model (exact Poisson deviance), its p-value / Gaussian-equivalent
for n_free parameters, and the largest marginal bias over all linear combinations, sqrt(d^T F d) (F = Fisher of the
blended model incl. priors).  Also the worst-biased combination."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import replib as R, r02_fits as F
J = json.load(open(f"{R.D}/r02_fits.json"))
SIGF, _ = F.fisher_sigmas()
for c in "ABC":
    for fs in (1.0, 0.4):
        r = [x for x in J["res"] if x["kind"] == "ignore" and x["case"] == c and x["fs"] == fs and x["start_lab"] == "truth" and x["mask"] is None][0]
        free = F.CASES[c][0]; d = np.array(r["d"]); Fm = SIGF[(c, fs)]["F"]
        dchi = r["chi2_truth"] - r["chi2"]
        p = stats.chi2.sf(dchi, len(free)); zeq = stats.norm.isf(p / 2)
        dist = np.sqrt(d @ Fm @ d)
        v = Fm @ d; v = v / np.linalg.norm(v)                 # direction of the most-biased combination
        print(f"case {c} fsky={fs}: Delta chi2(truth vs best fit) = {dchi:.1f} for {len(free)} params -> p = {p:.1e} "
              f"({zeq:.1f} sigma equiv.); max marginal bias over linear combinations sqrt(dFd) = {dist:.2f} sigma; "
              f"combination coeffs ({', '.join(R.PN[a] for a in free)}) = [{' '.join(f'{x:+.2f}' for x in v)}]")
