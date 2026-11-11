"""V4: profile-likelihood check of the Fisher errors and of the bias location (naive fit to blended Asimov data,
full sky).  For S1 and S2: fix Om0 (then sigma8) on a grid around the naive best fit, re-minimise all other free
parameters (least_squares), fit a parabola to -2lnL_profile, compare minimum and Delta(-2lnL)=1 half-width with
the Fisher marginal sigma at the best fit."""
import sys, time
import numpy as np
import v2_fits as F
import vlib as v

t0 = time.time()
popf = v.population(nd=F.ND)
NB = v.counts_blended(popf).ravel()
truth = {k: F.PAR[k][0] for k in F.PAR}
setups = sys.argv[1:] or ["S1", "S2"]
for S in setups:
    names = F.SETUPS[S]
    p, Fm, c2, _, _ = F.fit(NB, names, "naive", 1.0)
    sig = np.sqrt(np.diag(np.linalg.inv(Fm)))
    print(f"[{time.time()-t0:.0f}s] {S} free={names}: best {np.round(p,4)}, Fisher sigma {np.round(sig,4)}, -2lnL {c2:.3f}")
    for ip, pn in enumerate(["Om0", "sigma8"]):
        others = [n for n in names if n != pn]
        grid = p[ip] + sig[ip] * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        prof = []
        p0 = [p[names.index(n)] for n in others]
        for g in grid:
            old = F.PAR[pn]
            F.PAR[pn] = (g, old[1], None)
            po, _, c2o, _, _ = F.fit(NB, others, "naive", 1.0, p0=p0)
            F.PAR[pn] = old
            # fit() adds priors only for the free params; others' priors are included (pn has no prior)
            prof.append(c2o)
        prof = np.array(prof)
        a, b, c = np.polyfit(grid, prof, 2)
        xmin = -b / (2 * a); sprof = 1 / np.sqrt(a)
        print(f"   profile in {pn}: grid {np.round(grid,4)} -> -2lnL {np.round(prof,3)}")
        print(f"      parabola min {xmin:.5f} (fit {p[ip]:.5f}); profile sigma {sprof:.5f} vs Fisher {sig[ip]:.5f} "
              f"(ratio {sprof/sig[ip]:.3f}); bias/sigma_profile = {(xmin-truth[pn])/sprof:+.2f} (Fisher: {(p[ip]-truth[pn])/sig[ip]:+.2f});"
              f" cubic asymmetry: [-2lnL(+2s)-(-2lnL(-2s))] = {prof[-1]-prof[0]:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
