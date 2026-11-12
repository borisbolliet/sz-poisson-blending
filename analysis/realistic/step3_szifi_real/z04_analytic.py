"""z04: sigma(theta) of SZiFi's own filter with the true (analytic) CMB + noise covariance of our skies, in SZiFi's
units (central Compton y of its template, norm_type 'centre').  This is the denominator of r = sigma / sigma_analytic
for SZiFi (each filter with its own template, as the step-3a SZiFi check defined it), the analogue of the finder's
r = sigma_finder / sigma_Y.

How: SZiFi's mmf.filter_maps.find_clusters (its templates, Gaussian beams l(l+1), l range, and for S3 the
powspec_lmax1d cut of the inverse covariance) run on all-zero maps with the analytic inverse covariance
(z_szlib.analytic_inv_cov), detection_method 'maxima' (that branch fills sigma_vec at every scale).
Setups: 'S0' (shared by S0, S1, S2: lrange 100-20000, full-resolution covariance) and 'S3' (shared by S3 and S3i:
lrange 1-300000, inverse covariance on the l_max,1d = 10000 grid).
Usage: szifi-venv/bin/python -I -B z04_analytic.py
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

t0 = time.time()
out = {"thetas": Z.THETAS15}
for setup in ("S0", "S3"):
    for mode in ("baseline", "goal"):
        t1 = time.time()
        s, shape = Z.analytic_sigma(setup, mode)
        out[f"{mode}_{setup}"] = s
        print(f"{setup} setup, {mode}: inverse-covariance grid {shape}; sigma_analytic [y0] = "
              + " ".join(f"{x:.4e}" for x in s) + f"  [{time.time() - t1:.0f} s, peak RSS {Z.peak_rss_gb():.2f} GB]",
              flush=True)
print("\ntheta [']                          " + " ".join(f"{t:7.3f}" for t in Z.THETAS15))
for mode in ("baseline", "goal"):
    print(f"{mode:8s} sigma(S3 setup) / sigma(S0)  " + " ".join(f"{x:7.4f}" for x in out[f"{mode}_S3"] / out[f"{mode}_S0"]))
for setup in ("S0", "S3"):
    print(f"sigma goal / baseline, {setup} setup     " + " ".join(f"{x:7.4f}" for x in out[f"goal_{setup}"] / out[f"baseline_{setup}"]))
np.savez(os.path.join(Z.OUT, "analytic_sigma.npz"), **out)
print(f"done [{time.time() - t0:.0f} s]")
