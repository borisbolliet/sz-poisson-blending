"""z02b: the memory-only release of the iteration-0 inverse covariance (z_szlib.DROP_OLD_INV_COV) leaves SZiFi's
results unchanged.  Reruns a smoke configuration on the same tile with the release switched on and compares every
array with the smoke run made without it (runs/smokeCFG_baseline_real_000.npz, z02_smoke_CFG.log), bit for bit.
Usage: szifi-venv/bin/python -I -B z02b_memcheck.py CFG BUFFER_ARCMIN
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

cfg, buf = sys.argv[1], float(sys.argv[2])
ref = dict(np.load(os.path.join(Z.RUND, f"smoke{cfg}_baseline_real_000.npz")))
assert int(ref.get("n_drop_old_inv_cov", 0)) == 0
Z.DROP_OLD_INV_COV = True
r = Z.run_tile(cfg, "baseline", "real", 0, buf, save=True, tag=f"memcheck{cfg}")
print(f"z02b: {cfg} on tile 0 baseline with the release on: old inverse covariance released {r['n_drop_old_inv_cov']} time(s); "
      f"{r['t_run']:.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB (smoke run without it: {float(ref['peak_rss_gb']):.2f} GB)")
skip = {"t_load", "t_run", "peak_rss_gb", "tag", "n_drop_old_inv_cov"}
nd = 0
keys = sorted(k for k in r if k not in skip and not k.startswith("rec_t_"))
for k in keys:
    a, b = np.asarray(r[k]), np.asarray(ref[k])
    same = a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind not in "fc" else np.array_equal(a, b, equal_nan=True))
    if not same:
        nd += 1
        print(f"  DIFFERENT: {k}")
print(f"compared {len(keys)} arrays (catalogues, sigma, spectra, unmasked fraction, parameters): {nd} differ")
