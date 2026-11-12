"""z00b: the SZiFi environment (package versions, the SZiFi clone's commit and status) and, per configuration, every
SZiFi parameter that differs from its default dictionary (szifi.params_szifi_default) before the run (cluster_finder
then adds a_matrix and sets cmmf_type itself).  Also the instrument values the experiment object receives.
Usage: szifi-venv/bin/python -I -B z00b_env.py
"""
import os, sys, subprocess
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402
import importlib.metadata as md                       # noqa: E402

print(f"python {sys.version.split()[0]} ({sys.executable})")
for p in ("numpy", "scipy", "astropy", "healpy", "numba", "llvmlite", "matplotlib", "pixell", "pymaster", "scikit-learn",
          "setuptools", "orphics", "six", "h5py", "PyYAML", "ducc0"):
    try:
        print(f"  {p:14s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:14s} not installed")
git = lambda *a: subprocess.run(["git", "-C", Z.SZIFI] + list(a), capture_output=True, text=True).stdout.strip()
print(f"SZiFi clone {Z.SZIFI}: HEAD {git('rev-parse', '--short', 'HEAD')} ({git('log', '-1', '--format=%cd %s')}); "
      f"'git status --short' prints {len(git('status', '--short'))} characters; imported from {Z.szifi.__file__}")
print(f"environment of this process: NUMBA_CACHE_DIR={os.environ['NUMBA_CACHE_DIR']}, OMP/OPENBLAS/VECLIB/NUMBA threads "
      f"{os.environ['OMP_NUM_THREADS']}/{os.environ['OPENBLAS_NUM_THREADS']}/{os.environ['VECLIB_MAXIMUM_THREADS']}/"
      f"{os.environ['NUMBA_NUM_THREADS']}, sys.dont_write_bytecode={sys.dont_write_bytecode}")

inst = np.load(Z.INSTRUMENT)
e = Z.expt.experiment(experiment_name="SOgoal_simple", params_szifi=Z.make_params("S1"))
print("\nSZiFi's 'SOgoal_simple' experiment as shipped vs the values our survey file puts in its place:")
for k, ours in (("nu_eff", inst["nu_eff_Hz"]), ("FWHM", inst["fwhm_arcmin"]), ("noise_levels", inst["white_goal"]),
                ("tsz_sed", inst["tsz_sed_uK"])):
    a_, b_ = np.asarray(getattr(e, k), float), np.asarray(ours, float)
    print(f"  {k:12s} SZiFi {np.array2string(a_, precision=6, max_line_width=150)}")
    print(f"  {'':12s} ours  {np.array2string(b_, precision=6, max_line_width=150)}   max |ours/SZiFi - 1| = "
          f"{np.abs(b_ / a_ - 1).max():.2e}")
print(f"  noise_levels for 'baseline': {inst['white_baseline'].tolist()}; n_freqs {e.n_freqs} (recomputed: 6)")

print("\nparameters that differ from SZiFi's defaults, per configuration (before the run):")
for cfg in ("S0", "S1", "S2", "S3", "S3i", "S3it0"):
    d = Z.differs_from_default(Z.make_params(cfg))
    print(f"  {cfg}:")
    for k in sorted(d):
        v = d[k]
        if isinstance(v, np.ndarray):
            v = np.array2string(v, precision=4, max_line_width=150)
        dv = Z.szifi.params_szifi_default.get(k, "(not in the defaults)")
        if isinstance(dv, np.ndarray):
            dv = np.array2string(dv, precision=4, max_line_width=150)
        print(f"    {k:24s} = {v!s:40s} (default {dv})")
