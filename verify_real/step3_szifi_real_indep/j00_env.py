"""j00 (SZiFi's venv): environment, the clone's state, and the build's claim E2 (SZiFi's default dictionary cannot run).

Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j00_env.py  (after j01_maps.py, which writes the maps)
"""
import os, sys, copy, io, contextlib, traceback, subprocess, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
SZIFI = "/Users/boris/GitHub/szifi"
sys.path.insert(0, SZIFI)
import numpy as np

print(f"j00: python {sys.version.split()[0]} at {sys.executable}")
for name in ("numpy", "scipy", "astropy", "healpy", "numba", "matplotlib", "pixell", "pymaster", "sklearn", "orphics",
             "setuptools", "h5py", "six", "yaml"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:11s} {getattr(m, '__version__', '(no __version__)')}")
    except Exception as e:
        print(f"  {name:11s} not importable: {type(e).__name__}: {e}")
import szifi                                              # noqa: E402
print(f"  szifi imported from {os.path.dirname(szifi.__file__)}")
g = lambda *a: subprocess.run(["git", "-C", SZIFI] + list(a), capture_output=True, text=True).stdout.strip()
print(f"  clone: commit {g('rev-parse', '--short', 'HEAD')} ({g('log', '-1', '--format=%s')}); "
      f"'git status --short' prints {len(g('status', '--short'))} characters")

# ---- claim E2: SZiFi's default dictionary spells 'snr_weigthing', cluster_finder reads 'snr_weighting'
d = szifi.params.params_szifi_default
print(f"\nE2: 'snr_weigthing' in SZiFi's defaults: {'snr_weigthing' in d}; 'snr_weighting': {'snr_weighting' in d}")
p = copy.deepcopy(d)
ref = np.load(os.path.join(OUT, "j01_ref.npz"))
pid = int(ref["new_pids"][0])
p.update({"survey_file": os.path.join(HERE, "jsz_survey.py"), "path_data": os.path.join(OUT, "maps") + "/"})
pdata = {"field_ids": [pid], "j_map": os.path.join(OUT, "maps", f"noise_baseline_{pid:03d}.npy"), "j_buffer": 16,
         "j_fwhm": ref["fwhm_arcmin"], "j_white": ref["white_baseline"], "j_sed": ref["tsz_sed_uK"],
         "j_nu": ref["freqs_GHz"] * 1e9, "j_cm_name": os.path.join(OUT, "none.fits")}
try:
    data = szifi.input_data(params_szifi=p, params_data=pdata)
    cf = szifi.cluster_finder(params_szifi=p, data_file=data)
    with contextlib.redirect_stdout(io.StringIO()):
        cf.find_clusters()
    print("  cluster_finder.find_clusters with the default dictionary (only the survey file changed): ran")
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(f"  cluster_finder.find_clusters with the default dictionary (only the survey file changed): "
          f"{type(e).__name__}: {e} (raised in {os.path.basename(tb.filename)}, line {tb.lineno}, function {tb.name})")
print("done")
