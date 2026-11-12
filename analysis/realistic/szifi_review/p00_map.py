"""p00: regenerate one of step 3a's tiles (realistic, pid 0, SO baseline noise) from its seeds with the project's .venv,
check it bit for bit against the stored production-finder outputs, and write the float32 six-channel map (microK) that
SZiFi reads.  Same recipe as ../step3_szifi_real/zA00_maps.py, for one tile.  Used by p01_profile.py, p04_lrange_none.py,
p05_mixed_fix.py and p07_max_it.py.

Output: data_real/szifi_review/out/maps/baseline_real_000.npy (scratch, not committed; about 35 MB).
Usage (from the repository root): .venv/bin/python -I -B analysis/realistic/szifi_review/p00_map.py
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(ROOT, "data_real", "szifi_review", "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
import numpy as np
from scipy import fft as sfft
import s3lib as S                                   # noqa: E402
from szsim import finder as F                       # noqa: E402

PID, MODE = 0, "baseline"
MAPD = os.path.join(OUT, "maps")
os.makedirs(MAPD, exist_ok=True)

t0 = time.time()
G = F.grid()
TAB = F.build_paint_tables()
with np.load(os.path.join(F.STEP3_DATA, "tiles", f"tile_{PID:03d}.npz")) as d:
    st = {k: d[k] for k in d.files}
d = S.truth(PID)
X = (F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
     + F.sim_cmb(S.SEED_CMB + PID, G) + F.sim_noise(S.SEED_NOISE + PID, MODE, G))
its = F.run_finder(X, G)
ok = True
for k, it in enumerate(its):
    det = it["det"]
    for key in ("q", "iy", "ix", "j"):
        ok &= bool(np.array_equal(st[f"{MODE}_it{k}_{key}"], det[key]))
    ok &= bool(np.array_equal(st[f"{MODE}_it{k}_sig"], it["filt"].sigma_arcmin2))
print(f"tile {PID} {MODE}: production finder rerun identical to the stored outputs (both iterations): {ok}")
m = np.empty((F.N, F.N, X.shape[0]), np.float32)
for i in range(X.shape[0]):
    m[:, :, i] = sfft.irfft2(X[i], s=(F.N, F.N))
fn = os.path.join(MAPD, f"{MODE}_real_{PID:03d}.npy")
np.save(fn, m)
print(f"wrote {os.path.relpath(fn, ROOT)}: shape {m.shape}, {os.path.getsize(fn) / 1e6:.1f} MB")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
