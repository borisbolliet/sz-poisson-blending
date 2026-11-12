"""z03a: regenerate one tile map (realistic tile 0, SO baseline) for z03_decouple.py, exactly as z01_maps.py does,
check it against the stored production-finder outputs (bit for bit), and write it to out/maps/ (scratch, deleted
after use).
Usage: python -I -B z03a_map0.py
"""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
import numpy as np
from scipy import fft as sfft
import s3lib as S                                   # noqa: E402
from szsim import finder as F                       # noqa: E402

t0 = time.time()
G = F.grid()
pid, mode = 0, "baseline"
d = S.truth(pid)
X = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], F.build_paint_tables(), G), G) \
    + F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
its = F.run_finder(X, G)
with np.load(os.path.join(F.STEP3_DATA, "tiles", f"tile_{pid:03d}.npz")) as st:
    same = all(np.array_equal(st[f"{mode}_it{k}_q"], it["det"]["q"]) and np.array_equal(st[f"{mode}_it{k}_sig"],
               it["filt"].sigma_arcmin2) for k, it in enumerate(its))
os.makedirs(os.path.join(OUT, "out", "maps"), exist_ok=True)
m = np.empty((F.N, F.N, 6), np.float32)
for i in range(6):
    m[:, :, i] = sfft.irfft2(X[i], s=(F.N, F.N))
np.save(os.path.join(OUT, "out", "maps", f"{mode}_real_{pid:03d}.npy"), m)
print(f"tile {pid} {mode}: production finder rerun identical to the stored catalogues and sigma: {same}; map written "
      f"[{time.time() - t0:.0f} s]")
