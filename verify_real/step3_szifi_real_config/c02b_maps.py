"""c02b: regenerate more realistic tiles (as c02 part A: s3lib seeds and szsim.finder, checked with the production
finder against the stored outputs) and write their float32 maps to out/maps/real_MODE_PID.npy.
Usage: /Users/boris/GitHub/sz-poisson-blending/.venv/bin/python -I -B c02b_maps.py PID [PID ...]
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
import numpy as np                                   # noqa: E402
from scipy import fft as sfft                        # noqa: E402
import s3lib as S                                    # noqa: E402
from szsim import finder as F                        # noqa: E402

t0 = time.time()
MAPD = os.path.join(HERE, "out", "maps")
G = F.grid()
TAB = F.build_paint_tables()
n_ok = n_all = 0
for pid in [int(a) for a in sys.argv[1:]]:
    with np.load(os.path.join(F.STEP3_DATA, "tiles", f"tile_{pid:03d}.npz")) as z:
        st = {k: z[k] for k in z.files}
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
    for mode in S.MODES:
        X = Xt + F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        its = F.run_finder(X, G)
        ok = True
        for k, it in enumerate(its):
            dd = it["det"]
            ok &= bool(np.array_equal(st[f"{mode}_it{k}_q"], dd["q"]) and np.array_equal(st[f"{mode}_it{k}_iy"], dd["iy"])
                       and np.array_equal(st[f"{mode}_it{k}_ix"], dd["ix"]) and np.array_equal(st[f"{mode}_it{k}_j"], dd["j"])
                       and np.array_equal(st[f"{mode}_it{k}_sig"], it["filt"].sigma_arcmin2))
        n_ok += ok
        n_all += 1
        m = np.empty((F.N, F.N, 6), np.float32)
        for i in range(6):
            m[:, :, i] = sfft.irfft2(X[i], s=(F.N, F.N))
        np.save(os.path.join(MAPD, f"real_{mode}_{pid:03d}.npy"), m)
        print(f"  real {pid:3d} {mode:8s}: production finder rerun identical to the stored catalogues and sigma: {ok}", flush=True)
        del X, its, m
    del Xt, d, st
print(f"{n_ok} of {n_all} identical; done [{time.time() - t0:.0f} s]; peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
