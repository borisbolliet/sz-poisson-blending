"""zA00 (arbiter): regenerate the tiles the arbitration runs SZiFi on, from their seeds, with the project's .venv, the
same way z01_maps.py does (s3lib.truth, szsim.finder paint / CMB / noise), check each one bit for bit against the
stored production-finder outputs, and write the float32 six-channel maps (microK) that SZiFi reads.

Tiles (chosen before any arbiter run, blind to outcomes):
  realistic pid 0, 11, 22, 33, 44, 55 (the config check's six 'S1wg' tiles) and 88, 132, 176, 220, 264, 308
            (every fourth of the build's remaining S1 tiles): 12 per noise level, all among the build's 30 S1 tiles;
  noise-only pid 0, 22, 44, 66, 88 (the same CMB and noise without halos): 5 per noise level.
Output: data_real/step3_szifi_real/arbiter/out/maps/MODE_KIND_PID.npy (scratch; deleted after the runs).
Usage: .venv/bin/python -I -B zA00_maps.py
"""
import os, sys, time, resource
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
AOUT = os.path.join(ROOT, "data_real", "step3_szifi_real", "arbiter")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(AOUT, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
import numpy as np
from scipy import fft as sfft
import s3lib as S                                   # noqa: E402
from szsim import finder as F                       # noqa: E402

PIDS_REAL = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]
PIDS_NOISE = [0, 22, 44, 66, 88]
MAPD = os.path.join(AOUT, "out", "maps")
os.makedirs(MAPD, exist_ok=True)


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def write_map(X, fn):
    m = np.empty((F.N, F.N, X.shape[0]), np.float32)
    for i in range(X.shape[0]):
        m[:, :, i] = sfft.irfft2(X[i], s=(F.N, F.N))
    np.save(fn, m)


def same_as_stored(its, get):
    ok = True
    for k, it in enumerate(its):
        d = it["det"]
        ok &= bool(np.array_equal(get(f"it{k}_q"), d["q"]) and np.array_equal(get(f"it{k}_iy"), d["iy"])
                   and np.array_equal(get(f"it{k}_ix"), d["ix"]) and np.array_equal(get(f"it{k}_j"), d["j"])
                   and np.array_equal(get(f"it{k}_sig"), it["filt"].sigma_arcmin2))
    return ok


t0 = time.time()
G = F.grid()
TAB = F.build_paint_tables()
n_ok = n_all = 0
for pid in PIDS_REAL:
    st = load_npz(os.path.join(F.STEP3_DATA, "tiles", f"tile_{pid:03d}.npz"))
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
    for mode in S.MODES:
        X = Xt + F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        ok = same_as_stored(F.run_finder(X, G), lambda k: st[f"{mode}_{k}"])
        n_ok += ok; n_all += 1
        write_map(X, os.path.join(MAPD, f"{mode}_real_{pid:03d}.npy"))
        print(f"  real  {pid:3d} {mode:8s}: production finder rerun identical to the stored outputs: {ok}", flush=True)
        del X
    del Xt, d, st
for pid in PIDS_NOISE:
    st = load_npz(os.path.join(F.STEP3_DATA, "noise_only", f"noise_{pid:03d}.npz"))
    for mode in S.MODES:
        X = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        ok = same_as_stored(F.run_finder(X, G), lambda k: st[f"{mode}_whitened_{k}"])
        n_ok += ok; n_all += 1
        write_map(X, os.path.join(MAPD, f"{mode}_noise_{pid:03d}.npy"))
        print(f"  noise {pid:3d} {mode:8s}: production finder rerun identical to the stored outputs: {ok}", flush=True)
        del X
print(f"\n{n_ok} of {n_all} regenerated tiles reproduce the stored production-finder catalogues and sigma bit for bit")
print(f"maps written: {len(os.listdir(MAPD))} files, {sum(os.path.getsize(os.path.join(MAPD, f)) for f in os.listdir(MAPD)) / 1e9:.2f} GB")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
