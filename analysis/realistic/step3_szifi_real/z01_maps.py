"""z01: regenerate step 3a's tiles from their seeds (project .venv; exactly as verify_real/step3_szifi/v00_regen_check.py
and vsz_lib.make_tile do, through analysis/realistic/step3/s3lib.py and szsim.finder), check them bit-exactly against
the stored production-finder outputs, and write the six-channel maps that SZiFi reads.

Tile pid, noise level MODE:
  real  : tSZ (painted, szsim.finder.paint_y + tsz_channels) + CMB (seed 300000 + pid) + noise (seed 400000 + pid);
  noise : the same CMB and noise without halos (as t02 and the arbiter's a01).
Check: the production finder (szsim.finder.run_finder, whitened iteration) rerun on the regenerated tile must give the
stored iteration-0 and iteration-1 sigma, covariance and catalogues exactly (real: data_real/step3/tiles/; noise:
data_real/step3/noise_only/, whitened).  The noise-only catalogues are kept (finder_noise.npz) for the comparisons.
Maps: real space, (1200, 1200, 6) float32 in microK, axis 0 = y (pixel iy), axis 1 = x (pixel ix), sample (iy, ix) at
(0.5 iy, 0.5 ix) arcmin (the build's convention); written to data_real/step3_szifi_real/out/maps/MODE_KIND_PID.npy
(regenerable scratch, deleted after the runs).
Usage: python -I -B z01_maps.py
"""
import os, sys, time, resource
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
MAPD = os.path.join(OUT, "out", "maps")
os.makedirs(MAPD, exist_ok=True)
G = F.grid()
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
PIDS_REAL = [int(p) for p in REF["pids_real"]]
PIDS_NOISE = [int(p) for p in REF["pids_noise"]]
TAB = F.build_paint_tables()          # loads data_real/step3/paint_tables.npz (never rewritten)


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def write_map(X, fn):
    m = np.empty((F.N, F.N, X.shape[0]), np.float32)
    worst = 0.0
    for i in range(X.shape[0]):
        r = sfft.irfft2(X[i], s=(F.N, F.N))
        m[:, :, i] = r
        worst = max(worst, float(np.abs(m[:, :, i].astype(np.float64) - r).max() / r.std()))
    np.save(fn, m)
    return worst, m


def check(its, ref_get, lab):
    """Exact comparison of a production-finder rerun with stored outputs; returns a short status string."""
    msg, ok = [], True
    for k, it in enumerate(its):
        d = it["det"]
        q = ref_get(f"it{k}_q")
        same = q.size == d["q"].size and np.array_equal(q, d["q"]) and np.array_equal(ref_get(f"it{k}_iy"), d["iy"]) \
            and np.array_equal(ref_get(f"it{k}_ix"), d["ix"]) and np.array_equal(ref_get(f"it{k}_j"), d["j"])
        ds = float(np.abs(it["filt"].sigma_arcmin2 / ref_get(f"it{k}_sig") - 1).max())
        ok &= same and ds == 0.0
        msg.append(f"it{k}: {d['q'].size} det (stored {q.size}), identical catalogue {same}, max|sigma/stored-1| {ds:.1e}")
    return ok, f"{lab}: " + "; ".join(msg)


n_ok = n_all = 0
f32 = 0.0
fin = {}
print(f"z01: {len(PIDS_REAL)} realistic tiles and {len(PIDS_NOISE)} noise-only tiles per noise level -> {MAPD}", flush=True)
for pid in PIDS_REAL:
    t1 = time.time()
    st = load_npz(os.path.join(F.STEP3_DATA, "tiles", f"tile_{pid:03d}.npz"))
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], TAB, G), G)
    for mode in S.MODES:
        X = Xt + F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        its = F.run_finder(X, G)
        ok, msg = check(its, lambda k: st[f"{mode}_{k}"], f"real  {pid:3d} {mode:8s}")
        n_ok += ok; n_all += 1
        w, m = write_map(X, os.path.join(MAPD, f"{mode}_real_{pid:03d}.npy"))
        f32 = max(f32, w)
        if pid == PIDS_REAL[0]:
            rt = max(float(np.abs(sfft.rfft2(m[:, :, i].astype(np.float64)) - X[i]).max() / np.abs(X[i]).max()) for i in range(6))
            print(f"  round trip rfft2(map) vs X (float32 maps), tile {pid} {mode}: max |diff| / max |X| = {rt:.1e}")
        print("  " + msg + f"  [{time.time() - t1:.0f} s]", flush=True)
        del X, its, m
    del Xt, d, st
for pid in PIDS_NOISE:
    t1 = time.time()
    st = load_npz(os.path.join(F.STEP3_DATA, "noise_only", f"noise_{pid:03d}.npz"))
    for mode in S.MODES:
        X = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        its = F.run_finder(X, G)
        ok, msg = check(its, lambda k: st[f"{mode}_whitened_{k}"], f"noise {pid:3d} {mode:8s}")
        n_ok += ok; n_all += 1
        for k, it in enumerate(its):
            for key in ("q", "iy", "ix", "j"):
                fin[f"{mode}_{pid:03d}_it{k}_{key}"] = it["det"][key]
            fin[f"{mode}_{pid:03d}_it{k}_sig"] = it["filt"].sigma_arcmin2
        w, m = write_map(X, os.path.join(MAPD, f"{mode}_noise_{pid:03d}.npy"))
        f32 = max(f32, w)
        print("  " + msg + f"  [{time.time() - t1:.0f} s]", flush=True)
        del X, its, m
np.savez(os.path.join(OUT, "finder_noise.npz"), **fin)
print(f"\nregeneration: {n_ok} of {n_all} tile runs identical to the stored production outputs (catalogues and sigma)")
print(f"float32 maps: max |map32 - map64| / map std = {f32:.1e}")
print(f"maps written: {len(os.listdir(MAPD))} files, {sum(os.path.getsize(os.path.join(MAPD, f)) for f in os.listdir(MAPD)) / 1e9:.2f} GB")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
