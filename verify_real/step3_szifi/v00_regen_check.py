"""v00: can the build's stored tiles be regenerated exactly?  Rebuild tiles from their seeds with the build's own
simulator (szsim.finder via analysis/realistic/step3/s3lib.py) and rerun the build's finder; compare with the stored
catalogues (data_real/step3/tiles/tile_PID.npz).  No SZiFi code is run here.
Usage: python -I -B v00_regen_check.py PID [PID ...]
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
import vsz_lib as V

pids = [int(a) for a in sys.argv[1:]] or [0]
t0 = time.time()
for pid in pids:
    st = V.load_stored_tile(pid)
    for mode in V.MODES:
        X, comps = V.make_tile(pid, mode)
        its = V.F.run_finder(X, V.F.grid())
        for k, it in enumerate(its):
            d = it["det"]
            qs = st[f"{mode}_it{k}_q"]
            same_n = d["q"].size == qs.size
            dq = np.abs(d["q"] - qs).max() if same_n else np.nan
            same_pix = same_n and np.all(d["iy"] == st[f"{mode}_it{k}_iy"]) and np.all(d["ix"] == st[f"{mode}_it{k}_ix"]) \
                and np.all(d["j"] == st[f"{mode}_it{k}_j"])
            ds = np.abs(it["filt"].sigma_arcmin2 / st[f"{mode}_it{k}_sig"] - 1).max()
            dC = np.abs(it["Cbin"][V.F.grid().bins_used] / st[f"{mode}_it{k}_C"][V.F.grid().bins_used] - 1).max()
            print(f"tile {pid:3d} {mode:8s} it{k}: {d['q'].size} detections (stored {qs.size}); same pixels/scales {same_n and same_pix}; "
                  f"max |dq| {dq:.1e}; max |sigma/stored - 1| {ds:.1e}; max |C_bin/stored - 1| {dC:.1e}", flush=True)
        del X, comps, its
print(f"done [{time.time() - t0:.0f} s]; peak RSS {V.peak_rss_gb():.2f} GB")
