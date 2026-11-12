"""v04: check (b) -- the SZiFi-convention finder end to end on the build's realistic tiles (our emulation, vsz_lib).

Per tile and noise level: SZiFi's own isotropic covariance of the tile (binned periodogram, all modes per bin, nearest
bin), SZiFi's sampled Arnaud templates (15 scales 0.5'-15', as the build), S/N maps, SZiFi's peak rules; iteration 1
after masking q > 4 detections with 3 theta discs, with SZiFi's diffusive inpainting + cosine blend + lsep = 3000
correction (without inpainting SZiFi's iteration is a no-op, so its final catalogue is iteration 0's).
Convolution: 'per' = periodic (SZiFi on a sky without tile edges), 'lin' = SZiFi's zero-padded 'same' convolution of
the tile as given.  Output: out/v04_CONV_MODE_PID.npz.  Comparison with the build: v06_compare.py.
Usage: python -I -B v04_e2e.py CONV NPROC MODE:PID [MODE:PID ...]
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
import vsz_lib as V

VARIANT, NPROC = sys.argv[1], int(sys.argv[2])
CONV = VARIANT.split("_")[0]                 # 'per' or 'lin'; suffix '_noblend' = diagnostic without SZiFi's cosine blend
BLEND = "noblend" not in VARIANT.split("_")
LSEP = 12000.0 if "lsep12000" in VARIANT.split("_") else V.LSEP   # '_lsep12000': SZiFi's SO-example lsep
JOBS = [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[3:]]


def run(job):
    mode, spec = job
    src, pid = (spec.split("-") + [None])[:2] if "-" in spec else ("tile", spec)
    pid = int(pid)
    fn = os.path.join(V.OUTD, f"v04_{VARIANT}_{mode}_{src}{pid:03d}.npz")
    if os.path.exists(fn):
        return job, 0.0, V.peak_rss_gb(), "exists"
    t0 = time.time()
    if src == "tile":
        X, _ = V.make_tile(pid, mode)
    elif src == "noise":
        X = V.make_noise_only(pid, mode)
    elif src == "inj":
        X, _ = V.make_injection(mode, pid)
    else:
        raise ValueError(src)
    r = V.szifi_run2(X, conv=CONV, blend=BLEND, lsep=LSEP)
    del X
    out = dict(sig0=r["sig0"], C0=r["C0"], n_it=r["n_it"], sig_ana=V.szifi_sigma_analytic(mode))
    out.update(V.det_to_arrays(r["det0"], "det0"))
    if r["n_it"] == 2:
        out.update(sig1=r["sig1"], C1=r["C1"], C1raw=r["C1raw"], fsky=r["fsky"], nreg=r["nreg"],
                   wmean_hole=r["wmean_hole"], w2mean_hole=r["w2mean_hole"])
        out.update(V.det_to_arrays(r["det1"], "det1"))
    out["t"] = time.time() - t0
    np.savez(fn, **out)
    return job, out["t"], V.peak_rss_gb(), f"n_det it0 {r['det0']['q'].size}, it1 {r['det1']['q'].size if r['n_it'] == 2 else '-'}"


if __name__ == "__main__":
    t0 = time.time()
    print(f"v04: variant {VARIANT} (conv {CONV}, SZiFi cosine blend {BLEND}, lsep {LSEP:.0f}), {len(JOBS)} jobs, {NPROC} processes", flush=True)
    with Pool(NPROC, maxtasksperchild=4) as pool:
        for job, tt, rss, msg in pool.imap_unordered(run, JOBS):
            print(f"  {job[0]:8s} {job[1]:10s}: {tt:6.1f} s, worker peak RSS {rss:.2f} GB; {msg}  [elapsed {time.time() - t0:.0f} s]",
                  flush=True)
    print(f"v04 done [{time.time() - t0:.0f} s]", flush=True)
