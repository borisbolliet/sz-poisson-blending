"""p01: where SZiFi's time goes.  One run of SZiFi itself (commit 73626fb, unchanged; the build's driver
../step3_szifi_real/z_szlib.py) with its default masking settings as in step 3a (configuration S1: l range 100-20000,
15 scales 0.5'-15', two noise iterations, one thread, 8' edge buffer) on tile 0, SO baseline noise (map from
p00_map.py), under cProfile.

Prints:
  (1) whether this run reproduces the build's stored S1 run of the same tile (catalogues and sigma);
  (2) the time of SZiFi's main stages: cProfile's cumulative time next to the driver's own wall-clock timers of the
      same run, where the driver has one (a consistency check within one run, not an independent measurement);
  (3) inside the filtering: templates, band limits, filter construction, convolutions; counts and time of the 2D FFTs;
      other repeated work;
  (4) the 25 functions with the largest own time.
Only names of SZiFi's functions are printed (file:line(function)), never its code.

Usage (from the repository root):
  MallocLargeCache=0 /Users/boris/GitHub/szifi-venv/bin/python -I -B analysis/realistic/szifi_review/p01_profile.py
"""
import os, sys, io, cProfile, pstats
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data_real", "szifi_review", "out")
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3_szifi_real"))
import z_szlib as Z                                   # noqa: E402  (sets one thread, puts the clone on sys.path)
import numpy as np                                    # noqa: E402
import scipy.signal                                   # noqa: E402
szutils = sys.modules["szifi.utils"]                  # SZiFi's utils module (the package attribute is pixell's)

MAP = os.path.join(OUT, "maps", "baseline_real_000.npy")
STORED = os.path.join(ROOT, "data_real", "step3_szifi_real", "runs", "S1_baseline_real_000.npz")
Z.STDD = os.path.join(OUT, "stdout")                  # keep every file this run writes out of the build's folders
Z.RUND = os.path.join(OUT, "runs")
assert os.path.exists(MAP), "run p00_map.py first"
with np.load(STORED) as d:
    st = {k: d[k] for k in d.files}

pr = cProfile.Profile()
pr.enable()
r = Z.run_tile("S1", "baseline", "real", 0, float(st["buffer_arcmin"]), map_file=MAP, save=False, tag="p01")
pr.disable()
T = r["t_run"]
print(f"SZiFi run (S1, tile 0, baseline, edge buffer {float(st['buffer_arcmin']):.0f}') under cProfile: wall {T:.1f} s; "
      f"peak RSS {r['peak_rss_gb']:.2f} GB")

# (1) same results as the build's stored run?
keys = [k for k in st if k.startswith(("it0_", "it1_"))]
same = [k for k in keys if k in r and np.array_equal(np.asarray(st[k]), np.asarray(r[k]), equal_nan=True)]
print(f"(1) against the build's stored S1 run of the same tile: {len(same)} of {len(keys)} catalogue / sigma arrays "
      f"identical; q > 4 detections {r['it0_q'].size} (iteration 0) and {r['it1_q'].size} (iteration 1), "
      f"stored {st['it0_q'].size} and {st['it1_q'].size}")

S = pstats.Stats(pr).stats                            # (file, line, name) -> (cc, nc, tt, ct, callers)


def key(fn):
    fn = getattr(fn, "__func__", fn)
    c = fn.__code__
    return (c.co_filename, c.co_firstlineno, c.co_name)


def calls_cum(fn, caller=None):
    """(calls, cumulative s) of fn, in total or only for the calls made by caller."""
    v = S.get(key(fn))
    if v is None:
        return 0, 0.0
    if caller is None:
        return v[1], v[3]
    c = v[4].get(key(caller))
    return (c[1], c[3]) if c else (0, 0.0)


def own(pred):
    n, t = 0, 0.0
    for (f, _, fn), (cc, nc, tt, ct, _) in S.items():
        if pred(f, fn):
            n += nc
            t += tt
    return n, t


O = Z._orig                                            # SZiFi's own functions behind the driver's timers
FIND, QMAP = O["find_clusters"], Z.mmf.get_mmf_q_map
REC = {k: float(np.sum(r["rec_" + k])) for k in ("t_filter", "t_cross", "t_invcov", "t_inpaint", "t_blend", "t_mask")}
rows = [
    ("filtering + peak finding, 15 scales x 2 iterations (filter_maps.find_clusters)", calls_cum(FIND), REC["t_filter"]),
    ("  cluster templates (gnfw.get_t_map_convolved)", calls_cum(Z.model.gnfw.get_t_map_convolved, FIND), None),
    ("  band limit of each template and of the map (maps.filter_tmap)", calls_cum(Z.maps.filter_tmap, FIND), None),
    ("  matched filter and convolution (mmf.get_mmf_q_map)", calls_cum(QMAP, FIND), None),
    ("    template FFTs (maps.get_fft_f, from get_mmf_q_map)", calls_cum(Z.maps.get_fft_f, QMAP), None),
    ("    Fourier-space filter product C^-1 T (mmf.get_tem_conv_fft)", calls_cum(Z.mmf.get_tem_conv_fft, QMAP), None),
    ("      of which per-mode 6x6 matrix-vector products (utils.get_inv_cov_conjugate)", calls_cum(szutils.get_inv_cov_conjugate), None),
    ("    inverse FFTs of the filters (maps.get_ifft_f, from get_mmf_q_map)", calls_cum(Z.maps.get_ifft_f, QMAP), None),
    ("    SED weighting (maps.get_tmap_times_fvec, from get_mmf_q_map)", calls_cum(Z.maps.get_tmap_times_fvec, QMAP), None),
    ("    zero-padded convolutions, one per channel (scipy.signal.fftconvolve)", calls_cum(scipy.signal.fftconvolve, QMAP), None),
    ("noise re-estimate: spectra (cross_spec.get_cross_spec)", calls_cum(O["get_cross_spec"]), REC["t_cross"]),
    ("noise re-estimate: inverse covariance (cross_spec.get_inv_cov)", calls_cum(O["get_inv_cov"]), REC["t_invcov"]),
    ("  of which interpolation to the 2D grid (spec.interp_cov)", calls_cum(Z.spec.interp_cov), None),
    ("masking: cluster mask (mmf.get_cluster_mask)", calls_cum(O["get_cluster_mask"]), REC["t_mask"]),
    ("masking: inpainting (maps.inpaint_freq)", calls_cum(O["inpaint_freq"]), REC["t_inpaint"]),
    ("masking: blending the holes back (maps.circular_cosine_blend_adaptive)", calls_cum(O["blend"]), REC["t_blend"]),
]
print(f"\n(2) time per stage in this run of {T:.1f} s: cProfile's cumulative time and share, and the driver's own "
      f"wall-clock timer of the same stage where it has one")
print(f"  {'stage':80s} {'calls':>6s} {'cProfile [s]':>12s} {'share':>6s} {'timer [s]':>9s}")
for name, (n, t), rec in rows:
    tm = f"{rec:9.1f}" if rec is not None else "        -"
    print(f"  {name:80s} {n:6d} {t:12.1f} {t / T * 100:5.0f}% {tm}")
covered = calls_cum(FIND)[1] + sum(calls_cum(O[k])[1] for k in ("get_cross_spec", "get_inv_cov", "get_cluster_mask",
                                                                 "inpaint_freq", "blend"))
print(f"  {'the stages above together':80s} {'':6s} {covered:12.1f} {covered / T * 100:5.0f}%")

# (3) FFTs and other repeated work
n_np, _ = own(lambda f, fn: f.endswith("numpy/fft/_pocketfft.py") and fn in ("fft2", "ifft2"))
_, t_np = own(lambda f, fn: f.endswith("numpy/fft/_pocketfft.py") and fn == "_raw_fft")
n_sc, t_sc = own(lambda f, fn: f == "~" and "_duccfft" in fn and any(s in fn for s in (".r2c", ".c2r", ".c2c")))
print(f"\n(3) 2D Fourier transforms in numpy and scipy (NaMaster's own are not counted): {n_np} with numpy (fft2 / ifft2) and "
      f"{n_sc} with scipy inside fftconvolve (zero-padded grids), {n_np + n_sc} together; time in them {t_np:.1f} + {t_sc:.1f} = {t_np + t_sc:.1f} s "
      f"({(t_np + t_sc) / T * 100:.0f}% of the run)")
n_ein, t_ein = own(lambda f, fn: f == "~" and "c_einsum" in fn)
print(f"    numpy.einsum kernel: {n_ein} calls, {t_ein:.1f} s ({t_ein / T * 100:.0f}%)")
n_inv, t_inv = own(lambda f, fn: f.endswith("numpy/linalg/_linalg.py") and fn == "inv")
print(f"    numpy.linalg.inv: {n_inv} calls, {t_inv:.1f} s")
n_nmt, t_nmt = own(lambda f, fn: f == "~" and "_nmtlib" in fn)
print(f"    NaMaster (pymaster C library): {n_nmt} calls, {t_nmt:.1f} s")
print("    repeated work (parts of the stage times above, not in addition to them):")
for label, fn in (("the l grid of the tile, recomputed (maps.rmap.get_ell)", Z.maps.rmap.get_ell),
                  ("distance maps from the tile centre (maps.rmap.get_distance_map_wrt_centre)", Z.maps.rmap.get_distance_map_wrt_centre),
                  ("beam convolutions of templates (maps.get_gaussian_convolution)", Z.maps.get_gaussian_convolution),
                  ("masked discs, one per detection (maps.ps_mask.get_mask_map)", Z.maps.ps_mask.get_mask_map)):
    n, t = calls_cum(fn)
    print(f"      {label}: {n} calls, {t:.1f} s cumulative")

# (4) largest own times
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
print("\n(4) the 25 functions with the largest own time (pstats; paths shortened)")
started = False
for line in s.getvalue().splitlines():
    if line.strip().startswith("ncalls"):
        started = True
    if started and line.strip():
        for pre in ("/Users/boris/GitHub/szifi/", os.path.dirname(os.path.dirname(np.__file__)) + "/", ROOT + "/"):
            line = line.replace(pre, "")
        print("  " + line[:170])
