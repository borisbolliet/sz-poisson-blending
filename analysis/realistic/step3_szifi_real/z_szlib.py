"""Driver library for running SZiFi itself (github.com/inigozubeldia/szifi, commit 73626fb, the clone at
/Users/boris/GitHub/szifi, imported unchanged) on step 3a's tiles.  Own code; SZiFi is only imported and called.

Run with /Users/boris/GitHub/szifi-venv/bin/python -I -B.  Importing this module first sets sys.dont_write_bytecode,
NUMBA_CACHE_DIR (our out/ folder) and one thread for OMP / OpenBLAS / vecLib / numba, then puts the SZiFi clone on
sys.path.

What the driver adds around SZiFi (nothing inside SZiFi is changed):
  * params: SZiFi's default dictionary (deepcopy) updated with our settings (BASE) and the configuration (CONFIGS);
  * data: our survey file z_survey.py (SZiFi's input_data loads it);
  * recorders: thin wrappers around three SZiFi functions that only record what passes through them, so that SZiFi's
    own intermediate results can be stored: spec.cross_spec.get_cross_spec (the binned 6 x 6 spectra of each noise
    iteration, after SZiFi's inpainting correction), mmf.get_cluster_mask (unmasked fraction and number of masked
    detections) and the wall time of SZiFi's main steps.  Their return values are passed on unchanged.
  * 'fof' merging (cat.merge_detections mode 'fof') calls sklearn's AgglomerativeClustering(affinity=...); scikit-learn
    removed that keyword in 1.4 (installed: 1.9.1, where it is called 'metric').  sklearn_affinity_adapter() puts, in
    szifi.cat's namespace only, a factory in its place that passes affinity= on as metric= and returns scikit-learn's own
    estimator (same callable, same meaning: a callable metric is applied to the coordinates and returns the distance
    matrix).
  * SZiFi's merging also stores healpy's one-pair angular distance (shape (1,)) into a scalar, which numpy 2.5 refuses;
    healpy_angdist_adapter() returns that one-element result as a scalar in this process.
"""
import os, sys, time, copy, json, io, resource, contextlib, gc
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
SZIFI = "/Users/boris/GitHub/szifi"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "out", "numba_cache_szifi")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
if SZIFI not in sys.path:
    sys.path.insert(0, SZIFI)
import numpy as np                                     # noqa: E402
import szifi                                           # noqa: E402
from szifi import mmf, spec, maps, model, cat, expt   # noqa: E402

N = 1200
DX = 0.5 / 60. / 180. * np.pi
MAPD = os.path.join(OUT, "out", "maps")
RUND = os.path.join(OUT, "runs")
STDD = os.path.join(OUT, "out", "stdout")
SURVEY = os.path.join(HERE, "z_survey.py")
INSTRUMENT = os.path.join(OUT, "instrument.npz")
THETAS15 = np.geomspace(0.5, 15.0, 15)                 # szsim.finder.THETAS15 (checked against reference.npz below)
CM_FILE = os.path.join(OUT, "coupling_ones_bin04_lmax10000.fits")
_ref = np.load(os.path.join(OUT, "reference.npz"))
assert np.array_equal(_ref["thetas"], THETAS15)

# ------------------------------------------------------------------ configurations
BASE = {
    "theta_500_vec_arcmin": THETAS15,
    "q_th": 4.0, "q_th_noise": 4.0, "mask_radius": 3.0, "iterative": True, "max_it": 1,
    "beam": "gaussian", "integrate_bandpass": False, "get_lonlat": False, "save_and_load_template": False,
    "detection_method": "maxima_lomem", "extraction_mode": "find", "freqs": [0, 1, 2, 3, 4, 5],
    "snr_weighting": False,              # read by cluster_finder (its default dictionary spells it 'snr_weigthing')
    "survey_file": SURVEY, "path": SZIFI + "/", "path_data": MAPD + "/", "path_template": None,
}
CONFIGS = {
    "S0": {"cov_type": "isotropic", "powspec_bin_fac": 4, "powspec_lmax1d": None, "decouple_type": "none",
           "coupling_matrix_needed": False, "lrange": [100, 20000], "inpaint": False},
    "S1": {"cov_type": "isotropic", "powspec_bin_fac": 4, "powspec_lmax1d": None, "decouple_type": "none",
           "coupling_matrix_needed": False, "lrange": [100, 20000], "inpaint": True, "inpaint_type": "diffusive",
           "n_inpaint": 100, "lsep": 3000, "min_radius_mask_arcmin": 0.0},
    "S2": {"cov_type": "isotropic", "powspec_bin_fac": 4, "powspec_lmax1d": None, "decouple_type": "none",
           "coupling_matrix_needed": False, "lrange": [100, 20000], "inpaint": True, "inpaint_type": "diffusive",
           "n_inpaint": 100, "lsep": 12000, "min_radius_mask_arcmin": 0.0},
    "S3": {"cov_type": "mixed_gaussian", "cov_kernel_shape": [10, 10], "decouple_type": "master",
           "coupling_matrix_needed": True, "compute_coupling_matrix": False, "save_coupling_matrix": True,
           "powspec_lmax1d": 10000, "powspec_bin_fac": 4, "lrange": [1, 300000], "inpaint": True,
           "inpaint_type": "orphics", "n_inpaint": -1, "lsep": 12000, "min_radius_mask_arcmin": 1.0,
           "apod_type": "old"},
}
CONFIGS["S3i"] = dict(CONFIGS["S3"], cov_type="isotropic")
CONFIGS["S3it0"] = dict(CONFIGS["S3"], iterative=False)      # S3, noise iteration 0 only (diagnosis)


def make_params(cfg):
    p = copy.deepcopy(szifi.params_szifi_default)
    p.update(copy.deepcopy(BASE))
    p.update(copy.deepcopy(CONFIGS[cfg]))
    return p


def params_model():
    return copy.deepcopy(szifi.params_model_default)


def differs_from_default(p):
    d = szifi.params_szifi_default
    out = {}
    for k, v in p.items():
        if k not in d:
            out[k] = v
            continue
        a, b = v, d[k]
        try:
            same = np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)) if not isinstance(a, np.ndarray) \
                else (isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b))
        except Exception:
            same = a is b
        if not same:
            out[k] = v
    return out


def jsonable(p):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, type):
            return v.__name__
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        return v
    return {k: conv(v) for k, v in p.items()}


def print_params(p, label, fh=None):
    """The full parameter dictionary, one entry per line (fh: file-like, default stdout)."""
    out = fh or sys.stdout
    print(f"--- full SZiFi parameter dictionary used: {label}", file=out)
    for k in sorted(p):
        v = p[k]
        if isinstance(v, np.ndarray):
            v = np.array2string(v, precision=6, max_line_width=200, threshold=50)
        print(f"    {k!s:28s} = {v}", file=out)


# ------------------------------------------------------------------ recorders (wrap SZiFi functions, pass results on)
REC = {}
GC_BEFORE = False      # (tested: gc.collect() before SZiFi's steps does not lower the peak memory; off)


def rec_reset():
    REC.clear()
    REC.update(spectra=[], ell=[], fsky=[], n_masked=[], t_cross=[], t_invcov=[], t_filter=[], t_inpaint=[],
               t_blend=[], t_mask=[])


rec_reset()
_orig = dict(get_cross_spec=spec.cross_spec.get_cross_spec, get_inv_cov=spec.cross_spec.get_inv_cov,
             find_clusters=mmf.filter_maps.find_clusters, get_cluster_mask=mmf.get_cluster_mask,
             inpaint_freq=maps.inpaint_freq, blend=maps.circular_cosine_blend_adaptive)


def _w_cross(self, *a, **k):
    if GC_BEFORE:
        gc.collect()
    t = time.time()
    out = _orig["get_cross_spec"](self, *a, **k)
    REC["t_cross"].append(time.time() - t)
    REC["ell"].append(np.array(out[0], dtype=float))
    REC["spectra"].append(np.array(out[1], dtype=float))
    return out


DROP_OLD_INV_COV = True   # memory only: see _w_invcov


def _w_invcov(self, *a, **k):
    # Memory only (the 4 GB cap): when SZiFi re-estimates the covariance at noise iteration 1, the cluster_finder
    # still holds the iteration-0 inverse covariance (1200 x 1200 x 6 x 6 float64, 415 MB) in cluster_finder.inv_cov
    # and in its previous filter_maps object.  Both attributes are overwritten right after this call (inv_cov by its
    # return value, filtered_maps by a new filter_maps built from it) and nothing reads them in between, so they are
    # released here first.  Results are unchanged (checked bit for bit in z02_smoke_S1b.log).
    cf = REC.get("cf")
    if DROP_OLD_INV_COV and cf is not None and getattr(cf, "filtered_maps", None) is not None:
        cf.inv_cov = None
        cf.filtered_maps.inv_cov = None
        REC["n_drop"] = REC.get("n_drop", 0) + 1
    if GC_BEFORE:
        gc.collect()
    t = time.time()
    out = _orig["get_inv_cov"](self, *a, **k)
    REC["t_invcov"].append(time.time() - t)
    return out


def _w_find(self, *a, **k):
    if GC_BEFORE:
        gc.collect()
    t = time.time()
    out = _orig["find_clusters"](self, *a, **k)
    REC["t_filter"].append(time.time() - t)
    return out


def _w_mask(pix, catalogue, q_th_noise, mask_radius, **k):
    t = time.time()
    m = _orig["get_cluster_mask"](pix, catalogue, q_th_noise, mask_radius, **k)
    REC["t_mask"].append(time.time() - t)
    REC["fsky"].append(float(np.mean(m)))
    REC["n_masked"].append(int(np.sum(np.asarray(catalogue.catalogue["q_opt"]) > q_th_noise)))
    return m


def _w_inpaint(*a, **k):
    t = time.time()
    out = _orig["inpaint_freq"](*a, **k)
    REC["t_inpaint"].append(time.time() - t)
    return out


def _w_blend(*a, **k):
    t = time.time()
    out = _orig["blend"](*a, **k)
    REC["t_blend"].append(time.time() - t)
    return out


spec.cross_spec.get_cross_spec = _w_cross
spec.cross_spec.get_inv_cov = _w_invcov
mmf.filter_maps.find_clusters = _w_find
mmf.get_cluster_mask = _w_mask
maps.inpaint_freq = _w_inpaint
maps.circular_cosine_blend_adaptive = _w_blend


def sklearn_affinity_adapter():
    """Let szifi.cat's fof merging run with scikit-learn >= 1.4 (affinity= renamed metric=): in szifi.cat's namespace
    only, AgglomerativeClustering becomes a factory that passes affinity= on as metric= and returns scikit-learn's own
    estimator (same callable, same meaning: it is applied to the coordinates and returns the distance matrix)."""
    from sklearn.cluster import AgglomerativeClustering as _AC
    if getattr(cat.AgglomerativeClustering, "_szlib_adapter", False):
        return cat.AgglomerativeClustering

    def AgglomerativeClustering(*args, affinity=None, **kw):
        if affinity is not None:
            kw["metric"] = affinity
        return _AC(*args, **kw)
    AgglomerativeClustering._szlib_adapter = True
    cat.AgglomerativeClustering = AgglomerativeClustering
    return AgglomerativeClustering


def healpy_angdist_adapter():
    """Let szifi.cat's merging run with numpy 2.5: SZiFi's sphere.get_angdist (and cat.get_distance_sphere_lonlat)
    store healpy.rotator.angdist's result for ONE pair of directions (an array of shape (1,)) into a scalar slot, which
    numpy >= 2.x refuses ('only 0-dimensional arrays can be converted to Python scalars').  This wraps
    healpy.rotator.angdist in this process so that a one-element result is returned as a scalar (same value)."""
    import healpy as hp
    orig = hp.rotator.angdist
    if getattr(orig, "_szlib_adapter", False):
        return orig

    def angdist(dir1, dir2, lonlat=False):
        out = np.asarray(orig(dir1, dir2, lonlat=lonlat))
        return out.reshape(-1)[0] if out.size == 1 else out
    angdist._szlib_adapter = True
    hp.rotator.angdist = angdist
    return angdist


# ------------------------------------------------------------------ one SZiFi run on one tile
def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def cat_arrays(c):
    """SZiFi catalogue -> arrays; pixel indices from SZiFi's own coordinates (theta_x = (j + 1/2) dx, theta_y =
    (nx - i - 1/2) dx, mmf.filter_maps.find_clusters), i = row (our iy), j = column (our ix)."""
    d = c.catalogue
    q = np.asarray(d["q_opt"], float)
    tx = np.asarray(d["theta_x"], float); ty = np.asarray(d["theta_y"], float)
    jj = tx / DX - 0.5
    ii = N - 0.5 - ty / DX
    j = np.rint(jj).astype(np.int64); i = np.rint(ii).astype(np.int64)
    if q.size:
        assert np.abs(jj - j).max() < 1e-6 and np.abs(ii - i).max() < 1e-6
    th = np.asarray(d["theta_500"], float)
    jth = np.abs(np.log(th)[:, None] - np.log(THETAS15)[None, :]).argmin(1) if q.size else np.zeros(0, np.int64)
    return dict(q=q, y0=np.asarray(d["y0"], float), theta=th, j_theta=jth, i=i, j=j, theta_x=tx, theta_y=ty)


def run_tile(cfg, mode, kind, pid, buffer_arcmin, map_file=None, save=True, params_fh=None, tag=None):
    """Run SZiFi's cluster_finder (two noise iterations) on one tile.  Returns a dict of SZiFi's outputs and the
    recorded intermediates; saves it to runs/{tag or cfg}_{mode}_{kind}{pid:03d}.npz."""
    tag = tag or cfg
    fid = f"{mode}_{kind}_{pid:03d}"
    map_file = map_file or os.path.join(MAPD, f"{fid}.npy")
    p = make_params(cfg)
    pdata = {"data_set": "sz-poisson-blending step 3a tiles", "field_ids": [fid], "map_files": {fid: map_file},
             "noise_level": mode, "instrument_file": INSTRUMENT, "buffer_arcmin": float(buffer_arcmin),
             "coupling_matrix_name": CM_FILE if p["decouple_type"] == "master" else None}
    rec_reset()
    os.makedirs(STDD, exist_ok=True)
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if cfg.startswith("S3"):
            np.random.seed(1000003 * (1 + ("goal" == mode)) + 7919 * pid + (500 if kind == "noise" else 0))
        data = szifi.input_data(params_szifi=p, params_data=pdata)
        t_load = time.time() - t0
        cf = szifi.cluster_finder(params_szifi=p, params_model=params_model(), data_file=data, rank=0)
        REC["cf"] = cf
        cf.find_clusters()
        REC["cf"] = None
    t_run = time.time() - t0
    with open(os.path.join(STDD, f"{tag}_{fid}.txt"), "w") as fh:
        fh.write(buf.getvalue())
    res = cf.results_dict[fid]
    out = dict(cfg=cfg, tag=tag, mode=mode, kind=kind, pid=pid, buffer_arcmin=float(buffer_arcmin),
               params_json=json.dumps(jsonable(p)), t_load=t_load, t_run=t_run, peak_rss_gb=peak_rss_gb())
    n_it = 0
    for it in (0, 1):
        key = f"catalogue_find_{it}"
        if key in res.catalogues:
            n_it = it + 1
            ca = cat_arrays(res.catalogues[key])
            for k, v in ca.items():
                out[f"it{it}_{k}"] = v
            sv = np.asarray(res.sigma_vec[f"find_{it}"], float)
            out[f"it{it}_sigvec"] = sv
            # SZiFi's 'maxima_lomem' branch skips sigma_vec[0] (it 'continue's at the first scale), so sigma_vec[0]
            # stays 0.  sigma at every scale is also y0 / q of any detection at that scale (q = y / sigma in
            # mmf.get_mmf_q_map): median over the detections, from SZiFi's own catalogue (float32 maps).
            sc = np.full(THETAS15.size, np.nan)
            for jj in range(THETAS15.size):
                m = ca["j_theta"] == jj
                if m.any():
                    sc[jj] = np.median(ca["y0"][m] / ca["q"][m])
            out[f"it{it}_sig_cat"] = sc
            sig = sv.copy()
            sig[0] = sc[0]
            out[f"it{it}_sig"] = sig
    out["n_it"] = n_it
    out["n_drop_old_inv_cov"] = REC.get("n_drop", 0)
    for k in ("fsky", "n_masked", "t_cross", "t_invcov", "t_filter", "t_inpaint", "t_blend", "t_mask"):
        out["rec_" + k] = np.asarray(REC[k], float)
    for k, (e, s) in enumerate(zip(REC["ell"], REC["spectra"])):
        out[f"spec{k}_ell"] = e
        out[f"spec{k}"] = s
    out["n_spec"] = len(REC["spectra"])
    if params_fh is not None:
        print_params(p, f"{tag} (after the run on {fid}; cluster_finder adds a_matrix and sets cmmf_type in place)",
                     params_fh)
    if save:
        os.makedirs(RUND, exist_ok=True)
        np.savez(os.path.join(RUND, f"{tag}_{fid}.npz"), **out)
    del cf, data, res
    return out


# ------------------------------------------------------------------ analytic sigma with SZiFi's own templates
def analytic_inv_cov(mode, shape):
    """Inverse of the analytic CMB + noise covariance of our skies (z00_prep.py, uK^2 sr) on a grid of SZiFi's FFT
    layout (fftfreq order on both axes) with the 1200^2 tile's fundamental; shape = (n, n), n = 1200 or the degraded
    size of powspec_lmax1d.  The l = 0 entry is zero (the templates vanish there)."""
    sh = np.load(os.path.join(OUT, "analytic_shells.npz"))
    n2t, C = sh["n2"], sh[mode]
    Ci = np.linalg.inv(C)
    Ci[0] = 0.0
    n = shape[0]
    i0 = np.rint(np.fft.fftfreq(n) * n).astype(np.int64)
    n2 = i0[:, None] ** 2 + i0[None, :] ** 2
    idx = np.searchsorted(n2t, n2)
    assert np.array_equal(n2t[idx], n2)
    return Ci[idx]


def analytic_sigma(cfg, mode):
    """sigma(theta) [SZiFi's y0 units] that SZiFi's own filter (templates, beams, l range, lmax1d cut) has with the
    true CMB + noise covariance: mmf.filter_maps.find_clusters on zero maps with the analytic inverse covariance."""
    p = make_params(cfg)
    p["detection_method"] = "maxima"    # this branch of find_clusters sets sigma_vec at every scale (no detections here)
    pm = params_model()
    inst = np.load(INSTRUMENT)
    e = expt.experiment(experiment_name="SOgoal_simple", params_szifi=p)
    e.nu_eff = np.asarray(inst["nu_eff_Hz"], float); e.FWHM = np.asarray(inst["fwhm_arcmin"], float)
    e.noise_levels = np.asarray(inst[f"white_{mode}"], float); e.tsz_sed = np.asarray(inst["tsz_sed_uK"], float)
    e.n_freqs = len(e.nu_eff)
    a = np.zeros((6, 1)); a[:, 0] = e.tsz_sed
    p["a_matrix"] = a
    p["cmmf_type"] = "standard_mmf"
    pix = maps.pixel(N, DX)
    shape = (N, N)
    if p["powspec_lmax1d"] is not None:
        shape = maps.get_newshape_lmax1d((N, N), p["powspec_lmax1d"], DX)
    icov = analytic_inv_cov(mode, shape)
    ones = np.ones((N, N), dtype=np.float32)
    cmmf = mmf.scmmf_precomputation(pix=pix, freqs=p["freqs"], inv_cov=icov, lrange=p["lrange"], beam_type=p["beam"],
                                    exp=e, cmmf_type="standard_mmf", a_matrix=a, comp_to_calculate=[0],
                                    mmf_type="standard")
    fm = mmf.filter_maps(t_obs=np.zeros((N, N, 6), dtype=np.float32), inv_cov=icov, pix=pix,
                         cosmology=model.cosmological_model(p).cosmology, theta_500_vec=p["theta_500_vec_arcmin"],
                         params=p, field_id="analytic", i_it=0, mask_map=ones,
                         mask_select_dict={"field": ones, "tile": ones, "names": {"field": ["field"], "tile": ["tile"]}},
                         mask_peak_finding_dict={"field": ones, "tile": ones, "names": ["field", "tile"]},
                         rank=0, exp=e, cmmf=cmmf, params_model=pm)
    with contextlib.redirect_stdout(io.StringIO()):
        fm.find_clusters()
    return np.asarray(fm.sigma_vec, float), shape
