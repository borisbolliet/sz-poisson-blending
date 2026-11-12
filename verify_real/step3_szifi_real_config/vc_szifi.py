"""The config verifier's own minimal driver for SZiFi itself (clone /Users/boris/GitHub/szifi at 73626fb, imported
unchanged, never written to).  Written from scratch from reading SZiFi's code; it does not import the build's
z_szlib.py or z_survey.py.  Run with /Users/boris/GitHub/szifi-venv/bin/python -I -B.

What it adds around SZiFi:
  * parameter dictionaries: SZiFi's params_szifi_default (deep copy) updated with the settings below, written from
    the task text and SZiFi's code (S0 no inpainting; S1 SZiFi's iteration defaults; S2 lsep 12000; S3 the SO example
    test_files/run_szifi_so_car.py where it applies to flat periodic tiles; S3i S3 with an isotropic covariance;
    '...it0' = iterative False);
  * data: our own survey file vc_survey.py;
  * three pass-through recorders (results returned unchanged): spec.cross_spec.get_cross_spec (SZiFi's binned 6x6
    spectra of each noise iteration, after its inpainting correction), mmf.get_cluster_mask (mean of the cluster mask =
    unmasked fraction, and the number of masked pixels), and spec.cross_spec.get_inv_cov, which at noise iteration 1
    first drops the cluster finder's two references to the iteration-0 inverse covariance (memory only: SZiFi
    overwrites both right after this call and reads neither in between).
"""
import os, sys, io, copy, json, time, contextlib, resource
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SZIFI = "/Users/boris/GitHub/szifi"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
if SZIFI not in sys.path:
    sys.path.insert(0, SZIFI)
sys.path.insert(0, HERE)
import numpy as np                                    # noqa: E402
import szifi                                          # noqa: E402
from szifi import mmf, spec                           # noqa: E402
import vc_model as M                                  # noqa: E402

MAPD = os.path.join(HERE, "out", "maps")
RUND = os.path.join(HERE, "out", "runs")
STDD = os.path.join(HERE, "out", "stdout")
SURVEY = os.path.join(HERE, "vc_survey.py")
CM_FILE = os.path.join(HERE, "out", "cm_ones_556_bin4.fits")    # our own NaMaster coupling matrix (SZiFi writes it)

COMMON = {"theta_500_vec_arcmin": M.THETAS.copy(), "beam": "gaussian", "integrate_bandpass": False,
          "get_lonlat": False, "save_and_load_template": False, "snr_weighting": False,
          "survey_file": SURVEY, "path": SZIFI + "/", "path_data": MAPD + "/", "path_template": None}
CFG = {
    "S0": {"cov_type": "isotropic", "decouple_type": "none", "coupling_matrix_needed": False,
           "lrange": [100, 20000], "inpaint": False},
    "S1": {"cov_type": "isotropic", "decouple_type": "none", "coupling_matrix_needed": False,
           "lrange": [100, 20000], "inpaint": True, "inpaint_type": "diffusive", "n_inpaint": 100, "lsep": 3000},
    "S2": {"cov_type": "isotropic", "decouple_type": "none", "coupling_matrix_needed": False,
           "lrange": [100, 20000], "inpaint": True, "inpaint_type": "diffusive", "n_inpaint": 100, "lsep": 12000},
    "S3": {"cov_type": "mixed_gaussian", "cov_kernel_shape": [10, 10], "decouple_type": "master",
           "coupling_matrix_needed": True, "compute_coupling_matrix": False, "save_coupling_matrix": True,
           "powspec_lmax1d": 10000, "powspec_bin_fac": 4, "lrange": [1, 300000], "inpaint": True,
           "inpaint_type": "orphics", "n_inpaint": -1, "lsep": 12000, "min_radius_mask_arcmin": 1.},
}
CFG["S3i"] = dict(CFG["S3"], cov_type="isotropic")
for _k in list(CFG):
    CFG[_k + "it0"] = dict(CFG[_k], iterative=False)
# 'w' variants (diagnosis only): l range [100, 30600], i.e. every mode of the 1200^2 grid (max l 30547) in band, so
# that no binned spectrum is empty of power (with [100, 20000] the bins above l = 20016 and the grid corners hold
# only round-off of the l-filtered maps)
for _k in ("S1", "S1it0"):
    CFG[_k.replace("S1", "S1w")] = dict(CFG[_k], lrange=[100, 30600])
# 'g' variants: the same configuration with the 15 scales written as np.geomspace(0.5, 15, 15) (the build's and the
# production finder's THETAS15), which differs from exp(linspace(log 0.5, log 15, 15)) (SZiFi's own default form,
# used above) in the last bit of 6 of the 15 values.
for _k in list(CFG):
    CFG[_k + "g"] = dict(CFG[_k], theta_500_vec_arcmin=np.geomspace(0.5, 15.0, 15))
# 'f64' variants: maps and SZiFi's map_dtype in float64 instead of SZiFi's default float32 (tests round-off)
for _k in list(CFG):
    CFG[_k + "f64"] = dict(CFG[_k], map_dtype=np.float64)


def make_params(cfg, nch=6):
    p = copy.deepcopy(szifi.params_szifi_default)
    p.update(copy.deepcopy(COMMON))
    p.update(copy.deepcopy(CFG[cfg]))
    p["freqs"] = list(range(nch))
    return p


# ------------------------------------------------------------------ pass-through recorders
REC = {}
_orig = {"cross": spec.cross_spec.get_cross_spec, "invcov": spec.cross_spec.get_inv_cov, "mask": mmf.get_cluster_mask}


def _cross(self, *a, **k):
    out = _orig["cross"](self, *a, **k)
    REC.setdefault("ell", []).append(np.array(out[0], float))
    REC.setdefault("spec", []).append(np.array(out[1], float))
    return out


def _invcov(self, *a, **k):
    cf = REC.get("cf")
    if cf is not None and getattr(cf, "filtered_maps", None) is not None:      # noise iteration >= 1
        cf.inv_cov = None
        cf.filtered_maps.inv_cov = None
        REC["released"] = REC.get("released", 0) + 1
    return _orig["invcov"](self, *a, **k)


def _mask(*a, **k):
    m = _orig["mask"](*a, **k)
    REC.setdefault("fsky", []).append(float(np.mean(m)))
    REC.setdefault("n_masked_pix", []).append(int(np.sum(m == 0)))
    return m


spec.cross_spec.get_cross_spec = _cross
spec.cross_spec.get_inv_cov = _invcov
mmf.get_cluster_mask = _mask


def to_index(theta_x, theta_y):
    """SZiFi's catalogue coordinates -> (row i, column j) of the pixel it reports.  From mmf.filter_maps.find_clusters
    and maps.rmap: theta_x = (j + 1/2) dx, theta_y = nx dx - (i + 1/2) dx."""
    j = np.asarray(theta_x, float) / M.DX - 0.5
    i = M.N - 0.5 - np.asarray(theta_y, float) / M.DX
    ji, ii = np.rint(j).astype(np.int64), np.rint(i).astype(np.int64)
    if ji.size:
        assert np.abs(j - ji).max() < 1e-6 and np.abs(i - ii).max() < 1e-6
    return ii, ji


def run(cfg, map_name, mode, buffer_arcmin=8., channels=None, tag=None, seed=None, print_params=False):
    """One SZiFi cluster_finder run on out/maps/<map_name>.npy; returns and saves out/runs/<tag>.npz."""
    ch = list(range(6)) if channels is None else list(channels)
    p = make_params(cfg, len(ch))
    fid = map_name
    tag = tag or f"{cfg}_{map_name}"
    pdata = {"data_set": "verifier", "field_ids": [fid], "map_file": os.path.join(MAPD, map_name + ".npy"),
             "channels": ch, "noise_level": mode, "buffer_arcmin": float(buffer_arcmin),
             "coupling_matrix_name": CM_FILE if p["decouple_type"] == "master" else None}
    REC.clear()
    os.makedirs(STDD, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data = szifi.input_data(params_szifi=p, params_data=pdata)
        cf = szifi.cluster_finder(params_szifi=p, params_model=copy.deepcopy(szifi.params_model_default),
                                  data_file=data, rank=0)
        REC["cf"] = cf
        cf.find_clusters()
        REC["cf"] = None
    trun = time.time() - t0
    with open(os.path.join(STDD, tag + ".txt"), "w") as fh:
        fh.write(buf.getvalue())
    res = cf.results_dict[fid]
    out = dict(cfg=cfg, map_name=map_name, mode=mode, buffer_arcmin=float(buffer_arcmin), channels=np.array(ch),
               t_run=trun, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
               params_json=json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else
                                           (v.__name__ if isinstance(v, type) else
                                            (str(v) if isinstance(v, float) and not np.isfinite(v) else v)))
                                       for k, v in p.items()}),
               released=REC.get("released", 0))
    n_it = 0
    for it in (0, 1):
        key = f"catalogue_find_{it}"
        if key not in res.catalogues:
            continue
        n_it = it + 1
        c = res.catalogues[key].catalogue
        q = np.asarray(c["q_opt"], float)
        i, j = to_index(c["theta_x"], c["theta_y"])
        th = np.asarray(c["theta_500"], float)
        jt = np.abs(np.log(th)[:, None] - np.log(M.THETAS)[None, :]).argmin(1) if q.size else np.zeros(0, np.int64)
        out.update({f"it{it}_q": q, f"it{it}_y0": np.asarray(c["y0"], float), f"it{it}_theta": th,
                    f"it{it}_jt": jt, f"it{it}_i": i, f"it{it}_j": j,
                    f"it{it}_sigvec": np.asarray(res.sigma_vec[f"find_{it}"], float)})
    out["n_it"] = n_it
    for k, (e, s) in enumerate(zip(REC.get("ell", []), REC.get("spec", []))):
        out[f"spec{k}_ell"], out[f"spec{k}"] = e, s
    out["n_spec"] = len(REC.get("spec", []))
    out["fsky"] = np.array(REC.get("fsky", []))
    out["n_masked_pix"] = np.array(REC.get("n_masked_pix", []))
    os.makedirs(RUND, exist_ok=True)
    np.savez(os.path.join(RUND, tag + ".npz"), **out)
    if print_params:
        print(f"--- full parameter dictionary of {tag} (after the run; SZiFi adds a_matrix and cmmf_type in place)")
        for k in sorted(p):
            v = p[k]
            v = np.array2string(v, precision=6, max_line_width=250) if isinstance(v, np.ndarray) else v
            print(f"    {k:26s} = {v}")
    del cf, data, res
    return out
