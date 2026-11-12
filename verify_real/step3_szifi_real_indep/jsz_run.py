"""The verifier's own driver: one SZiFi run (two noise iterations unless stated) on one tile map (lens 'indep').

Usage (SZiFi's venv): /Users/boris/GitHub/szifi-venv/bin/python -I -B jsz_run.py CFG MODE KIND PID [norelease]
  CFG  : S1 (SZiFi's iteration defaults), S2 (S1 with l_sep 12000), S3 (SZiFi's SO example settings), S3i (S3 with
         an isotropic covariance),
         S3it0 (S3, iteration 0 only), S0 (no inpainting, no edge buffer)
  MODE : baseline | goal;  KIND : real | noise;  PID : tile id (map out/maps/KIND_MODE_PID.npy from j01_maps.py)
SZiFi (/Users/boris/GitHub/szifi, commit 73626fb) is imported unchanged and run through its own input_data and
cluster_finder.find_clusters.  Around it, own code only:
  * recorders: the std returned by each mmf.get_mmf_q_map call (sigma at every scale, also the first one, which
    SZiFi's 'maxima_lomem' branch never stores), the spectra each spec.cross_spec.get_cross_spec call returns, and the
    mean of the mask mmf.get_cluster_mask returns (the unmasked fraction f); results are passed on unchanged;
  * memory: before SZiFi computes the iteration-1 inverse covariance, the two references it still holds to the old
    one (cluster_finder.inv_cov, cluster_finder.filtered_maps.inv_cov) are dropped; both are reassigned right after
    and never read in between (mmf.py find_clusters).  'norelease' switches this off.
Output: out/runs/CFG_MODE_KIND_PID.npz (catalogues of both iterations with pixel indices, sigma, spectra, f, params).
"""
import os, sys, time, json, copy, io, contextlib, resource
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
SZIFI = "/Users/boris/GitHub/szifi"
if SZIFI not in sys.path:
    sys.path.insert(0, SZIFI)
import numpy as np
import szifi                                   # noqa: E402
from szifi import mmf, spec                    # noqa: E402

N = 1200
DXR = 0.5 / 60.0 / 180.0 * np.pi
THETAS = np.geomspace(0.5, 15.0, 15)           # step 3a's 15 scales


def params_for(cfg):
    """SZiFi's default dictionary with the settings of each configuration (the build's stated table)."""
    p = copy.deepcopy(szifi.params.params_szifi_default)
    p.update({"theta_500_vec_arcmin": THETAS.copy(), "beam": "gaussian", "integrate_bandpass": False,
              "get_lonlat": False, "save_and_load_template": False, "path_template": None,
              "snr_weighting": False,   # cluster_finder reads this key; SZiFi's defaults spell it 'snr_weigthing'
              "path": SZIFI + "/", "path_data": os.path.join(OUT, "maps") + "/",
              "survey_file": os.path.join(HERE, "jsz_survey.py")})
    if cfg in ("S0", "S1", "S2"):
        p.update({"lrange": [100, 20000], "decouple_type": "none", "coupling_matrix_needed": False})
        if cfg == "S0":
            p["inpaint"] = False
        if cfg == "S2":
            p["lsep"] = 12000
    elif cfg in ("S3", "S3it0", "S3i"):
        # test_files/run_szifi_so_car.py: everything that applies to a flat 'healpix'-type tile with our channels
        p.update({"lrange": [1, 300000], "decouple_type": "master", "coupling_matrix_needed": True,
                  "save_coupling_matrix": True, "compute_coupling_matrix": False, "powspec_lmax1d": 10000,
                  "cov_type": "mixed_gaussian", "cov_kernel_shape": [10, 10], "inpaint_type": "orphics",
                  "n_inpaint": -1, "lsep": 12000, "min_radius_mask_arcmin": 1.0})
        if cfg == "S3it0":
            p["iterative"] = False
        if cfg == "S3i":                     # the build's S3i: S3 with SZiFi's isotropic covariance
            p["cov_type"] = "isotropic"
    else:
        raise ValueError(cfg)
    return p


def jsonable(p):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and np.isinf(v):
            return "inf"
        if isinstance(v, type):
            return v.__name__
        return v
    return {k: conv(v) for k, v in p.items()}


def main():
    cfg, mode, kind, pid = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    release = not (len(sys.argv) > 5 and sys.argv[5] == "norelease")
    tag = f"{cfg}_{mode}_{kind}_{pid:03d}" + ("" if release else "_norelease")
    ref = np.load(os.path.join(OUT, "j01_ref.npz"))
    params = params_for(cfg)
    params_data = {"data_set": "sz-poisson-blending step 3a tiles", "field_ids": [pid],
                   "j_map": os.path.join(OUT, "maps", f"{kind}_{mode}_{pid:03d}.npy"),
                   "j_buffer": 0 if cfg == "S0" else 16,
                   "j_fwhm": ref["fwhm_arcmin"], "j_white": ref[f"white_{mode}"], "j_sed": ref["tsz_sed_uK"],
                   "j_nu": ref["freqs_GHz"] * 1e9,
                   "j_cm_name": os.path.join(OUT, "coupling_ones_bin04_lmax10000.fits")}
    REC = {"std": [], "t_std": [], "spec": [], "ell": [], "fsky": [], "n_release": 0}
    HOLD = {}
    o_q, o_cs, o_cm, o_inv = mmf.get_mmf_q_map, spec.cross_spec.get_cross_spec, mmf.get_cluster_mask, spec.cross_spec.get_inv_cov

    def q_rec(*a, **k):
        out = o_q(*a, **k)
        REC["std"].append(float(out[2]))
        REC["t_std"].append(time.time())
        return out

    def cs_rec(self_cs, *a, **k):
        ell, st = o_cs(self_cs, *a, **k)
        REC["ell"].append(np.array(ell, copy=True))
        REC["spec"].append(np.array(st, copy=True))
        return ell, st

    def cm_rec(*a, **k):
        m = o_cm(*a, **k)
        REC["fsky"].append(float(np.mean(m)))
        return m

    def inv_release(self_cs, *a, **k):
        cf = HOLD.get("cf")
        if release and cf is not None and getattr(cf, "inv_cov", None) is not None:
            cf.inv_cov = None
            fm = getattr(cf, "filtered_maps", None)
            if fm is not None:
                fm.inv_cov = None
            REC["n_release"] += 1
        return o_inv(self_cs, *a, **k)

    mmf.get_mmf_q_map, spec.cross_spec.get_cross_spec, mmf.get_cluster_mask = q_rec, cs_rec, cm_rec
    spec.cross_spec.get_inv_cov = inv_release
    cm_before = os.path.exists(params_data["j_cm_name"])
    t0 = time.time()
    data = szifi.input_data(params_szifi=params, params_data=params_data)
    t_load = time.time() - t0
    cf = szifi.cluster_finder(params_szifi=params, params_model=copy.deepcopy(szifi.params.params_model_default),
                              data_file=data, rank=0)
    HOLD["cf"] = cf
    with open(os.path.join(OUT, "stdout", tag + ".txt"), "w") as fo, contextlib.redirect_stdout(fo):
        cf.find_clusters()
    t_run = time.time() - t0
    res = cf.results_dict[pid]
    out = {"cfg": cfg, "mode": mode, "kind": kind, "pid": pid, "release": release, "t_load": t_load, "t_run": t_run,
           "params_json": json.dumps(jsonable(params)), "thetas": THETAS, "n_release": REC["n_release"],
           "cm_existed_before": cm_before}
    n_it = 0
    for k in (0, 1):
        key = f"catalogue_find_{k}"
        if key not in res.catalogues:
            continue
        n_it += 1
        c = res.catalogues[key].catalogue
        q = np.asarray(c["q_opt"], float)
        tx, ty = np.asarray(c["theta_x"], float), np.asarray(c["theta_y"], float)
        jj = tx / DXR - 0.5                      # theta_x = (j + 1/2) dx (column)
        ii = N - 0.5 - ty / DXR                  # theta_y = (N - i - 1/2) dx (row)
        ij_err = float(max(np.abs(jj - np.rint(jj)).max(initial=0), np.abs(ii - np.rint(ii)).max(initial=0)))
        th = np.asarray(c["theta_500"], float)
        jt = np.array([int(np.argmin(np.abs(THETAS - t))) for t in th], dtype=np.int64)
        out.update({f"it{k}_q": q, f"it{k}_y0": np.asarray(c["y0"], float), f"it{k}_theta": th, f"it{k}_jt": jt,
                    f"it{k}_i": np.rint(ii).astype(np.int64), f"it{k}_j": np.rint(jj).astype(np.int64),
                    f"it{k}_theta_x": tx, f"it{k}_theta_y": ty, f"it{k}_ij_err": ij_err,
                    f"it{k}_sigvec": np.asarray(res.sigma_vec[f"find_{k}"], float)})
    out["n_it"] = n_it
    std = np.array(REC["std"])
    out["std_rec"] = std
    nth = THETAS.size
    for k in range(n_it):
        out[f"it{k}_sig_rec"] = std[k * nth:(k + 1) * nth]
    out["t_std"] = np.array(REC["t_std"]) - t0
    for k, (e, s) in enumerate(zip(REC["ell"], REC["spec"])):
        out[f"spec{k}_ell"], out[f"spec{k}"] = e, s
    out["n_spec"] = len(REC["spec"])
    out["fsky_rec"] = np.array(REC["fsky"])
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    np.savez(os.path.join(OUT, "runs", tag + ".npz"), **out)
    s0 = out.get("it0_sig_rec", np.zeros(1))
    print(f"{tag}: {n_it} iteration(s); it0 {out.get('it0_q', np.zeros(0)).size} det" +
          (f", it1 {out['it1_q'].size} det, f {REC['fsky'][0]:.4f}" if n_it > 1 else "") +
          f"; sigma(0.5') it0 {s0[0]:.4e}; SZiFi sigma_vec[0] it0 {out.get('it0_sigvec', np.zeros(1))[0]:.1e}; "
          f"releases {REC['n_release']}; coupling matrix existed before: {cm_before}; "
          f"{t_run:.0f} s; peak RSS {out['peak_rss_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
