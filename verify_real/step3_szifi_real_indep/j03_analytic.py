"""j03 (SZiFi's venv): sigma_analytic of SZiFi's own filter (its pixel-sampled template) for S1/S2's settings.

sigma_analytic(theta) is the noise of SZiFi's matched filter when its inverse covariance is the TRUE CMB + noise
covariance of our skies (no tSZ, no estimation noise).  r = sigma_SZiFi / sigma_analytic then measures only what the
tile's estimated covariance adds.  The verifier builds the true covariance itself (j01: szsim's analytic CMB + white
noise + atmosphere on every in-band |k| shell, in SZiFi's units microK^2 sr per mode), places it on SZiFi's FFT grid
(full fft2 layout, l from fftfreq as SZiFi computes it; modes outside l = 100-20000 get a zero inverse, where SZiFi's
l-filtered template vanishes anyway), and lets SZiFi's own mmf.filter_maps.find_clusters filter an all-zero map with it.
The std of every mmf.get_mmf_q_map call is recorded (own wrapper).  Nothing is copied from SZiFi; the build's z04 is not
used (its values are compared at the end, as data).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j03_analytic.py
"""
import os, sys, time, copy, io, contextlib, resource
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
import jsz_run as J                       # own driver: SZiFi import (clone on sys.path) and the parameter table
from szifi import mmf, maps, model, expt  # noqa: E402

t0 = time.time()
N, DXR = J.N, J.DXR
ref = np.load(os.path.join(OUT, "j01_ref.npz"))
pix = maps.pixel(N, DXR)
i0 = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
n2 = i0[:, None] ** 2 + i0[None, :] ** 2                         # SZiFi's full fft2 layout, both axes fftfreq
ell = maps.rmap(pix).get_ell()
print(f"j03: SZiFi's l grid vs 36 sqrt(n2): max |diff| {np.abs(ell - 36.0 * np.sqrt(n2)).max():.2e}")
band = (ell >= 100.0) & (ell <= 20000.0)
shell_n2 = ref["shell_n2"]
pos = np.searchsorted(shell_n2, n2[band])
assert np.array_equal(shell_n2[pos], n2[band]), "an in-band mode has no shell"
std_rec = []
o_q = mmf.get_mmf_q_map


def q_rec(*a, **k):
    out = o_q(*a, **k)
    std_rec.append(float(out[2]))
    return out


mmf.get_mmf_q_map = q_rec
res = {}
for mode in ("baseline", "goal"):
    t1 = time.time()
    Cinv_sh = np.linalg.inv(ref[f"shellC_{mode}"])               # (nshell, 6, 6)
    inv_cov = np.zeros((N, N, 6, 6))
    inv_cov[band] = Cinv_sh[pos]
    del Cinv_sh
    p = J.params_for("S1")
    ex = expt.experiment(experiment_name="SOgoal_simple", params_szifi=p)
    ex.nu_eff, ex.FWHM = ref["freqs_GHz"] * 1e9, ref["fwhm_arcmin"].copy()
    ex.noise_levels, ex.tsz_sed = ref[f"white_{mode}"].copy(), ref["tsz_sed_uK"].copy()
    ex.n_freqs = 6
    a = np.zeros((6, 1))
    a[:, 0] = ex.tsz_sed
    p["a_matrix"], p["cmmf_type"] = a, "standard_mmf"
    cmmf = mmf.scmmf_precomputation(pix=pix, freqs=p["freqs"], inv_cov=inv_cov, lrange=p["lrange"], beam_type=p["beam"],
                                    exp=ex, cmmf_type=p["cmmf_type"], a_matrix=a, comp_to_calculate=p["comp_to_calculate"],
                                    mmf_type=p["mmf_type"])
    ones = np.ones((N, N), dtype=np.float32)
    msel = {"field": ones, "tile": ones, "names": {"field": ["field"], "tile": ["tile"]}}
    mpf = {"field": ones, "tile": ones, "names": ["field", "tile"]}
    fm = mmf.filter_maps(t_obs=np.zeros((N, N, 6), dtype=np.float32), inv_cov=inv_cov, pix=pix,
                         cosmology=model.cosmological_model(p).cosmology, theta_500_vec=p["theta_500_vec_arcmin"],
                         params=p, params_model=copy.deepcopy(J.szifi.params.params_model_default), field_id=0, i_it=0,
                         mask_map=ones, mask_select_dict=msel, mask_peak_finding_dict=mpf, rank=0, exp=ex, cmmf=cmmf)
    std_rec.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        fm.find_clusters()
    s = np.array(std_rec)
    res[f"sigana_{mode}"] = s
    res[f"sigvec_{mode}"] = np.asarray(fm.sigma_vec, float)
    print(f"  {mode:8s}: sigma_analytic [y0] = " + " ".join(f"{v:.4e}" for v in s) +
          f"  [{time.time() - t1:.0f} s; detections on the zero map: {sum(len(v.catalogue['q_opt']) for k, v in fm.results.items() if k != 'empty')}]")
    del inv_cov, fm, cmmf
res["thetas"] = J.THETAS
np.savez(os.path.join(OUT, "j03_analytic.npz"), **res)

# ---- compare with the build's z04 (data_real/step3_szifi_real/analytic_sigma.npz, read as data)
b = np.load("/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/analytic_sigma.npz")
for mode in ("baseline", "goal"):
    d = res[f"sigana_{mode}"] / b[f"{mode}_S0"] - 1
    print(f"  {mode:8s}: verifier / build (z04, S0 setup) - 1: max |.| {np.abs(d).max():.1e} (" +
          " ".join(f"{v:+.1e}" for v in d[[0, 4, 9, 14]]) + " at 0.5, 1.32, 4.45, 15')")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
