"""j13 (SZiFi's venv): how much of SZiFi's iteration-1 sigma comes from the tile edges?  (follows j11, j12)

For tiles 41 (cluster mask on 20% of the tile boundary) and 83 (3%), baseline, the verifier repeats SZiFi's iteration-1
covariance estimate step by step with SZiFi's own functions, for l_sep = 3000 (S1) and 12000 (S2):
  noise maps: A = as SZiFi runs it (bare tile), B = inpainting and blend on the tile extended periodically (j12);
  spectra: spec.power_spectrum / spec.cross_spec.get_cross_spec ('none' decoupling, bin_fac 4, correct_for_inpainting
           with SZiFi's mask), inverse covariance spec.cross_spec.get_inv_cov (isotropic, 'nearest');
  sigma: SZiFi's own mmf.filter_maps.find_clusters on an all-zero map with that inverse covariance, at 0.5', 4.45', 15'.
Check: sigma_A must equal the sigma SZiFi itself reported at iteration 1 of the verifier's S1 / S2 runs.
Then sigma_A / sigma_B is the part of SZiFi's sigma that the bare-tile edges add.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j13_edge_sigma.py   (run alone: about 1.5 GB)
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
import jsz_run as J                                  # own driver: SZiFi on sys.path, parameter table
from szifi import maps, mmf, cat, spec, model, expt  # noqa: E402

t0 = time.time()
N, DX, PAD = J.N, J.DXR, 100
pix = maps.pixel(N, DX)
pixP = maps.pixel(N + 2 * PAD, DX)
ref = np.load(os.path.join(OUT, "j01_ref.npz"))
SC = [0, 9, 14]
TH3 = J.THETAS[SC]


def inpaint_blend(t, mask, p):
    inp = maps.inpaint_freq(t, mask, n_inpaint=100, pix=p, noise=None, inpaint_type="diffusive")
    for k in range(t.shape[2]):
        inp[:, :, k] = maps.circular_cosine_blend_adaptive(t[:, :, k], inp[:, :, k], mask, inner_zero_frac=0.)
    return inp


def sigma3(tnoi, mask, lsep, mode):
    p = J.params_for("S1")
    p["lsep"] = lsep
    ps = spec.power_spectrum(pix, mask=np.ones((N, N), dtype=np.float32), cm_compute=False, bin_fac=4)
    cs = spec.cross_spec(np.arange(6))
    cs.get_cross_spec(pix, t_map=tnoi, ps=ps, decouple_type="none", inpaint_flag=True, mask_point=mask, lsep=lsep, bin_fac=4)
    inv_cov = cs.get_inv_cov(pix, t_map=tnoi, interp_type="nearest", bin_fac=4, mask=np.ones((N, N), dtype=np.float32),
                             cov_type="isotropic", cov_kernel_shape=[4, 4])
    ex = expt.experiment(experiment_name="SOgoal_simple", params_szifi=p)
    ex.nu_eff, ex.FWHM = ref["freqs_GHz"] * 1e9, ref["fwhm_arcmin"].copy()
    ex.noise_levels, ex.tsz_sed, ex.n_freqs = ref[f"white_{mode}"].copy(), ref["tsz_sed_uK"].copy(), 6
    a = np.zeros((6, 1))
    a[:, 0] = ex.tsz_sed
    p["a_matrix"], p["cmmf_type"], p["theta_500_vec_arcmin"] = a, "standard_mmf", TH3.copy()
    cmmf = mmf.scmmf_precomputation(pix=pix, freqs=p["freqs"], inv_cov=inv_cov, lrange=p["lrange"], beam_type=p["beam"],
                                    exp=ex, cmmf_type=p["cmmf_type"], a_matrix=a, comp_to_calculate=[0], mmf_type="standard")
    ones = np.ones((N, N), dtype=np.float32)
    fm = mmf.filter_maps(t_obs=np.zeros((N, N, 6), dtype=np.float32), inv_cov=inv_cov, pix=pix,
                         cosmology=model.cosmological_model(p).cosmology, theta_500_vec=TH3, params=p,
                         params_model=copy.deepcopy(J.szifi.params.params_model_default), field_id=0, i_it=1, mask_map=ones,
                         mask_select_dict={"field": ones, "tile": ones, "names": {"field": ["field"], "tile": ["tile"]}},
                         mask_peak_finding_dict={"field": ones, "tile": ones, "names": ["field", "tile"]}, rank=0, exp=ex, cmmf=cmmf)
    with contextlib.redirect_stdout(io.StringIO()):
        fm.find_clusters()
    s = np.asarray(fm.sigma_vec, float).copy()
    del inv_cov, fm, cmmf, cs
    return s


print("j13: SZiFi's iteration-1 sigma from the bare tile (A, as SZiFi) and from the periodically extended tile (B)")
for pid in (41, 83):
    mode = "baseline"
    t1 = time.time()
    tmap = np.load(os.path.join(OUT, "maps", f"real_{mode}_{pid:03d}.npy"))
    r1 = np.load(os.path.join(OUT, "runs", f"S1_{mode}_real_{pid:03d}.npz"))
    c = cat.cluster_catalogue()
    for k, v in (("q_opt", "it0_q"), ("theta_500", "it0_theta"), ("theta_x", "it0_theta_x"), ("theta_y", "it0_theta_y")):
        c.catalogue[k] = np.asarray(r1[v], float)
    with contextlib.redirect_stdout(io.StringIO()):
        mask = mmf.get_cluster_mask(pix, c, 4.0, 3.0)
    t_or = maps.filter_tmap(maps.select_freqs(tmap, [0, 1, 2, 3, 4, 5]), pix, [100, 20000])
    A = maps.filter_tmap(inpaint_blend(t_or, mask, pix), pix, [100, 20000])
    tP = np.pad(t_or, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="wrap")
    mP = np.pad(mask, PAD, mode="constant", constant_values=1.0)
    B = maps.filter_tmap(inpaint_blend(tP, mP, pixP)[PAD:PAD + N, PAD:PAD + N], pix, [100, 20000])
    del tP, mP
    print(f"\n  tile {pid} {mode}: unmasked fraction {mask.mean():.4f}")
    for cfg, lsep in (("S1", 3000), ("S2", 12000)):
        rr = np.load(os.path.join(OUT, "runs", f"{cfg}_{mode}_real_{pid:03d}.npz"))
        s_szifi = rr["it1_sig_rec"][SC]
        sA = sigma3(A, mask, lsep, mode)
        sB = sigma3(B, mask, lsep, mode)
        print(f"    {cfg} (l_sep {lsep}): sigma_A / SZiFi's own iteration-1 sigma - 1 at 0.5, 4.45, 15': "
              + " ".join(f"{v:+.1e}" for v in sA / s_szifi - 1))
        print(f"    {cfg} (l_sep {lsep}): sigma_A / sigma_B at 0.5, 4.45, 15': " + " ".join(f"{v:.4f}" for v in sA / sB)
              + f"  [{time.time() - t1:.0f} s]")
print(f"\ndone [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
