"""j10 (SZiFi's venv): why S3 (SZiFi's SO example settings) breaks down; the build's diagnosis of claim 6
('mixed_gaussian' covariance not positive definite in 45.5% of the modes, MMF weight negative in 6.1%), checked by the
verifier on two of its own new tiles with SZiFi's own covariance functions.
Steps, as SZiFi's cluster_finder does them at iteration 0 for S3 (mmf.py): the tile is l-filtered to [1, 300000]
(maps.filter_tmap); spec.power_spectrum on the grid degraded to l_max,1d = 10000 with the ones mask and the coupling
matrix of out/ (written by SZiFi in the verifier's first S3 run); spec.cross_spec.get_cross_spec with 'master'
decoupling; spec.cross_spec.get_cov with cov_type 'mixed_gaussian' (kernel [10, 10]) and 'isotropic'.  (The inpainting
step of iteration 0 has no holes and is skipped here.)  Then, per 2D mode: the smallest eigenvalue of the 6 x 6
matrix, the correlation coefficients C_ij / sqrt(C_ii C_jj), and the MMF weight a^T C^-1 a with a_i = SED_i x SZiFi's
Gaussian beam B_i(l) (its sign does not depend on the cluster template).  Own code around SZiFi's calls.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j10_mixed.py   (run alone: about 0.7 GB)
"""
import os, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/boris/GitHub/szifi")
import numpy as np
from szifi import maps, spec       # noqa: E402

t0 = time.time()
N, DX = 1200, 0.5 / 60.0 / 180.0 * np.pi
ref = np.load(os.path.join(OUT, "j01_ref.npz"))
pix = maps.pixel(N, DX)
new_shape = maps.get_newshape_lmax1d((N, N), 10000, pix.dx)
pix_d = maps.degrade_pix(pix, new_shape)
ell_d = maps.rmap(pix_d).get_ell()
ones = np.ones((N, N))
print(f"j10: SZiFi's covariance estimates for S3's settings; degraded grid {tuple(int(v) for v in new_shape)}")
for kind, mode, pid in (("real", "baseline", int(ref["new_pids"][0])), ("noise", "goal", int(ref["new_pids"][0]))):
    t1 = time.time()
    tmap = np.load(os.path.join(OUT, "maps", f"{kind}_{mode}_{pid:03d}.npy"))
    tf = maps.filter_tmap(tmap, pix, [1, 300000])
    ps = spec.power_spectrum(pix, mask=ones, cm_compute=True, cm_save=False,
                             cm_name=os.path.join(OUT, "coupling_ones_bin04_lmax10000.fits"), bin_fac=4, new_shape=new_shape)
    cs = spec.cross_spec(np.arange(6))
    cs.get_cross_spec(pix, t_map=tf, ps=ps, decouple_type="master", inpaint_flag=True, mask_point=ones, lsep=12000,
                      bin_fac=4, new_shape=new_shape)
    res = {}
    for ct in ("isotropic", "mixed_gaussian"):
        C = cs.get_cov(pix, t_map=tf, mask=ones, bin_fac=4, new_shape=new_shape, ps=ps, decouple_type="master",
                       interp_type="nearest", cov_type=ct, cov_kernel_shape=[10, 10])
        ok = ell_d > 0
        Cm = C[ok]
        ev = np.linalg.eigvalsh(Cm)[:, 0]
        d = np.sqrt(np.abs(np.einsum("nii->ni", Cm)))
        rho = Cm / (d[:, :, None] * d[:, None, :])
        iu = np.triu_indices(6, 1)
        rr = rho[:, iu[0], iu[1]]
        B = np.stack([maps.get_bl(fw, ell_d[ok]) for fw in ref["fwhm_arcmin"]], axis=1)
        a = B * ref["tsz_sed_uK"][None, :]
        w = np.einsum("ni,ni->n", a, np.linalg.solve(Cm, a[..., None])[..., 0])
        res[ct] = w
        print(f"  {kind} {mode} {pid}, {ct:14s}: modes {ev.size}; smallest eigenvalue <= 0 in {100 * np.mean(ev <= 0):.1f}%; "
              f"correlation coefficients outside [-1, 1]: {100 * np.mean(np.abs(rr) > 1):.1f}%; MMF weight a^T C^-1 a < 0 in "
              f"{100 * np.mean(w < 0):.1f}%")
        del C, Cm, rho
    hi = ell_d[ell_d > 0] > 3000
    q = res["mixed_gaussian"][hi] / res["isotropic"][hi]
    print(f"    l > 3000: MMF weight mixed / isotropic, 1st-99th percentile {np.percentile(q, 1):+.1f} to {np.percentile(q, 99):+.1f}"
          f"  [{time.time() - t1:.0f} s]")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
