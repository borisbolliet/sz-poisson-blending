"""z06b: why S3 (cov_type 'mixed_gaussian') finds tens of thousands of peaks -- SZiFi's own covariance estimate of one
tile (realistic tile 0, SO baseline) at noise iteration 0 with S3's settings, isotropic vs mixed_gaussian.
The steps are SZiFi's functions called in cluster_finder's order (select_freqs, filter_tmap with lrange 1-300000,
power_spectrum with powspec_lmax1d 10000 and the stored 'master' coupling matrix, cross_spec.get_cross_spec,
cross_spec.get_cov).  Per 2D mode of the covariance grid (556 x 556): the smallest eigenvalue of the 6 x 6 covariance
(is it positive definite?), and the MMF weight w = a^T C^-1 a (a_i = tSZ SED x SZiFi's Gaussian beam), mixed / isotropic.
Usage: szifi-venv/bin/python -I -B z06b_mixed.py
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402
from szifi import maps, spec                          # noqa: E402

t0 = time.time()
N, DX = Z.N, Z.DX
p = Z.make_params("S3")
pix = maps.pixel(N, DX)
t = maps.select_freqs(np.load(os.path.join(Z.MAPD, "baseline_real_000.npy")), p["freqs"])
lmin, lmax = p["lrange"]
ell = maps.rmap(pix).get_ell()
ind = tuple(np.asarray(np.where((ell < lmin) | (ell > lmax)), dtype=np.int32))
tn = maps.filter_tmap(t, pix, p["lrange"], indices_filter=ind)
ns = maps.get_newshape_lmax1d((N, N), p["powspec_lmax1d"], DX)
ones = np.ones((N, N), dtype=np.float32)
ps = spec.power_spectrum(pix, mask=ones, cm_compute=True, cm_save=False, cm_name=Z.CM_FILE, bin_fac=4, new_shape=ns)
cs = spec.cross_spec(np.arange(6))
cs.get_cross_spec(pix, t_map=tn, ps=ps, decouple_type="master", inpaint_flag=True, mask_point=ones, lsep=p["lsep"],
                  bin_fac=4, new_shape=ns)
C = {}
for ct in ("isotropic", "mixed_gaussian"):
    C[ct] = cs.get_cov(pix, t_map=tn, mask=ones, bin_fac=4, new_shape=ns, ps=ps, cov_type=ct,
                       cov_kernel_shape=p["cov_kernel_shape"])
print(f"z06b: tile 0 baseline, S3 settings, noise iteration 0; covariance grid {C['isotropic'].shape} [{time.time() - t0:.0f} s]")
inst = np.load(Z.INSTRUMENT)
pd = maps.degrade_pix(pix, ns)
L = maps.rmap(pd).get_ell()
B = np.stack([maps.get_bl(fw, L) for fw in inst["fwhm_arcmin"]], -1)
a = B * inst["tsz_sed_uK"][None, None, :]
w = {}
for ct, Cc in C.items():
    ev = np.linalg.eigvalsh(Cc)
    lam = ev[..., 0] / ev[..., -1]
    w[ct] = np.einsum("xyi,xyij,xyj->xy", a, np.linalg.inv(Cc), a)
    sel = L > 0
    print(f"  {ct:15s}: modes with a non-positive eigenvalue {100 * np.mean(ev[..., 0][sel] <= 0):6.2f}%; smallest "
          f"eigenvalue / largest, median {np.median(lam[sel]):.1e}; MMF weight a^T C^-1 a < 0 in {100 * np.mean(w[ct][sel] < 0):.2f}% of the modes")
sel = L > 0
rat = w["mixed_gaussian"][sel] / w["isotropic"][sel]
for lo, hi in ((1, 1000), (1000, 3000), (3000, 6000), (6000, 10500)):
    m = (L[sel] >= lo) & (L[sel] < hi)
    print(f"  l {lo:5d}-{hi:5d}: weight mixed / isotropic: median {np.median(rat[m]):+.3f}, 1-99% "
          f"[{np.percentile(rat[m], 1):+.2e}, {np.percentile(rat[m], 99):+.2e}], |ratio| > 10 in {100 * np.mean(np.abs(rat[m]) > 10):.2f}%")
d = np.arange(6)
iu = np.triu_indices(6, 1)
for ct in ("isotropic", "mixed_gaussian"):
    Cc = C[ct][sel]
    cr = Cc[:, iu[0], iu[1]] / np.sqrt(np.abs(Cc[:, iu[0], iu[0]] * Cc[:, iu[1], iu[1]]))
    print(f"  {ct:15s}: correlation coefficients C_ij / sqrt(C_ii C_jj) outside [-1, 1] in {100 * np.mean(np.abs(cr) > 1):.2f}% "
          f"of the (mode, pair) entries; diagonal mixed / isotropic median {np.median(C['mixed_gaussian'][sel][:, d, d] / C['isotropic'][sel][:, d, d]):.3f}")
print(f"done [{time.time() - t0:.0f} s]")
