"""p05: is there a positive-definite version of SZiFi's 'mixed_gaussian' covariance?  (Review item E5.)

SZiFi's 'mixed_gaussian' divides each channel pair's 2D cross-periodogram by that pair's isotropic cross-spectrum,
smooths the ratio and multiplies back.  Each element is rescaled differently, so the 6 x 6 result need not be positive
semi-definite.  Two variants written here (our own code) smooth, with one non-negative kernel, a matrix that is
positive semi-definite at every mode, and so stay positive semi-definite by construction:
  'direct'   : smooth the real part of the cross-periodogram matrix P(k) with SZiFi's Gaussian kernel (10 pixels),
               no isotropic division (what 'anisotropic_gaussian' does, here on the same reduced grid);
  'whitened' : smooth W(k) P(k) W(k), with W = C_iso^(-1/2) the inverse matrix square root of SZiFi's isotropic
               covariance, then transform back with C_iso^(1/2).  This keeps the idea of 'mixed_gaussian' (smooth the
               departure from isotropy) but whitens with the whole matrix instead of dividing pair by pair.  It needs a
               positive-definite isotropic covariance (its eigenvalues are clipped at zero here).
Same tile, settings and spectra as ../step3_szifi_real/z06b_mixed.py (tile 0, SO baseline, S3 settings, noise
iteration 0, covariance grid 556 x 556); SZiFi's own functions give the maps, the spectra and its two covariances.
Not run through the cluster finder: only the covariances are compared.

Usage (from the repository root):
  MallocLargeCache=0 /Users/boris/GitHub/szifi-venv/bin/python -I -B analysis/realistic/szifi_review/p05_mixed_fix.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data_real", "szifi_review", "out")
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3_szifi_real"))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402
from scipy import ndimage                             # noqa: E402
from szifi import maps, spec                          # noqa: E402

t0 = time.time()
N, DX = Z.N, Z.DX
p = Z.make_params("S3")
pix = maps.pixel(N, DX)
t = maps.select_freqs(np.load(os.path.join(OUT, "maps", "baseline_real_000.npy")), p["freqs"])
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
C = {ct: cs.get_cov(pix, t_map=tn, mask=ones, bin_fac=4, new_shape=ns, ps=ps, cov_type=ct,
                    cov_kernel_shape=p["cov_kernel_shape"]) for ct in ("isotropic", "mixed_gaussian")}
shape = C["isotropic"].shape[:2]
sig = np.array(p["cov_kernel_shape"], float)

# real part of the 6 x 6 cross-periodogram at every mode, with SZiFi's FFT normalisation, on the same reduced grid
F = [maps.get_fft(tn[:, :, i] * ones, pix) for i in range(6)]
P = np.zeros(shape + (6, 6))
for i in range(6):
    for j in range(i, 6):
        P[..., i, j] = maps.reshape_ell_matrix(np.real(np.conjugate(F[i]) * F[j]), shape)
        P[..., j, i] = P[..., i, j]
del F


def smooth(A):
    """Gaussian smoothing of every element with the same kernel, on the centred (fftshift) grid with reflecting
    boundaries, as SZiFi smooths its spectra."""
    out = np.empty_like(A)
    for i in range(6):
        for j in range(i, 6):
            centred = np.fft.fftshift(A[..., i, j])
            centred = ndimage.gaussian_filter(centred, sigma=sig, mode="reflect")
            out[..., i, j] = out[..., j, i] = np.fft.ifftshift(centred)
    return out


C["direct (own)"] = smooth(P)
lam, V = np.linalg.eigh(C["isotropic"])
lam = np.clip(lam, 0, None)
Wm = np.einsum("...ik,...k,...jk->...ij", V, 1 / np.sqrt(np.where(lam > 0, lam, np.inf)), V)   # C_iso^(-1/2)
Wp = np.einsum("...ik,...k,...jk->...ij", V, np.sqrt(lam), V)                                     # C_iso^(1/2)
C["whitened (own)"] = np.einsum("...ij,...jk,...kl->...il", Wp, smooth(np.einsum("...ij,...jk,...kl->...il", Wm, P, Wm)), Wp)
print(f"p05: tile 0 baseline, S3 settings, noise iteration 0; covariance grid {shape} [{time.time() - t0:.0f} s]")

inst = np.load(Z.INSTRUMENT)
pd = maps.degrade_pix(pix, ns)
L = maps.rmap(pd).get_ell()
B = np.stack([maps.get_bl(fw, L) for fw in inst["fwhm_arcmin"]], -1)
a = B * inst["tsz_sed_uK"][None, None, :]
sel = L > 0
w = {}
print(f"modes with l > 0 on the covariance grid: {int(sel.sum())}")
print(f"  {'covariance':16s} {'modes with an eigenvalue <= 0 (count)':>38s} {'a^T C^-1 a < 0':>15s} {'weight / isotropic: median, 1-99%':>40s}")
for ct, Cc in C.items():
    ev = np.linalg.eigvalsh(Cc[sel])
    w[ct] = np.einsum("ni,nij,nj->n", a[sel], np.linalg.inv(Cc[sel]), a[sel])
    r = w[ct] / w.get("isotropic", w[ct])
    print(f"  {ct:16s} {100 * np.mean(ev[:, 0] <= 0):21.2f}% ({int(np.sum(ev[:, 0] <= 0)):6d}) {100 * np.mean(w[ct] < 0):14.2f}%"
          f"   {np.median(r):+.3f}  [{np.percentile(r, 1):+.2e}, {np.percentile(r, 99):+.2e}]")
d = np.arange(6)
print(f"  diagonal / isotropic, median: " + ", ".join(f"{ct} {np.median(C[ct][sel][:, d, d] / C['isotropic'][sel][:, d, d]):.3f}"
                                                   for ct in C if ct != "isotropic"))
print(f"done [{time.time() - t0:.0f} s]")
