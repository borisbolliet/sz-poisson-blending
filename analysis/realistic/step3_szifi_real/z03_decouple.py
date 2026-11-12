"""z03: S0's covariance estimate on one tile (realistic tile 0, SO baseline): SZiFi's binned cross spectra of the
l-filtered tile (NaMaster flat sky, power-spectrum mask = ones, bin_fac 4) with decouple_type 'none' and 'fsky' at
full resolution, and 'none' and 'master' on S3's l_max,1d = 10000 grid with the coupling matrix SZiFi computed there
for the ones mask (data_real/step3_szifi_real/coupling_ones_bin04_lmax10000.fits), against
  (a) SZiFi's own non-NaMaster path (power_spectrum.get_power_spectrum(implementation='custom'));
  (b) our own binned periodogram of the same l-filtered maps: mean of Re(F_i^* F_j) over ALL modes with
      l0 <= l < lf (F = SZiFi's maps.get_fft normalisation, E|F|^2 = C_l);
  (c) the production finder's stored iteration-0 covariance (data_real/step3/tiles/tile_000.npz, rfft units converted
      to uK^2 sr), which averages each bin over its in-band modes (100 <= l <= 20000) only;
  (d) the spectra SZiFi recorded in the S0 smoke run (runs/smokeS0_baseline_real_000.npz).
The l-filtered maps are made with SZiFi's own functions, as cluster_finder.find_clusters makes them.
Usage: szifi-venv/bin/python -I -B z03_decouple.py
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402
from szifi import maps, spec                          # noqa: E402

t0 = time.time()
N, DX = Z.N, Z.DX
pix = maps.pixel(N, DX)
lr = [100, 20000]
t = np.load(os.path.join(Z.MAPD, "baseline_real_000.npy"))
t = maps.select_freqs(t, [0, 1, 2, 3, 4, 5])
ellmap = maps.rmap(pix).get_ell()
ind = tuple(np.asarray(np.where((ellmap < lr[0]) | (ellmap > lr[1])), dtype=np.int32))
tf = maps.filter_tmap(t, pix, lr, indices_filter=ind)
ones = np.ones((N, N), dtype=np.float32)
print(f"z03: tile 0 baseline, l-filtered maps {tf.shape} {tf.dtype}")

out = {}
for dec in ("none", "fsky"):
    t1 = time.time()
    ps = spec.power_spectrum(pix, mask=ones, cm_compute=False, cm_save=False, cm_name=None, bin_fac=4,
                             new_shape=None, cm_compute_scratch=False)
    cs = spec.cross_spec(np.arange(6))
    ell, C = cs.get_cross_spec(pix, t_map=tf, ps=ps, decouple_type=dec, inpaint_flag=False, mask_point=ones,
                               lsep=3000, bin_fac=4, new_shape=None)
    out[dec] = np.array(C)
    print(f"  decouple '{dec}': {C.shape[0]} bins (first bin dropped), {time.time() - t1:.1f} s")
ell = np.asarray(ell)
l0, lf = ps.l0_bins[1:], ps.lf_bins[1:]
# 'master' with a computed coupling matrix: on S3's l_max,1d = 10000 grid (556 x 556, 139 bins), where SZiFi computed
# the coupling matrix of the ones mask once (Z.CM_FILE); 'none' on the same grid for comparison.  (A full-resolution
# coupling matrix, 1200^2 and 300 bins, was not computed: NaMaster's flat-sky computation on one thread is far slower
# than on the 556^2 grid.)
nsd = maps.get_newshape_lmax1d((N, N), 10000, DX)
outd = {}
for dec in ("none", "master"):
    t1 = time.time()
    psd = spec.power_spectrum(pix, mask=ones, cm_compute=(dec == "master"), cm_save=False,
                              cm_name=Z.CM_FILE if dec == "master" else None, bin_fac=4, new_shape=nsd)
    csd = spec.cross_spec(np.arange(6))
    elld, Cd = csd.get_cross_spec(pix, t_map=tf, ps=psd, decouple_type=dec, inpaint_flag=False, mask_point=ones,
                                  lsep=3000, bin_fac=4, new_shape=nsd)
    outd[dec] = np.array(Cd)
    print(f"  decouple '{dec}' on the {nsd} grid: {Cd.shape[0]} bins, {time.time() - t1:.1f} s")
out["master"] = None
print(f"  NaMaster effective l vs bin centres (l0 + lf)/2: max |diff| = {np.abs(ell - 0.5 * (l0 + lf)).max():.2e}; "
      f"bin width {lf[0] - l0[0]:.4f}")

# (a) SZiFi's custom implementation
Cc = np.zeros_like(out["none"])
for i in range(6):
    for j in range(i, 6):
        e_c, c = ps.get_power_spectrum(tf[:, :, i], map2=tf[:, :, j], implementation="custom")
        Cc[:, i, j] = Cc[:, j, i] = np.real(c)[1:]
# (b) own binned periodogram
Fm = [maps.get_fft(tf[:, :, i].astype(np.float64), pix) for i in range(6)]
b = np.floor(ellmap / (lf[0] - l0[0])).astype(np.int64).ravel()
nb = l0.size + 1
cnt = np.bincount(b, minlength=nb)[:nb]
Co = np.zeros_like(out["none"])
for i in range(6):
    for j in range(i, 6):
        v = (np.conj(Fm[i]) * Fm[j]).real.ravel()
        s = np.bincount(b, weights=v, minlength=nb)[:nb] / np.maximum(cnt, 1)
        Co[:, i, j] = Co[:, j, i] = s[1:]
# (c) the finder's stored covariance
st = np.load(os.path.join(Z.ROOT, "data_real", "step3", "tiles", "tile_000.npz"))
Cf_r = st["baseline_it0_C"]                                         # (nbin, 6, 6), rfft units, bin = floor(l / 72)
Om = (0.5 / 60. / 180. * np.pi) ** 2
Cf = np.full_like(out["none"], np.nan)
nfb = min(Cf_r.shape[0] - 1, Cf.shape[0])
Cf[:nfb] = Cf_r[1:nfb + 1] * Om / (N * N)
# (d) the S0 smoke run
sm = np.load(os.path.join(Z.RUND, "smokeS0_baseline_real_000.npz"))

dd = np.arange(6)


def cmp(A, B, lab, sel=None):
    sel = np.ones(A.shape[0], bool) if sel is None else sel
    au = np.abs(A[sel][:, dd, dd] / B[sel][:, dd, dd] - 1)
    iu = np.triu_indices(6, 1)
    cr = np.abs(A[sel][:, iu[0], iu[1]] - B[sel][:, iu[0], iu[1]]) / np.sqrt(B[sel][:, iu[0], iu[0]] * B[sel][:, iu[1], iu[1]])
    print(f"  {lab:62s}: auto max |ratio - 1| = {au.max():.2e} (median {np.median(au):.1e}); cross max |diff|/sqrt(C_ii C_jj) = {cr.max():.2e}")
    return au


inb = (l0 >= 144) & (lf <= 19944)                                    # bins entirely inside 100 <= l <= 20000
lo_b = np.flatnonzero(l0 == 72)[0]; hi_b = np.flatnonzero(l0 == 19944)[0]
print("comparisons, all bins up to l = 20016 that hold in-band modes (bins 72-144 ... 19944-20016):")
allb = (l0 >= 72) & (l0 < 20000)
cmp(out["fsky"], out["none"], "'fsky' vs 'none'", allb)
l0d = np.arange(1, outd["none"].shape[0] + 1) * 72.0
au = cmp(outd["master"], outd["none"], "'master' vs 'none' (556^2 grid, bins 72-9972)", np.ones(outd["none"].shape[0], bool))
mx = np.argmax(np.max(au, axis=1))
print(f"  'master' vs 'none': largest auto deviation in the bin {l0d[mx]:.0f}-{l0d[mx] + 72:.0f}")
print(f"  'master' vs 'none' by l range: " + ", ".join(
    f"{a}-{b_}: {np.abs(outd['master'][(l0d >= a) & (l0d + 72 <= b_)][:, dd, dd] / outd['none'][(l0d >= a) & (l0d + 72 <= b_)][:, dd, dd] - 1).max():.1e}"
    for a, b_ in ((144, 1000), (1000, 3000), (3000, 10000))))
cmp(Cc, out["none"], "(a) SZiFi 'custom' path vs NaMaster 'none'", allb)
cmp(Co, out["none"], "(b) own binned periodogram vs NaMaster 'none'", allb)
cmp(sm["spec0"], out["none"], "(d) S0 smoke run's recorded iteration-0 spectra vs 'none'", allb)
cmp(out["none"], Cf, "'none' vs (c) the finder, bins fully inside 100-20000", inb)
for lab, k in (("lowest", lo_b), ("highest", hi_b)):
    r = out["none"][k, dd, dd] / Cf[k, dd, dd]
    print(f"  'none' / finder in the {lab} band-edge bin ({l0[k]:.0f}-{lf[k]:.0f}): auto ratios {np.round(r, 4).tolist()} "
          f"(SZiFi averages over all modes of the bin, the finder over its in-band modes)")
# where do NaMaster's bins differ from l0 <= l < lf?  modes whose l lies exactly on a bin edge (l = 72 k <=> n^2 = (2k)^2)
dev = np.max(np.abs(out["none"][:, dd, dd] / Co[:, dd, dd] - 1), axis=1)
bad = np.flatnonzero(allb & (dev > 1e-6))
n2 = np.rint((ellmap / 36.0) ** 2).astype(np.int64)
on_edge = np.zeros(l0.size, np.int64)
for k in range(l0.size):
    on_edge[k] = np.sum(n2 == int(round((l0[k] / 36.0) ** 2)))
print(f"  bins where NaMaster's 'none' differs from the periodogram by > 1e-6: {bad.size} of {allb.sum()}; their lower "
      f"edges l0 = {np.round(l0[bad]).astype(int).tolist()[:12]}{' ...' if bad.size > 12 else ''}; largest |ratio - 1| per bin "
      f"{np.round(dev[bad], 4).tolist()[:12]}")
print(f"  modes exactly on the lower edge of those bins: {on_edge[bad].tolist()[:12]}; bins with modes on their lower edge "
      f"and no deviation: {int(np.sum((on_edge > 0) & allb & (dev <= 1e-6)))} (NaMaster assigns some edge modes to the "
      f"bin below)")
above = l0 >= 20016
print(f"  bins entirely above l = 20000 ({above.sum()}): 'none' auto spectra / their value at l = 3000: max "
      f"{np.max(out['none'][above][:, dd, dd] / out['none'][np.argmin(np.abs(ell - 3000))][dd, dd]):.1e} (float32 rounding of the l-filtered maps)")
np.savez(os.path.join(Z.OUT, "z03_decouple.npz"), ell=ell, l0=l0, lf=lf, none=out["none"], fsky=out["fsky"],
         none_556=outd["none"], master_556=outd["master"], custom=Cc, own=Co, finder=Cf)
print(f"done [{time.time() - t0:.0f} s]")
