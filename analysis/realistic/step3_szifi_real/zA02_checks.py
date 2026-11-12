"""zA02 (arbiter): checks of the two diagnostic interventions in zA_lib.py, before the production runs.

(1) 'periodic' (SZiFi's inpainting and blend on a periodically padded tile).  For the arbiter tiles, SZiFi's iteration-1
    cluster mask is rebuilt with SZiFi's own mmf.get_cluster_mask from the build's stored S1 iteration-0 catalogue, and
    the fraction of the tile's boundary pixels it covers is printed.  On the tile with the largest coverage, SZiFi's
    noise-map step (maps.inpaint_freq, diffusive, 100 sweeps; maps.circular_cosine_blend_adaptive; maps.filter_tmap)
    is run (A) on the bare tile, as SZiFi does, (B) through zA_lib's periodic wrappers with a 200-pixel pad and (C) with
    a 300-pixel pad.  Printed: B vs C (the pad is wide enough if identical); A vs B on the holes that do not touch the
    tile edge (the wrappers change nothing there if identical) and on those that do; the mean squared jump across the
    periodic boundary (row 0 vs row N-1, column 0 vs column N-1) relative to that between neighbouring rows inside the
    tile, for the unmasked map, A and B; the spectra of A and B relative to the unmasked map in five l ranges.
(2) 'zero_oob' (SZiFi's inverse covariance set to 0 outside lrange).  SZiFi's iteration 0 (cluster_finder with
    iterative False) on realistic tile 0, baseline, as configured (S1) and with zero_oob.  A recorder on
    szifi.utils.get_inv_cov_conjugate (the product C^-1 a tau that makes each scale's filter in Fourier space) prints,
    per scale, the energy of the template and of the filter outside lrange relative to inside it, and the size of
    SZiFi's inverse covariance outside lrange relative to inside (l 15000-20000).  Then the two catalogues are compared.
Usage: szifi-venv/bin/python -I -B zA02_checks.py
"""
import os, sys, time, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zA_lib as A                                     # noqa: E402
Z = A.Z
import numpy as np                                     # noqa: E402
from scipy.ndimage import label                        # noqa: E402
from szifi import maps, mmf, cat                       # noqa: E402
utils = sys.modules["szifi.utils"]                    # (the package attribute 'utils' is pixell's)
assert mmf.utils is utils

t0 = time.time()
N, DX = Z.N, Z.DX
pix = maps.pixel(N, DX)
LR = [100, 20000]
PR = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]
BUILD_RUNS = os.path.join(Z.OUT, "runs")


def build_run(cfg, mode, kind, pid):
    with np.load(os.path.join(BUILD_RUNS, f"{cfg}_{mode}_{kind}_{pid:03d}.npz")) as d:
        return {k: d[k] for k in d.files}


def mask_from_catalogue(r):
    c = cat.cluster_catalogue()
    c.catalogue = {"q_opt": r["it0_q"], "theta_x": r["it0_theta_x"], "theta_y": r["it0_theta_y"],
                   "theta_500": r["it0_theta"]}
    return A._prev["get_cluster_mask"](pix, c, 4.0, 3.0)


def boundary(m):
    return np.concatenate([m[0, :], m[-1, :], m[1:-1, 0], m[1:-1, -1]])


print("(1) PERIODIC WRAPPERS: SZiFi's iteration-1 mask rebuilt from the build's S1 iteration-0 catalogue")
cov = {}
for mode in ("baseline", "goal"):
    for pid in PR:
        r = build_run("S1", mode, "real", pid)
        m = mask_from_catalogue(r)
        cov[(mode, pid)] = (float(np.mean(boundary(m) < 0.5)), float(m.mean()), float(r["rec_fsky"][0]))
    print(f"  {mode}: boundary pixels inside the mask, per tile: "
          + ", ".join(f"{p}: {cov[(mode, p)][0]:.3f}" for p in PR))
    print(f"     unmasked fraction rebuilt vs recorded by SZiFi: max |diff| "
          f"{max(abs(cov[(mode, p)][1] - cov[(mode, p)][2]) for p in PR):.1e}")
mode, pid = max(cov, key=lambda k: cov[k][0])
print(f"  largest coverage: {mode} tile {pid} ({cov[(mode, pid)][0]:.3f})")
r = build_run("S1", mode, "real", pid)
mask = mask_from_catalogue(r)
tmap = np.load(os.path.join(Z.MAPD, f"{mode}_real_{pid:03d}.npy"))
tf = maps.filter_tmap(tmap, pix, LR)                   # SZiFi's band-limited noise map (t_noi_original)
noise = np.asarray(np.load(Z.INSTRUMENT)[f"white_{mode}"], float)


def noise_step(periodic, pad):
    A.FLAGS.update(periodic=periodic, pad=pad)
    t1 = time.time()
    ti = maps.inpaint_freq(tf, mask, n_inpaint=100, pix=pix, noise=noise, inpaint_type="diffusive")
    for ll in range(6):
        ti[:, :, ll] = maps.circular_cosine_blend_adaptive(tf[:, :, ll], ti[:, :, ll], mask, inner_zero_frac=0.)
    out = maps.filter_tmap(ti, pix, LR)
    return out, ti, time.time() - t1


Amap, Apre, tA = noise_step(False, 200)
Bmap, Bpre, tB = noise_step(True, 200)
Cmap, Cpre, tC = noise_step(True, 300)
A.FLAGS.update(periodic=False, pad=200)
print(f"  noise-map step: bare {tA:.1f} s, periodic pad 200 {tB:.1f} s, pad 300 {tC:.1f} s")
print(f"  pad 200 vs pad 300: max |diff| / map std = {np.abs(Bmap - Cmap).max() / tf.std():.1e} (after the final l "
      f"filter), {np.abs(Bpre - Cpre).max() / tf.std():.1e} (before)")
lab, nreg = label(mask < 0.5, structure=np.ones((3, 3), bool))
edge_lab = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
edge_lab = edge_lab[edge_lab > 0]
in_edge_holes = np.isin(lab, edge_lab)
inner_holes = (lab > 0) & ~in_edge_holes
dP = np.abs(Apre - Bpre).max(axis=2) / tf.std()
dA = np.abs(Amap - Bmap).max(axis=2) / tf.std()
print(f"  holes: {nreg}, of which {edge_lab.size} touch the tile edge ({in_edge_holes.sum()} pixels; other holes "
      f"{inner_holes.sum()} pixels)")
print(f"  bare vs periodic, after inpainting and blend (before SZiFi's final l filter), max |diff| / map std: holes away "
      f"from the edge {dP[inner_holes].max():.1e}; unmasked pixels {dP[mask > 0.5].max():.1e}; holes at the edge: max "
      f"{dP[in_edge_holes].max():.2f}, rms {np.sqrt(np.mean(dP[in_edge_holes] ** 2)):.3f}")
print(f"  same after the final l filter (a global operation, so edge-hole changes spread): holes away from the edge "
      f"max {dA[inner_holes].max():.1e}, unmasked pixels max {dA[mask > 0.5].max():.1e}, rms over the tile "
      f"{np.sqrt(np.mean(dA ** 2)):.1e}")


def jump(x):
    """mean squared jump across the periodic boundary / mean squared difference of neighbouring rows and columns."""
    jb = np.mean((x[0] - x[-1]) ** 2) + np.mean((x[:, 0] - x[:, -1]) ** 2)
    ji = np.mean((x[1:] - x[:-1]) ** 2) + np.mean((x[:, 1:] - x[:, :-1]) ** 2)
    return jb / ji


for lab_, x in (("unmasked map", tf), ("A, bare (SZiFi)", Amap), ("B, periodic pad", Bmap)):
    print(f"  mean squared jump across the periodic boundary / inside, per channel: {lab_:18s} "
          + " ".join(f"{jump(x[:, :, i]):.2f}" for i in range(6)))
ell = maps.rmap(pix).get_ell()
LR5 = ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
F0 = [np.abs(np.fft.fft2(tf[:, :, i].astype(np.float64))) ** 2 for i in range(6)]
for lab_, x in (("A, bare (SZiFi)", Amap), ("B, periodic pad", Bmap)):
    Fx = [np.abs(np.fft.fft2(x[:, :, i].astype(np.float64))) ** 2 for i in range(6)]
    rat = [np.mean([Fx[i][(ell >= lo) & (ell < hi)].sum() / F0[i][(ell >= lo) & (ell < hi)].sum() for i in range(6)])
           for lo, hi in LR5]
    print(f"  auto power / unmasked, mean of 6 channels, l {LR5}: {lab_:18s} " + " ".join(f"{v:.4f}" for v in rat))
del Amap, Bmap, Cmap, Apre, Bpre, Cpre, tf, tmap

print("\n(2) ZERO_OOB: SZiFi's iteration 0 on realistic tile 0, baseline, as configured and with zero_oob")
ell = maps.rmap(pix).get_ell()
oob = (ell < LR[0]) | (ell > LR[1])
hi_in = (ell >= 15000) & (ell <= 20000)
REC2 = []
_orig_conj = utils.get_inv_cov_conjugate


def _rec_conj(tem_fft, inv_cov):
    out = _orig_conj(tem_fft, inv_cov)
    if tem_fft.shape[:2] == (N, N):
        et = np.sum(np.abs(tem_fft) ** 2, axis=2)
        ef = np.sum(np.abs(out) ** 2, axis=2)
        REC2.append((et[oob].sum() / et[~oob].sum(), ef[oob].sum() / ef[~oob].sum()))
    return out


utils.get_inv_cov_conjugate = _rec_conj
Z.CONFIGS["S1it0"] = dict(Z.CONFIGS["S1"], iterative=False)
icv = {}
_orig_icv = A._w_invcov


def _rec_icv(self, pix_, *a, **k):
    out = _orig_icv(self, pix_, *a, **k)
    d = np.abs(out[:, :, 3, 3])
    icv["ratio"] = (np.median(d[oob & (ell > LR[1])]) / np.median(d[hi_in]) if np.any(d[oob & (ell > LR[1])] > 0)
                    else 0.0)
    return out


from szifi import spec                                 # noqa: E402
spec.cross_spec.get_inv_cov = _rec_icv
res = {}
for lab_, zo in (("S1 as configured", False), ("S1 + zero_oob", True)):
    REC2.clear()
    A.FLAGS.update(zero_oob=zo, periodic=False, lrange=LR)
    out = Z.run_tile("S1it0", "baseline", "real", 0, 8.0, map_file=os.path.join(Z.MAPD, "baseline_real_000.npy"),
                     save=False, tag="zA02_" + ("z" if zo else "s1"))
    res[lab_] = out
    e = np.array(REC2)
    print(f"  {lab_}: {out['t_run']:.0f} s; median |C^-1_(145 GHz)| at l > 20000 / at l 15000-20000 = {icv['ratio']:.1e}")
    print(f"     template energy outside / inside lrange, per scale: " + " ".join(f"{v:.0e}" for v in e[:, 0]))
    print(f"     filter energy outside / inside lrange, per scale:   " + " ".join(f"{v:.0e}" for v in e[:, 1]))
spec.cross_spec.get_inv_cov = A._w_invcov
utils.get_inv_cov_conjugate = _orig_conj
a, b = res["S1 as configured"], res["S1 + zero_oob"]
br = build_run("S1", "baseline", "real", 0)
same = all(np.array_equal(a[f"it0_{k}"], br[f"it0_{k}"]) for k in ("q", "i", "j", "j_theta", "sigvec"))
print(f"  'as configured' iteration 0 identical to the build's stored S1 run (q, positions, scales, sigma): {same}")
ka = {(i, j, t): q for i, j, t, q in zip(a["it0_i"], a["it0_j"], a["it0_j_theta"], a["it0_q"])}
kb = {(i, j, t): q for i, j, t, q in zip(b["it0_i"], b["it0_j"], b["it0_j_theta"], b["it0_q"])}
com = sorted(set(ka) & set(kb))
dq = np.array([kb[k] - ka[k] for k in com])
print(f"  detections q > 4: as configured {len(ka)}, zero_oob {len(kb)}; same (pixel, scale) {len(com)}; on those "
      f"|dq| max {np.abs(dq).max():.3f}, sd {dq.std():.4f}")
for lab_, x in (("as configured", a), ("zero_oob", b)):
    big = x["it0_j_theta"] >= 11
    stacked = 0
    for k in np.where(big & (x["it0_q"] > 4))[0]:
        near = (np.abs(x["it0_i"] - x["it0_i"][k]) <= 2) & (np.abs(x["it0_j"] - x["it0_j"][k]) <= 2) & \
               (x["it0_j_theta"] < x["it0_j_theta"][k])
        stacked += bool(near.any())
    print(f"  {lab_:14s}: q > 4 detections at theta >= 7.24': {int(np.sum(big & (x['it0_q'] > 4)))}, of which "
          f"{stacked} sit within 2 pixels of a smaller-scale detection of the same run")
rs = b["it0_sig"] / a["it0_sig"] - 1
print(f"  sigma(zero_oob) / sigma(as configured) - 1 at the 15 scales: " + " ".join(f"{v:+.1e}" for v in rs))
print(f"done [{time.time() - t0:.0f} s]; peak RSS {Z.peak_rss_gb():.2f} GB")
