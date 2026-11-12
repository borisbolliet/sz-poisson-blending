"""Arbiter driver: runs SZiFi itself (clone at /Users/boris/GitHub/szifi, commit 73626fb, imported unchanged) through
the build's driver z_szlib.py, plus two optional in-process interventions that serve as diagnostics.  Own code; no SZiFi
code is copied.  Run with /Users/boris/GitHub/szifi-venv/bin/python -I -B.

Why: the two checks disagree about why SZiFi's default iteration (S1) ends with a larger sigma than step 3a's emulation.
  * config check: a numerical artifact.  With lrange [100, 20000] on 0.5' pixels, 32.7% of the grid's modes lie outside
    the band.  SZiFi zeroes them in the maps, so the bins there hold only float32 round-off, and its inverse covariance
    there is ~1e15 too large.  This amplifies round-off into the filters and creates spurious scale-axis maxima.
  * independent check: an edge effect of running SZiFi on bare periodic tiles.  The diffusive inpainting and the cosine
    blend stop at the tile edge, while SZiFi's spectra are FFTs that treat the tile as periodic.

Interventions (flags; each one is a wrapper around a SZiFi function that SZiFi looks up at call time):
  zero_oob : after SZiFi builds its inverse covariance (spec.cross_spec.get_inv_cov), set it to 0 at every mode outside
             lrange, i.e. at exactly the modes SZiFi zeroes in the maps (same rule as its indices_filter:
             l < lmin or l > lmax).  In exact arithmetic this changes nothing, because SZiFi's band-limited templates
             vanish there; it removes only the amplified round-off.  In-band modes are untouched.
  periodic : SZiFi's inpainting (maps.inpaint_freq) and cosine blend (maps.circular_cosine_blend_adaptive) are called on
             the tile padded periodically by PAD pixels on each side (np.pad mode 'wrap'), and the centre is cut out
             again.  Holes that do not touch the tile edge are filled exactly as before (zA02 checks this); holes that
             do are filled from both sides, as on a periodic sky.  The cluster mask itself (SZiFi's discs, not wrapped)
             and everything else are SZiFi's.
Variants: S1z (S1 + zero_oob), S1p (S1 + periodic), S1zp (both), S2z and S2zp (the same for S2, l_sep 12000), S1zL (S1z with the 15 scales built as
exp(linspace) instead of geomspace: values differ in the last bit), S1 (as configured; reproduction check).
Outputs: data_real/step3_szifi_real/arbiter/runs/VAR_MODE_KIND_PID.npz (same keys as the build's runs, plus 'variant',
'zero_oob', 'periodic', 'pad', 'n_oob', 'edge_cov' = fraction of the tile's boundary pixels inside the iteration-1
cluster mask, 'mask_boundary' = the mask on the 4 boundary rows/columns).
"""
import os, sys, time, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
AOUT = os.path.join(ROOT, "data_real", "step3_szifi_real", "arbiter")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import z_szlib as Z                                     # noqa: E402  (sets threads, puts the SZiFi clone on sys.path)
os.environ["NUMBA_CACHE_DIR"] = os.path.join(AOUT, "out", "numba_cache")   # SZiFi's one numba function is not cached
import numpy as np                                      # noqa: E402
from szifi import spec, maps, mmf                      # noqa: E402

Z.RUND = os.path.join(AOUT, "runs")
Z.STDD = os.path.join(AOUT, "out", "stdout")
Z.MAPD = os.path.join(AOUT, "out", "maps")
Z.BASE["path_data"] = Z.MAPD + "/"
for _d in (Z.RUND, Z.STDD, Z.MAPD):
    os.makedirs(_d, exist_ok=True)

FLAGS = dict(zero_oob=False, periodic=False, pad=200, lrange=None)
AREC = {}

# the functions as the build's driver left them (its recorders and its memory release), wrapped once more here
_prev = dict(get_inv_cov=spec.cross_spec.get_inv_cov, inpaint_freq=maps.inpaint_freq,
             blend=maps.circular_cosine_blend_adaptive, get_cluster_mask=mmf.get_cluster_mask)


def _ell_of(pix, shape):
    if shape[0] != pix.nx:
        raise ValueError("zero_oob is written for full-resolution covariances only")
    return maps.rmap(pix).get_ell()


def _w_invcov(self, pix, *a, **k):
    out = _prev["get_inv_cov"](self, pix, *a, **k)
    if FLAGS["zero_oob"]:
        lmin, lmax = FLAGS["lrange"]
        ell = _ell_of(pix, out.shape)
        oob = (ell < lmin) | (ell > lmax)
        out[oob] = 0.0
        AREC["n_oob"] = int(oob.sum())
    return out


def _wrap(a, P):
    return np.pad(a, [(P, P), (P, P)] + [(0, 0)] * (a.ndim - 2), mode="wrap")


def _w_inpaint(tmap, mask, n_inpaint=None, pix=None, noise=None, inpaint_type="diffusive"):
    if not FLAGS["periodic"] or np.all(np.asarray(mask) == 1):
        return _prev["inpaint_freq"](tmap, mask, n_inpaint, pix, noise, inpaint_type=inpaint_type)
    P = FLAGS["pad"]
    pixp = maps.pixel(tmap.shape[0] + 2 * P, pix.dx) if pix is not None else None
    out = _prev["inpaint_freq"](_wrap(tmap, P), _wrap(np.asarray(mask), P), n_inpaint, pixp, noise,
                                inpaint_type=inpaint_type)
    AREC["n_periodic_inpaint"] = AREC.get("n_periodic_inpaint", 0) + 1
    return np.ascontiguousarray(out[P:-P, P:-P])


def _w_blend(original_map, inpainted_map, mask, inner_zero_frac=0., min_radius=0., pad=2):
    if not FLAGS["periodic"] or np.all(np.asarray(mask) > 0.5):
        return _prev["blend"](original_map, inpainted_map, mask, inner_zero_frac=inner_zero_frac,
                              min_radius=min_radius, pad=pad)
    P = FLAGS["pad"]
    out = _prev["blend"](_wrap(original_map, P), _wrap(inpainted_map, P), _wrap(np.asarray(mask), P),
                         inner_zero_frac=inner_zero_frac, min_radius=min_radius, pad=pad)
    AREC["n_periodic_blend"] = AREC.get("n_periodic_blend", 0) + 1
    return np.ascontiguousarray(out[P:-P, P:-P])


def _w_mask(pix, catalogue, q_th_noise, mask_radius, **k):
    m = _prev["get_cluster_mask"](pix, catalogue, q_th_noise, mask_radius, **k)
    b = np.concatenate([m[0, :], m[-1, :], m[1:-1, 0], m[1:-1, -1]])
    AREC["edge_cov"] = float(np.mean(b < 0.5))
    AREC["mask_boundary"] = np.stack([m[0, :], m[-1, :], m[:, 0], m[:, -1]]).astype(np.uint8)
    return m


spec.cross_spec.get_inv_cov = _w_invcov
maps.inpaint_freq = _w_inpaint
maps.circular_cosine_blend_adaptive = _w_blend
mmf.get_cluster_mask = _w_mask

THETAS_LIN = np.exp(np.linspace(np.log(0.5), np.log(15.0), 15))
VARIANTS = {
    "S1": ("S1", False, False, None),
    "S1z": ("S1", True, False, None),
    "S1p": ("S1", False, True, None),
    "S1zp": ("S1", True, True, None),
    "S1zL": ("S1", True, False, THETAS_LIN),
    "S1zp300": ("S1", True, True, None),      # S1zp with a 300-pixel periodic pad (padding check)
    "S2z": ("S2", True, False, None),         # l_sep 12000 (S2) with zero_oob
    "S2zp": ("S2", True, True, None),         # S2 with zero_oob and periodic inpainting
}


def run_variant(var, mode, kind, pid, buf=8.0, save=True):
    cfg, zoob, per, thetas = VARIANTS[var]
    FLAGS.update(zero_oob=zoob, periodic=per, pad=300 if var.endswith("300") else 200,
                 lrange=list(Z.CONFIGS[cfg]["lrange"]))
    AREC.clear()
    th_saved = Z.BASE["theta_500_vec_arcmin"]
    if thetas is not None:
        Z.BASE["theta_500_vec_arcmin"] = thetas
    try:
        out = Z.run_tile(cfg, mode, kind, pid, buf, map_file=os.path.join(Z.MAPD, f"{mode}_{kind}_{pid:03d}.npy"),
                         save=False, tag=var)
    finally:
        Z.BASE["theta_500_vec_arcmin"] = th_saved
    out.update(variant=var, zero_oob=zoob, periodic=per, pad=FLAGS["pad"], n_oob=AREC.get("n_oob", -1),
               n_periodic_inpaint=AREC.get("n_periodic_inpaint", 0), n_periodic_blend=AREC.get("n_periodic_blend", 0),
               edge_cov=AREC.get("edge_cov", np.nan),
               mask_boundary=AREC.get("mask_boundary", np.zeros((4, Z.N), np.uint8)))
    if save:
        np.savez(os.path.join(Z.RUND, f"{var}_{mode}_{kind}_{pid:03d}.npz"), **out)
    return out
