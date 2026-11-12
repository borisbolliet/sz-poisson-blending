"""Step 3: a lean SZiFi-like cluster finder for periodic flat-sky tiles, the six-channel SO tile simulator it runs
on, and the exact signal decomposition used to measure blending.

Own implementation.  The algorithm follows SZiFi (Zubeldia et al. 2023a, b).  Its code (github.com/inigozubeldia/
szifi, no licence) was read to learn the algorithm; nothing was copied and it was not run.  What is reproduced:
  * covariance (spec.py: get_cov, cov_type 'isotropic'): 6x6 auto and cross spectra of the tile itself, in bins of
    width bin_fac * pi / (n dx) (bin_fac = 4 -> 72 in l here), each Fourier mode taking the value of its own bin
    ('nearest', SZiFi's default interp_type); l in [100, 20000] as in steps 1-2;
  * iteration (mmf.py: cluster_finder.find_clusters, get_cluster_mask; params.py defaults q_th_noise = 4,
    mask_radius = 3, max_it = 1): detections with q > 4 are masked with binary discs of radius 3 x their estimated
    theta500 (pixels at distance <= r), the spectra are re-estimated from the masked tile and the filter is rerun
    once on the unmasked tile.  The masked pseudo-spectra are divided by the unmasked fraction ('fsky'
    decoupling, as the step-3 task specifies), after whitening (masked_cov method 'whitened', the default; the
    literal binary estimate leaks low-l power).  SZiFi itself reaches the covariance with the cluster mask only
    through inpainting: diffusive inpainting of the holes, a cosine blend back towards the original map, then the
    spectra above l_sep = 3000 divided by the unmasked fraction (mmf.py find_clusters, maps.py, spec.py
    correct_for_inpainting).  masked_cov method 'szifi' is an own implementation of those steps (added by the step-3
    arbiter); with SZiFi's defaults it inflates the high-l spectra by 14-20% on SO tiles;
  * S/N maps per scale, sigma(theta) = 1/sqrt(norm) from the estimated covariance (mmf.py get_mmf_q_map);
  * detections (mmf.py make_detections2, get_maxima_mask_2d; 'maxima_lomem'): q > q_th = 4, strictly larger than the
    two neighbours along each map axis, q_j > q_{j-1} and q_j >= q_{j+1} along the scale axis with q_{-1} = 0 and
    only q_{n-1} > q_{n-2} at the last scale (argmax tie rules).  Neighbours are periodic here (SZiFi pads the tile
    edges with -inf).  Catalogue order: scale, then row-major pixel (the order merge_detections sees);
  * merging (cat.py merge_detections): 'closest' (anchor = first remaining entry in catalogue order, keep the
    highest q within the radius of the anchor, drop the group) and 'fof' (single linkage, keep the highest q).
Differences by design: periodic tiles (no apodisation, tiling, point-source masks, inpainting or NaMaster); the
templates are analytic in Fourier space (unit-Y500 Arnaud template x Gaussian beams x tSZ SED, no pixel window --
the simulated maps have none either); a bin's power is averaged over its in-band modes only.

Template.  gt(u) = the unit-Y500 Arnaud Hankel transform of szsim.signal.template_hankel (truncated at 5 R500,
gt(0) = RATIO_5R500), interpolated with a cubic spline in ln u instead of signal.py's linear interpolation.  The
smooth version makes painting, filtering and the response integrals mutually exact; the two differ by <= 1e-4 of
g at the u that matter and sigma_Y by much less (checked in analysis/realistic/step3).  So q means what it meant in
steps 1-2.

Geometry.  Tile 600' x 600', N = 1200 pixels of 0.5'; sample (iy, ix) sits at (y, x) = (0.5 iy, 0.5 ix) arcmin;
array axis 0 is y, axis 1 is x.  Fourier modes: k0 (axis 0, full) and k1 (axis 1, rfft half plane); |k| = 36 sqrt(n2)
exactly, n2 = i0^2 + i1^2 (fundamental 2 pi / 600' = 36).  rfft coefficients X(k) = sum_p x(p) exp(-i k.p); a
field of power C_l has E|X|^2 = Npx C_l / Omega_pix ('rfft units'; the covariances below are in these units).
Halos are painted at 0.25' (beam 0.8') and Fourier-cropped to the 0.5' grid (as analysis/visual/v01_patches.py
Part B), then the painting beam is divided out.

Units: arcmin at the interface (positions, theta500, Y500 in arcmin^2), radians / sr inside the Fourier code.
"""
import os
import numpy as np
import numba as nb
from scipy import fft as sfft
from scipy.interpolate import CubicSpline
from scipy.special import j0

from . import instrument as ins
from . import signal as sg

AR = ins.ARCMIN
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP3_DATA = os.path.join(ROOT, "data_real", "step3")

N, PIX = 1200, 0.5                 # coarse grid
NF, PIXF = 2400, 0.25              # painting grid
LBOX = N * PIX                     # 600 arcmin
LMIN, LMAX = 100.0, 20000.0
DK = 2 * np.pi / (LBOX * AR)       # 36.0 exactly (2 pi / (pi/18))
BIN_FAC = 4
DBIN = BIN_FAC * np.pi / (N * PIX * AR)   # 72.0: SZiFi bin width bin_fac * pi / (nx dx)
N2MIN = int(np.ceil((LMIN / 36.0) ** 2))  # |k| >= 100  <=>  n2 >= 8
N2MAX = int(np.floor((LMAX / 36.0) ** 2))  # |k| <= 20000 <=> n2 <= 308641
FWHM_PAINT = 0.8
SIG_PAINT = FWHM_PAINT / np.sqrt(8 * np.log(2))      # arcmin
THETAS15 = np.geomspace(0.5, 15.0, 15)
Q_TH, Q_TH_NOISE, MASK_RADIUS, MAX_IT = 4.0, 4.0, 3.0, 1


# ============================================================================ template
_UG, _GG = sg.template_hankel()
_LNU = np.log(_UG[1:])
_CS = CubicSpline(_LNU, _GG[1:])
_CS_X = _CS.x.copy()
_CS_C = _CS.c.copy()                        # (4, n-1) piecewise-cubic coefficients in ln u
_LNU0, _DLNU = _LNU[0], (_LNU[-1] - _LNU[0]) / (len(_LNU) - 1)
_G0, _G1, _U1, _UMAX = _GG[0], _GG[1], _UG[1], _UG[-1]


@nb.njit(cache=True)
def _gt_scalar(u, csx, csc, lnu0, dlnu, g0, g1, u1, umax):
    if u < u1:
        return g0 + (g1 - g0) * (u / u1) ** 2
    if u > umax:
        return 0.0
    lu = np.log(u)
    i = int((lu - lnu0) / dlnu)
    if i < 0:
        i = 0
    n = csx.size - 1
    if i > n - 1:
        i = n - 1
    # guard against rounding at knots
    while i > 0 and lu < csx[i]:
        i -= 1
    while i < n - 1 and lu >= csx[i + 1]:
        i += 1
    t = lu - csx[i]
    return ((csc[0, i] * t + csc[1, i]) * t + csc[2, i]) * t + csc[3, i]


@nb.njit(cache=True)
def _gt_array(u, out, csx, csc, lnu0, dlnu, g0, g1, u1, umax):
    for i in range(u.size):
        out[i] = _gt_scalar(u[i], csx, csc, lnu0, dlnu, g0, g1, u1, umax)


def gt(u):
    """Smooth unit-Y500 Arnaud Hankel transform g(u), u = l theta500 [rad]."""
    u = np.asarray(u, float)
    out = np.empty(u.size)
    _gt_array(u.ravel(), out, _CS_X, _CS_C, _LNU0, _DLNU, _G0, _G1, _U1, _UMAX)
    return out.reshape(u.shape)


def gt_lin(u):
    """szsim.signal's own (linear) interpolation of the same table: step 1's MMF template."""
    return sg.template_hankel(u)


# ============================================================================ grid
class Grid:
    """Fourier grid of the 1200^2 tile (rfft layout) with SZiFi binning and the SO channels."""

    def __init__(self):
        i0 = np.fft.fftfreq(N, d=1.0 / N).round().astype(np.int64)        # integer wavenumbers, axis 0
        i1 = np.arange(N // 2 + 1, dtype=np.int64)                        # axis 1 (rfft)
        self.i0, self.i1 = i0, i1
        self.n2 = i0[:, None] ** 2 + i1[None, :] ** 2
        self.k0 = i0 * DK
        self.k1 = i1 * DK
        self.L = DK * np.sqrt(self.n2.astype(float))
        self.Om = (PIX * AR) ** 2
        self.Npx = N * N
        wcol = np.full(N // 2 + 1, 2.0)
        wcol[0] = 1.0
        wcol[-1] = 1.0
        self.wcol = wcol
        self.W = np.broadcast_to(wcol[None, :], self.L.shape)
        self.sel = (self.n2 >= N2MIN) & (self.n2 <= N2MAX)
        isq = np.floor(np.sqrt(self.n2.astype(float))).astype(np.int64)
        isq -= (isq * isq > self.n2)                                       # exact integer sqrt
        isq += ((isq + 1) * (isq + 1) <= self.n2)
        self.ib = isq // 2                                                 # floor(|k| / 72)
        self.nbin = int(self.ib[self.sel].max()) + 1
        self.bins_used = np.unique(self.ib[self.sel])
        # SO LAT channels
        self.freqs = ins.SO_LAT["freqs"]
        self.fwhm = ins.SO_LAT["fwhm"]
        self.sigb = self.fwhm * AR / np.sqrt(8 * np.log(2))
        self.f = ins.tsz_f_uK(self.freqs)
        self.Bp = np.exp(-0.5 * (self.L * SIG_PAINT * AR) ** 2)
        # mode list (in-band modes only), used by the numba kernels
        idx = np.flatnonzero(self.sel.ravel())
        self.m_idx = idx
        self.m_i0 = (idx // (N // 2 + 1)).astype(np.int64)
        self.m_i1 = (idx % (N // 2 + 1)).astype(np.int64)
        self.m_L = self.L.ravel()[idx]
        self.m_w = self.W.ravel()[idx]
        self.m_bin = self.ib.ravel()[idx]
        self.m_k0 = self.k0[self.m_i0]
        self.m_k1 = self.k1[self.m_i1]
        n2s = self.n2.ravel()[idx]
        self.shell_n2, self.m_shell = np.unique(n2s, return_inverse=True)  # distinct |k| shells
        self.shell_L = DK * np.sqrt(self.shell_n2.astype(float))
        self._a_modes = None

    def beams(self, Lv):
        return np.exp(-0.5 * (np.asarray(Lv)[..., None] * self.sigb) ** 2)         # (..., 6)

    def a_modes(self):
        """a_i(k) = B_i(|k|) f_i for the in-band modes, shape (nmodes, 6)."""
        if self._a_modes is None:
            self._a_modes = self.beams(self.m_L) * self.f[None, :]
        return self._a_modes

    def analytic_cov_bands(self, Lv, mode):
        """Analytic C_ij(l) = B_i B_j C_l^CMB + N_ij(l) in rfft units, shape (len(Lv), 6, 6)."""
        Lv = np.asarray(Lv, float)
        B = self.beams(Lv)
        Nn = np.moveaxis(ins.so_lat_noise(Lv, mode), 2, 0)
        C = Nn + B[:, :, None] * B[:, None, :] * ins.cmb_cl_TT(Lv)[:, None, None]
        return C * (self.Npx / self.Om)


_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = Grid()
    return _GRID


# ============================================================================ CMB + instrument noise
def sim_cmb(seed, G=None):
    """Lensed CMB in the six channels (beam-convolved), rfft units [uK], shape (6, N, N/2+1)."""
    G = G or grid()
    rng = np.random.default_rng(seed)
    g = sfft.rfft2(rng.standard_normal((N, N))) * np.sqrt(np.where(G.L > 0, ins.cmb_cl_TT(G.L.ravel()).reshape(G.L.shape), 0.0) / G.Om)
    B = np.exp(-0.5 * (G.L[None] * G.sigb[:, None, None]) ** 2)
    return B * g[None]


def sim_noise(seed, mode, G=None):
    """White noise + atmosphere (0.9 in-tube correlation), rfft units [uK], shape (6, N, N/2+1).  The same seed
    gives the same unit draws for baseline and goal (only the white amplitudes differ)."""
    G = G or grid()
    rng = np.random.default_rng(seed)
    wl = ins.SO_LAT["white"][mode]
    X = np.empty((6, N, N // 2 + 1), complex)
    for i in range(6):
        X[i] = sfft.rfft2(rng.standard_normal((N, N)) * (wl[i] / PIX))
    with np.errstate(divide="ignore"):
        shp = np.where(G.L > 0, (G.L / ins.SO_LAT["ell_knee"]) ** ins.SO_LAT["alpha_knee"], 0.0)
    red = ins.SO_LAT["N_red"] * 4 * np.pi * ins.SO_LAT["fsky"] / ins.so_t_eff()
    r = ins.SO_LAT["atm_corr"]
    for (i, j) in ins.SO_LAT["tubes"]:
        xa = sfft.rfft2(rng.standard_normal((N, N))) / np.sqrt(G.Om)
        xb = sfft.rfft2(rng.standard_normal((N, N))) / np.sqrt(G.Om)
        X[i] += np.sqrt(red[i] * shp) * xa
        X[j] += np.sqrt(red[j] * shp) * (r * xa + np.sqrt(1 - r * r) * xb)
    return X


def tsz_channels(Ysig, G=None):
    """tSZ in the six channels from the painted y map Ysig (rfft units, painting beam included), (6, N, N/2+1)."""
    G = G or grid()
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.where(G.sel, 1.0 / G.Bp, 0.0)
    B = np.exp(-0.5 * (G.L[None] * G.sigb[:, None, None]) ** 2)
    return (B * G.f[:, None, None]) * (Ysig * dec)[None]


# ============================================================================ painting
PT_DLN, PT_TH0, PT_THMAX, PT_NR = 0.04, 0.24, 62.0, 16384


def paint_table_nodes():
    nth = int(np.ceil(np.log(PT_THMAX / PT_TH0) / PT_DLN)) + 2
    th = PT_TH0 * np.exp(PT_DLN * np.arange(nth))
    rP = 5.0 * th + 7.0 * SIG_PAINT
    return th, rP


def _paint_profile_one(th, rP, nr=None):
    """P(r; th) = (1/2 pi) int dl l gt(l th) B_paint(l) J0(l r), r on [0, rP] (nr nodes), arcmin^-2 per unit Y500
    (l in arcmin^-1).  Gauss-Legendre panels (10 nodes) short enough that l r changes by <= 3 rad per panel at
    r = rP; l up to 9.5 / sigma_paint (B = exp(-45))."""
    nr = nr or PT_NR
    lmax = 9.5 / SIG_PAINT
    npan = int(np.ceil(lmax / (3.0 / rP)))
    xg, wg = np.polynomial.legendre.leggauss(10)
    edges = np.linspace(0.0, lmax, npan + 1)
    lo, hi = edges[:-1], edges[1:]
    l = (0.5 * (hi - lo)[:, None] * xg[None, :] + 0.5 * (hi + lo)[:, None]).ravel()
    w = (0.5 * (hi - lo)[:, None] * wg[None, :]).ravel()
    v = w * l * gt(l * th) * np.exp(-0.5 * (l * SIG_PAINT) ** 2) / (2 * np.pi)
    r = np.linspace(0.0, rP, nr)
    out = np.empty(nr)
    for a in range(0, nr, 256):
        out[a:a + 256] = j0(np.outer(r[a:a + 256], l)) @ v
    return out


def build_paint_tables(path=None, nproc=1):
    """Paint tables on the theta nodes (cached).  P in arcmin^-2 per unit Y500 (Y500 in arcmin^2)."""
    path = path or os.path.join(STEP3_DATA, "paint_tables.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    th, rP = paint_table_nodes()
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as p:
            tabs = p.starmap(_paint_profile_one, [(t, r) for t, r in zip(th, rP)])
    else:
        tabs = [_paint_profile_one(t, r) for t, r in zip(th, rP)]
    out = dict(tab=np.array(tabs), th=th, rP=rP, lnth0=np.log(th[0]), dln=PT_DLN, sig_paint=SIG_PAINT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


@nb.njit(cache=True, inline="always")
def _tv(tab, i, jj, nr):
    if jj < 0:
        jj = -jj
    return tab[i, jj] if jj < nr else 0.0


@nb.njit(cache=True, inline="always")
def _cub(tab, i, u, nr):
    """4-point Lagrange interpolation of the radial table row i at fractional node u >= 0 (even extension at 0,
    zero beyond the last node)."""
    j = int(u)
    t = u - j
    v0 = _tv(tab, i, j - 1, nr)
    v1 = _tv(tab, i, j, nr)
    v2 = _tv(tab, i, j + 1, nr)
    v3 = _tv(tab, i, j + 2, nr)
    return (-t * (t - 1.0) * (t - 2.0) / 6.0 * v0 + (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0 * v1
            - (t + 1.0) * t * (t - 2.0) / 2.0 * v2 + (t + 1.0) * t * (t - 1.0) / 6.0 * v3)


@nb.njit(cache=True)
def theta_node(lnth, lnth0, dln, nth):
    t = (lnth - lnth0) / dln
    if t < 0.0:
        t = 0.0
    if t > nth - 1.000001:
        t = nth - 1.000001
    i0 = int(t)
    return i0, t - i0


@nb.njit(cache=True)
def paint_fine(m, ys, xs, Ys, lnth, tab, rP, lnth0, dln, pix):
    """Add sum_h Y_h [(1-w) P_i0 + w P_i0+1](|p - x_h|) to the periodic fine map m (sample j at j * pix)."""
    n = m.shape[0]
    nth, nr = tab.shape
    for h in range(xs.size):
        i0, w = theta_node(lnth[h], lnth0, dln, nth)
        R0, R1 = rP[i0], rP[i0 + 1]
        R = max(R0, R1)
        s0, s1 = (nr - 1) / R0, (nr - 1) / R1
        y, x, Y = ys[h], xs[h], Ys[h]
        jy0 = int(np.floor((y - R) / pix)); jy1 = int(np.ceil((y + R) / pix))
        jx0 = int(np.floor((x - R) / pix)); jx1 = int(np.ceil((x + R) / pix))
        for jy in range(jy0, jy1 + 1):
            dy = jy * pix - y
            iy = jy % n
            for jx in range(jx0, jx1 + 1):
                dx = jx * pix - x
                r = np.sqrt(dx * dx + dy * dy)
                if r >= R:
                    continue
                v = 0.0
                if r < R0:
                    v += (1.0 - w) * _cub(tab, i0, r * s0, nr)
                if r < R1:
                    v += w * _cub(tab, i0 + 1, r * s1, nr)
                m[iy, jx % n] += Y * v


def paint_y(ys, xs, Ys, ths, tables, G=None):
    """Painted y map of the halos (painting beam included), as rfft coefficients on the 1200^2 grid."""
    G = G or grid()
    m = np.zeros((NF, NF))
    paint_fine(m, np.asarray(ys, float), np.asarray(xs, float), np.asarray(Ys, float), np.log(np.asarray(ths, float)),
               tables["tab"], tables["rP"], float(tables["lnth0"]), float(tables["dln"]), PIXF)
    F = sfft.rfft2(m)
    del m
    rows = np.r_[0:N // 2, NF - N // 2:NF]
    return F[rows][:, : N // 2 + 1] / 4.0


# ============================================================================ covariance and filter
def band_limit_maps(X, G=None):
    """Real-space channel maps band-limited to [LMIN, LMAX] (SZiFi filter_tmap)."""
    G = G or grid()
    return np.stack([sfft.irfft2(X[i] * G.sel, s=(N, N)) for i in range(X.shape[0])])


def binned_cov(X, G=None):
    """Binned 6x6 cross spectra C_b (rfft units): average of Re(X_i X_j^*) over the in-band modes of each bin
    (rfft multiplicities, i.e. the full plane)."""
    G = G or grid()
    nb_ = G.nbin
    bsel = G.ib[G.sel]
    wsel = G.W[G.sel]
    cnt = np.bincount(bsel, weights=wsel, minlength=nb_)
    C = np.zeros((nb_, 6, 6))
    Xs = [X[i][G.sel] for i in range(X.shape[0])]
    for i in range(6):
        for j in range(i, 6):
            v = np.bincount(bsel, weights=wsel * (Xs[i].real * Xs[j].real + Xs[i].imag * Xs[j].imag), minlength=nb_)
            v = v / np.maximum(cnt, 1e-300)
            C[:, i, j] = v
            C[:, j, i] = v
    return C


def _whitening(Cref, G):
    Wm = np.zeros_like(Cref)
    Wi = np.zeros_like(Cref)
    ev, U = np.linalg.eigh(Cref[G.bins_used])
    Wm[G.bins_used] = np.einsum("bij,bj,bkj->bik", U, ev ** -0.5, U)
    Wi[G.bins_used] = np.einsum("bij,bj,bkj->bik", U, ev ** 0.5, U)
    return Wm, Wi


@nb.njit(cache=True)
def _inpaint_diffusive(img, hy, hx, n_iter):
    """In place.  Diffusive inpainting as SZiFi's default ('diffusive', n_inpaint sweeps; own implementation, read in
    maps.diffusive_inpaint): the hole pixels start at 0 and each sweep replaces every hole pixel, in row-major order,
    by the mean of its 8 neighbours, using values already updated in the same sweep (Gauss-Seidel).  Periodic
    neighbours here (SZiFi drops neighbours outside the tile)."""
    n0, n1 = img.shape
    for k in range(hy.size):
        img[hy[k], hx[k]] = 0.0
    for _ in range(n_iter):
        for k in range(hy.size):
            y, x = hy[k], hx[k]
            ym, yp = (y - 1) % n0, (y + 1) % n0
            xm, xp = (x - 1) % n1, (x + 1) % n1
            img[y, x] = (img[ym, xm] + img[ym, x] + img[ym, xp] + img[y, xm] + img[y, xp]
                         + img[yp, xm] + img[yp, x] + img[yp, xp]) / 8.0


def szifi_blend_weights(mask):
    """Weights of SZiFi's circular_cosine_blend_adaptive (inner_zero_frac = 0; own implementation, read in maps.py):
    for each 8-connected hole of area A (pixels), R = sqrt(A / pi); a hole pixel at Euclidean distance r from the
    nearest kept pixel gets w = (1 + cos(pi r / R)) / 2 if r <= R, else 0; the blended map is w x original +
    (1 - w) x inpainted in the holes.  Holes and distances are periodic here.  Returns w (0 outside the holes)."""
    from scipy import ndimage
    hole = mask < 0.5
    n = hole.shape[0]
    if not hole.any():
        return np.zeros(hole.shape)
    lab, nl = ndimage.label(hole, structure=np.ones((3, 3), bool))
    parent = np.arange(nl + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for d in (-1, 0, 1):                       # periodic 8-connectivity across the four edges
        for a, b in ((lab[0, :], np.roll(lab[-1, :], d)), (lab[:, 0], np.roll(lab[:, -1], d))):
            for u, v in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
                ru, rv = root(u), root(v)
                if ru != rv:
                    parent[max(ru, rv)] = min(ru, rv)
    roots = np.array([root(a) for a in range(nl + 1)])
    lab = roots[lab]
    area = np.bincount(lab.ravel(), minlength=nl + 1).astype(float)
    area[0] = 0.0
    R = np.sqrt(area / np.pi)
    P = int(np.ceil(R.max())) + 3
    r = ndimage.distance_transform_edt(np.pad(hole, P, mode="wrap"))[P:P + n, P:P + n]
    Rp = R[lab]
    w = np.zeros(hole.shape)
    m = hole & (r <= Rp)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * r[m] / Rp[m]))
    return w


def szifi_masked_maps(X, mask, G=None, n_inpaint=100, blend=True):
    """SZiFi's noise maps at the iteration (mmf.py find_clusters, inpaint = True): band-limited channel maps, holes
    inpainted (diffusive, n_inpaint sweeps), then cosine-blended back towards the original map (blend = True), then
    band-limited again.  Returns the rffts (6, N, N/2+1) and the mean of w^2 over the hole pixels."""
    G = G or grid()
    maps = band_limit_maps(X, G)
    hole = mask < 0.5
    hy, hx = np.nonzero(hole)
    hy, hx = hy.astype(np.int64), hx.astype(np.int64)
    w = szifi_blend_weights(mask) if blend else np.zeros(mask.shape)
    wh = w[hole]
    Xm = np.empty_like(X)
    for i in range(maps.shape[0]):
        inp = maps[i].copy()
        _inpaint_diffusive(inp, hy, hx, int(n_inpaint))
        b = maps[i].copy()
        b[hole] = wh * maps[i][hole] + (1.0 - wh) * inp[hole]
        Xm[i] = sfft.rfft2(b)
    return Xm, (float(np.mean(wh ** 2)) if wh.size else 0.0)


def masked_cov(X, mask, G=None, Cref=None, method="whitened", lsep=3000.0, n_inpaint=100, blend=True):
    """Covariance of the tile with the cluster mask applied (binary discs, 1 = kept), pseudo-spectra divided by the
    unmasked fraction f.
      method 'binary'   : the band-limited channel maps are masked as they are (the literal 'fsky' estimate).  The
                          sharp disc edges leak the steep low-l CMB and atmosphere power to high l (biased).
      method 'whitened' : each in-band mode is first whitened with the unmasked estimate Cref of its bin
                          (Z = Cref^-1/2 X), the whitened maps are masked, their binned pseudo-spectra divided by f,
                          and the result is un-whitened (Cref^1/2 C_w Cref^1/2).  The whitened field is close to white,
                          so the mask does not leak power across l.  Default.
      method 'szifi'    : SZiFi's own estimate with its defaults (added by the step-3 arbiter): holes inpainted
                          (diffusive, n_inpaint sweeps) and cosine-blended back towards the original map (blend), full-
                          tile binned spectra, bins centred above l = lsep divided by f (spec.correct_for_inpainting).
    Returns (C_b, f)."""
    G = G or grid()
    f = float(mask.mean())
    if method == "binary":
        maps = band_limit_maps(X, G)
        Xm = np.stack([sfft.rfft2(maps[i] * mask) for i in range(maps.shape[0])])
        return binned_cov(Xm, G) / f, f
    if method == "szifi":
        Xm, _ = szifi_masked_maps(X, mask, G, n_inpaint=n_inpaint, blend=blend)
        C = binned_cov(Xm, G)
        C[(np.arange(G.nbin) + 0.5) * DBIN > lsep] /= f
        return C, f
    Wm, Wi = _whitening(Cref, G)
    Xs = np.stack([X[i].ravel()[G.m_idx] for i in range(X.shape[0])], 1)          # (nm, 6)
    Zs = np.empty_like(Xs)
    _apply_bin_mat(Wm, G.m_bin, Xs, Zs)
    Zm = np.empty_like(X)
    for i in range(X.shape[0]):
        Z = np.zeros(G.L.size, complex)
        Z[G.m_idx] = Zs[:, i]
        Zm[i] = sfft.rfft2(sfft.irfft2(Z.reshape(G.L.shape), s=(N, N)) * mask)
    Cw = binned_cov(Zm, G) / f
    C = np.zeros_like(Cw)
    C[G.bins_used] = np.einsum("bij,bjk,bkl->bil", Wi[G.bins_used], Cw[G.bins_used], Wi[G.bins_used])
    return C, f


_ABC = {}


def analytic_bin_cov(mode, G=None):
    """The analytic covariance averaged over the in-band modes of each bin (rfft units; cached per mode)."""
    G = G or grid()
    if mode in _ABC:
        return _ABC[mode]
    C = np.zeros((G.nbin, 6, 6))
    cnt = np.bincount(G.m_bin, weights=G.m_w, minlength=G.nbin)
    for a in range(0, G.m_L.size, 50_000):
        sl = slice(a, a + 50_000)
        Cm = G.analytic_cov_bands(G.m_L[sl], mode)
        for i in range(6):
            for j in range(i, 6):
                v = np.bincount(G.m_bin[sl], weights=G.m_w[sl] * Cm[:, i, j], minlength=G.nbin)
                C[:, i, j] += v
                if j != i:
                    C[:, j, i] += v
    C /= np.maximum(cnt, 1e-300)[:, None, None]
    _ABC[mode] = C
    return C


@nb.njit(cache=True)
def _apply_bin_mat(Mb, mbin, x, out):
    """out[m] = Mb[bin(m)] @ x[m] (complex x)."""
    for m in range(x.shape[0]):
        b = mbin[m]
        for i in range(x.shape[1]):
            s = 0.0 + 0.0j
            for j in range(x.shape[1]):
                s += Mb[b, i, j] * x[m, j]
            out[m, i] = s


@nb.njit(cache=True)
def _apply_bin_inv(Ci, mbin, a, out):
    for m in range(a.shape[0]):
        b = mbin[m]
        for i in range(a.shape[1]):
            s = 0.0
            for j in range(a.shape[1]):
                s += Ci[b, i, j] * a[m, j]
            out[m, i] = s


class Filter:
    """Per-mode MMF weights: Cia = C(k)^-1 a(k), aCa = a^T C^-1 a, on the in-band modes."""

    def __init__(self, G, Cbin=None, mode=None, analytic=False, template="lin", thetas=THETAS15):
        self.G = G
        self.analytic_mode = mode if analytic else None
        a = G.a_modes()
        if analytic:
            Cm = G.analytic_cov_bands(G.m_L, mode)
            x = np.linalg.solve(Cm, a[..., None])[..., 0]
            self.Cbin = None
        else:
            self.Cbin = Cbin
            Ci = np.zeros_like(Cbin)
            Ci[G.bins_used] = np.linalg.inv(Cbin[G.bins_used])
            self.Cinv = Ci
            x = np.empty_like(a)
            _apply_bin_inv(Ci, G.m_bin, a, x)
        self.Cia = x                                       # (nmodes, 6)
        self.aCa = np.einsum("ni,ni->n", a, x)             # (nmodes,)
        self.thetas = np.asarray(thetas, float)
        self.gfun = gt if template == "cub" else gt_lin
        self.template = template
        self.tau = np.empty((len(self.thetas), G.m_L.size))                              # (nth, nmodes)
        self.norm = np.empty(len(self.thetas))
        wa = G.m_w * self.aCa
        for j, t in enumerate(self.thetas):
            self.tau[j] = self.gfun(G.m_L * t * AR) / G.Om
            self.norm[j] = np.dot(wa, self.tau[j] ** 2)
        self.sigma = 1.0 / np.sqrt(self.norm)              # sr
        self.sigma_arcmin2 = self.sigma / AR ** 2

    def CX(self, X):
        """sum_i Cia_i X_i on the in-band modes (complex, nmodes)."""
        G = self.G
        out = np.zeros(G.m_L.size, complex)
        for i in range(X.shape[0]):
            out += self.Cia[:, i] * X[i].ravel()[G.m_idx]
        return out

    def yhat_map(self, cx, j):
        """Y500 estimate map [sr] at scale j from CX."""
        G = self.G
        Z = np.zeros(G.L.size, complex)
        Z[G.m_idx] = self.tau[j] * cx / self.norm[j]
        return sfft.irfft2(Z.reshape(G.L.shape), s=(N, N)) * G.Npx

    def q_cube(self, cx):
        q = np.empty((len(self.thetas), N, N))
        for j in range(len(self.thetas)):
            q[j] = self.yhat_map(cx, j)
            q[j] /= self.sigma[j]
        return q

    def release(self):
        """Drop the per-mode arrays (keep sigma, norm, Cbin)."""
        self.tau = None
        self.Cia = None


# ============================================================================ detections
@nb.njit(cache=True)
def find_peaks(q, qth):
    """SZiFi 'maxima_lomem' peaks of the (nth, N, N) S/N cube, periodic in the two map axes.  Returns (j, iy, ix) in
    catalogue order (scale, then row-major pixel)."""
    nth, n0, n1 = q.shape
    out_j = []; out_y = []; out_x = []
    for j in range(nth):
        for iy in range(n0):
            for ix in range(n1):
                v = q[j, iy, ix]
                if v <= qth:
                    continue
                if not (v > q[j, (iy + 1) % n0, ix] and v > q[j, (iy - 1) % n0, ix]
                        and v > q[j, iy, (ix + 1) % n1] and v > q[j, iy, (ix - 1) % n1]):
                    continue
                if j == 0:
                    ok = (v > 0.0) and (nth == 1 or v >= q[1, iy, ix])
                elif j == nth - 1:
                    ok = v > q[j - 1, iy, ix]
                else:
                    ok = (v > q[j - 1, iy, ix]) and (v >= q[j + 1, iy, ix])
                if ok:
                    out_j.append(j); out_y.append(iy); out_x.append(ix)
    return np.array(out_j, dtype=np.int64), np.array(out_y, dtype=np.int64), np.array(out_x, dtype=np.int64)


@nb.njit(cache=True)
def peaks_one_scale(qm, q0, qp, j, nth, qth):
    """Peaks at scale j given the maps at j-1 (qm), j (q0), j+1 (qp) -- the same rules as find_peaks."""
    n0, n1 = q0.shape
    out_y = []; out_x = []
    for iy in range(n0):
        for ix in range(n1):
            v = q0[iy, ix]
            if v <= qth:
                continue
            if not (v > q0[(iy + 1) % n0, ix] and v > q0[(iy - 1) % n0, ix]
                    and v > q0[iy, (ix + 1) % n1] and v > q0[iy, (ix - 1) % n1]):
                continue
            if j == 0:
                ok = (v > 0.0) and (nth == 1 or v >= qp[iy, ix])
            elif j == nth - 1:
                ok = v > qm[iy, ix]
            else:
                ok = (v > qm[iy, ix]) and (v >= qp[iy, ix])
            if ok:
                out_y.append(iy); out_x.append(ix)
    return np.array(out_y, dtype=np.int64), np.array(out_x, dtype=np.int64)


@nb.njit(cache=True)
def disc_mask(iy, ix, rad_arcmin, n, pix):
    """Binary mask (1 = kept): 0 within distance <= rad of each pixel centre (periodic), as SZiFi's ps_mask."""
    m = np.ones((n, n))
    for k in range(iy.size):
        R = rad_arcmin[k] / pix
        Ri = int(np.ceil(R))
        for dy in range(-Ri, Ri + 1):
            for dx in range(-Ri, Ri + 1):
                if dx * dx + dy * dy <= R * R:
                    m[(iy[k] + dy) % n, (ix[k] + dx) % n] = 0.0
    return m


def detect(filt, cx, qth=Q_TH, keep_cube=False):
    """Peaks of the S/N cube.  Without keep_cube the scales are processed with a rolling window of three maps
    (SZiFi's 'maxima_lomem'); the result is identical to find_peaks on the full cube."""
    nth = len(filt.thetas)
    if keep_cube:
        q = filt.q_cube(cx)
        j, iy, ix = find_peaks(q, qth)
        qv = q[j, iy, ix]
    else:
        q = None
        qmap = lambda k: filt.yhat_map(cx, k) / filt.sigma[k]
        prev, cur = None, qmap(0)
        J, IY, IX, QV = [], [], [], []
        for k in range(nth):
            nxt = qmap(k + 1) if k + 1 < nth else cur
            a, b = peaks_one_scale(prev if prev is not None else cur, cur, nxt, k, nth, qth)
            J.append(np.full(a.size, k, np.int64)); IY.append(a); IX.append(b); QV.append(cur[a, b])
            prev, cur = cur, nxt
        j, iy, ix, qv = np.concatenate(J), np.concatenate(IY), np.concatenate(IX), np.concatenate(QV)
    det = dict(j=j, iy=iy, ix=ix, q=qv, theta=filt.thetas[j], Y=qv * filt.sigma_arcmin2[j], y=iy * PIX, x=ix * PIX)
    return det, q


def run_finder(X, G=None, thetas=THETAS15, max_it=MAX_IT, analytic_mode=None, template="lin", keep_cube=False,
               mask_method="whitened", mask_kw=None):
    """SZiFi-like iterative MMF on one tile.  X: channel rffts (6, N, N/2+1).  Returns a list (one dict per
    iteration) with the filter, the binned covariance, the unmasked fraction, the catalogue and (optionally) the S/N
    cube of the last iteration.  analytic_mode: use the analytic CMB + noise covariance per mode instead (no
    iteration).  template: 'lin' = step 1's MMF template (signal.template_hankel), 'cub' = its smooth version.
    mask_method / mask_kw: how the masked covariance is estimated (masked_cov; default 'whitened')."""
    G = G or grid()
    its = []
    if analytic_mode is not None:
        filt = Filter(G, mode=analytic_mode, analytic=True, template=template, thetas=thetas)
        det, q = detect(filt, filt.CX(X), keep_cube=keep_cube)
        its.append(dict(filt=filt, Cbin=None, fsky=1.0, det=det, q=q))
        return its
    mask = None
    C0 = None
    for it in range(max_it + 1):
        if mask is None:
            Cb, fs = binned_cov(X, G), 1.0
            C0 = Cb
        else:
            Cb, fs = masked_cov(X, mask, G, Cref=C0, method=mask_method, **(mask_kw or {}))
        filt = Filter(G, Cbin=Cb, template=template, thetas=thetas)
        last = it == max_it
        det, q = detect(filt, filt.CX(X), keep_cube=(keep_cube and last))
        if its:
            its[-1]["filt"].release()
        its.append(dict(filt=filt, Cbin=Cb, fsky=fs, det=det, q=q))
        if last:
            break
        sel = det["q"] > Q_TH_NOISE
        if not np.any(sel):
            break
        mask = disc_mask(det["iy"][sel], det["ix"][sel], MASK_RADIUS * det["theta"][sel], N, PIX)
    return its


# ============================================================================ merging
def _pdist(y, x, y0, x0):
    dy = (y - y0 + LBOX / 2) % LBOX - LBOX / 2
    dx = (x - x0 + LBOX / 2) % LBOX - LBOX / 2
    return np.hypot(dy, dx)


def merge_closest(y, x, q, radius):
    """SZiFi merge_detections(mode='closest'): indices kept, in the order they are produced."""
    rem = np.arange(len(q))
    keep = []
    while rem.size:
        a = rem[0]
        grp = rem[_pdist(y[rem], x[rem], y[a], x[a]) < radius]
        keep.append(grp[np.argmax(q[grp])])
        rem = np.setdiff1d(rem, grp, assume_unique=True)
    return np.array(keep, dtype=np.int64)


def merge_fof(y, x, q, radius):
    """Friends-of-friends groups (single linkage, linking length = radius); keep the highest q of each group."""
    n = len(q)
    lab = -np.ones(n, dtype=np.int64)
    g = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = g
        stack = [s]
        while stack:
            a = stack.pop()
            nb_ = np.flatnonzero((lab < 0) & (_pdist(y, x, y[a], x[a]) < radius))
            lab[nb_] = g
            stack.extend(nb_.tolist())
        g += 1
    keep = [np.flatnonzero(lab == k)[np.argmax(q[lab == k])] for k in range(g)]
    return np.array(sorted(keep), dtype=np.int64)


MERGES = {"none": None, "closest1": ("closest", 1.0), "closest2": ("closest", 2.0), "closest10": ("closest", 10.0),
          "fof10": ("fof", 10.0)}


def apply_merge(det, name, qcut):
    """Indices of the catalogue after the q cut and the merge rule (SZiFi's Planck scripts cut, then merge)."""
    idx = np.flatnonzero(det["q"] > qcut)
    rule = MERGES[name]
    if rule is None or idx.size == 0:
        return idx
    fn = merge_closest if rule[0] == "closest" else merge_fof
    k = fn(det["y"][idx], det["x"][idx], det["q"][idx], rule[1])
    return idx[k]


# ============================================================================ responses and decomposition
def lattice_gauss_rules(G=None, nmax=12):
    """Per SZiFi bin, an n-point Gauss rule for the discrete measure sum_{in-band modes k in bin} w_k delta(l - |k|)
    (w = rfft multiplicity).  sum_b sum_m W_bm f(l_bm) = sum_k w_k f(|k|) exactly for f a polynomial of degree
    <= 2n-1 within each bin.  Returns nodes l (rad^-1), weights, bin index."""
    G = G or grid()
    nodes, wts, bins = [], [], []
    for b in G.bins_used:
        m = G.m_bin == b
        sh, inv = np.unique(G.m_shell[m], return_inverse=True)
        w = np.bincount(inv, weights=G.m_w[m])
        x = G.shell_L[sh]
        n = min(nmax, x.size)
        # Lanczos on diag(x) with start vector sqrt(w): Jacobi matrix of the discrete measure
        c = 0.5 * (x.max() + x.min()); h = max(0.5 * (x.max() - x.min()), 1e-12)
        xs = (x - c) / h
        q0 = np.sqrt(w / w.sum())
        Q = [q0]
        alpha, beta = [], []
        qprev = np.zeros_like(q0); bprev = 0.0
        for k in range(n):
            v = xs * Q[-1]
            a_ = Q[-1] @ v
            v = v - a_ * Q[-1] - bprev * qprev
            for qq in Q:                                  # full reorthogonalisation
                v -= (qq @ v) * qq
            alpha.append(a_)
            bk = np.linalg.norm(v)
            if k < n - 1:
                beta.append(bk)
                qprev = Q[-1]; bprev = bk
                Q.append(v / bk)
        J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        ev, evec = np.linalg.eigh(J)
        nodes.append(c + h * ev)
        wts.append(w.sum() * evec[0] ** 2)
        bins.append(np.full(n, b))
    return np.concatenate(nodes), np.concatenate(wts), np.concatenate(bins)


R_NEAR_MAX, R_NEAR_N = 40.0, 2001                  # arcmin, 0.02' spacing
R_FAR_MIN, R_FAR_MAX, R_FAR_N = 39.0, 427.0, 1941   # 0.2' spacing
R_SWITCH = 39.9                                     # near table below, far table above (interior stencils)


class ResponseBasis:
    """Things that depend only on the grid: lattice Gauss nodes, J0 matrices, template values at the nodes."""

    def __init__(self, G=None, tables=None):
        G = G or grid()
        self.G = G
        self.ln, self.lw, self.lb = lattice_gauss_rules(G)
        self.rn = np.linspace(0.0, R_NEAR_MAX, R_NEAR_N)
        self.rf = np.linspace(R_FAR_MIN, R_FAR_MAX, R_FAR_N)
        self.Jn = np.empty((self.ln.size, self.rn.size))
        self.Jf = np.empty((self.ln.size, self.rf.size))
        for a in range(0, self.ln.size, 256):
            self.Jn[a:a + 256] = j0(np.outer(self.ln[a:a + 256], self.rn * AR))
            self.Jf[a:a + 256] = j0(np.outer(self.ln[a:a + 256], self.rf * AR))
        self.an = G.beams(self.ln) * G.f[None, :]
        tables = tables or build_paint_tables()
        self.th_nodes = tables["th"]
        self.lnth0, self.dln = float(tables["lnth0"]), float(tables["dln"])
        self.Gn = np.stack([gt(self.ln * t * AR) / G.Om for t in self.th_nodes])        # (nth, nl)
        # template values on the distinct |k| shells (exact lattice sums), filled lazily per theta node
        self.Gs = np.zeros((len(self.th_nodes), G.shell_L.size))
        self._gs_done = np.zeros(len(self.th_nodes), bool)

    def gs_rows(self, i0):
        for i in (i0, i0 + 1):
            if not self._gs_done[i]:
                self.Gs[i] = gt(self.G.shell_L * self.th_nodes[i] * AR) / self.G.Om
                self._gs_done[i] = True
        return self.Gs


def response_tables(filt, basis):
    """Angle-averaged lattice response R[j, i, r] of the tile's filter at scale j to a unit-Y500 template on theta node
    i, on the near and far r grids (exact for the angle average: lattice Gauss rule per bin)."""
    G = basis.G
    if filt.Cbin is not None:
        aCa_n = np.einsum("ni,nij,nj->n", basis.an, filt.Cinv[basis.lb], basis.an)
    else:
        Cn = G.analytic_cov_bands(basis.ln, filt.analytic_mode)
        aCa_n = np.einsum("ni,ni->n", basis.an, np.linalg.solve(Cn, basis.an[..., None])[..., 0])
    nth = len(filt.thetas)
    Rn = np.empty((nth, len(basis.th_nodes), basis.rn.size))
    Rf = np.empty((nth, len(basis.th_nodes), basis.rf.size))
    for j in range(nth):
        Wj = basis.lw * filt.gfun(basis.ln * filt.thetas[j] * AR) / G.Om * aCa_n / filt.norm[j]
        M = basis.Gn * Wj[None, :]
        Rn[j] = M @ basis.Jn
        Rf[j] = M @ basis.Jf
    return Rn, Rf


@nb.njit(cache=True)
def _resp_eval(Rn, Rf, j, i0, w, r, rswitch, drn, rf_min, drf):
    nn = Rn.shape[2]
    nf = Rf.shape[2]
    if r < rswitch:
        u = r / drn
        return (1.0 - w) * _cub(Rn[j], i0, u, nn) + w * _cub(Rn[j], i0 + 1, u, nn)
    u = (r - rf_min) / drf
    if u >= nf - 3:
        return 0.0
    return (1.0 - w) * _cub(Rf[j], i0, u, nf) + w * _cub(Rf[j], i0 + 1, u, nf)


@nb.njit(cache=True)
def table_contributions(dj, dy, dx, hy, hx, hY, hlnth, Rn, Rf, lnth0, dln, rswitch, drn, rf_min, drf, box,
                        topk, out_sum, out_top_idx, out_top_val, out_n):
    """For each detection: sum over ALL halos of the table contribution Y_h R(r_h) [arcmin^2 Y units as hY], and
    the topk largest contributions (indices, values)."""
    nth_nodes = Rn.shape[1]
    for d in range(dj.size):
        j = dj[d]
        s = 0.0
        tv = np.full(topk, -1e300)
        ti = np.full(topk, -1, np.int64)
        cnt = 0
        for h in range(hY.size):
            ddy = (hy[h] - dy[d] + 0.5 * box) % box - 0.5 * box
            ddx = (hx[h] - dx[d] + 0.5 * box) % box - 0.5 * box
            r = np.sqrt(ddy * ddy + ddx * ddx)
            i0, w = theta_node(hlnth[h], lnth0, dln, nth_nodes)
            c = hY[h] * _resp_eval(Rn, Rf, j, i0, w, r, rswitch, drn, rf_min, drf)
            s += c
            if c != 0.0:
                cnt += 1
            if c > tv[topk - 1]:
                k = topk - 1
                while k > 0 and c > tv[k - 1]:
                    tv[k] = tv[k - 1]; ti[k] = ti[k - 1]
                    k -= 1
                tv[k] = c; ti[k] = h
        out_sum[d] = s
        out_n[d] = cnt
        for k in range(topk):
            out_top_idx[d, k] = ti[k]
            out_top_val[d, k] = tv[k]


@nb.njit(cache=True)
def lattice_contribution(m_k0, m_k1, m_w, m_shell, aCa, tau_j, norm_j, Gs, i0, w, Y, dy, dx):
    """Exact lattice response: Y sum_k w_k tau_j(k) aCa(k) G(k) cos(k.D) / norm_j, D = detection - halo [rad],
    G = (1-w) Gs[i0] + w Gs[i0+1] on the shells (the painted node mixture)."""
    s = 0.0
    for m in range(m_w.size):
        sh = m_shell[m]
        g = (1.0 - w) * Gs[i0, sh] + w * Gs[i0 + 1, sh]
        s += m_w[m] * tau_j[m] * aCa[m] * g * np.cos(m_k0[m] * dy + m_k1[m] * dx)
    return Y * s / norm_j


def exact_contrib(filt, basis, j, Y, theta, dy_arcmin, dx_arcmin):
    """Exact lattice contribution [same units as Y] of a halo (Y, theta) at offset D = detection - halo (arcmin)."""
    G = basis.G
    i0, w = theta_node(np.log(theta), basis.lnth0, basis.dln, len(basis.th_nodes))
    return lattice_contribution(G.m_k0, G.m_k1, G.m_w, G.m_shell, filt.aCa, filt.tau[j], filt.norm[j],
                                basis.gs_rows(i0), i0, w, Y, dy_arcmin * AR, dx_arcmin * AR)
