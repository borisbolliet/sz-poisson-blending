"""Verifier library, step 3a, lens 'szifi'.

Running SZiFi's own code (/Users/boris/GitHub/szifi) was refused by this session's permission classifier
("Code from External"), so nothing from the clone is imported, run or copied.  This module is our OWN emulation of the
conventions SZiFi's code uses (written from reading mmf.py, spec.py, maps.py, model.py, cat.py), applied to the
build's tiles.  Every SZiFi behaviour emulated here is listed with the SZiFi function it comes from:

  T  template (mmf.filter_maps.find_clusters -> model.gnfw.get_t_map_convolved, eval_type 'standard'):
       Arnaud+10 GNFW pressure, truncated at 5 R500 (3D), integrated along the line of sight (our own quadrature),
       tabulated on 1000 equally spaced radii 0..5 theta500, linearly interpolated at the pixel CENTRES of the tile,
       with the template centre at index N/2 - 0.5 (between four pixels); Gaussian beam B = exp(-fwhm^2 l(l+1)/(16 ln 2))
       applied by FFT; normalised to the central y (q does not depend on it); modes outside [lmin, lmax] zeroed.
  Q  S/N map (mmf.get_mmf_q_map): filter_i = IFFT(sum_j C^-1_ij a_j T_j), y = sum_i LINEAR convolution
       (scipy.signal.fftconvolve mode 'same', zero padded, not periodic) of the l-filtered map with filter_i,
       norm = sum(tem a * filter), sigma = norm^-1/2, q = y/sigma.  With N even the 'same' crop makes the S/N at pixel n
       the matched value for a cluster centred at n - 1/2 (both axes).
  C  isotropic covariance (spec.cross_spec.get_cross_spec / get_cov, cov_type 'isotropic', bin_fac 4): binned
       periodogram of the l-filtered maps over ALL modes with l0 <= l < lf, bins of width 4 pi/(N dx) = 72; first bin
       dropped; each mode takes the bin whose centre is nearest (scipy interp1d 'nearest', ties to the lower bin).
       (SZiFi computes the pseudo-spectra with NaMaster; for an all-ones power-spectrum mask with decouple 'none',
       'fsky' or 'master' this is the binned periodogram, which is also what SZiFi's own 'custom' implementation
       computes.  NaMaster is not installed, so its binning could not be checked.)
  D  detections ('maxima_lomem', mmf.make_detections2 / get_maxima_mask_2d): q > 4, strictly above the in-map
       neighbours along both axes (tile edges padded with -inf, not periodic), q_j > q_{j-1} (q_{-1} = 0) and
       q_j >= q_{j+1}; last scale q > previous.  Order: scale, then row-major.
  I  iteration (mmf.cluster_finder.find_clusters, get_cluster_mask; maps.inpaint_freq, diffusive_inpaint,
       circular_cosine_blend_adaptive; spec.correct_for_inpainting): discs of radius 3 theta_est around q > 4
       detections (pixel centres within distance <= r of the detection pixel, not periodic); WITHOUT inpainting the
       cluster mask is never used by the covariance estimate (it only enters inpainting and correct_for_inpainting),
       so the iteration reproduces iteration 0.  WITH inpainting (SZiFi default inpaint=True, inpaint_type
       'diffusive', n_inpaint 100, lsep 3000): every masked pixel is replaced, 100 times in row-major order, by the
       mean of its in-map 8 neighbours (in place); each 8-connected hole is then blended back towards the original
       map, w = 0.5 (1 + cos(pi d / R)), d = distance to the nearest unmasked pixel (in the hole's bounding box
       padded by 2), R = sqrt(area / pi), value = w original + (1 - w) inpainted; the result is l-filtered and its
       spectra divided by the unmasked fraction only in bins with centre l > 3000.

The tiles are regenerated with the build's own simulator (szsim.finder, analysis/realistic/step3/s3lib.py) from their
seeds; v00_regen_check.py shows that this reproduces the stored catalogues.
"""
import os, sys, resource
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "out", "numba_cache"))
sys.dont_write_bytecode = True
import numpy as np
import numba as nb
from scipy import fft as sfft
from scipy import integrate, ndimage
from scipy.interpolate import interp1d

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
for _p in (os.path.join(ROOT, "analysis", "q2_maps"), os.path.join(ROOT, "analysis", "realistic", "step3"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s3lib as S                                     # noqa: E402  (the build's helpers)
from szsim import finder as F, instrument as ins      # noqa: E402  (the build's finder and instrument)

MODES = S.MODES
N = F.N
PIXA = F.PIX                                   # 0.5 arcmin
DX = 0.5 / 60. / 180. * np.pi                  # SZiFi's dx [rad] (dx_arcmin / 60 / 180 * pi)
AR = ins.ARCMIN
THETAS = F.THETAS15
LMIN, LMAX = 100.0, 20000.0
BIN_FAC = 4
LSEP = 3000.0
N_INPAINT = 100
Q_TH, Q_TH_NOISE, MASK_RADIUS = 4.0, 4.0, 3.0
OUTD = os.path.join(HERE, "out")
DATA3 = F.STEP3_DATA
GN = dict(c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)   # Arnaud+10 (SZiFi params_sz and step 1)
FWHM = np.asarray(ins.SO_LAT["fwhm"], float)                     # arcmin
SED = ins.tsz_f_uK(ins.SO_LAT["freqs"])                          # uK per unit y (the build's tSZ SED)
HALF = N // 2 + 1


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9      # macOS: bytes


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


# ============================================================================ the build's tiles
_TAB = {}


def tables():
    if "t" not in _TAB:
        _TAB["t"] = F.build_paint_tables()     # loads data_real/step3/paint_tables.npz (exists; never rewritten)
    return _TAB["t"]


def load_stored_tile(pid):
    return load_npz(os.path.join(DATA3, "tiles", f"tile_{pid:03d}.npz"))


def tsz_tile(pid):
    """Painted tSZ channels of realistic tile pid (rfft, the build's units) and its truth table."""
    G = F.grid()
    d = S.truth(pid)
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables(), G), G)
    return Xt, d


def make_tile(pid, mode, Xt=None):
    """Six-channel rfft of realistic tile pid at a noise level (tSZ + CMB seed 300000+pid + noise seed 400000+pid)."""
    G = F.grid()
    if Xt is None:
        Xt, d = tsz_tile(pid)
    else:
        d = None
    Xc = F.sim_cmb(S.SEED_CMB + pid, G)
    Xn = F.sim_noise(S.SEED_NOISE + pid, mode, G)
    return Xt + Xc + Xn, dict(tsz=Xt, cmb=Xc, noise=Xn, truth=d)


def make_noise_only(pid, mode):
    G = F.grid()
    return F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)


def make_injection(mode, k):
    """Injection tile k (stored layout data_real/step3/injections/inj_MODE_K.npz; CMB 500000+k, noise 600000+k)."""
    G = F.grid()
    r = load_npz(os.path.join(DATA3, "injections", f"inj_{mode}_{k:03d}.npz"))
    th = float(r["theta_inj"])
    Xt = F.tsz_channels(F.paint_y(r["y"], r["x"], r["Y"], np.full(r["y"].size, th), tables(), G), G)
    X = Xt + F.sim_cmb(S.SEED_INJ_CMB + k, G) + F.sim_noise(S.SEED_INJ_NOISE + k, mode, G)
    return X, r


# ============================================================================ SZiFi-convention grid (rfft half plane)
class SZGrid:
    """l of every rfft mode computed as SZiFi does (fftfreq(n, dx) 2 pi), SZiFi's bins, band and nearest-bin map."""

    def __init__(self):
        lx = np.fft.fftfreq(N, DX) * 2. * np.pi
        ly = np.fft.rfftfreq(N, DX) * 2. * np.pi
        self.L = np.sqrt(lx[:, None] ** 2 + ly[None, :] ** 2)
        w = np.full(HALF, 2.0)
        w[0] = 1.0
        w[-1] = 1.0
        self.W = np.broadcast_to(w[None, :], self.L.shape)
        self.band = (self.L >= LMIN) & (self.L <= LMAX)            # filter_tmap keeps lmin <= l <= lmax
        self.l0 = np.arange(N / BIN_FAC) * BIN_FAC * np.pi / (N * DX)
        self.lf = (np.arange(N / BIN_FAC) + 1) * BIN_FAC * np.pi / (N * DX)
        self.lc = 0.5 * (self.l0 + self.lf)
        self.nb = self.l0.size
        b = np.searchsorted(self.l0, self.L, side="right") - 1      # l0[b] <= l < lf[b]
        b[self.L >= self.lf[-1]] = -1                               # beyond the last bin: in no bin
        self.binm = b
        near = interp1d(self.lc[1:], np.arange(self.nb - 1, dtype=float), kind="nearest", bounds_error=False,
                        fill_value="extrapolate")(self.L.ravel())
        self.near = (np.rint(near).astype(np.int64) + 1).reshape(self.L.shape)   # index into the full bin list
        idx = np.flatnonzero(self.band.ravel())
        self.m_idx = idx
        self.m_L = self.L.ravel()[idx]
        self.m_w = self.W.ravel()[idx]
        self.m_near = self.near.ravel()[idx]
        i0 = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
        i1 = np.arange(HALF, dtype=np.int64)
        self.m_i0 = i0[idx // HALF]
        self.m_i1 = i1[idx % HALF]
        # template centre at index N/2 - 1/2: phase exp(-i k.c)
        c = N / 2 - 0.5
        self.m_phase_c = np.exp(-2j * np.pi * (self.m_i0 + self.m_i1) * c / N)
        # half-pixel evaluation shift (value at n - 1/2): exp(-i k.(1/2, 1/2))
        self.m_phase_half = np.exp(-2j * np.pi * (self.m_i0 + self.m_i1) * 0.5 / N)


_SZG = {}


def szgrid():
    if "g" not in _SZG:
        _SZG["g"] = SZGrid()
    return _SZG["g"]


def to_modes(Xfull, g=None):
    """(c, N, HALF) -> (c, nm) on the in-band modes."""
    g = g or szgrid()
    return np.stack([Xfull[i].ravel()[g.m_idx] for i in range(Xfull.shape[0])])


def from_modes(Zm, g=None):
    g = g or szgrid()
    Z = np.zeros(N * HALF, complex)
    Z[g.m_idx] = Zm
    return Z.reshape(N, HALF)


# ============================================================================ SZiFi-convention template
def gnfw_P(x):
    cx = GN["c500"] * x
    return 1.0 / (cx ** GN["gamma"] * (1.0 + cx ** GN["alpha"]) ** ((GN["beta"] - GN["gamma"]) / GN["alpha"]))


_PT = {}


def projected_table(n=1000, X=5.0):
    """Our own line-of-sight integral of the GNFW pressure truncated at X R500 (3D), on n equally spaced radii
    s = 0..X (units of R500), as SZiFi tabulates it (theta_vec = linspace(0, 5 theta500, 1000))."""
    key = (n, X)
    if key in _PT:
        return _PT[key]
    s = np.linspace(0.0, X, n)
    p = np.zeros(n)
    for i, si in enumerate(s):
        if si >= X:
            continue
        Lz = np.sqrt(X * X - si * si)
        if si > 0:
            p[i] = 2 * integrate.quad(lambda z: gnfw_P(np.sqrt(si * si + z * z)), 0.0, Lz, limit=500, epsabs=0,
                                      epsrel=1e-12)[0]
        else:   # cusp z^-gamma at 0: z = Lz t^2
            p[i] = 2 * integrate.quad(lambda t: 2 * Lz * t * gnfw_P(Lz * t * t), 0.0, 1.0, limit=500, epsabs=0,
                                      epsrel=1e-12)[0]
    _PT[key] = (s, p)
    return s, p


def szifi_ymap(theta_arcmin, pix_centre=N / 2 - 0.5):
    """SZiFi's real-space template (no beam): the tabulated profile interpolated at the pixel centres, centre at index
    N/2 - 1/2, zero beyond 5 theta500; normalised to the central value (y0 = 1)."""
    s, p = projected_table()
    th = theta_arcmin / 60. / 180. * np.pi
    c = (np.arange(N) - pix_centre) * DX
    r = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    return np.interp(r, s * th, p, right=0.0) / p[0]


def szifi_beam(Lv, fwhm_arcmin):
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.) ** 2 / (16. * np.log(2.)) * Lv * (Lv + 1.))


def template_modes(theta_arcmin, kind, g=None):
    """Per-channel template DFT on the in-band modes, (6, nm), unnormalised-DFT units, centred at N/2 - 1/2.
    kind 'szifi'  : SZiFi's real-space sampled template x SZiFi Gaussian beam (l(l+1));
         'szifi_bb': the same sampled template x the build's beams (l^2) -- isolates the beam convention;
         'build'  : the build's Fourier template (step 1's g_lin(l theta500) / Omega_pix) x the build's beams, with
                    the SZiFi centring phase (what SZiFi's filtering does with the build's template)."""
    g = g or szgrid()
    if kind in ("szifi", "szifi_bb"):
        Yk = sfft.rfft2(szifi_ymap(theta_arcmin)).ravel()[g.m_idx]
        if kind == "szifi":
            B = np.stack([szifi_beam(g.m_L, fw) for fw in FWHM])
        else:
            B = np.exp(-0.5 * (g.m_L[None, :] * (FWHM[:, None] * AR / np.sqrt(8 * np.log(2)))) ** 2)
        return Yk[None, :] * B
    if kind == "build":
        G = F.grid()
        assert np.array_equal(G.m_idx, g.m_idx)
        tau = F.gt_lin(G.m_L * theta_arcmin * AR) / G.Om
        return (tau * g.m_phase_c)[None, :] * G.beams(G.m_L).T
    raise ValueError(kind)


# ============================================================================ covariance
def binned_spectra(Xr, g=None):
    """SZiFi-style binned periodogram: mean of Re(X_i^* X_j) over ALL modes (full plane) with l0 <= l < lf.
    Xr: (6, N, HALF) rfft2 of the l-filtered maps.  Returns (nb, 6, 6) in |rfft2|^2 units."""
    g = g or szgrid()
    b = g.binm.ravel()
    ok = b >= 0
    bb = b[ok]
    wv = g.W.ravel()[ok]
    cnt = np.bincount(bb, weights=wv, minlength=g.nb)
    Xs = [Xr[i].ravel()[ok] for i in range(Xr.shape[0])]
    C = np.zeros((g.nb, 6, 6))
    for i in range(6):
        for j in range(i, 6):
            v = np.bincount(bb, weights=wv * (Xs[i].real * Xs[j].real + Xs[i].imag * Xs[j].imag), minlength=g.nb) / cnt
            C[:, i, j] = v
            C[:, j, i] = v
    return C


def lsep_correct(C, fsky, g=None, lsep=LSEP):
    """spec.correct_for_inpainting: divide by the unmasked fraction only where the bin centre exceeds lsep."""
    g = g or szgrid()
    C = C.copy()
    C[g.lc > lsep] /= fsky
    return C


def inv_bins(C, bins_needed):
    Ci = np.zeros_like(C)
    Ci[bins_needed] = np.linalg.inv(C[bins_needed])
    return Ci


@nb.njit(cache=True)
def _apply_bin_inv(Ci, mbin, a, out):
    for m in range(a.shape[1]):
        b = mbin[m]
        for i in range(a.shape[0]):
            s = 0.0 + 0.0j
            for j in range(a.shape[0]):
                s += Ci[b, i, j] * a[j, m]
            out[i, m] = s


class SZFilter:
    """Per-mode MMF weights for one covariance: binned (Cbin + per-mode bin index) or analytic per mode."""

    def __init__(self, Cbin=None, mbin=None, analytic_mode=None, g=None):
        self.g = g or szgrid()
        self.Cbin, self.mbin, self.analytic_mode = Cbin, mbin, analytic_mode
        if Cbin is not None:
            self.Ci = inv_bins(Cbin, np.unique(mbin))
        else:
            G = F.grid()
            Cm = G.analytic_cov_bands(G.m_L, analytic_mode)          # (nm, 6, 6), |rfft2|^2 units
            self.Cim = np.linalg.inv(Cm)

    def weights(self, T):
        """F_i = sum_j C^-1_ij a_j T_j (6, nm) and D = sum_k w (aT)^dag C^-1 (aT) = sigma^-2."""
        aT = SED[:, None] * T
        if self.Cbin is not None:
            Fm = np.empty_like(aT)
            _apply_bin_inv(self.Ci, self.mbin, aT, Fm)
        else:
            Fm = np.einsum("mij,jm->im", self.Cim, aT)
        D = float(np.sum(self.g.m_w * np.sum((np.conj(aT) * Fm).real, axis=0)))
        return Fm, D


def sigma_from(T, filt):
    return 1.0 / np.sqrt(filt.weights(T)[1])


# ============================================================================ S/N maps
def q_circular(Fm, D, Xm, shift_half=False, g=None):
    """Circular (periodic) matched filter: Y(n) = sum_k sum_i F_i X_i e^{ik.n}; q = Y / sqrt(D).
    With the SZiFi-centred template (phase c) Y(n) matches a cluster at n - c."""
    g = g or szgrid()
    Z = np.sum(Fm * Xm, axis=0)
    if shift_half:
        Z = Z * g.m_phase_half
    return sfft.irfft2(from_modes(Z, g), s=(N, N)) * (N * N) / np.sqrt(D)


class LinearConv:
    """SZiFi's y map: sum_i fftconvolve(map_i, filter_i, mode='same') (linear, zero padded; N even -> offset)."""

    def __init__(self, maps):
        self.P = 2 * N                                              # >= 2N - 1: no wrap
        self.M = [sfft.rfft2(maps[i], s=(self.P, self.P)) for i in range(maps.shape[0])]
        self.s0 = (2 * N - 1 - N) // 2                              # scipy 'same' start index (N/2 - 1 for even N)

    def q(self, Fm, D, g=None):
        g = g or szgrid()
        acc = None
        for i in range(Fm.shape[0]):
            filt = sfft.irfft2(from_modes(Fm[i], g), s=(N, N)) * (N * N)   # filter_i(x) = sum_k F_i(k) e^{ikx}
            Fp = sfft.rfft2(filt, s=(self.P, self.P))
            acc = Fp * self.M[i] if acc is None else acc + Fp * self.M[i]
        full = sfft.irfft2(acc, s=(self.P, self.P))
        return full[self.s0:self.s0 + N, self.s0:self.s0 + N] / np.sqrt(D)


def lfiltered_maps(X, g=None):
    """Real-space maps of the l-filtered tile (SZiFi filter_tmap), from the build's rfft X (6, N, HALF)."""
    g = g or szgrid()
    return np.stack([sfft.irfft2(X[i] * g.band, s=(N, N)) for i in range(X.shape[0])])


# ============================================================================ detections (SZiFi rules, -inf edges)
@nb.njit(cache=True)
def szifi_peaks_scale(qm, q0, qp, j, nth, qth):
    n0, n1 = q0.shape
    oy = []
    ox = []
    for iy in range(n0):
        for ix in range(n1):
            v = q0[iy, ix]
            if not v > qth:
                continue
            if iy + 1 < n0 and not v > q0[iy + 1, ix]:
                continue
            if iy - 1 >= 0 and not v > q0[iy - 1, ix]:
                continue
            if ix + 1 < n1 and not v > q0[iy, ix + 1]:
                continue
            if ix - 1 >= 0 and not v > q0[iy, ix - 1]:
                continue
            if j == 0:
                ok = (v > 0.0) and (nth == 1 or v >= qp[iy, ix])
            elif j == nth - 1:
                ok = v > qm[iy, ix]
            else:
                ok = (v > qm[iy, ix]) and (v >= qp[iy, ix])
            if ok:
                oy.append(iy)
                ox.append(ix)
    return np.array(oy, dtype=np.int64), np.array(ox, dtype=np.int64)


def detect_cube(qmaps, thetas, qth=Q_TH):
    """qmaps: list/array of (N, N) S/N maps per scale.  Returns dict(j, n0, n1, q, theta) in SZiFi order."""
    nth = len(qmaps)
    J, Y, Xx, Qv = [], [], [], []
    for j in range(nth):
        qm = qmaps[j - 1] if j > 0 else qmaps[j]
        qp = qmaps[j + 1] if j + 1 < nth else qmaps[j]
        a, b = szifi_peaks_scale(qm, qmaps[j], qp, j, nth, qth)
        J.append(np.full(a.size, j, np.int64)); Y.append(a); Xx.append(b); Qv.append(qmaps[j][a, b])
    j = np.concatenate(J)
    return dict(j=j, n0=np.concatenate(Y), n1=np.concatenate(Xx), q=np.concatenate(Qv), theta=np.asarray(thetas)[j])


# ============================================================================ SZiFi-style iteration (inpainting)
def cluster_mask(det, n=N):
    """get_cluster_mask: 0 within distance <= 3 theta_est of each q > 4 detection pixel (not periodic)."""
    m = np.ones((n, n))
    ar = np.arange(n)
    for k in np.flatnonzero(det["q"] > Q_TH_NOISE):
        R = MASK_RADIUS * det["theta"][k] / PIXA          # pixels
        Ri = int(np.ceil(R)) + 1
        y0, y1 = max(det["n0"][k] - Ri, 0), min(det["n0"][k] + Ri + 1, n)
        x0, x1 = max(det["n1"][k] - Ri, 0), min(det["n1"][k] + Ri + 1, n)
        dy = (ar[y0:y1] - det["n0"][k])[:, None]
        dx = (ar[x0:x1] - det["n1"][k])[None, :]
        sub = m[y0:y1, x0:x1]
        sub[dy * dy + dx * dx <= R * R] = 0.0
    return m


@nb.njit(cache=True)
def diffusive_fill(img, hr, hc, nsweep):
    """In place: nsweep passes over the hole pixels (given order); each becomes the mean of its in-map 8 neighbours."""
    n0, n1 = img.shape
    for s in range(nsweep):
        for k in range(hr.size):
            x = hr[k]
            y = hc[k]
            v = 0.0
            c = 0
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    if ddx == 0 and ddy == 0:
                        continue
                    xx = x + ddx
                    yy = y + ddy
                    if 0 <= xx < n0 and 0 <= yy < n1:
                        v += img[xx, yy]
                        c += 1
            img[x, y] = v / c


def blend_weights(mask, pad=2):
    """w per pixel for the cosine blend (0 outside holes): w = 0.5 (1 + cos(pi d / R)) for hole pixels with d <= R."""
    hole = mask < 0.5
    lab, nreg = ndimage.label(hole, structure=np.ones((3, 3), bool))
    w = np.zeros(mask.shape)
    for r, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, mask.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, mask.shape[1])
        box = lab[y0:y1, x0:x1] == r
        area = box.sum()
        R = np.sqrt(area / np.pi)
        d = ndimage.distance_transform_edt(box)
        ann = box & (d > 0) & (d <= R)
        sub = w[y0:y1, x0:x1]
        sub[ann] = 0.5 * (1 + np.cos(np.pi * d[ann] / R))
    return w, nreg


def inpaint_blend(maps_lf, mask, nsweep=N_INPAINT, blend=True):
    """SZiFi's iteration-1 noise maps: diffusive inpainting of the holes, cosine blend (blend=False: diagnostic
    without the blend, i.e. holes purely inpainted), l-filter (real maps out)."""
    hr, hc = np.nonzero(mask == 0.0)                     # row-major order (np.where)
    hr = hr.astype(np.int64); hc = hc.astype(np.int64)
    w, nreg = blend_weights(mask)
    if not blend:
        w = np.zeros_like(w)
    hole = mask == 0.0
    out = np.empty_like(maps_lf)
    for i in range(maps_lf.shape[0]):
        img = maps_lf[i] * mask
        diffusive_fill(img, hr, hc, nsweep)
        bl = maps_lf[i].copy()
        bl[hole] = w[hole] * maps_lf[i][hole] + (1 - w[hole]) * img[hole]
        out[i] = bl
    return out, nreg, w


# ============================================================================ matching
def szifi_pos_build_units(det):
    """Matched cluster position of a SZiFi-convention detection in the build's pixel units (index - 1/2)."""
    return det["n0"] - 0.5, det["n1"] - 0.5


def interior(y_pix, x_pix, margin_arcmin):
    m = margin_arcmin / PIXA
    return (y_pix >= m) & (y_pix <= N - 1 - m) & (x_pix >= m) & (x_pix <= N - 1 - m)


def match(ya, xa, yb, xb, radius_pix):
    """For each a, the nearest b within radius (non-periodic distances); -1 if none."""
    out = np.full(ya.size, -1, np.int64)
    dist = np.full(ya.size, np.nan)
    if ya.size == 0 or yb.size == 0:
        return out, dist
    from scipy.spatial import cKDTree
    tr = cKDTree(np.c_[yb, xb])
    d, i = tr.query(np.c_[ya, xa], k=1, distance_upper_bound=radius_pix)
    ok = np.isfinite(d)
    out[ok] = i[ok]
    dist[ok] = d[ok]
    return out, dist


# ============================================================================ SZiFi-convention pipeline on one tile
_TC = {}


def szifi_template_dft(j, g=None):
    """Cached DFT (in-band modes) of SZiFi's sampled template at scale j (no beam, y0 = 1)."""
    g = g or szgrid()
    if j not in _TC:
        _TC[j] = sfft.rfft2(szifi_ymap(THETAS[j])).ravel()[g.m_idx]
    return _TC[j]


_BC = {}


def szifi_beams(g=None):
    g = g or szgrid()
    if "b" not in _BC:
        _BC["b"] = np.stack([szifi_beam(g.m_L, fw) for fw in FWHM])
    return _BC["b"]


def szifi_T(j, g=None):
    return szifi_template_dft(j, g)[None, :] * szifi_beams(g)


_SA = {}


def szifi_sigma_analytic(mode, g=None):
    """sigma(theta) of SZiFi's template with the analytic CMB + noise covariance per mode (y0 units)."""
    if mode not in _SA:
        filt = SZFilter(analytic_mode=mode, g=g)
        _SA[mode] = np.array([sigma_from(szifi_T(j, g), filt) for j in range(len(THETAS))])
        del filt
    return _SA[mode]


def szifi_filter_pass(maps_lf, Cbin, g=None, keep_q=False):
    """One SZiFi-convention filtering pass: per-mode nearest-bin covariance, SZiFi templates, linear convolution,
    SZiFi peak rules.  Returns (det, sigma[15], qmaps or None)."""
    g = g or szgrid()
    filt = SZFilter(Cbin=Cbin, mbin=g.m_near, g=g)
    lin = LinearConv(maps_lf)
    qs, sig = [], np.zeros(len(THETAS))
    for j in range(len(THETAS)):
        Fm, D = filt.weights(szifi_T(j, g))
        qs.append(lin.q(Fm, D, g))
        sig[j] = 1.0 / np.sqrt(D)
    det = detect_cube(qs, THETAS)
    del lin
    return det, sig, (qs if keep_q else None)


def szifi_run(X, inpaint=True, g=None, keep_q=False):
    """SZiFi-convention cluster finder on one tile (rfft X in the build's units): iteration 0 with the tile's own
    isotropic covariance; iteration 1 after masking q > 4 detections WITH SZiFi's inpainting + blending + lsep
    correction.  (Without inpainting SZiFi's iteration 1 equals iteration 0: the mask is not used.)"""
    g = g or szgrid()
    maps = lfiltered_maps(X, g)
    Xr = np.stack([X[i] * g.band for i in range(X.shape[0])])
    C0 = binned_spectra(Xr, g)
    del Xr
    det0, sig0, q0 = szifi_filter_pass(maps, C0, g, keep_q)
    out = dict(C0=C0, det0=det0, sig0=sig0, q0=q0, n_it=1)
    if not inpaint or not np.any(det0["q"] > Q_TH_NOISE):
        return out
    mask = cluster_mask(det0)
    mib, nreg, w = inpaint_blend(maps, mask)
    Xr1 = np.stack([sfft.rfft2(mib[i]) * g.band for i in range(mib.shape[0])])
    del mib
    fsky = float(mask.mean())
    C1raw = binned_spectra(Xr1, g)
    del Xr1
    C1 = lsep_correct(C1raw, fsky, g)
    det1, sig1, q1 = szifi_filter_pass(maps, C1, g, keep_q)
    out.update(C1=C1, C1raw=C1raw, det1=det1, sig1=sig1, q1=q1, fsky=fsky, nreg=nreg,
               wmean_hole=float(w[mask == 0].mean()), w2mean_hole=float((w[mask == 0] ** 2).mean()), n_it=2)
    return out


def det_to_arrays(det, prefix):
    return {f"{prefix}_{k}": np.asarray(v) for k, v in det.items()}


# ============================================================================ periodic / linear switch for the pipeline
def szifi_filter_pass2(maps_lf, Xm, Cbin, conv, g=None):
    """As szifi_filter_pass, conv = 'lin' (SZiFi's zero-padded 'same' convolution on the tile as given) or 'per'
    (periodic convolution, rolled so that pixel n again refers to position n - 1/2: SZiFi run on an unbounded,
    periodically continued sky, i.e. without the artificial tile edges)."""
    g = g or szgrid()
    filt = SZFilter(Cbin=Cbin, mbin=g.m_near, g=g)
    lin = LinearConv(maps_lf) if conv == "lin" else None
    qs, sig = [], np.zeros(len(THETAS))
    for j in range(len(THETAS)):
        Fm, D = filt.weights(szifi_T(j, g))
        if conv == "lin":
            qs.append(lin.q(Fm, D, g))
        else:
            qs.append(np.roll(q_circular(Fm, D, Xm, g=g), (-(N // 2 - 1), -(N // 2 - 1)), axis=(0, 1)))
        sig[j] = 1.0 / np.sqrt(D)
        del Fm
    det = detect_cube(qs, THETAS)
    del lin, qs
    return det, sig


def szifi_run2(X, conv="per", inpaint=True, g=None, blend=True, lsep=LSEP):
    """SZiFi-convention finder (iteration 0, and iteration 1 with SZiFi's inpainting, blend and lsep correction)."""
    g = g or szgrid()
    maps = lfiltered_maps(X, g)
    Xm = to_modes(X, g)
    Xr = np.stack([X[i] * g.band for i in range(X.shape[0])])
    C0 = binned_spectra(Xr, g)
    del Xr
    det0, sig0 = szifi_filter_pass2(maps, Xm, C0, conv, g)
    out = dict(C0=C0, det0=det0, sig0=sig0, n_it=1)
    if not inpaint or not np.any(det0["q"] > Q_TH_NOISE):
        return out
    mask = cluster_mask(det0)
    mib, nreg, w = inpaint_blend(maps, mask, blend=blend)
    Xr1 = np.stack([sfft.rfft2(mib[i]) * g.band for i in range(mib.shape[0])])
    del mib
    fsky = float(mask.mean())
    C1raw = binned_spectra(Xr1, g)
    del Xr1
    C1 = lsep_correct(C1raw, fsky, g, lsep=lsep)
    det1, sig1 = szifi_filter_pass2(maps, Xm, C1, conv, g)
    hole = mask == 0
    out.update(C1=C1, C1raw=C1raw, det1=det1, sig1=sig1, fsky=fsky, nreg=nreg, wmean_hole=float(w[hole].mean()),
               w2mean_hole=float((w[hole] ** 2).mean()), n_it=2)
    return out
