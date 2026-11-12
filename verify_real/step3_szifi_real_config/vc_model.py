"""Own numerics of the config verifier (numpy / scipy only; imported by scripts in either venv).  Written from scratch:
nothing is imported from the build (analysis/realistic/step3_szifi_real), from szsim or from SZiFi.

Contents
  * the SO LAT instrument as typed from SO19 (arXiv:1808.07445) Tables 1-2 and Sec. 2.2, and the non-relativistic tSZ
    SED at the band centres (own formula);
  * the Arnaud+10 GNFW pressure shape truncated at 5 R500 (3D), normalised to the SPHERICAL Y500 (step 1's
    convention: y(theta) = Y500 tau(theta), int tau d^2theta = Y(5R500)/Y500):
      - central value f(s -> 0) of the projected shape (y0 = Y500 f / theta500^2),
      - its 2D Fourier transform g(u) by the projection-slice theorem: g(u) = 4 pi int_0^5 r^2 P(r) sinc(u r) dr / Ysph(1);
  * painting of isolated clusters in the tiles' own conventions (Fourier painting on the rfft grid of the 1200^2
    tile, sample (iy, ix) at (0.5 iy, 0.5 ix) arcmin, beams exp(-l^2 sigma^2 / 2), tSZ band-limited to 100 <= l <= 20000,
    rfft units X(k) = sum_p x(p) exp(-i k.p));
  * SZiFi's filter normalisation re-derived from its code (mmf.get_mmf_q_map, mmf.get_tem_conv_fft,
    model.gnfw.get_t_map_convolved, maps.get_bl, spec.interp_cov 'nearest'):
      norm(theta) = (dx^2 / N^2) sum_k T(k)^dagger C^-1(k) T(k),  T_i(k) = a_i B_i(l) Yhat(k) (in band),
      sigma = norm^-1/2, with Yhat the FFT of the template SAMPLED at the pixel centres (centre at index N/2 - 1/2),
      normalised to y(r = 1e-8 R500), B_i(l) = exp(-(fwhm_i)^2 l (l + 1) / (16 ln 2)), C^-1(k) the inverse of the
      binned spectrum of the bin whose centre is nearest to l(k) (scipy interp1d 'nearest', extrapolated);
    and the response kappa = (SZiFi's estimate of y0 for a painted unit-y0 cluster at a half-integer pixel position) /
    1, i.e. kappa = cross / norm with cross = (dx^2 / N^2) sum_k Ghat(k) Yhat(k) a^T B_paint C^-1 B_szifi a.
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

N = 1200
PIXA = 0.5                                   # arcmin
ARC = np.pi / 180. / 60.                     # rad per arcmin
DX = PIXA * ARC                              # rad (SZiFi: dx_arcmin / 60 / 180 * pi)
OMEGA_PIX_ARCMIN2 = PIXA * PIXA
_LN_LO, _LN_HI = np.log(0.5), np.log(15.0)
THETAS = np.exp(np.linspace(_LN_LO, _LN_HI, 15))     # 15 log-spaced scales 0.5'..15'
DK = 2. * np.pi / (N * DX)                   # 36.0: fundamental of the 600' tile

# ----------------------------------------------------------------------------- SO LAT (SO19 Tables 1-2, Sec. 2.2)
# SO19 Table 1 per band: (frequency GHz, FWHM arcmin, white noise baseline / goal uK-arcmin)
_SO19_T1 = ((27., 7.4, 71., 52.), (39., 5.1, 36., 27.), (93., 2.2, 8., 5.8), (145., 1.4, 10., 6.3),
            (225., 1.0, 22., 15.), (280., 0.9, 54., 37.))
FREQS_GHZ = np.array([r[0] for r in _SO19_T1])
FWHM = np.array([r[1] for r in _SO19_T1])
WHITE = {"baseline": np.array([r[2] for r in _SO19_T1]), "goal": np.array([r[3] for r in _SO19_T1])}
N_RED = np.array([100., 39., 230., 1500., 17000., 31000.])      # uK^2 s
ELL_KNEE, ALPHA_KNEE, ATM_CORR = 1000., -3.5, 0.9
TUBES = ((0, 1), (2, 3), (4, 5))
FSKY, T_EFF = 0.4, 5. * 365.25 * 86400. * 0.2 * 0.85
T_CMB = 2.7255
H_PL, K_B = 6.62607015e-34, 1.380649e-23


def tsz_sed_uK(nu_ghz):
    """Delta T [uK_CMB] per unit Compton y, non-relativistic, monochromatic."""
    x = H_PL * np.asarray(nu_ghz, float) * 1e9 / (K_B * T_CMB)
    return T_CMB * 1e6 * (x / np.tanh(x / 2.) - 4.)


# ----------------------------------------------------------------------------- Arnaud+10 GNFW
GN = dict(c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
XT = 5.0


def P3(x):
    cx = GN["c500"] * np.asarray(x, float)
    return cx ** (-GN["gamma"]) * (1. + cx ** GN["alpha"]) ** (-(GN["beta"] - GN["gamma"]) / GN["alpha"])


def ysph(X):
    """4 pi int_0^X x^2 P(x) dx."""
    return integrate.quad(lambda x: 4 * np.pi * x * x * P3(x), 0., X, limit=500, epsabs=0., epsrel=1e-13)[0]


YSPH1 = ysph(1.0)
RATIO5 = ysph(XT) / YSPH1


def proj(s):
    """Projected shape f(s) = 2 int_0^sqrt(25 - s^2) P(sqrt(s^2 + z^2)) dz / Ysph(1) (substitution z = Z t^3)."""
    s = np.atleast_1d(np.asarray(s, float))
    out = np.zeros(s.size)
    for i, si in enumerate(s):
        if si >= XT:
            continue
        Z = np.sqrt(XT * XT - si * si)
        fz = lambda t: 3. * Z * t * t * P3(np.sqrt(si * si + (Z * t ** 3) ** 2))   # noqa: E731
        out[i] = 2. * integrate.quad(fz, 0., 1., limit=500, epsabs=0., epsrel=1e-13)[0]
    return out / YSPH1


F_CENTRE = float(proj(0.0)[0])               # y0 = Y500 * F_CENTRE / theta500^2   (Y500 arcmin^2, theta500 arcmin)
F_NORM = float(proj(1e-8)[0])                # SZiFi normalises its template to y(r = 1e-8 R500)

# Gauss-Legendre nodes on r in [0, 5] (r = 5 t^3; 256 panels x 16 nodes) for g(u)
_xg, _wg = np.polynomial.legendre.leggauss(16)
_ed = np.linspace(0., 1., 257)
_t = (0.5 * (_ed[1:] - _ed[:-1])[:, None] * _xg[None, :] + 0.5 * (_ed[1:] + _ed[:-1])[:, None]).ravel()
_wt = (0.5 * (_ed[1:] - _ed[:-1])[:, None] * _wg[None, :]).ravel()
_R = XT * _t ** 3
_WR = _wt * 15. * _t ** 2 * 4. * np.pi * _R * _R * P3(_R) / YSPH1


def ghat(u):
    """2D Fourier transform of the unit-Y500 template, u = l theta500 [rad]; g(0) = Y(5R500)/Y500."""
    u = np.atleast_1d(np.asarray(u, float))
    out = np.empty(u.size)
    for a in range(0, u.size, 2000):
        ur = np.outer(u[a:a + 2000], _R)
        out[a:a + 2000] = (np.sinc(ur / np.pi) * _WR[None, :]).sum(1)
    return out


# ----------------------------------------------------------------------------- tile grids
def rfft_grid():
    """Integer wavenumbers and |k| on the rfft half plane of the 1200^2 tile (axis 0 full, axis 1 half)."""
    m0 = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
    m1 = np.arange(N // 2 + 1, dtype=np.int64)
    n2 = m0[:, None] ** 2 + m1[None, :] ** 2
    return m0, m1, n2


def paint_tsz_rfft(py, px, Y500, th500, mode_fwhm=FWHM, sed=None):
    """tSZ channel maps of isolated clusters as rfft coefficients (6, N, N/2+1) in uK, in the tiles' conventions:
    cluster h at index position (py[h], px[h]) (axis 0, axis 1; continuous), Y500[h] arcmin^2, th500[h] arcmin;
    X(k) = Y500 / Omega_pix * g(|k| th500) * exp(-i 2 pi (m0 py + m1 px) / N), times B_i(|k|) = exp(-l^2 sigma_i^2 / 2)
    and the SED, kept only for 100 <= |k| <= 20000 (n^2 in [8, 308641], as the tiles' tSZ)."""
    sed = tsz_sed_uK(FREQS_GHZ) if sed is None else np.asarray(sed, float)
    m0, m1, n2 = rfft_grid()
    band = (n2 >= 8) & (n2 <= 308641)
    L = DK * np.sqrt(n2.astype(float))
    un, inv = np.unique(n2[band], return_inverse=True)
    Lu = DK * np.sqrt(un.astype(float))
    Yk = np.zeros((N, N // 2 + 1), complex)
    for th in np.unique(th500):
        sel = np.flatnonzero(th500 == th)
        G = np.zeros((N, N // 2 + 1))
        G[band] = ghat(Lu * th * ARC)[inv]
        S = np.zeros((N, N // 2 + 1), complex)
        for h in sel:
            e0 = np.exp(-2j * np.pi * m0 * py[h] / N)
            e1 = np.exp(-2j * np.pi * m1 * px[h] / N)
            S += (Y500[h] / OMEGA_PIX_ARCMIN2) * np.outer(e0, e1)
        Yk += G * S
    sig = np.asarray(mode_fwhm, float) * ARC / np.sqrt(8. * np.log(2.))
    B = np.exp(-0.5 * (L[None] * sig[:, None, None]) ** 2)
    return B * sed[:, None, None] * Yk[None]


# ----------------------------------------------------------------------------- SZiFi-convention filter quantities
def szifi_ell(n=N, dx=DX):
    """l of every FFT mode as SZiFi computes it (maps.rmap.get_ell)."""
    f = np.fft.fftfreq(n, dx) * 2. * np.pi
    a = f[:, None] * f[:, None] + 0. * f[None, :]       # same IEEE operations as SZiFi's l map: sqrt(lx^2 + ly^2)
    b = 0. * f[:, None] + f[None, :] * f[None, :]
    return np.sqrt(a + b)


_PROF = {}


def szifi_profile_table():
    """Projected shape on SZiFi's 1000 radial nodes linspace(0, 5, 1000) (units of theta500), /f(1e-8)."""
    if "t" not in _PROF:
        s = np.linspace(0., XT, 1000)
        _PROF["t"] = (s, proj(s) / F_NORM)
    return _PROF["t"]


def szifi_template(theta_arcmin):
    """SZiFi's real-space template (no beam): profile interpolated at the pixel centres, centre at index N/2 - 1/2,
    zero beyond 5 theta500, normalised to y(1e-8 R500)."""
    s, f = szifi_profile_table()
    c = (np.arange(N) - N * 0.5 + 0.5) * DX
    r = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    return np.interp(r, s * theta_arcmin * ARC, f, right=0.)


def beam_szifi(ell, fwhm_arcmin):
    """SZiFi's Gaussian beam convention: exp(-sigma^2 l (l + 1) / 2), sigma = FWHM / sqrt(8 ln 2)."""
    s2 = (fwhm_arcmin * ARC) ** 2 / (8. * np.log(2.))
    return np.exp(-0.5 * s2 * ell * (ell + 1.))


def beam_paint(ell, fwhm_arcmin):
    return np.exp(-0.5 * (ell * fwhm_arcmin * ARC / np.sqrt(8. * np.log(2.))) ** 2)


def nearest_bin_index(ell_c, ell_map):
    """Index of the bin whose centre SZiFi's interp1d('nearest', extrapolate) assigns to each l."""
    f = interp1d(ell_c, np.arange(ell_c.size, dtype=float), kind="nearest", bounds_error=False,
                 fill_value="extrapolate")
    return np.rint(f(ell_map)).astype(np.int64)


def square_cut_mask(n_new, n=N):
    """Modes kept by SZiFi's maps.reshape_ell_matrix when the inverse covariance lives on an n_new^2 grid."""
    keep = np.zeros(n, bool)
    keep[:n_new // 2] = True
    keep[n - (n_new - n_new // 2):] = True
    return keep[:, None] & keep[None, :]


def filter_quantities(ell_c, spec, sed, fwhm, thetas=THETAS, lrange=(100., 20000.), n_cov=None, paint_fwhm=FWHM,
                      paint_band=(100., 20000.), want_kappa=True):
    """sigma(theta) [y0 units] of SZiFi's MMF for binned spectra spec (nb, nf, nf) [uK^2 sr] at bin centres ell_c,
    and kappa(theta) = SZiFi's y0 estimate for a painted unit-y0 cluster (beams of the tiles, tSZ band-limited to
    paint_band) at a half-integer position.  n_cov: side of SZiFi's degraded covariance grid (powspec_lmax1d), or None.
    Uses the full 1200^2 FFT plane (SZiFi's layout)."""
    sed = np.asarray(sed, float)
    fwhm = np.asarray(fwhm, float)
    nf = sed.size
    ell = szifi_ell()
    inband = (ell >= lrange[0]) & (ell <= lrange[1])
    if n_cov is None:
        b = nearest_bin_index(ell_c, ell)
        cut = np.ones_like(inband)
    else:
        dxd = DX * (N / n_cov)
        elld = szifi_ell(n_cov, dxd)
        bd = nearest_bin_index(ell_c, elld)
        cut = square_cut_mask(n_cov)
        b = np.zeros(ell.shape, np.int64)
        idx = np.flatnonzero(np.r_[np.ones(n_cov // 2, bool), np.zeros(N - n_cov, bool), np.ones(n_cov - n_cov // 2, bool)])
        b[np.ix_(idx, idx)] = bd
    use = inband & cut
    Ci = np.linalg.inv(spec)                                       # (nb, nf, nf)
    Bs = np.stack([beam_szifi(ell, fw) for fw in fwhm])            # (nf, N, N)
    Wss = np.zeros(ell.shape)
    for i in range(nf):
        for j in range(nf):
            Wss += sed[i] * sed[j] * Bs[i] * Bs[j] * Ci[:, i, j][b]
    Wss[~use] = 0.
    if want_kappa:
        Bp = np.stack([beam_paint(ell, fw) for fw in paint_fwhm])
        Wps = np.zeros(ell.shape)
        for i in range(nf):
            for j in range(nf):
                Wps += sed[i] * sed[j] * Bp[i] * Bs[j] * Ci[:, i, j][b]
        pb = (ell >= paint_band[0]) & (ell <= paint_band[1])
        Wps[~(use & pb)] = 0.
        del Bp
    del Bs
    # template centred at N/2 - 1/2 on both axes: Re(FFT x exp(+i k.c)) with signed wavenumbers
    m = np.rint(np.fft.fftfreq(N) * N)
    ph = np.exp(2j * np.pi * (N / 2 - 0.5) * m / N)
    if want_kappa:
        n2 = (m[:, None] ** 2 + m[None, :] ** 2).astype(np.int64)
        need = Wps != 0.
        n2need = n2[need]
    sig = np.zeros(len(thetas))
    kap = np.full(len(thetas), np.nan)
    for t, th in enumerate(thetas):
        F = np.fft.fft2(szifi_template(th))
        norm = (DX * DX / N ** 2) * np.sum((F.real ** 2 + F.imag ** 2) * Wss)
        sig[t] = 1. / np.sqrt(norm)
        if want_kappa:
            Yc = (F * ph[:, None] * ph[None, :]).real
            G = np.zeros(ell.shape)
            un, gu = ghat_shells(th)
            G[need] = (th * th / F_NORM) / OMEGA_PIX_ARCMIN2 * gu[np.searchsorted(un, n2need)]
            cross = (DX * DX / N ** 2) * np.sum(G * Yc * Wps)
            kap[t] = cross / norm
    return sig, kap


_GH = {}


def ghat_shells(th):
    """g(|k| theta500) on every distinct |k|^2 = n^2 of the 1200^2 grid with |k| <= 20000 (cached per theta)."""
    if th not in _GH:
        if "un" not in _GH:
            mm = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
            n2 = mm[:, None] ** 2 + mm[None, :] ** 2
            _GH["un"] = np.unique(n2[n2 <= 308641])
        un = _GH["un"]
        _GH[th] = (un, ghat(DK * np.sqrt(un.astype(float)) * th * ARC))
    return _GH[th]


# ----------------------------------------------------------------------------- analytic covariance of the tiles
def so_noise(ell, mode):
    """N_ij(l) [uK^2 sr], white + atmosphere (0.9 within tubes), not beam-deconvolved."""
    ell = np.asarray(ell, float)
    w = WHITE[mode] * ARC
    Nn = np.zeros(ell.shape + (6, 6))
    with np.errstate(divide="ignore"):
        shp = np.where(ell > 0, (ell / ELL_KNEE) ** ALPHA_KNEE, 0.)
    red = N_RED * 4. * np.pi * FSKY / T_EFF
    for i in range(6):
        Nn[..., i, i] = w[i] ** 2 + red[i] * shp
    for (i, j) in TUBES:
        Nn[..., i, j] = Nn[..., j, i] = ATM_CORR * np.sqrt(red[i] * red[j]) * shp
    return Nn


def cmb_cl(ell, path="/Users/boris/GitHub/sz-poisson-blending/data_real/cmb_lensed_TT_camb.npz"):
    d = np.load(path)
    out = np.interp(np.asarray(ell, float), d["ell"].astype(float), d["cl_uK2"])
    out[np.asarray(ell) < 2] = 0.
    return out


def analytic_cov(ell, mode):
    """C_ij(l) = B_i B_j C_l^CMB + N_ij(l) [uK^2 sr] with the tiles' beams (l^2 convention)."""
    ell = np.asarray(ell, float)
    B = np.stack([beam_paint(ell, fw) for fw in FWHM], -1)
    return B[..., :, None] * B[..., None, :] * cmb_cl(ell)[..., None, None] + so_noise(ell, mode)
