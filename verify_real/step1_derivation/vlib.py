"""Independent re-derivation library for the step-1 verification (lens: derivation).

Written from scratch WITHOUT importing szsim/instrument.py or any analysis/realistic code.
Allowed project pieces: szsim.sampler (halo catalogues) only; colossus for the cosmology/HMF model definition.

Sources used here (fetched by the verifier itself, 2026-11-11):
  [SO19]  arXiv:1808.07445v2 (PDF text): Table 1 (LAT f_sky=0.4: FWHM, white noise baseline/goal),
          Table 2 (LAT temperature N_red), Sec. 2.2 (l_knee = 1000, alpha = -3.5, 5 yr, 20% time, 15% edge
          discard, correlation 0.9 between the two bands of one optics tube).
  [SOcode] simonsobs/so_noise_models so_models_v3/SO_Noise_Calculator_Public_v3_0_0.py (read, not executed):
          AN_T = C (l/1000)^-3.5 A_SR / t / (2 N_tubes),  C = 200, 77, 1800, 12000, 68000, 124000;
          t = 5 * 365.25 * 24 * 3600 * 0.2 * 0.85 s; white = S_LA / sqrt(t) * sqrt(A_arcmin);
          S_LA baseline 48, 24, 5.4, 6.7, 15, 36; goal 35, 18, 3.9, 4.2, 10, 25 uK sqrt(s); r_atm = 0.9.
Units: angles in radians internally unless suffixed _am (arcmin); spectra in uK^2 sr or y^2 sr.
"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AM = np.pi / 180.0 / 60.0                  # 1 arcmin in rad
TCMB = 2.7255
HPL, KB = 6.62607015e-34, 1.380649e-23

# ----------------------------------------------------------------------------- [SO19] numbers (my transcription)
FREQ = np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0])
FWHM_AM = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
WHITE = {"baseline": np.array([71.0, 36.0, 8.0, 10.0, 22.0, 54.0]),
         "goal": np.array([52.0, 27.0, 5.8, 6.3, 15.0, 37.0])}
NRED_PAPER = np.array([100.0, 39.0, 230.0, 1500.0, 17000.0, 31000.0])   # uK^2 s, Table 2 LAT T
L_KNEE, ALPHA_KNEE, R_ATM = 1000.0, -3.5, 0.9
TUBES = [(0, 1), (2, 3), (4, 5)]
FSKY = 0.4
T_EFF = 5.0 * 365.25 * 24 * 3600 * 0.2 * 0.85          # [SOcode]
A_SR = 4 * np.pi * FSKY
# [SOcode] values used for the code-algorithm route
SO_C = np.array([200.0, 77.0, 1800.0, 12000.0, 68000.0, 124000.0])
SO_NTUBE = np.array([1, 1, 4, 4, 2, 2], float)
SO_S = {"baseline": np.array([48.0, 24.0, 5.4, 6.7, 15.0, 36.0]),
        "goal": np.array([35.0, 18.0, 3.9, 4.2, 10.0, 25.0])}


def white_from_code(mode):
    A_arcmin = A_SR * (180 / np.pi) ** 2 * 3600.0
    return SO_S[mode] / np.sqrt(T_EFF) * np.sqrt(A_arcmin)


def nred_from_code():
    return SO_C / (2.0 * SO_NTUBE)


# ----------------------------------------------------------------------------- tSZ SED
def gx(nu_GHz):
    x = HPL * np.asarray(nu_GHz, float) * 1e9 / (KB * TCMB)
    return x / np.tanh(x / 2) - 4.0


def f_uK(nu_GHz):
    return TCMB * 1e6 * gx(nu_GHz)


# ----------------------------------------------------------------------------- CMB (CLASS, built by v02)
CMB_FILE = os.path.join(HERE, "cmb_lensed_TT_class.npz")
_CMB = {}


def cmb_cl(ell, source="class"):
    """Lensed TT C_l [uK^2] at real-valued l (linear interpolation between integer l); 0 for l < 2."""
    if source not in _CMB:
        if source == "class":
            d = np.load(CMB_FILE)
            _CMB[source] = (d["ell"].astype(float), d["cl_uK2"])
        elif source == "camb_builder":            # builder's stored file, read as data only (comparison runs)
            d = np.load(os.path.join(ROOT, "data_real", "cmb_lensed_TT_camb.npz"))
            _CMB[source] = (d["ell"].astype(float), d["cl_uK2"])
        elif source == "none":
            _CMB[source] = (np.array([0.0, 1e9]), np.array([0.0, 0.0]))
    le, cl = _CMB[source]
    ell = np.asarray(ell, float)
    out = np.interp(ell, le, cl, right=np.nan)
    if np.any(np.isnan(out)):
        raise ValueError("l beyond stored CMB range")
    return np.where(ell < 2, 0.0, out)


# ----------------------------------------------------------------------------- effective y noise (Woodbury)
def inv_Ny(ell, mode="baseline", cmb="class", atm=True, r_atm=R_ATM, nred=None, white=None, channels=None):
    """1/N_y(l) = f^T C'^-1 f [1/(y^2 sr)], C' = beam-deconvolved noise + C_CMB 1 1^T.
    Noise: block-diagonal (2x2 per optics tube) -> analytic block inverse; CMB rank-1 -> Woodbury.
    channels: subset of channel indices (the others get infinite noise)."""
    ell = np.asarray(ell, float)
    nred = NRED_PAPER if nred is None else nred
    w2 = ((WHITE[mode] if white is None else white) * AM) ** 2                 # uK^2 sr
    sig = FWHM_AM * AM / np.sqrt(8 * np.log(2))
    f = f_uK(FREQ)
    with np.errstate(divide="ignore", over="ignore"):
        shape = np.where(ell > 0, (ell / L_KNEE) ** ALPHA_KNEE, np.inf)
    Ninv_f = np.zeros((6,) + ell.shape)      # N'^-1 f
    Ninv_1 = np.zeros((6,) + ell.shape)      # N'^-1 1
    use = np.ones(6, bool) if channels is None else np.isin(np.arange(6), channels)
    for (i, j) in TUBES:
        Bi = np.exp(-0.5 * ell ** 2 * sig[i] ** 2); Bj = np.exp(-0.5 * ell ** 2 * sig[j] ** 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Ri = nred[i] * A_SR / T_EFF * shape if atm else 0.0 * shape
            Rj = nred[j] * A_SR / T_EFF * shape if atm else 0.0 * shape
            a = (w2[i] + Ri) / Bi ** 2                         # N'_ii
            b = (w2[j] + Rj) / Bj ** 2                         # N'_jj
            c = r_atm * np.sqrt(Ri * Rj) / (Bi * Bj)           # N'_ij
            if not use[i]:
                a = np.full_like(ell, np.inf); c = np.zeros_like(ell)
            if not use[j]:
                b = np.full_like(ell, np.inf); c = np.zeros_like(ell)
            # inverse of [[a, c], [c, b]] written to stay finite when a or b -> inf
            # inv = [[1/a / (1 - c^2/(ab)), -c/(ab - c^2)], [.., 1/b / (1 - c^2/(ab))]]
            rho2 = np.where(np.isfinite(a) & np.isfinite(b), c * c / (a * b), 0.0)
            den = 1.0 - rho2
            ia = np.where(np.isfinite(a), 1.0 / a, 0.0) / den
            ib = np.where(np.isfinite(b), 1.0 / b, 0.0) / den
            iab = np.where(np.isfinite(a) & np.isfinite(b), -c / (a * b), 0.0) / den
        ia = np.nan_to_num(ia, nan=0.0, posinf=0.0); ib = np.nan_to_num(ib, nan=0.0, posinf=0.0)
        iab = np.nan_to_num(iab, nan=0.0, posinf=0.0, neginf=0.0)
        Ninv_f[i] = ia * f[i] + iab * f[j]; Ninv_f[j] = iab * f[i] + ib * f[j]
        Ninv_1[i] = ia + iab; Ninv_1[j] = iab + ib
    F = (f[:, None] * Ninv_f.reshape(6, -1)).sum(0).reshape(ell.shape)
    G = Ninv_f.sum(0)
    H = Ninv_1.sum(0)
    Ccmb = cmb_cl(ell, cmb) if cmb != "none" else np.zeros_like(ell)
    out = F - Ccmb * G * G / (1.0 + Ccmb * H)
    return np.where(ell > 0, out, 0.0)


def inv_Ny_solve(ell, mode="baseline", cmb="class"):
    """Brute-force cross-check: build the full 6x6 (not beam-deconvolved) C and solve a^T C^-1 a."""
    ell = np.atleast_1d(np.asarray(ell, float))
    w2 = (WHITE[mode] * AM) ** 2
    sig = FWHM_AM * AM / np.sqrt(8 * np.log(2))
    f = f_uK(FREQ)
    out = np.empty(len(ell))
    for k, l in enumerate(ell):
        B = np.exp(-0.5 * l * l * sig ** 2)
        R = NRED_PAPER * A_SR / T_EFF * (l / L_KNEE) ** ALPHA_KNEE
        N = np.diag(w2 + R)
        for (i, j) in TUBES:
            N[i, j] = N[j, i] = R_ATM * np.sqrt(R[i] * R[j])
        C = N + np.outer(B, B) * cmb_cl(np.array([l]), cmb)[0]
        a = B * f
        out[k] = a @ np.linalg.solve(C, a)
    return out


def inv_Ny_white(ell, noise_y_am=4e-6, fwhm_am=1.4):
    sb = fwhm_am * AM / np.sqrt(8 * np.log(2))
    return np.exp(-(np.asarray(ell, float) * sb) ** 2) / (noise_y_am * AM) ** 2


# ----------------------------------------------------------------------------- Arnaud GNFW template (own projection)
C500, GAM, ALP, BET = 1.177, 0.3081, 1.0510, 5.4905
XT = 5.0


def P3(x):
    cx = C500 * np.asarray(x, float)
    return cx ** (-GAM) * (1 + cx ** ALP) ** (-(BET - GAM) / ALP)


def Ysph(X):
    # substitution x = X v^(1/(3-gam)) style split to handle the cusp: integrate 4 pi x^2 P in pieces
    pts = [0.0, 1e-4 * X, 1e-2 * X, 0.1 * X, X]
    return sum(quad(lambda x: 4 * np.pi * x * x * P3(x), a, b, epsabs=0, epsrel=1e-12, limit=500)[0]
               for a, b in zip(pts[:-1], pts[1:]))


RATIO = Ysph(XT) / Ysph(1.0)


def proj_shape(s):
    """f(s) = 2 int_0^sqrt(X^2-s^2) P(sqrt(s^2+l^2)) dl / Ysph(1)   (unit spherical Y500).
    Computed with the substitution l = sqrt(X^2-s^2) * sin(phi) to remove the sqrt edge."""
    s = np.atleast_1d(np.asarray(s, float))
    out = np.zeros_like(s)
    norm = Ysph(1.0)
    for i, si in enumerate(s):
        if si >= XT:
            continue
        Lm = np.sqrt(XT * XT - si * si)
        if si > 0:
            # integrand varies on the scale si near l = 0: geometric break points si * 10^k
            edges = [0.0] + [si * 10.0 ** k for k in range(0, 9) if si * 10.0 ** k < Lm] + [Lm]
        else:   # cusp l^-gam at l = 0 (integrable): geometric break points down to 1e-14
            edges = [0.0] + [10.0 ** k for k in range(-14, 1)] + [Lm]
        g = lambda l: P3(np.sqrt(si * si + l * l))
        v = sum(quad(g, a, b, epsabs=0, epsrel=1e-12, limit=500)[0] for a, b in zip(edges[:-1], edges[1:]))
        out[i] = 2 * v / norm
    return out


_PROF = None


def prof_table():
    """Tabulated projected shape on s in [0, 5] (cached in this directory)."""
    global _PROF
    if _PROF is None:
        fn = os.path.join(HERE, "proj_shape_table.npz")
        if os.path.exists(fn):
            d = np.load(fn); _PROF = (d["s"], d["f"])
        else:
            s = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, 1500), np.linspace(1.0, XT, 2001)[1:]])
            s = np.unique(np.concatenate([s, XT - np.geomspace(1e-9, 0.5, 400)]))
            f = proj_shape(s)
            np.savez(fn, s=s, f=f); _PROF = (s, f)
    return _PROF


def tau_shape(srel):
    """projected unit-Y500 shape at s = theta/theta500 (linear interpolation of the dense table)."""
    s, f = prof_table()
    return np.interp(srel, s, f, right=0.0)


# ----------------------------------------------------------------------------- template FT g(u) from 2D FFTs
GU_FILE = os.path.join(HERE, "g_of_u_2dfft.npz")
_GU = None


def g_u(u):
    """g(u) = 2D Fourier transform of the unit-Y500 template at u = l theta500 (from v03's 2D FFT tables)."""
    global _GU
    if _GU is None:
        d = np.load(GU_FILE)
        from scipy.interpolate import CubicSpline
        _GU = (CubicSpline(d["u"], d["g"]), d["u"][-1])
    sp, umax = _GU
    u = np.asarray(u, float)
    return np.where(u <= umax, sp(np.minimum(u, umax)), 0.0)


# ----------------------------------------------------------------------------- 1D filter integrals (own code)
def ell_grid(lmin=100.0, lmax=20000.0, dl=1.0):
    return np.arange(lmin, lmax + dl / 2, dl)


def trap_w(x):
    w = np.empty_like(x)
    d = np.diff(x)
    w[1:-1] = 0.5 * (d[:-1] + d[1:]); w[0] = 0.5 * d[0]; w[-1] = 0.5 * d[-1]
    return w


def sigmaY_1d(theta_am, ell, invN):
    """sigma_Y [arcmin^2] by the 1D radial integral (used only for tables / cross-check of the 2D route)."""
    th = np.atleast_1d(np.asarray(theta_am, float)) * AM
    w = trap_w(ell) * ell / (2 * np.pi) * invN
    out = np.array([np.sum(w * g_u(ell * t) ** 2) for t in th])
    return 1.0 / np.sqrt(out) / AM ** 2


# ----------------------------------------------------------------------------- 2D l-grid sums
class LGrid2D:
    """Half-plane 2D l-grid (rfft layout) restricted to lmin <= |l| <= lmax, spacing dl.
    Sums of radially symmetric integrands: S = (dl^2 / (2 pi)^2) sum_modes w F(|l|) approximates
    int d^2l/(2pi)^2 F."""

    def __init__(self, dl=10.0, lmin=100.0, lmax=20000.0):
        n = int(np.ceil(lmax / dl)) + 1
        kx = np.arange(0, n)
        ky = np.arange(-n, n + 1)
        LX, LY = np.meshgrid(kx * dl, ky * dl, indexing="ij")
        L = np.hypot(LX, LY)
        sel = (L >= lmin) & (L <= lmax)
        self.lx = LX[sel]; self.ly = LY[sel]; self.l = L[sel]
        # rfft half plane: kx = 0 column counts once but contains both +ky and -ky
        self.w = np.where(self.lx == 0, 1.0, 2.0)
        self.norm = dl * dl / (2 * np.pi) ** 2
        self.dl, self.lmin, self.lmax = dl, lmin, lmax

    def sum(self, F):
        return self.norm * np.sum(self.w * F)


# ----------------------------------------------------------------------------- scaling relation (own code)
_COS = None


def cosmo():
    global _COS
    if _COS is None:
        from colossus.cosmology import cosmology
        _COS = cosmology.setCosmology("planck18")
    return _COS


def DA_Mpc(z):
    c = cosmo()
    return c.angularDiameterDistance(np.asarray(z, float)) / c.h


def Y500_med_am2(M, z):
    """Planck 2013 XX, (1-b)=0.8, no h70 factor; median Y500 [arcmin^2]."""
    c = cosmo()
    E = c.Ez(np.asarray(z, float))
    YD2 = 1e-4 * 10 ** (-0.19) * E ** (2.0 / 3.0) * (0.8 * np.asarray(M, float) / 6e14) ** 1.79   # Mpc^2
    return YD2 / DA_Mpc(z) ** 2 / AM ** 2


def theta500_am(M, z):
    c = cosmo()
    rho_c = c.rho_c(np.asarray(z, float)) * 1e9 * c.h ** 2          # Msun / Mpc^3
    R = (3 * np.asarray(M, float) / (4 * np.pi * 500 * rho_c)) ** (1 / 3)
    return R / DA_Mpc(z) / AM


def set_threads(n=4):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)


# ----------------------------------------------------------------------------- memory-safe helpers (5 GB budget)
def inv_Ny_chunked(ell, chunk=1_000_000, **kw):
    ell = np.asarray(ell, float)
    out = np.empty_like(ell)
    for a in range(0, ell.size, chunk):
        out[a:a + chunk] = inv_Ny(ell[a:a + chunk], **kw)
    return out


def grid_modes_chunks(dl, lmin, lmax, rows=200):
    """Yield (lx, ly, l, w) for the half-plane 2D l-grid in chunks of kx rows (never the full grid)."""
    n = int(np.ceil(lmax / dl)) + 1
    ky = np.arange(-n, n + 1) * dl
    for k0 in range(0, n, rows):
        kx = np.arange(k0, min(n, k0 + rows)) * dl
        LX, LY = np.meshgrid(kx, ky, indexing="ij")
        L = np.hypot(LX, LY)
        sel = (L >= lmin) & (L <= lmax)
        lx = LX[sel]
        yield lx, LY[sel], L[sel], np.where(lx == 0, 1.0, 2.0)


class LGridRadial:
    """The same half-plane 2D l-grid as LGrid2D, with identical terms grouped: modes with the same integer
    i^2 + j^2 have the same |l| and therefore identical summands for radially symmetric integrands.
    sum(F) = (dl^2/(2pi)^2) sum_u m_u F(l_u) is EXACTLY the 2D grid sum (only the summation order differs)."""

    def __init__(self, dl=10.0, lmin=100.0, lmax=20000.0):
        n = int(np.ceil(lmax / dl)) + 1
        mult = {}
        k2 = []
        cnt = []
        for k0 in range(0, n, 250):
            kx = np.arange(k0, min(n, k0 + 250))
            ky = np.arange(-n, n + 1)
            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            r2 = (KX.astype(np.int64) ** 2 + KY.astype(np.int64) ** 2)
            L = np.sqrt(r2) * dl
            sel = (L >= lmin) & (L <= lmax)
            w = np.where(KX[sel] == 0, 1.0, 2.0)
            k2.append(r2[sel]); cnt.append(w)
        k2 = np.concatenate(k2); cnt = np.concatenate(cnt)
        u, inv = np.unique(k2, return_inverse=True)
        self.m = np.bincount(inv, weights=cnt)
        self.l = np.sqrt(u.astype(float)) * dl
        self.norm = dl * dl / (2 * np.pi) ** 2
        self.nmodes = len(k2)
        self.dl, self.lmin, self.lmax = dl, lmin, lmax

    def sum(self, F):
        return self.norm * np.sum(self.m * F)
