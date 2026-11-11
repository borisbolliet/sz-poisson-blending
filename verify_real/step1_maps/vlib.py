"""Independent MAP-LEVEL verification of realistic step 1 (SO LAT multi-frequency matched filter): shared code.

Everything that GENERATES the simulated sky is written here from scratch (not imported from the build):
  * SO LAT channels and noise typed from Ade et al. 2019 (arXiv:1808.07445): Table 1 (freq, FWHM, white
    baseline/goal), Table 2 / Eq. 1 (temperature N_red, l_knee = 1000, alpha = -3.5), Sec. 2.2 (5 yr, 0.2 x 0.85,
    f_sky = 0.4, atmospheric correlation 0.9 inside an optics tube).
  * Atmosphere correlated inside each tube through a COMMON-MODE + independent decomposition
    n_a = sqrt(r N_a) c + sqrt((1-r) N_a) e_a  (the build's s04 used a Cholesky form instead).
  * White noise generated in PIXEL space; CMB = Gaussian field from OUR OWN CAMB run (v01_camb.py).
  * tSZ SED written as x (e^x+1)/(e^x-1) - 4 (non-relativistic, band centres).
  * Arnaud+10 GNFW clusters truncated at 5 R500 and normalised to the spherical Y500 (the project convention):
    real-space line-of-sight projection + Gaussian (I0-kernel) pre-smoothing for painting (v02), and an
    independent Fourier template through the 3D projection-slice theorem (sinc transform of the 3D profile).
  * Poisson halo population drawn per periodic flat patch from the HMF table (own sampler; only the tabulated
    Tinker08 d2N/dz/dlnM and the project's Y-M relation are shared, they are not claims under test).
The build's szsim.instrument is imported ONLY for the filter: N_y(l), ILC weights w_i(l) and sigma_Y(theta)
(MMFModel.sigma_Y_exact).  The filter template is the build's szsim.signal.template_hankel.

Conventions: maps (n, n) indexed [iy, ix], pixel j centred at (j + 1/2) pix; rfft2 layout (n, n//2+1);
l in rad^-1; angles in arcmin at the interface; Y500 in arcmin^2; y dimensionless; channel maps in uK_CMB.
"""
import os
NTHREADS = 4
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(NTHREADS)
import numpy as np
import scipy.fft as sfft
from scipy.integrate import quad
from scipy.special import i0e, j0
import numba as nb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ARCMIN = np.pi / 10800.0
TCMB = 2.7255
H_PL, K_B = 6.62607015e-34, 1.380649e-23

# ----------------------------------------------------------------------------------- SO LAT (paper numbers)
FREQ = np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0])
FWHM = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
WHITE = {"baseline": np.array([71.0, 36.0, 8.0, 10.0, 22.0, 54.0]),
         "goal": np.array([52.0, 27.0, 5.8, 6.3, 15.0, 37.0])}
NRED = np.array([100.0, 39.0, 230.0, 1500.0, 17000.0, 31000.0])      # uK^2 s
ELL_KNEE, ALPHA_KNEE = 1000.0, -3.5
TUBES = ((0, 1), (2, 3), (4, 5))
R_ATM = 0.9
FSKY = 0.4
T_EFF = 5.0 * 365.25 * 86400.0 * 0.2 * 0.85


def sed_uK(nu_GHz):
    """Delta T [uK_CMB] per unit y (non-relativistic tSZ)."""
    x = H_PL * np.asarray(nu_GHz, float) * 1e9 / (K_B * TCMB)
    return TCMB * 1e6 * (x * (np.exp(x) + 1.0) / np.expm1(x) - 4.0)


def beam(ell, fwhm_arcmin):
    s = fwhm_arcmin * ARCMIN / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * (np.asarray(ell, float) * s) ** 2)


def red_power(ell, i):
    """Atmospheric (red) noise power of channel i [uK^2 sr]."""
    ell = np.asarray(ell, float)
    amp = NRED[i] * 4.0 * np.pi * FSKY / T_EFF
    out = np.zeros_like(ell)
    pos = ell > 0
    out[pos] = amp * (ell[pos] / ELL_KNEE) ** ALPHA_KNEE
    return out


def white_power(mode, i):
    return (WHITE[mode][i] * ARCMIN) ** 2                                  # uK^2 sr


def channel_cov(ell, mode, cl_cmb):
    """Total per-mode covariance C_ij(l) of the 6 beam-convolved channel maps [uK^2 sr], shape (nl, 6, 6)."""
    ell = np.asarray(ell, float)
    B = np.array([beam(ell, f) for f in FWHM])                           # (6, nl)
    C = B.T[:, :, None] * B.T[:, None, :] * cl_cmb[:, None, None]
    red = np.array([red_power(ell, i) for i in range(6)])
    for i in range(6):
        C[:, i, i] += white_power(mode, i) + red[i]
    for a, b in TUBES:
        x = R_ATM * np.sqrt(red[a] * red[b])
        C[:, a, b] += x
        C[:, b, a] += x
    return C


# ----------------------------------------------------------------------------------- CMB (own CAMB run)
class CMB:
    def __init__(self, path=os.path.join(OUT, "cmb_camb_mine.npz")):
        d = np.load(path)
        self.ell, self.cl = d["ell"].astype(float), d["cl_uK2"]
        self.lcut, self.slope, self.lnamp = float(d["lcut"]), float(d["slope"]), float(d["lnamp"])

    def __call__(self, ell):
        ell = np.asarray(ell, float)
        out = np.interp(ell, self.ell, self.cl)
        hi = ell > self.lcut
        out[hi] = np.exp(self.lnamp + self.slope * np.log(ell[hi]))
        out[ell < 2] = 0.0
        return out


# ----------------------------------------------------------------------------------- flat periodic grid
class Grid:
    def __init__(self, n=2400, pix=0.25):
        self.n, self.pix, self.L = n, pix, n * pix
        self.omega = (pix * ARCMIN) ** 2
        self.ly = 2 * np.pi * np.fft.fftfreq(n, d=pix * ARCMIN)
        self.lx = 2 * np.pi * np.fft.rfftfreq(n, d=pix * ARCMIN)
        self.ell = np.hypot(self.ly[:, None], self.lx[None, :])
        m = np.full(self.lx.size, 2.0)
        m[0] = 1.0
        if n % 2 == 0:
            m[-1] = 1.0
        self.mult = np.broadcast_to(m[None, :], self.ell.shape)          # rfft half-plane multiplicity
        self.area = n * n * self.omega                                    # sr

    def rfft(self, m):
        return sfft.rfft2(m, workers=NTHREADS)

    def irfft(self, k):
        return sfft.irfft2(k, s=(self.n, self.n), workers=NTHREADS)

    def modesum(self, a):
        """(1/A) sum over all 2D modes of a(l) (rfft layout, Hermitian partners counted) = int d^2l/(2pi)^2 a."""
        return float((a * self.mult).sum() / self.area)


class SkySim:
    """Gaussian CMB + white (pixel space) + atmosphere for the 6 SO LAT channels, returned as rfft2 arrays [uK]."""

    def __init__(self, grid, cmb):
        self.g = grid
        L = grid.ell
        self.B = np.stack([beam(L, f) for f in FWHM])                     # (6, n, n/2+1)
        self.sq_cmb = np.sqrt(cmb(L.ravel()).reshape(L.shape) / grid.omega)
        self.sq_red = np.stack([np.sqrt(red_power(L, i) / grid.omega) for i in range(6)])

    def realise(self, rng, modes=("baseline", "goal"), cmb=True, atm=True, white=True):
        """Dict mode -> (6, n, n/2+1) complex: same CMB and atmosphere for every mode, white noise from the same
        unit fields scaled by each mode's level (baseline/goal differ only in white levels)."""
        g = self.g
        n = g.n
        common = np.zeros((6,) + g.ell.shape, complex)
        if cmb:
            c = g.rfft(rng.standard_normal((n, n))) * self.sq_cmb
            common += self.B * c[None]
        if atm:
            for a, b in TUBES:
                cm = g.rfft(rng.standard_normal((n, n)))
                ea = g.rfft(rng.standard_normal((n, n)))
                eb = g.rfft(rng.standard_normal((n, n)))
                common[a] += self.sq_red[a] * (np.sqrt(R_ATM) * cm + np.sqrt(1 - R_ATM) * ea)
                common[b] += self.sq_red[b] * (np.sqrt(R_ATM) * cm + np.sqrt(1 - R_ATM) * eb)
        out = {}
        if white:
            unit = [g.rfft(rng.standard_normal((n, n))) for _ in range(6)]
        for mode in modes:
            X = common.copy()
            if white:
                for i in range(6):
                    X[i] += unit[i] * (WHITE[mode][i] / g.pix)          # uK per pixel = w[uK'] / pix[']
            out[mode] = X
        return out


# ----------------------------------------------------------------------------------- the build's filter
class BuildMMF:
    """The build's MMF from szsim.instrument mapped onto the 2D grid (sharp l cut kept exactly as in the build).
    For a channel data set D_i(l) (rfft of maps in uK):  Y_hat(l) = sigma^2 g(l theta) inv_Ny(l) sum_i w_i(l) D_i(l).
    mode 'white' = the old single-frequency y map (4e-6', 1.4' beam), sigma from szsim.signal.SZModel."""

    def __init__(self, grid, mode):
        from szsim import instrument as ins, signal as sg
        self.g, self.mode, self.sg = grid, mode, sg
        L = grid.ell
        if mode == "white":
            self.inst = ins.Instrument.white_y()
            self.model = sg.SZModel()
            self.inv2d = np.interp(L, self.inst.ell, self.inst.inv_Ny)
            self.w2d = None
        else:
            self.inst = ins.Instrument.so_lat(mode)
            self.model = ins.MMFModel(self.inst)
            lmin, lmax = self.inst.meta["lmin"], self.inst.meta["lmax"]
            sel = (L >= lmin) & (L <= lmax)
            self.inv2d = np.where(sel, np.interp(L, self.inst.ell, self.inst.inv_Ny), 0.0)
            self.w2d = np.stack([np.where(sel, np.interp(L, self.inst.ell, self.inst.weights[:, i]), 0.0)
                                 for i in range(6)])
        self._gcache = {}

    def sigma(self, theta):
        """Claimed sigma_Y [arcmin^2] (continuum Hankel integral of the build)."""
        return float(self.model.sigma_Y_exact(theta)[0])

    def template(self, theta):
        key = round(float(theta), 10)
        if key not in self._gcache:
            self._gcache[key] = self.sg.template_hankel(self.g.ell * theta * ARCMIN)
        return self._gcache[key]

    def ycomb(self, D):
        """inv_Ny(l) sum_i w_i(l) D_i(l)  (channel data -> beam-deconvolved y estimate x 1/N_y)."""
        if self.mode == "white":
            raise ValueError("white mode has no channel data; use yresp_transfer")
        return self.inv2d * np.einsum("ixy,ixy->xy", self.w2d, D)

    def yresp_transfer(self, extra=None):
        """Transfer T(l) such that Z = T * y(l) for a pure-y sky y(l): inv_Ny * sum_i w_i B_i f_i (MMF, with OUR
        beams and SED) or inv_Ny (white, single y map).  'extra' divides out a painting pre-smoothing kernel."""
        if self.mode == "white":
            T = self.inv2d.copy()
        else:
            a = np.stack([beam(self.g.ell, FWHM[i]) * sed_uK(FREQ[i]) for i in range(6)])
            T = self.inv2d * np.einsum("ixy,ixy->xy", self.w2d, a)
        if extra is not None:
            T = T / extra
        return T

    def filt(self, Z, theta):
        """Y_hat map [arcmin^2] from Z = inv_Ny sum_i w_i D_i (or T y)."""
        s = self.sigma(theta) * ARCMIN ** 2
        return self.g.irfft(Z * (s * s * self.template(theta))) / ARCMIN ** 2

    def sigma_disc(self, theta):
        """Discrete-grid normalisation: (int d^2l/(2pi)^2 g^2 inv_Ny)^-1/2 summed over the grid modes [arcmin^2]."""
        gt = self.template(theta)
        return (1.0 / np.sqrt(self.g.modesum(gt * gt * self.inv2d))) / ARCMIN ** 2


# ----------------------------------------------------------------------------------- Arnaud GNFW (own code)
GNFW = dict(c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)        # Arnaud+10
XT = 5.0


def gnfw_core(x):
    """Smooth part of the GNFW: P(x) = x^-gamma * gnfw_core(x)  (finite at x = 0)."""
    c = GNFW["c500"] * np.asarray(x, float)
    return GNFW["c500"] ** (-GNFW["gamma"]) * (1.0 + c ** GNFW["alpha"]) ** (-(GNFW["beta"] - GNFW["gamma"]) / GNFW["alpha"])


def gnfw(x):
    return np.asarray(x, float) ** (-GNFW["gamma"]) * gnfw_core(x)


def ysph(X):
    return quad(lambda x: 4 * np.pi * x ** (2 - GNFW["gamma"]) * gnfw_core(x), 0.0, X, limit=500, epsabs=0,
                epsrel=1e-12)[0]


YSPH1 = ysph(1.0)
RATIO = ysph(XT) / YSPH1


def g_projslice(u):
    """2D Fourier transform of the unit-Y500 projected template = 3D transform at k_z = 0 (projection-slice):
    g(u) = 4 pi int_0^5 x^2 P(x) sin(u x)/(u x) dx / Y_sph(1),  u = l * theta500 [rad]."""
    u = np.atleast_1d(np.asarray(u, float))
    out = np.empty_like(u)
    for k, uk in enumerate(u):
        if uk < 1e-6:
            out[k] = 4 * np.pi * quad(lambda x: x ** (2 - GNFW["gamma"]) * gnfw_core(x), 0, XT, limit=500,
                                      epsabs=0, epsrel=1e-11)[0]
        elif uk < 2.0:
            out[k] = 4 * np.pi * quad(lambda x: x ** (2 - GNFW["gamma"]) * gnfw_core(x) * np.sinc(uk * x / np.pi),
                                      0, XT, limit=500, epsabs=0, epsrel=1e-11)[0]
        else:
            out[k] = 4 * np.pi / uk * quad(lambda x: x ** (1 - GNFW["gamma"]) * gnfw_core(x), 0, XT, weight="sin",
                                           wvar=uk, limit=2000, epsabs=1e-13, epsrel=1e-10)[0]
    return out / YSPH1


def proj_profile(x):
    """Line-of-sight projection of the 5R500-truncated GNFW, unit spherical Y500: int 2 pi x f dx = RATIO."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.zeros_like(x)
    for k, xk in enumerate(x):
        if xk >= XT:
            continue
        lmax = np.sqrt(XT * XT - xk * xk)
        if xk < 1e-12:
            # weight 'alg' multiplies the integrand by l^-gamma: pass the smooth part gnfw_core(l)
            out[k] = 2 * quad(lambda l: gnfw_core(l), 0, lmax, weight="alg", wvar=(-GNFW["gamma"], 0.0), limit=500,
                              epsrel=1e-11)[0]
        else:
            out[k] = 2 * quad(lambda l: gnfw(np.sqrt(xk * xk + l * l)), 0, lmax, limit=500, epsrel=1e-11)[0]
    return out / YSPH1


# ----------------------------------------------------------------------------------- painting
SIG_S = 0.75 / np.sqrt(8 * np.log(2))       # arcmin: Gaussian pre-smoothing of painted profiles (FWHM 0.75')


@nb.njit(cache=True)
def _smooth_profile(xs, fx, theta, sig, rgrid, out):
    """P_s(r) = int_0^{5 theta} (s/sig^2) f_theta(s) exp(-(r-s)^2/2sig^2) i0e(r s/sig^2) ds on a uniform s grid
    (trapezoid), banded |r - s| < 9 sig; f_theta(s) = f(s/theta)/theta^2 by linear interpolation of (xs, fx).
    i0e is evaluated with a numba-friendly series/asymptotic form."""
    R5 = 5.0 * theta
    h = min(sig / 12.0, theta / 600.0)
    ns = int(np.ceil(R5 / h)) + 1
    h = R5 / (ns - 1)
    s = np.empty(ns)
    f = np.empty(ns)
    for j in range(ns):
        s[j] = j * h
        xv = s[j] / theta
        # linear interpolation in the tabulated projected profile
        lo = 0
        hi = xs.size - 1
        if xv >= xs[hi]:
            f[j] = 0.0
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= xv:
                lo = mid
            else:
                hi = mid
        w = (xv - xs[lo]) / (xs[hi] - xs[lo])
        f[j] = ((1 - w) * fx[lo] + w * fx[hi]) / (theta * theta)
    for k in range(rgrid.size):
        r = rgrid[k]
        j0_ = max(0, int((r - 9.0 * sig) / h))
        j1_ = min(ns - 1, int((r + 9.0 * sig) / h) + 1)
        acc = 0.0
        for j in range(j0_, j1_ + 1):
            z = r * s[j] / (sig * sig)
            # i0e(z) = exp(-z) I0(z)
            if z < 15.0:
                t = 1.0
                term = 1.0
                q = 0.25 * z * z
                for m in range(1, 80):
                    term *= q / (m * m)
                    t += term
                    if term < 1e-17 * t:
                        break
                ie = t * np.exp(-z)
            else:
                iz = 1.0 / z
                ie = (1.0 + iz / 8.0 + 9.0 * iz * iz / 128.0 + 225.0 * iz ** 3 / 3072.0
                      + 11025.0 * iz ** 4 / 98304.0 + 893025.0 * iz ** 5 / 3932160.0) / np.sqrt(2.0 * np.pi * z)
            wgt = h if (j > 0 and j < ns - 1) else 0.5 * h
            d = r - s[j]
            acc += wgt * s[j] / (sig * sig) * f[j] * np.exp(-0.5 * d * d / (sig * sig)) * ie
        out[k] = acc


def rmax_paint(theta):
    return 5.0 * theta + 7.0 * SIG_S


@nb.njit(cache=True, inline="always")
def _tab(tab, rmax, lnth0, dlnth, lnth, r):
    nth = tab.shape[0]
    nr = tab.shape[1]
    t = (lnth - lnth0) / dlnth
    if t < 0.0:
        t = 0.0
    if t > nth - 1.000001:
        t = nth - 1.000001
    i0 = int(t)
    w = t - i0
    v = 0.0
    for a in range(2):
        ia = i0 + a
        wa = w if a == 1 else 1.0 - w
        u = r / rmax[ia] * (nr - 1)
        if u < nr - 1:
            j = int(u)
            fr = u - j
            v += wa * ((1.0 - fr) * tab[ia, j] + fr * tab[ia, j + 1])
    return v


@nb.njit(cache=True)
def paint(m, xs, ys, Ys, ths, tab, rmax, lnth0, dlnth, pix):
    """Add sum_h Y_h P_s(|x - x_h|; theta_h) to the periodic map m (pixel centres at (j+1/2) pix)."""
    n = m.shape[0]
    nth = tab.shape[0]
    for h in range(xs.size):
        lnth = np.log(ths[h])
        t = (lnth - lnth0) / dlnth
        t = min(max(t, 0.0), nth - 1.000001)
        i0 = int(t)
        R = max(rmax[i0], rmax[i0 + 1])
        x = xs[h]
        y = ys[h]
        Y = Ys[h]
        jx0 = int(np.floor((x - R) / pix - 0.5))
        jx1 = int(np.ceil((x + R) / pix - 0.5))
        jy0 = int(np.floor((y - R) / pix - 0.5))
        jy1 = int(np.ceil((y + R) / pix - 0.5))
        for jy in range(jy0, jy1 + 1):
            dy = (jy + 0.5) * pix - y
            iy = jy % n
            for jx in range(jx0, jx1 + 1):
                dx = (jx + 0.5) * pix - x
                r = np.sqrt(dx * dx + dy * dy)
                if r < R:
                    m[iy, jx % n] += Y * _tab(tab, rmax, lnth0, dlnth, lnth, r)


class Painter:
    def __init__(self, path=os.path.join(OUT, "paint_tables.npz")):
        d = np.load(path)
        self.tab, self.rmax = np.ascontiguousarray(d["tab"]), d["rmax"]
        self.lnth0, self.dlnth = float(d["lnth0"]), float(d["dlnth"])
        self.sig_s = float(d["sig_s"])

    def paint(self, grid, xs, ys, Ys, ths, m=None):
        if m is None:
            m = np.zeros((grid.n, grid.n))
        paint(m, np.ascontiguousarray(xs, float), np.ascontiguousarray(ys, float), np.ascontiguousarray(Ys, float),
              np.ascontiguousarray(ths, float), self.tab, self.rmax, self.lnth0, self.dlnth, grid.pix)
        return m

    def presmooth(self, grid):
        """Fourier transfer of the painting pre-smoothing (to be divided out): exp(-l^2 sig_s^2 / 2)."""
        return np.exp(-0.5 * (grid.ell * self.sig_s * ARCMIN) ** 2)


# ----------------------------------------------------------------------------------- histogram of delta
H_NLOG, H_LIN_LO, H_LIN_HI, H_DX = 300, -20.0, 20.0, 0.002
H_NLIN = int(round((H_LIN_HI - H_LIN_LO) / H_DX))
H_NB = 2 * H_NLOG + H_NLIN


def hist_edges():
    neg = -np.logspace(np.log10(20.0) + 3.0, np.log10(20.0), H_NLOG + 1)[:-1]
    lin = np.linspace(H_LIN_LO, H_LIN_HI, H_NLIN + 1)
    pos = np.logspace(np.log10(20.0), np.log10(20.0) + 3.0, H_NLOG + 1)[1:]
    return np.concatenate([neg, lin, pos])          # H_NB + 1 edges


@nb.njit(parallel=True, cache=True)
def hist_accum(v, scale, cnt, s1, nthr):
    """Accumulate x = v*scale into (cnt, s1) [count, sum x] over the hybrid bins; returns (sum x, sum x^2, min, max)."""
    n = v.size
    C = np.zeros((nthr, cnt.size))
    S = np.zeros((nthr, cnt.size))
    m1 = np.zeros(nthr)
    m2 = np.zeros(nthr)
    mn = np.full(nthr, 1e300)
    mx = np.full(nthr, -1e300)
    chunk = (n + nthr - 1) // nthr
    for t in nb.prange(nthr):
        for k in range(t * chunk, min(n, (t + 1) * chunk)):
            x = v[k] * scale
            m1[t] += x
            m2[t] += x * x
            if x < mn[t]:
                mn[t] = x
            if x > mx[t]:
                mx[t] = x
            if x >= -20.0 and x < 20.0:
                b = 300 + int((x + 20.0) / 0.002)
                if b > 300 + 19999:
                    b = 300 + 19999
            elif x >= 20.0:
                b = 300 + 20000 + int(np.log10(x / 20.0) * 100.0)
                if b > 2 * 300 + 20000 - 1:
                    b = 2 * 300 + 20000 - 1
            else:
                b = 299 - int(np.log10(-x / 20.0) * 100.0)
                if b < 0:
                    b = 0
            C[t, b] += 1.0
            S[t, b] += x
    for t in range(nthr):
        cnt += C[t]
        s1 += S[t]
    return m1.sum(), m2.sum(), mn.min(), mx.max()


# ----------------------------------------------------------------------------------- own Poisson patch sampler
class PatchSampler:
    """Poisson halos M500 in [1e13, 1e16], 0.01 < z < 3 on a periodic flat patch of side L [arcmin]:
    cell counts ~ Poisson(lambda_cell * A_patch / 4pi) on the HMF table cells (trapezoid cell integral),
    (z, lnM) uniform inside a cell, positions uniform; Y500 = median (project Y-M) x lognormal(0.173)."""

    def __init__(self, L_arcmin):
        from szsim import cosmo_hmf as ch, signal as sg
        T = ch.default_table()
        self.z, self.lnM, d2N = T.z, T.lnM, T.d2N
        dz, dl = self.z[1] - self.z[0], self.lnM[1] - self.lnM[0]
        cell = 0.25 * (d2N[:-1, :-1] + d2N[1:, :-1] + d2N[:-1, 1:] + d2N[1:, 1:]) * dz * dl
        self.frac = (L_arcmin * ARCMIN) ** 2 / (4 * np.pi)
        self.lam = cell * self.frac
        self.N_expected = float(self.lam.sum())
        self.N_table = T.N_total() * self.frac
        self.dz, self.dl = dz, dl
        self.L = L_arcmin
        self.model = sg.SZModel()
        # thin interpolation tables in z for Y_med(M,z)/M^alpha and theta500/M^(1/3) (both exact power laws in M)
        zt = np.linspace(self.z[0], self.z[-1], 6001)
        m = self.model
        self.zt = zt
        self.ly0 = np.log(m.Y500_med(1e14, zt))
        self.lt0 = np.log(m.theta500(1e14, zt))
        self.alpha = m.alpha
        self.sig = m.sigma_lnY

    def draw(self, rng):
        N = rng.poisson(self.lam)
        iz, im = np.nonzero(N)
        k = N[iz, im]
        iz = np.repeat(iz, k)
        im = np.repeat(im, k)
        n = iz.size
        z = self.z[iz] + rng.random(n) * self.dz
        lnM = self.lnM[im] + rng.random(n) * self.dl
        x = rng.random(n) * self.L
        y = rng.random(n) * self.L
        dlm = lnM - np.log(1e14)
        lY = np.interp(z, self.zt, self.ly0) + self.alpha * dlm + self.sig * rng.standard_normal(n)
        th = np.exp(np.interp(z, self.zt, self.lt0) + dlm / 3.0)
        return dict(x=x, y=y, z=z, M=np.exp(lnM), Y=np.exp(lY), th=th)


class Tee:
    """Print to stdout and a log file."""

    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
