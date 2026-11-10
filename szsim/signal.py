"""SZ signal model: Y500-M500 scaling relation, Arnaud+10 GNFW template, matched-filter noise.

Conventions
-----------
* M500 in Msun (h-free); Y500 = spherical Compton-Y inside R500, returned in arcmin^2;
  theta500 = R500 / D_A in arcmin.
* Scaling relation (Planck 2013 XX, used exactly as specified, i.e. WITHOUT the
  (h/0.7)^(-2+alpha) factor unless ``h70_factor=True``):
      E(z)^(-2/3) D_A^2 Y500 / (1e-4 Mpc^2) = 10^-0.19 [ (1-b) M500 / 6e14 Msun ]^1.79
  The relation gives the MEDIAN Y500 (= exp<lnY>); intrinsic scatter is lognormal,
  sigma_lnY = 0.173.
* Template: Arnaud+10 universal GNFW pressure profile, truncated at 5 R500 (3D sphere),
  projected along the line of sight.  NORMALISATION CHOICE: the template is scaled so that
  the *spherical* Y inside R500 equals Y500; therefore the total (cylindrical = spherical,
  because of the 5R500 truncation) integral of the template is
      int tau d^2theta = Y5R500/Y500 = RATIO_5R500 (~1.8, computed numerically).
  i.e. y(theta) = Y500 * tau(theta; theta500).
* Instrument: Gaussian beam (FWHM 1.4 arcmin), white noise in y of 4e-6 arcmin
  (N_l = (4e-6 arcmin)^2 converted to sr, flat).
* Matched filter (flat sky, Hankel transform; no CMB/foregrounds, so the filter is the
  beam-convolved template itself):
      sigma_Y^-2(theta500) = int d^2l/(2pi)^2 |B_l tau_l|^2 / N_l
                           = (1/2pi) int dl l B_l^2 g(l theta500)^2 / N_w ,
  with g(u) = 2 pi int_0^5 s ds f(s) J0(u s) the Hankel transform of the unit-Y500
  profile shape (g(0) = RATIO_5R500).
* Mean SNR q = Y500 / sigma_Y(theta500); observed q_obs = q + N(0,1).
"""
import os
import numpy as np
from scipy.special import j0, ndtr
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from . import cosmo_hmf as ch

# Arnaud+10 GNFW shape parameters
GNFW = dict(c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
XTRUNC = 5.0  # truncation radius in units of R500


def gnfw_P(x, c500=GNFW["c500"], gamma=GNFW["gamma"], alpha=GNFW["alpha"], beta=GNFW["beta"]):
    """Dimensionless GNFW pressure shape P(x), x = r/R500 (P0 dropped)."""
    cx = c500 * np.asarray(x, dtype=float)
    return 1.0 / (cx ** gamma * (1.0 + cx ** alpha) ** ((beta - gamma) / alpha))


def Ysph_shape(X):
    """int_0^X 4 pi x^2 P(x) dx (shape units)."""
    return quad(lambda x: 4 * np.pi * x * x * gnfw_P(x), 0, X, limit=400, epsabs=0, epsrel=1e-11)[0]


RATIO_5R500 = Ysph_shape(XTRUNC) / Ysph_shape(1.0)


def projected_shape(s):
    """Line-of-sight projection of P truncated at XTRUNC: f(s) = 2 int_0^sqrt(X^2-s^2) P(sqrt(s^2+l^2)) dl,
    normalised to the spherical Y500, i.e. int_0^X 2 pi s f(s) ds = RATIO_5R500."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.zeros_like(s)
    norm = Ysph_shape(1.0)
    for i, si in enumerate(s):
        if si >= XTRUNC:
            continue
        L = np.sqrt(XTRUNC ** 2 - si ** 2)
        if si > 0:
            out[i] = 2 * quad(lambda l: gnfw_P(np.sqrt(si * si + l * l)), 0, L, limit=400, epsrel=1e-10)[0]
        else:
            out[i] = 2 * quad(lambda l: gnfw_P(l), 0, L, limit=400, epsrel=1e-10)[0]
    return out / norm


_HANKEL_CACHE = os.path.join(ch.DATA_DIR, "template_hankel_arnaud_5R500.npz")
_HANKEL_MEM = None


def template_real(theta, theta500):
    """Unit-Y500 template tau(theta) [arcmin^-2] (NOT beam-convolved): y(theta) = Y500 * tau.
    theta, theta500 in arcmin; integrates to RATIO_5R500 over the disc of radius 5 theta500."""
    template_hankel()  # make sure the cache (incl. the projected profile) exists
    d = np.load(_HANKEL_CACHE)
    th5 = np.asarray(theta500, dtype=float)
    return np.interp(np.asarray(theta, dtype=float) / th5, d["sn"], d["fn"], right=0.0) / th5 ** 2


def template_hankel(u=None, cache=True):
    """g(u): 2D Fourier (Hankel) transform of the unit-Y500 template, u = l * theta500 [rad].
    Returns (u_grid, g_grid) if u is None, else interpolated g(u)."""
    global _HANKEL_MEM
    if _HANKEL_MEM is not None:
        ug, gg = _HANKEL_MEM
    elif cache and os.path.exists(_HANKEL_CACHE):
        d = np.load(_HANKEL_CACHE)
        ug, gg = d["u"], d["g"]
        _HANKEL_MEM = (ug, gg)
    else:
        # profile on a fine grid (cusp at 0, sqrt edge at X): tabulate f on nodes then spline in s
        sn = np.concatenate([[0.0], np.geomspace(1e-5, XTRUNC * (1 - 1e-9), 1500)])
        fn = projected_shape(sn)
        # fine quadrature grid for the oscillatory integral
        ns = 400001
        t = np.linspace(0, 1, ns)
        s = XTRUNC * (1 - (1 - t) ** 2)  # cluster nodes near the sqrt edge at s=X
        ds_dt = XTRUNC * 2 * (1 - t)
        fs = np.interp(s, sn, fn)
        w = 2 * np.pi * s * fs * ds_dt
        dt = t[1] - t[0]
        ug = np.concatenate([[0.0], np.geomspace(1e-3, 3e3, 700)])
        gg = np.empty_like(ug)
        for k, uk in enumerate(ug):
            integrand = w * j0(uk * s)
            gg[k] = dt * (integrand.sum() - 0.5 * (integrand[0] + integrand[-1]))
        if cache:
            os.makedirs(ch.DATA_DIR, exist_ok=True)
            np.savez(_HANKEL_CACHE, u=ug, g=gg, ratio=RATIO_5R500, sn=sn, fn=fn)
    if u is None:
        return ug, gg
    u = np.asarray(u, dtype=float)
    return np.interp(u, ug, gg, right=0.0)


class SZModel:
    """Scaling relation + instrument + matched filter.  All defaults = baseline model."""

    def __init__(self, logA=-0.19, alpha=1.79, beta_E=2.0 / 3.0, one_minus_b=0.8, M_pivot=6e14,
                 sigma_lnY=0.173, h70_factor=False, beam_fwhm_arcmin=1.4, noise_y_arcmin=4e-6):
        self.logA, self.alpha, self.beta_E = logA, alpha, beta_E
        self.one_minus_b, self.M_pivot, self.sigma_lnY = one_minus_b, M_pivot, sigma_lnY
        self.h70_factor = h70_factor
        self.beam_fwhm_arcmin, self.noise_y_arcmin = beam_fwhm_arcmin, noise_y_arcmin
        self._sigY_spline = None

    # ---------------- scaling relation ----------------
    def Y500_Mpc2_med(self, M500, z):
        """Median Y500 * D_A^2 [Mpc^2]."""
        A = 1e-4 * 10 ** self.logA
        if self.h70_factor:
            A *= (ch.h() / 0.7) ** (-2 + self.alpha)
        return A * ch.Ez(z) ** self.beta_E * (self.one_minus_b * np.asarray(M500) / self.M_pivot) ** self.alpha

    def Y500_med(self, M500, z):
        """Median Y500 [arcmin^2]."""
        return self.Y500_Mpc2_med(M500, z) / ch.DA_Mpc(z) ** 2 / ch.ARCMIN ** 2

    @staticmethod
    def R500_Mpc(M500, z):
        return (3 * np.asarray(M500) / (4 * np.pi * 500 * ch.rho_crit_Msun_Mpc3(z))) ** (1.0 / 3.0)

    def theta500(self, M500, z):
        """theta500 = R500/D_A [arcmin]."""
        return self.R500_Mpc(M500, z) / ch.DA_Mpc(z) / ch.ARCMIN

    # ---------------- matched filter ----------------
    @property
    def sigma_beam_rad(self):
        return self.beam_fwhm_arcmin * ch.ARCMIN / np.sqrt(8 * np.log(2))

    @property
    def N_white_sr(self):
        return (self.noise_y_arcmin * ch.ARCMIN) ** 2

    def sigma_Y_exact(self, theta500_arcmin, lmax=None, nl=20000):
        """sigma_Y [arcmin^2] by direct l-integration (vectorised over theta500)."""
        th = np.atleast_1d(np.asarray(theta500_arcmin, dtype=float)) * ch.ARCMIN
        sb = self.sigma_beam_rad
        ug, _ = template_hankel()
        out = np.empty_like(th)
        for i, t in enumerate(th):
            # B_l^2 = exp(-81) at 9/sigma_b; g(u) tabulated to u=ug[-1] (template ~fully resolved there)
            lm = min(9.0 / sb, ug[-1] / t) if lmax is None else lmax
            l = np.linspace(0, lm, nl)
            B2 = np.exp(-(l * sb) ** 2)
            g = template_hankel(l * t)
            out[i] = np.trapezoid(l * B2 * g * g, l) / (2 * np.pi) / self.N_white_sr
        return (1.0 / np.sqrt(out)) / ch.ARCMIN ** 2

    def _build_sigY(self):
        tg = np.geomspace(1e-3, 3e3, 300)
        self._sigY_tab = (tg, self.sigma_Y_exact(tg))
        self._sigY_spline = CubicSpline(np.log(tg), np.log(self._sigY_tab[1]))

    def sigma_Y(self, theta500_arcmin):
        """Interpolated matched-filter sigma_Y(theta500) [arcmin^2] (log-log cubic spline)."""
        if self._sigY_spline is None:
            self._build_sigY()
        tg = self._sigY_tab[0]
        lt = np.log(np.clip(np.asarray(theta500_arcmin, dtype=float), tg[0], tg[-1]))
        return np.exp(self._sigY_spline(lt))

    def cross_response(self, theta_A, theta_B, sep_arcmin, nl=20000):
        """Matched-filter response: expected Y_hat_A (filter tuned to theta_A, centred on A) produced by
        a cluster of unit Y500 with size theta_B at separation sep [arcmin]:
            rho = sigma_Y(A)^2 * (1/2pi) int dl l B_l^2 g(l thA) g(l thB) J0(l sep) / N_w .
        rho(thA, thA, 0) = 1.  Blended estimate: Y_hat_A = Y_A + sum_B rho_AB * Y_B (+ noise).
        Scalar theta_A, theta_B; sep may be an array."""
        from scipy.special import j0 as _j0
        sb = self.sigma_beam_rad
        ug, _ = template_hankel()
        tA, tB = theta_A * ch.ARCMIN, theta_B * ch.ARCMIN
        lm = min(9.0 / sb, ug[-1] / max(tA, tB))
        l = np.linspace(0, lm, nl)
        w = l * np.exp(-(l * sb) ** 2) * template_hankel(l * tA) * template_hankel(l * tB)
        sep = np.atleast_1d(np.asarray(sep_arcmin, dtype=float)) * ch.ARCMIN
        I = np.trapezoid(w[None, :] * _j0(np.outer(sep, l)), l, axis=1) / (2 * np.pi) / self.N_white_sr
        sA = self.sigma_Y_exact(theta_A, nl=nl)[0] * ch.ARCMIN ** 2
        return I * sA ** 2

    @property
    def sigma_Y_point(self):
        """Point-source limit: sqrt(4 pi sigma_b^2 N_w) / RATIO_5R500 [arcmin^2]."""
        return np.sqrt(4 * np.pi * self.sigma_beam_rad ** 2 * self.N_white_sr) / RATIO_5R500 / ch.ARCMIN ** 2

    def q_mean(self, M500, z, lnY_dev=0.0):
        """Mean matched-filter SNR for a halo with lnY = lnY_med + lnY_dev."""
        return self.Y500_med(M500, z) * np.exp(lnY_dev) / self.sigma_Y(self.theta500(M500, z))

    # ---------------- selection ----------------
    def p_det(self, M500, z, q_cut, n_gh=48):
        """P(q_obs > q_cut | M500, z), marginalised over lognormal Y scatter and unit Gaussian noise."""
        x, w = np.polynomial.hermite_e.hermegauss(n_gh)
        w = w / w.sum()
        q0 = self.q_mean(M500, z)
        q0 = np.asarray(q0)[..., None] * np.exp(self.sigma_lnY * x)
        return (ndtr(q0 - q_cut) * w).sum(-1)

    def expected_detections(self, q_cut, table=None, Mmin=None, Mmax=None, zmin=None, zmax=None):
        """Full-sky expected number of detections with q_obs > q_cut."""
        table = table or ch.default_table()
        M = np.exp(table.lnM)
        lnMlo = np.log(Mmin) if Mmin else table.lnM[0]
        lnMhi = np.log(Mmax) if Mmax else table.lnM[-1]
        wM = table._weights(table.lnM, lnMlo, lnMhi)
        wz = table._weights(table.z, zmin or table.zmin, zmax or table.zmax)
        q_cut = np.atleast_1d(q_cut)
        out = []
        for qc in q_cut:
            P = np.array([self.p_det(M, zz, qc) for zz in table.z])
            out.append(float(wz @ (table.d2N * P) @ wM))
        return np.array(out) if len(out) > 1 else out[0]
