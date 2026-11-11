"""Realistic instrument + multi-frequency matched filter (MMF) for the tSZ blending study (step 1).

Sources (fetched 2026-11-11; text extracted from the arXiv PDF and the public SO noise code)
---------------------------------------------------------------------------------------------
[SO19]  The Simons Observatory Collaboration (Ade et al. 2019), "The Simons Observatory: science goals and
        forecasts", JCAP 02 (2019) 056, arXiv:1808.07445.
        * Table 1 (LAT, f_sky = 0.4), temperature white noise for a homogeneous hits map:
              freq [GHz]           27    39    93   145   225   280
              FWHM [arcmin]        7.4   5.1   2.2   1.4   1.0   0.9
              baseline [uK-arcmin] 71    36    8.0   10    22    54
              goal     [uK-arcmin] 52    27    5.8   6.3   15    37
        * Sec. 2.2 / Eq. (1): N_l = N_red (l / l_knee)^alpha_knee + N_white.  LAT temperature: l_knee = 1000,
          alpha_knee = -3.5; Table 2 "LAT Temperature N_red [uK^2 s]": 100, 39, 230, 1500, 17000, 31000.
          The two bands of one optics tube (27/39 LF, 93/145 MF, 225/280 UHF) have atmospheric noise with
          correlation coefficient 0.9 ("we assign it a correlation coefficient of 0.9"); no correlation
          between tubes.  Footnote 7: the paper's forecasts used N_red = 4 instead of 39 uK^2 s at 39 GHz
          (we use 39, as in their corrected Fig. 2).
        * Survey: 5 years, 20% of time used for science, times 0.85 for the discarded noisy map edges.
[SOcode] github.com/simonsobs/so_noise_models, so_models_v3/SO_Noise_Calculator_Public_v3_0_0.py (the code
        released with [SO19]): N_l,red [uK^2 sr] = N_red (l/1000)^-3.5 * A_sky / t_eff with A_sky = 4 pi f_sky
        [sr], t_eff = 5 * 365.25 * 86400 s * 0.2 * 0.85; the code's C/(2 N_tubes) reproduces Table 2
        (100, 38.5, 225, 1500, 17000, 31000; Table 2 rounds 38.5 -> 39 and 225 -> 230).  The atmosphere
        conversion uses A_sky / t_eff, the same dilution as the white noise.
        This module uses the PAPER numbers (Tables 1, 2).  analysis/realistic/step1/s02_noise_check.py
        compares them with an independent re-implementation of the SO code's algorithm (SOLatV3 of
        SO_Noise_Calculator_Public_v3_1_2.py, atmospheric model version 0 = the paper model), built from the
        per-tube detector NETs; the downloaded SO code was read, not executed.  Agreement: white levels within
        the paper's rounding (<= 1.5%), all N_ij(l) within <= 2.9% over l = 2..20000 (max 2.86%: baseline 225 GHz at
        high l, where N_ii -> white and (22 / 22.32 uK-arcmin)^2 - 1 = -2.87%) [arbiter correction: the first check
        stopped at l = 8000 and quoted <= 2.8%].

Choices (stated)
----------------
* Temperature only.  No foregrounds other than primary CMB (no CIB, radio, kSZ, Galaxy): optimistic.
* Atmospheric correlation 0.9 within each optics tube is INCLUDED (atm_corr=0.9; atm_corr=0 to switch off).
* Primary CMB: lensed TT from CAMB at the project cosmology (data_real/cmb_lensed_TT_camb.npz, built by
  analysis/realistic/step1/s01_camb_cmb.py; non-linear lensing with CAMB's default HMcode-2020 ('mead2020');
  numerically converged to 0.2% at l<=6000 and 3.6% at l=20000, where it is < 1e-7 of the beam-deconvolved noise of
  the quietest channel; 4-9e-3 at l=6000).  Model uncertainty of the lensed tail [arbiter, s10 (C)]: halofit
  'takahashi' instead of mead2020 changes C_l by +1.2% at l=4000 and +3-4% at l=5000-8000, CLASS (halofit) by up to
  +5% at l=6000; the resulting change of the MMF sigma_Y is <= 8e-4.  Fully correlated across channels in CMB
  thermodynamic units.
* tSZ SED: non-relativistic, monochromatic at the band centres (no bandpass integration):
      Delta T / T_CMB = g(x) y,  g(x) = x coth(x/2) - 4,  x = h nu / (k T_CMB),  T_CMB = 2.7255 K.
* Beams: Gaussian, B_l = exp(-l^2 sigma_b^2 / 2) (flat sky; same convention as szsim.signal).
* Flat sky, l range [lmin, lmax] with a SHARP cut (default 100..20000).  lmin = 100 is the default; a
  patch-based lmin = 2 pi / 10 deg = 36 is the alternative tested in the scripts.

Matched filter algebra
----------------------
Data per channel i (beam-convolved, uK):   d_i(l) = B_i f_i Y tau(l) + B_i T_cmb(l) + n_i(l),
with f_i = T_CMB[uK] g(x_i).  Covariance  C_ij(l) = B_i B_j C_l^CMB + N_ij(l) (N = white + atmosphere).
MMF (Herranz+2002, Melin+2006):  Psi_i(l) = sigma_Y^2 tau_A(l) [C^-1 a]_i,  a_i = B_i f_i, and
      sigma_Y^-2 = int d^2l/(2pi)^2 |tau_A(l)|^2 a^T C^-1 a .
Everything about the instrument therefore enters through ONE function, the effective y-noise power
      N_y(l) = 1 / (a^T C^-1 a)       [sr, beam-deconvolved, in y units],
and the MMF is the single-map matched filter with white-noise-over-beam N_w/B_l^2 replaced by N_y(l).
Because the y signal of any other cluster B has the same SED, the MMF response to it is
      rho_AB(s) = sigma_A^2 (1/2pi) int dl l g(l thA) g(l thB) J0(l s) / N_y(l)
(same semantics as szsim.signal.SZModel.cross_response; rho(thA, thA, 0) = 1), its response to a uniform
y background is int d^2s rho = sigma_A^2 RATIO_5R500^2 / N_y(l=0) = 0 for lmin > 0 (compensated filter),
and the Poisson confusion variance of Y_hat_A is (Parseval)
      var_conf(A) = sigma_A^4 (1/2pi) int dl l g(l thA)^2 C_l^yy,P / N_y(l)^2,
      C_l^yy,P = sum_B n_B Y_B^2 g(l thB)^2   (Poisson / 1-halo y power of the neighbour population).
Y conventions are those of szsim.signal: the template is the unit-Y500 Arnaud profile truncated at 5 R500
(integral RATIO_5R500), so Y_hat estimates Y500 exactly as SZModel.sigma_Y does.

Units: angles in arcmin at the interface (theta500, sep), radians/sr internally; sigma_Y in arcmin^2.
"""
import os
import json
import numpy as np
from scipy.special import j0

from . import cosmo_hmf as ch
from . import signal as sg

T_CMB = 2.7255                     # K
H_PLANCK = 6.62607015e-34          # J s
K_BOLTZ = 1.380649e-23             # J / K
ARCMIN = ch.ARCMIN
DATA_REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_real")
CMB_FILE = os.path.join(DATA_REAL, "cmb_lensed_TT_camb.npz")

# ----------------------------------------------------------------------------- SO LAT [SO19]
SO_LAT = dict(
    freqs=np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0]),          # GHz, Table 1
    fwhm=np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),                    # arcmin, Table 1
    white={"baseline": np.array([71.0, 36.0, 8.0, 10.0, 22.0, 54.0]),  # uK-arcmin, Table 1
           "goal": np.array([52.0, 27.0, 5.8, 6.3, 15.0, 37.0])},
    N_red=np.array([100.0, 39.0, 230.0, 1500.0, 17000.0, 31000.0]),   # uK^2 s, Table 2 (LAT temperature)
    ell_knee=1000.0, alpha_knee=-3.5,                                 # Sec. 2.2
    tubes=[(0, 1), (2, 3), (4, 5)],                                   # LF, MF, UHF band pairs
    atm_corr=0.9,                                                     # Sec. 2.2
    fsky=0.4, years=5.0, eff=0.2 * 0.85,                              # Sec. 2.2 / SOcode
)


def so_t_eff():
    return SO_LAT["years"] * 365.25 * 86400.0 * SO_LAT["eff"]


def tsz_g(nu_GHz):
    """Non-relativistic tSZ SED in CMB temperature units: Delta T / T_CMB = g(x) y."""
    x = H_PLANCK * np.asarray(nu_GHz, float) * 1e9 / (K_BOLTZ * T_CMB)
    return x / np.tanh(x / 2) - 4.0


def tsz_f_uK(nu_GHz):
    """Delta T [uK_CMB] per unit y."""
    return T_CMB * 1e6 * tsz_g(nu_GHz)


def beam_l(ell, fwhm_arcmin):
    sb = np.asarray(fwhm_arcmin, float) * ARCMIN / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * np.outer(sb ** 2, np.asarray(ell, float) ** 2))   # (nch, nl)


def so_lat_noise(ell, mode="baseline", atm=True, atm_corr=None, channels=None):
    """Instrument noise covariance N_ij(l) [uK^2 sr], NOT beam-deconvolved, shape (nch, nch, nl)."""
    ch_idx = np.arange(6) if channels is None else np.asarray(channels)
    r = SO_LAT["atm_corr"] if atm_corr is None else atm_corr
    ell = np.asarray(ell, float)
    w = SO_LAT["white"][mode] * ARCMIN                                 # uK rad
    N = np.zeros((6, 6, len(ell)))
    for i in range(6):
        N[i, i] += w[i] ** 2
    if atm:
        A = 4 * np.pi * SO_LAT["fsky"]
        with np.errstate(divide="ignore"):
            shape = np.where(ell > 0, (ell / SO_LAT["ell_knee"]) ** SO_LAT["alpha_knee"], 0.0)
        red = SO_LAT["N_red"] * A / so_t_eff()                         # uK^2 sr at l_knee
        for i in range(6):
            N[i, i] += red[i] * shape
        for (i, j) in SO_LAT["tubes"]:
            N[i, j] += r * np.sqrt(red[i] * red[j]) * shape
            N[j, i] += r * np.sqrt(red[i] * red[j]) * shape
    return N[np.ix_(ch_idx, ch_idx)]


# ----------------------------------------------------------------------------- CMB
_CMB = None


def cmb_cl_TT(ell):
    """Lensed CMB TT C_l [uK^2] (CAMB, s01_camb_cmb.py); power-law beyond the stored l (> 22000)."""
    global _CMB
    if _CMB is None:
        if not os.path.exists(CMB_FILE):
            raise FileNotFoundError(f"{CMB_FILE} missing: run analysis/realistic/step1/s01_camb_cmb.py")
        d = np.load(CMB_FILE)
        _CMB = (d["ell"].astype(float), d["cl_uK2"], float(d["ext_slope"]), float(d["ext_lnamp"]), int(d["l_trust"]))
    le, cl, sl, amp, lt = _CMB
    ell = np.asarray(ell, float)
    out = np.interp(ell, le, cl)
    hi = ell > lt
    out[hi] = np.exp(amp + sl * np.log(ell[hi]))
    out[ell < 2] = 0.0
    return out


# ----------------------------------------------------------------------------- instrument
class Instrument:
    """Effective y-noise power N_y(l) on a uniform l grid; 1/N_y = 0 outside [lmin, lmax]."""

    def __init__(self, ell, inv_Ny, name, meta=None):
        self.ell = np.asarray(ell, float)
        self.inv_Ny = np.asarray(inv_Ny, float)
        self.name = name
        self.meta = meta or {}
        self.dl = self.ell[1] - self.ell[0]

    @classmethod
    def so_lat(cls, mode="baseline", lmin=100.0, lmax=20000.0, dl=1.0, cmb=True, atm=True, atm_corr=None,
               channels=None, extra_y_power=None):
        """SO LAT MMF.  extra_y_power: callable l -> C_l^yy [sr] added to the y covariance (e.g. the tSZ
        power itself, if the filter's noise model includes it): N_y -> N_y + C^yy (Sherman-Morrison)."""
        ell = np.arange(0.0, lmax + dl / 2, dl)
        ch_idx = np.arange(6) if channels is None else np.asarray(channels)
        B = beam_l(ell, SO_LAT["fwhm"][ch_idx])                              # (nch, nl)
        f = tsz_f_uK(SO_LAT["freqs"][ch_idx])
        N = so_lat_noise(ell, mode, atm=atm, atm_corr=atm_corr, channels=ch_idx)
        C = N.copy()
        if cmb:
            C += B[:, None, :] * B[None, :, :] * cmb_cl_TT(ell)[None, None, :]
        a = (B * f[:, None]).T                                               # (nl, nch)
        Ct = np.moveaxis(C, 2, 0)                                            # (nl, nch, nch)
        sel = (ell >= lmin) & (ell <= lmax) & (ell > 0)
        inv = np.zeros(len(ell))
        x = np.linalg.solve(Ct[sel], a[sel][..., None])[..., 0]
        inv[sel] = np.einsum("li,li->l", a[sel], x)
        w = np.zeros((len(ell), len(ch_idx)))
        w[sel] = x / inv[sel][:, None]                                       # ILC-like weights (y per uK)
        if extra_y_power is not None:
            Pyy = extra_y_power(ell)
            inv[sel] = 1.0 / (1.0 / inv[sel] + Pyy[sel])
        meta = dict(mode=mode, lmin=lmin, lmax=lmax, cmb=cmb, atm=atm, channels=[float(v) for v in SO_LAT["freqs"][ch_idx]],
                    atm_corr=SO_LAT["atm_corr"] if atm_corr is None else atm_corr, extra_y_power=extra_y_power is not None)
        obj = cls(ell, inv, f"SO-{mode}", meta)
        obj.weights = w
        return obj

    @classmethod
    def white_y(cls, noise_y_arcmin=4e-6, beam_fwhm_arcmin=1.4, lmin=0.0, lmax=None, nl=200001):
        """The project's original single-frequency setup: white y noise, Gaussian beam."""
        sb = beam_fwhm_arcmin * ARCMIN / np.sqrt(8 * np.log(2))
        lmax = 9.0 / sb if lmax is None else lmax
        ell = np.linspace(0.0, lmax, nl)
        Nw = (noise_y_arcmin * ARCMIN) ** 2
        inv = np.exp(-(ell * sb) ** 2) / Nw
        inv[(ell < lmin)] = 0.0
        return cls(ell, inv, "white", dict(noise_y_arcmin=noise_y_arcmin, beam=beam_fwhm_arcmin, lmin=lmin, lmax=lmax))

    def Ny(self, ell=None):
        with np.errstate(divide="ignore"):
            v = 1.0 / self.inv_Ny
        return v if ell is None else np.interp(ell, self.ell, v)

    def equivalent_white_y(self, ell_eval):
        """sqrt(N_y(l) B_1.4'(l)^2) in arcmin: the white y-noise level a 1.4' single-frequency map would need
        to have the same N_y at l (diagnostic only)."""
        sb = 1.4 * ARCMIN / np.sqrt(8 * np.log(2))
        return np.sqrt(self.Ny(ell_eval) * np.exp(-(np.asarray(ell_eval) * sb) ** 2)) / ARCMIN


# ----------------------------------------------------------------------------- MMF model
def _trap(y, x):
    return np.trapezoid(y, x)


class MMFModel(sg.SZModel):
    """SZModel with the matched filter defined by an Instrument (any N_y(l)).  sigma_Y, q_mean, p_det,
    expected_detections are inherited and use the MMF sigma_Y; cross_response is the MMF y-response."""

    def __init__(self, instrument, **kw):
        kw.pop("beam_fwhm_arcmin", None); kw.pop("noise_y_arcmin", None)
        super().__init__(**kw)
        self.inst = instrument

    # the single-map white-noise attributes of SZModel have no meaning here: fail loudly instead of
    # silently returning the old 4e-6 / 1.4' values (e.g. in blendlib.dc_response; use self.dc_response)
    @property
    def N_white_sr(self):
        raise AttributeError("MMFModel has no single white-noise level: use self.inst.Ny(l) / self.dc_response")

    @property
    def sigma_beam_rad(self):
        raise AttributeError("MMFModel has no single beam: the beams are inside self.inst.Ny(l)")

    def _lgrid(self, thmax_rad):
        ug, _ = sg.template_hankel()
        lm = min(self.inst.ell[-1], ug[-1] / thmax_rad)
        k = np.searchsorted(self.inst.ell, lm, side="right")
        return self.inst.ell[:k], self.inst.inv_Ny[:k]

    def sigma_Y_exact(self, theta500_arcmin, lmax=None, nl=None):
        th = np.atleast_1d(np.asarray(theta500_arcmin, dtype=float)) * ARCMIN
        out = np.empty_like(th)
        for i, t in enumerate(th):
            l, w = self._lgrid(t)
            g = sg.template_hankel(l * t)
            out[i] = _trap(l * w * g * g, l) / (2 * np.pi)
        return (1.0 / np.sqrt(out)) / ARCMIN ** 2

    @property
    def sigma_Y_point(self):
        l, w = self.inst.ell, self.inst.inv_Ny
        return 1.0 / np.sqrt(_trap(l * w, l) / (2 * np.pi)) / sg.RATIO_5R500 / ARCMIN ** 2

    def cross_response(self, theta_A, theta_B, sep_arcmin, nl=None, chunk=64):
        tA, tB = theta_A * ARCMIN, theta_B * ARCMIN
        l, w = self._lgrid(max(tA, tB))
        kern = l * w * sg.template_hankel(l * tA) * sg.template_hankel(l * tB)
        sep = np.atleast_1d(np.asarray(sep_arcmin, dtype=float)) * ARCMIN
        I = np.empty(len(sep))
        for a in range(0, len(sep), chunk):
            s = sep[a:a + chunk]
            I[a:a + chunk] = _trap(kern[None, :] * j0(np.outer(s, l)), l) / (2 * np.pi)
        sA = self.sigma_Y_exact(theta_A)[0] * ARCMIN ** 2
        return I * sA ** 2

    def dc_response(self, theta_A):
        """int d^2s rho_AB(s) [arcmin^2] = sigma_A^2 RATIO^2 / N_y(l=0): the Y_hat per unit uniform y."""
        sA = self.sigma_Y_exact(theta_A) * ARCMIN ** 2
        return sA ** 2 * sg.RATIO_5R500 ** 2 * self.inst.inv_Ny[0] / ARCMIN ** 2

    def rho2_area(self, theta_A, theta_B):
        """int d^2s rho_AB(s)^2 [arcmin^2] by Parseval."""
        tA, tB = theta_A * ARCMIN, theta_B * ARCMIN
        l, w = self._lgrid(min(tA, tB))
        gA, gB = sg.template_hankel(l * tA), sg.template_hankel(l * tB)
        sA = self.sigma_Y_exact(theta_A)[0] * ARCMIN ** 2
        return sA ** 4 * _trap(l * (w * gA * gB) ** 2, l) / (2 * np.pi) / ARCMIN ** 2

    def confusion_var(self, theta_A, Cyy):
        """Poisson confusion variance of Y_hat_A [arcmin^4] for a y power C_l^yy [sr] given on self.inst.ell
        (or a callable of l).  Divide by sigma_Y(theta_A)^2 for noise units."""
        th = np.atleast_1d(np.asarray(theta_A, float))
        P = Cyy(self.inst.ell) if callable(Cyy) else np.asarray(Cyy)
        out = np.empty(len(th))
        for i, t in enumerate(th):
            l, w = self._lgrid(t * ARCMIN)
            g = sg.template_hankel(l * t * ARCMIN)
            sA = self.sigma_Y_exact(t)[0] * ARCMIN ** 2
            out[i] = sA ** 4 * _trap(l * (w * g) ** 2 * P[:len(l)], l) / (2 * np.pi)
        return out / ARCMIN ** 4


def poisson_Cyy(ell, thB, YB, wB, chunk=4000):
    """Poisson (1-halo) y power [sr]: C_l = sum_B n_B Y_B^2 g(l thB)^2, with thB [arcmin], YB [arcmin^2] and
    wB = expected number per arcmin^2 (blendlib.neighbour_population convention)."""
    ell = np.asarray(ell, float)
    out = np.zeros(len(ell))
    # bin the population in ln(theta_B) to limit the cost: g(l theta) is smooth in ln theta on a fine grid
    lt = np.log(thB)
    edges = np.linspace(lt.min() - 1e-9, lt.max() + 1e-9, 1201)
    idx = np.digitize(lt, edges) - 1
    S2 = np.bincount(idx, weights=wB * YB ** 2, minlength=len(edges) - 1)        # arcmin^4 / arcmin^2
    S2th = np.bincount(idx, weights=wB * YB ** 2 * lt, minlength=len(edges) - 1)
    nz = S2 > 0
    thc = np.exp(S2th[nz] / S2[nz])                                               # Y^2-weighted mean theta
    for a in range(0, nz.sum(), 50):
        g = sg.template_hankel(np.outer(thc[a:a + 50] * ARCMIN, ell))
        out += (S2[nz][a:a + 50, None] * g * g).sum(0)
    return out * ARCMIN ** 2          # arcmin^4/arcmin^2 = arcmin^2 -> sr
