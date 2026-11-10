"""Sensitivity configurations for the robustness study.

Baseline: beam 1.4', y-noise 4e-6 arcmin, sigma_lnY = 0.173, neighbours M500 > 1e13, zmax = 3, Tinker08.
Each PASS is one family of full-sky draws (one neighbour-sum computation per sky); its CONFIGS are
derived from the same draws:
  * noise x f only rescales sigma_Y (sigma_Y is exactly proportional to the white-noise level, and rho
    does not depend on it), so q -> q/f, delta -> delta/f, mu -> mu/f;
  * the neighbour mass floor keeps only some neighbour mass classes;
  * zmax keeps only candidates and neighbours with z < zmax (a Poisson process restricted to a subset
    is the Poisson process of the restricted intensity).
Planck-like: one 143 GHz-like channel, FWHM 7.3', white noise 33 uK-arcmin (approx. Planck HFI 143 GHz
level), converted to Compton y with the non-relativistic tSZ spectral function; no CMB (as in the
user's homogeneous-noise setup), so this is optimistic for Planck itself.
"""
import numpy as np

T_CMB_UK = 2.7255e6


def y_noise_from_uK(uK_arcmin, nu_GHz):
    x = nu_GHz / 56.7798  # h nu / k T_CMB  (T = 2.7255 K)
    f = x / np.tanh(x / 2) - 4.0
    return uK_arcmin / (T_CMB_UK * abs(f)), f


PLANCK_NOISE_Y, F143 = y_noise_from_uK(33.0, 143.0)

MCLASS_EDGES = [1e13, 3e13, 1e14, np.inf]   # neighbour mass classes
ZCLASS_EDGES = [0.0, 1.0, 2.0, np.inf]      # neighbour z classes
G = 9                                       # group = 3 * mclass + zclass


def groups_for(floor, zmax):
    """Neighbour groups kept for a mass floor and zmax (zmax must be a z-class edge: 1, 2 or 3)."""
    return [3 * m + zc for m in range(3) for zc in range(3)
            if MCLASS_EDGES[m] >= floor * 0.999 and ZCLASS_EDGES[zc] < zmax - 1e-9]


BASE = dict(f=1.0, floor=1e13, zmax=3.0)
PASSES = {
    "fam": dict(beam=1.4, noise=4e-6, sig=0.173, hmf="tinker08", configs={
        "base": BASE,
        "noise_x0.5": dict(BASE, f=0.5),
        "noise_x2": dict(BASE, f=2.0),
        "floor_3e13": dict(BASE, floor=3e13),
        "floor_1e14": dict(BASE, floor=1e14),
        "zmax_1": dict(BASE, zmax=1.0),
        "zmax_2": dict(BASE, zmax=2.0)}),
    "beam5": dict(beam=5.0, noise=4e-6, sig=0.173, hmf="tinker08", configs={"beam_5": BASE}),
    "planck": dict(beam=7.3, noise=PLANCK_NOISE_Y, sig=0.173, hmf="tinker08", configs={"planck_like": BASE}),
    "scat0": dict(beam=1.4, noise=4e-6, sig=0.0, hmf="tinker08", configs={"scatter_0": BASE}),
    "scat03": dict(beam=1.4, noise=4e-6, sig=0.3, hmf="tinker08", configs={"scatter_0.3": BASE}),
    "boc16": dict(beam=1.4, noise=4e-6, sig=0.173, hmf="bocquet16", configs={"hmf_bocquet16": BASE}),
    "des16": dict(beam=1.4, noise=4e-6, sig=0.173, hmf="despali16", configs={"hmf_despali16": BASE}),
}

QCUTS = [4.0, 5.0, 6.0, 7.0]
EPS = [0.05, 0.1, 0.2, 0.5]
THETAS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]   # arcmin, for B1 / Q1
KMAX = 3                                     # P(>= k other detections), k = 1..KMAX
QPRIM = 0.5                                  # primaries: true mean SNR q_A > 0.5 (as in analysis/blending)
NSIDE_JK = 2                                 # 48 jackknife regions per sky
