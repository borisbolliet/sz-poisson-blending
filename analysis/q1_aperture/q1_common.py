"""Shared helpers for the Q1 (N clusters in an aperture) analysis.

Conventions
-----------
* angles in arcmin unless stated; ARCMIN = pi/(180*60) rad
* aperture area: exact spherical cap A(theta) = 2 pi (1 - cos theta) [sr]
* surface density Sigma = Nbar / (4 pi) [sr^-1], Nbar = expected full-sky count of the population
* mu(theta) = Sigma * A(theta) = Nbar * (1 - cos theta) / 2
"""
import os
import numpy as np
from scipy.stats import poisson
from scipy.special import gammainc

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
DATA = os.path.join(ROOT, "data", "q1_aperture")
FIG = os.path.join(ROOT, "figures", "q1_aperture")
ARCMIN = np.pi / (180 * 60)
SKY_DEG2 = 4 * np.pi * (180 / np.pi) ** 2

# theta grid used for all aperture statistics (arcmin), 0.5' .. 1 deg
THETAS = np.array([0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60.0])
KMAX = 6          # kNN depth: gives P(N=0..5) and P(N>=6) exactly
ZSLICES = [(0.01, 0.5), (0.5, 1.0), (1.0, 3.0)]
QCUTS = [4, 5, 6]


def cap_area(theta_arcmin):
    """Exact spherical-cap solid angle [sr]."""
    t = np.asarray(theta_arcmin, float) * ARCMIN
    return 2 * np.pi * (1 - np.cos(t))     # numerically fine down to 1e-4 arcmin in float64


def cap_area_stable(theta_arcmin):
    t = np.asarray(theta_arcmin, float) * ARCMIN
    return 4 * np.pi * np.sin(t / 2) ** 2


def flat_area(theta_arcmin):
    t = np.asarray(theta_arcmin, float) * ARCMIN
    return np.pi * t ** 2


def mu_of(Nbar, theta_arcmin, flat=False):
    A = flat_area(theta_arcmin) if flat else cap_area_stable(theta_arcmin)
    return Nbar * A / (4 * np.pi)


def P_N(N, mu):
    return poisson.pmf(N, mu)


def P_geq(N, mu):
    """P(Pois(mu) >= N)."""
    return poisson.sf(N - 1, mu)


def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def random_vec(n, rng):
    z = 2 * rng.random(n) - 1
    phi = 2 * np.pi * rng.random(n)
    s = np.sqrt(1 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


def chord(theta_arcmin):
    return 2 * np.sin(np.asarray(theta_arcmin, float) * ARCMIN / 2)


def chord_to_arcmin(c):
    return 2 * np.arcsin(np.clip(np.asarray(c) / 2, 0, 1)) / ARCMIN


def knn_u(dk, k, Nbar):
    """PIT of the k-th neighbour distance under Poisson: P(d_k <= d) = P(Pois(mu(d)) >= k)."""
    return gammainc(k, mu_of(Nbar, dk))
