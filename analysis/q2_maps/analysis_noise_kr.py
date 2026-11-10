"""Kac-Rice expected density of local maxima above nu for the unit-variance filtered noise field at scale k."""
import numpy as np
import q2lib as L
from szsim import signal as sg

lg = np.linspace(0, 12 / L.SIGB, 200001)

def moments(k):
    T2 = (sg.template_hankel(lg * L.THETA_K[k]) * np.exp(-0.5 * (lg * L.SIGB) ** 2)) ** 2
    s = [np.trapezoid(lg ** (1 + 2 * j) * T2, lg) / (2 * np.pi) for j in range(3)]
    return s[1] / (2 * s[0]), s[2] / s[0]

def kr_counts(k, nus, nmc=400000, seed=1):
    lam, s2 = moments(k)
    rng = np.random.default_rng(seed)
    nug = np.linspace(min(nus), 9.0, 241)
    a, b = 3 * s2 / 8, s2 / 8
    C = np.array([[a - lam ** 2, b - lam ** 2], [b - lam ** 2, a - lam ** 2]])
    z = rng.standard_normal((nmc, 2)) @ np.linalg.cholesky(C).T; h12 = rng.standard_normal(nmc) * np.sqrt(b)
    dens = []
    for nu in nug:
        H11 = -lam * nu + z[:, 0]; H22 = -lam * nu + z[:, 1]
        det = H11 * H22 - h12 ** 2
        dens.append(np.mean(det * ((H11 < 0) & (det > 0))) * np.exp(-nu ** 2 / 2) / np.sqrt(2 * np.pi) / (2 * np.pi * lam))
    dens = np.array(dens)
    return np.array([np.trapezoid(dens[nug >= nu], nug[nug >= nu]) for nu in nus])
