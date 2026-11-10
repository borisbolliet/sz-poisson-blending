"""Convergence of the analytic neighbour-rate histogram with the population discretisation
(z/lnM subsampling of the HMF table and number of Gauss-Hermite scatter nodes), at fixed theta_A.
Reports mean, P(delta>t) for the neighbour floor 1e14 and 1e13."""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg

tab = bl.load_rho_table(); m = sg.SZModel()
T_TEST = np.array([0.05, 1.0, 10.0, 20.0, 50.0])
pdc = bl.PD(N=2 ** 20, dd=0.001, x_small=0.002)
for thA_t in (0.5, 1.0, 4.0):
    i = int(np.argmin(np.abs(np.log(tab.th / thA_t)))); thA = tab.th[i]; sA = float(m.sigma_Y(thA))
    for zsub, msub, ngh in ((6, 6, 16), (2, 6, 16), (6, 2, 16), (6, 6, 48), (1, 3, 32)):
        pop = bl.neighbour_population(zsub=zsub, msub=msub, ngh=ngh)
        W, WX, WX2, CAP, CAPX = bl.rate_histogram(thA, sA, pop, tab)
        out = []
        for fl in ("1e13", "1e14"):
            cl = list(bl.FLOORS[fl])
            p = pdc.pdf(W[cl].sum(0), WX[cl].sum(0), WX2[cl].sum(0)) * np.exp(-CAP[cl].sum())
            tail = 1 - np.cumsum(p)
            mu = WX[cl].sum() + CAPX[cl].sum()
            out.append(f"floor {fl}: mean {mu:.5f} " + " ".join(f"P(>{t:g})={np.interp(t, pdc.grid, tail):.4e}" for t in T_TEST))
        print(f"thA={thA:.3f}' zsub={zsub} msub={msub} ngh={ngh} (n_pop={len(pop[0])}): " + " | ".join(out), flush=True)
